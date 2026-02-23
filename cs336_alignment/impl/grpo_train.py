import torch
import logging
import json
import copy
import time
from typing import Literal
from pathlib import Path

from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cs336_alignment.impl.mathBaseline import evaluate_vllm
from cs336_alignment.impl.sft_train import init_vllm, load_r1_zero_prompt_template, load_policy_into_vllm_instance
from cs336_alignment.impl.util import get_response_log_probs, \
    compute_group_normalized_rewards, grpo_microbatch_train_step
from torch.nn.utils.rnn import pad_sequence
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

TRAIN_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:1"
EVALUATION_STEP = 10
EVAL_SAMPLING_PARAM = SamplingParams(
        temperature=0.7,  # Lower temperature for more focused outputs
        top_p=0.95,       # Slightly lower top_p
        max_tokens=2048,  # More tokens for reasoning
        stop=["</answer>", "</answer >", "User:", "\n\n\n"],  # Multiple stop sequences
        include_stop_str_in_output=True
    )


class Timer:
    """Accumulating timer for profiling GRPO training phases.

    Tracks per-step totals so that timers called many times within a step
    (e.g. microbatch_forward ×128) are correctly summed rather than
    double-counted with a parent timer.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        # per_step_totals[name] = list of per-step cumulative times
        self.per_step_totals: dict[str, list[float]] = {}
        # _current_step_accum[name] = running total for the current step
        self._current_step_accum: dict[str, float] = {}
        self._start_stack: dict[str, float] = {}

    def start(self, name: str):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self._start_stack[name] = time.perf_counter()

    def stop(self, name: str):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_stack.pop(name)
        self._current_step_accum[name] = self._current_step_accum.get(name, 0.0) + elapsed

    def end_step(self):
        """Flush current step accumulators into per-step history."""
        if not self.enabled:
            return
        for name, total in self._current_step_accum.items():
            self.per_step_totals.setdefault(name, []).append(total)
        self._current_step_accum.clear()

    def log_step_summary(self, step: int):
        if not self.enabled:
            return
        # Flush first so current step data is available
        self.end_step()
        logger.info(f"--- TIMING BREAKDOWN (rollout step {step}) ---")
        total = 0.0
        for name, step_vals in self.per_step_totals.items():
            t = step_vals[-1]  # This step's total
            total += t
            logger.info(f"  {name:40s}: {t:8.3f}s")
        logger.info(f"  {'TOTAL':40s}: {total:8.3f}s")
        logger.info(f"--- END TIMING ---")

    def log_final_summary(self):
        if not self.enabled:
            return
        # Flush any remaining data
        if self._current_step_accum:
            self.end_step()
        logger.info("=" * 70)
        logger.info("FINAL TIMING SUMMARY (cumulative across all rollout steps)")
        logger.info("=" * 70)
        grand_total = 0.0
        rows = []
        for name, step_vals in self.per_step_totals.items():
            cumulative = sum(step_vals)
            avg = cumulative / len(step_vals)
            grand_total += cumulative
            rows.append((name, cumulative, avg, len(step_vals)))
        # Sort by cumulative time descending
        rows.sort(key=lambda r: r[1], reverse=True)
        for name, cumulative, avg, count in rows:
            pct = 100.0 * cumulative / grand_total if grand_total > 0 else 0
            logger.info(f"  {name:40s}: {cumulative:8.3f}s total ({pct:5.1f}%) | {avg:8.3f}s avg | {count} steps")
        logger.info(f"  {'GRAND TOTAL':40s}: {grand_total:8.3f}s")
        logger.info("=" * 70)


def _tokenize_unpacked(prompt_strs: list[str], output_strs: list[str], tokenizer):
    """Tokenize prompt+output pairs WITHOUT padding. Returns per-sample lists."""
    tokenized_prompts = tokenizer(prompt_strs)['input_ids']
    tokenized_outputs = tokenizer(output_strs)['input_ids']
    combined = [torch.tensor(tokenized_prompts[i] + tokenized_outputs[i]) for i in range(len(prompt_strs))]
    prompt_lengths = [len(tokenized_prompts[i]) for i in range(len(prompt_strs))]
    return combined, prompt_lengths


def _pad_microbatch(combined_seqs: list[torch.Tensor], prompt_lengths: list[int], pad_token_id: int, device):
    """Pad a microbatch of sequences and build input_ids, labels, response_mask."""
    padded = pad_sequence(combined_seqs, batch_first=True, padding_value=pad_token_id)
    input_ids = padded[:, :-1].to(device)
    labels = padded[:, 1:].to(device)
    seq_len = labels.shape[1]
    batch_size = len(combined_seqs)

    response_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        # Response tokens start after the prompt (shifted by 1 for next-token prediction)
        resp_start = prompt_lengths[i] - 1
        resp_end = len(combined_seqs[i]) - 1  # exclude padding
        response_mask[i, resp_start:resp_end] = True

    return input_ids, labels, response_mask


def grpo_train( policy: PreTrainedModel,
                optimizer: Optimizer,
                vllm_instance: LLM,
                n_grpo_steps: int = 200,
                advantage_eps: float = 1e-6,
                rollout_batch_size: int = 256,
                group_size: int = 8,
                sampling_temperature: float = 1.0,
                sampling_min_tokens: int = 4, # As in Expiter, disallow empty string responses
                sampling_max_tokens: int = 1024,
                epochs_per_rollout_batch: int = 1, # On-policy
                train_batch_size: int = 256,
                gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100
                clip_range = 0.2,
                loss_type: Literal[
                "no_baseline",
                "reinforce_with_baseline",
                "grpo_clip",
                ] = "reinforce_with_baseline",
                use_std_normalization: bool = True,
                normalize_constant: float | None = None,
                output_dir: str = None,
                lr_tag: str = None,
                debug: bool = False,
                ) -> dict:
    """
    Train policy using GRPO algorithm.

    Returns:
        dict containing:
            - 'eval_history': list of (step, accuracy, format_reward) tuples
            - 'final_accuracy': final validation accuracy
            - 'diverged': bool indicating if training diverged
    """
    timer = Timer(enabled=debug)
    training_start_time = time.perf_counter()

    eval_history = []
    diverged = False
    policy.train()  # Set to training mode
    if torch.cuda.is_available():
        policy = policy.to(TRAIN_DEVICE)

    timer.start("dataset_load")
    ds = load_dataset("gsm8k", "main")
    prompt_template = load_r1_zero_prompt_template()
    valid_prompts = [prompt_template.format(question=example['question']) for example in ds['test']]
    valid_ground_truths = [example['answer'] for example in ds['test']]
    timer.stop("dataset_load")

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    # Format training data with r1_zero template
    raw_questions = [example['question'] for example in ds['train']]
    raw_answers = [example['answer'] for example in ds['train']]


    # Apply r1_zero formatting to prompts only (answers used raw for reward function)
    formatted_prompts = [prompt_template.format(question=q) for q in raw_questions]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,  # Generate group_size outputs per prompt
    )
    global_step = 0

    logger.info(f"Starting GRPO training with {n_grpo_steps} steps")
    logger.info(f"Loss type: {loss_type}, rollout_batch_size: {rollout_batch_size}, group_size: {group_size}")
    logger.info(f"train_batch_size: {train_batch_size}, micro_batch_size: {micro_train_batch_size}, grad_accum_steps: {gradient_accumulation_steps}")
    logger.info(f"Token aggregation: {'masked_normalize(constant=' + str(normalize_constant) + ')' if normalize_constant is not None else 'masked_mean'}")
    if debug:
        logger.info("DEBUG TIMING ENABLED - will log per-phase timings each rollout step")

    for i in range(n_grpo_steps):
        timer.start("load_policy_to_vllm")
        load_policy_into_vllm_instance(policy, vllm_instance)
        timer.stop("load_policy_to_vllm")

        current_batch_prompts = formatted_prompts[i * n_prompts_per_rollout_batch: (i + 1) * n_prompts_per_rollout_batch]
        current_batch_answer = raw_answers[i * n_prompts_per_rollout_batch: (i + 1) * n_prompts_per_rollout_batch]
        repeated_batch_prompt = []
        repeated_ground_truths = []
        for j in range(n_prompts_per_rollout_batch):
            repeated_ground_truths+=[current_batch_answer[j]]* group_size
            repeated_batch_prompt+=[current_batch_prompts[j]]* group_size

        timer.start("vllm_generate")
        vllm_outputs = vllm_instance.generate(current_batch_prompts, sampling_params)
        timer.stop("vllm_generate")

        rollout_responses = []
        for request_output in vllm_outputs:  # One per input prompt
            for completion in request_output.outputs:  # group_size completions
                rollout_responses.append(completion.text)

        # Compute rollout-level entropy and response length for logging
        rollout_token_counts = []
        for request_output in vllm_outputs:
            for completion in request_output.outputs:
                rollout_token_counts.append(len(completion.token_ids))
        rollout_mean_length = sum(rollout_token_counts) / len(rollout_token_counts) if rollout_token_counts else 0.0

        # vLLM provides per-token logprobs if requested; use token_id count as proxy for length.
        # Entropy is computed during the training forward pass below and accumulated.
        rollout_entropy_sum = 0.0
        rollout_entropy_count = 0

        timer.start("compute_rewards")
        advantage, raw_rewards, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, use_std_normalization)
        timer.stop("compute_rewards")

        logger.info(f"Rollout batch {i}: mean_reward={metadata['mean_reward']:.4f}, std_reward={metadata['std_reward']:.4f}")
        advantage = advantage.unsqueeze(-1).to(TRAIN_DEVICE)
        raw_rewards = raw_rewards.unsqueeze(-1).to(TRAIN_DEVICE)

        timer.start("tokenize")
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        combined_seqs, prompt_lengths = _tokenize_unpacked(repeated_batch_prompt, rollout_responses, tokenizer)
        seq_lengths = [len(s) for s in combined_seqs]
        timer.stop("tokenize")

        # Sort by sequence length so similar-length samples are in the same microbatch
        timer.start("sort_by_length")
        sorted_indices = sorted(range(len(combined_seqs)), key=lambda k: seq_lengths[k])
        combined_seqs = [combined_seqs[j] for j in sorted_indices]
        prompt_lengths = [prompt_lengths[j] for j in sorted_indices]
        seq_lengths = [seq_lengths[j] for j in sorted_indices]
        advantage = advantage[sorted_indices]
        raw_rewards = raw_rewards[sorted_indices]
        timer.stop("sort_by_length")

        if debug:
            avg_resp_tokens = sum(seq_lengths[j] - prompt_lengths[j] for j in range(len(seq_lengths))) / len(seq_lengths)
            logger.info(f"  seq lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}")
            logger.info(f"  avg response tokens per sample: {avg_resp_tokens:.1f}")

        # Compute old_log_probs for off-policy GRPO-Clip loss.
        # Always computed; used only when loss_type == "grpo_clip".
        # Stored as a list of per-sample tensors (variable length) so they can be
        # permuted consistently with epoch shuffling.
        timer.start("old_log_probs")
        policy.eval()
        old_log_probs_per_sample = []
        chunk_size = micro_train_batch_size * 4
        with torch.inference_mode():
            for chunk_start in range(0, rollout_batch_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, rollout_batch_size)
                chunk_input_ids, chunk_labels, _ = _pad_microbatch(
                    combined_seqs[chunk_start:chunk_end],
                    prompt_lengths[chunk_start:chunk_end],
                    pad_token_id, TRAIN_DEVICE)
                chunk_log_probs = get_response_log_probs(
                    policy, chunk_input_ids, chunk_labels, False
                )['log_probs']  # (chunk, seq_len)
                # Store each sample's log_probs trimmed to its actual sequence length
                for k in range(chunk_end - chunk_start):
                    actual_len = seq_lengths[chunk_start + k]
                    old_log_probs_per_sample.append(chunk_log_probs[k, :actual_len - 1].cpu())
        timer.stop("old_log_probs")

        policy.train()
        for _ in range(epochs_per_rollout_batch):
            # Shuffle rollout data each epoch for unbiased coverage
            perm = torch.randperm(len(combined_seqs))
            epoch_seqs = [combined_seqs[j] for j in perm]
            epoch_prompt_lengths = [prompt_lengths[j] for j in perm]
            epoch_advantage = advantage[perm]
            epoch_raw_rewards = raw_rewards[perm]
            epoch_old_log_probs = [old_log_probs_per_sample[j] for j in perm]

            # Iterate over the full rollout in train_batch_size chunks,
            # one optimizer step per chunk (gradient_accumulation_steps microbatches each)
            for train_batch_start in range(0, rollout_batch_size, train_batch_size):
                accumulated_loss = 0.0

                for idx, micro_start in enumerate(range(train_batch_start, train_batch_start + train_batch_size, micro_train_batch_size)):
                    # Pad this microbatch independently (only to its own max length)
                    timer.start("microbatch_pad")
                    mb_input_ids, mb_labels, mb_response_mask = _pad_microbatch(
                        epoch_seqs[micro_start:micro_start+micro_train_batch_size],
                        epoch_prompt_lengths[micro_start:micro_start+micro_train_batch_size],
                        pad_token_id, TRAIN_DEVICE)
                    timer.stop("microbatch_pad")

                    timer.start("microbatch_forward")
                    fwd_out = get_response_log_probs(policy, mb_input_ids, mb_labels, False)
                    new_log_prob = fwd_out['log_probs']
                    # Accumulate mean response-token entropy for logging (first epoch, first microbatch only)
                    if _ == 0 and micro_start == train_batch_start:
                        with torch.no_grad():
                            ent_out = get_response_log_probs(policy, mb_input_ids, mb_labels, True)
                            ent = ent_out['token_entropy']
                            mask = mb_response_mask.float()
                            valid_tokens = mask.sum().item()
                            if valid_tokens > 0:
                                rollout_entropy_sum += (ent * mask).sum().item()
                                rollout_entropy_count += int(valid_tokens)
                            del ent_out, ent
                    timer.stop("microbatch_forward")

                    # Align old_log_probs to this microbatch's padded seq_len
                    mb_seq_len = mb_input_ids.shape[1]
                    mb_old_list = epoch_old_log_probs[micro_start:micro_start+micro_train_batch_size]
                    mb_old_log_probs = torch.zeros(len(mb_old_list), mb_seq_len, device=TRAIN_DEVICE)
                    for k, old_lp in enumerate(mb_old_list):
                        copy_len = min(len(old_lp), mb_seq_len)
                        mb_old_log_probs[k, :copy_len] = old_lp[:copy_len].to(TRAIN_DEVICE)

                timer.start("microbatch_loss_backward")
                loss, loss_metadata = grpo_microbatch_train_step(new_log_prob, mb_response_mask,
                                           gradient_accumulation_steps, loss_type, epoch_raw_rewards[micro_start:micro_start+micro_train_batch_size],
                                           epoch_advantage[micro_start:micro_start+micro_train_batch_size],
                                           mb_old_log_probs, clip_range, normalize_constant)
                timer.stop("microbatch_loss_backward")

                accumulated_loss += loss.item()

                # Check for divergence (NaN or very large loss)
                if torch.isnan(loss) or torch.isinf(loss) or abs(loss.item()) > 1e6:
                    logger.warning(f"Training diverged at step {global_step}! Loss: {loss.item()}")
                    diverged = True
                    timer.log_final_summary()
                    return {
                        'eval_history': eval_history,
                        'final_accuracy': eval_history[-1][1] if eval_history else 0.0,
                        'diverged': True
                    }

                if (idx + 1) % gradient_accumulation_steps == 0:
                    timer.start("optimizer_step")
                    clip_grad_norm_(policy.parameters(), 1.0)
                    global_step += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    timer.stop("optimizer_step")
                    accumulated_loss = 0.0

        grpo_step = i + 1
        logger.info(f"GRPO step {grpo_step}: mean_reward={metadata['mean_reward']:.4f}")
        timer.log_step_summary(i)

        if grpo_step % EVALUATION_STEP == 0:
            logger.info(f"Running evaluation at GRPO step {grpo_step}...")
            timer.start("evaluation")
            load_policy_into_vllm_instance(policy, vllm_instance)
            step_mean_entropy = rollout_entropy_sum / rollout_entropy_count if rollout_entropy_count > 0 else None
            eval_metrics = evaluate_vllm(
                vllm_instance, r1_zero_reward_fn, valid_prompts, valid_ground_truths,
                EVAL_SAMPLING_PARAM, step_number=grpo_step, output_dir=output_dir, lr_tag=lr_tag,
                wall_clock_time=time.perf_counter() - training_start_time,
                mean_entropy=step_mean_entropy,
                mean_response_length=rollout_mean_length,
            )
            timer.stop("evaluation")
            logger.info(f"Evaluation results - accuracy: {eval_metrics['accuracy']:.4f}, format_reward: {eval_metrics['format_reward']:.4f}")
            eval_history.append((grpo_step, eval_metrics['accuracy'], eval_metrics['format_reward']))

    # Final evaluation (if last step wasn't already evaluated)
    if n_grpo_steps % EVALUATION_STEP != 0:
        logger.info("Running final evaluation...")
        timer.start("final_evaluation")
        load_policy_into_vllm_instance(policy, vllm_instance)
        final_step_mean_entropy = rollout_entropy_sum / rollout_entropy_count if rollout_entropy_count > 0 else None
        final_metrics = evaluate_vllm(
            vllm_instance, r1_zero_reward_fn, valid_prompts, valid_ground_truths,
            EVAL_SAMPLING_PARAM, step_number=n_grpo_steps, output_dir=output_dir, lr_tag=lr_tag,
            wall_clock_time=time.perf_counter() - training_start_time,
            mean_entropy=final_step_mean_entropy,
            mean_response_length=rollout_mean_length,
        )
        timer.stop("final_evaluation")
        eval_history.append((n_grpo_steps, final_metrics['accuracy'], final_metrics['format_reward']))
        logger.info(f"Final evaluation - accuracy: {final_metrics['accuracy']:.4f}, format_reward: {final_metrics['format_reward']:.4f}")

    timer.log_final_summary()

    return {
        'eval_history': eval_history,
        'final_accuracy': eval_history[-1][1] if eval_history else 0.0,
        'diverged': False
    }


def run_learning_rate_sweep(
    checkpoint_path: str,
    learning_rates: list[float],
    n_grpo_steps: int = 200,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    output_dir: str = "lr_sweep_results",
    debug: bool = False,
    **grpo_kwargs
) -> dict:
    """
    Run GRPO training with multiple learning rates and collect results.

    Args:
        checkpoint_path: Path to the model checkpoint to start from
        learning_rates: List of learning rates to sweep over
        n_grpo_steps: Number of GRPO steps per run
        loss_type: Type of policy gradient loss
        output_dir: Directory to save results
        debug: Enable timing instrumentation
        **grpo_kwargs: Additional arguments to pass to grpo_train

    Returns:
        dict mapping learning_rate -> results dict
    """
    import matplotlib.pyplot as plt
    from transformers import AutoConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Load model once and cache state dict to avoid checkpoint corruption
    logger.info("Loading base model into CPU memory (will clone for each LR)...")
    base_policy = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    base_state_dict = {k: v.cpu().clone() for k, v in base_policy.state_dict().items()}
    model_config = base_policy.config
    del base_policy
    torch.cuda.empty_cache()

    for lr in learning_rates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting sweep with learning_rate={lr}")
        logger.info(f"{'='*60}\n")

        # Create fresh model from cached state dict (no disk access)
        policy = AutoModelForCausalLM.from_config(model_config)
        policy.load_state_dict(base_state_dict)

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        # Initialize vLLM instance (needs checkpoint path for tokenizer/config)
        vllm_instance = init_vllm(checkpoint_path, seed=42, device=VLLM_DEVICE, gpu_memory_utilization=0.85)

        lr_tag = f"{lr:.0e}".replace("+", "").replace("-0", "-")  # e.g., "1e-06"

        try:
            results = grpo_train(
                policy=policy,
                optimizer=optimizer,
                vllm_instance=vllm_instance,
                n_grpo_steps=n_grpo_steps,
                loss_type=loss_type,
                output_dir=output_dir,
                lr_tag=lr_tag,
                debug=debug,
                **grpo_kwargs
            )
            all_results[lr] = results
        except Exception as e:
            logger.error(f"Training failed for lr={lr}: {e}")
            all_results[lr] = {
                'eval_history': [],
                'final_accuracy': 0.0,
                'diverged': True,
                'error': str(e)
            }

        # Clean up to free memory
        del policy, optimizer, vllm_instance
        torch.cuda.empty_cache()

        # Save results incrementally after each learning rate
        results_file = output_path / "sweep_results.json"
        serializable_results = {
            str(lr_key): {
                'eval_history': result['eval_history'],
                'final_accuracy': result['final_accuracy'],
                'diverged': result['diverged']
            }
            for lr_key, result in all_results.items()
        }
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved intermediate results to {results_file}")

    # Save results to JSON
    results_file = output_path / "sweep_results.json"
    serializable_results = {
        str(lr): {
            'eval_history': result['eval_history'],
            'final_accuracy': result['final_accuracy'],
            'diverged': result['diverged']
        }
        for lr, result in all_results.items()
    }
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved sweep results to {results_file}")

    # Generate summary report
    logger.info("\n" + "="*60)
    logger.info("LEARNING RATE SWEEP RESULTS SUMMARY")
    logger.info("="*60)
    for lr in learning_rates:
        result = all_results[lr]
        if result['diverged']:
            logger.info(f"lr={lr:.1e}: DIVERGED")
        else:
            logger.info(f"lr={lr:.1e}: final_accuracy={result['final_accuracy']:.4f}")
    logger.info("="*60 + "\n")

    # Plot validation reward curves
    plot_validation_curves(all_results, output_path / "validation_curves.png")

    return all_results


def plot_validation_curves(results: dict, output_path: str | Path):
    """
    Plot validation accuracy curves for multiple learning rates.

    Args:
        results: dict mapping learning_rate -> results dict with 'eval_history'
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for lr, result in results.items():
        if result['diverged'] and not result['eval_history']:
            continue

        eval_history = result['eval_history']
        if not eval_history:
            continue

        steps = [h[0] for h in eval_history]
        accuracies = [h[1] for h in eval_history]

        label = f"lr={lr:.1e}"
        if result['diverged']:
            label += " (diverged)"

        plt.plot(steps, accuracies, marker='o', label=label, markersize=4)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('GRPO Training: Validation Accuracy vs Learning Rate', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved validation curves plot to {output_path}")


def plot_from_saved_results(results_file: str | Path, output_path: str | Path = None):
    """
    Load saved sweep results and generate the validation curves plot.

    Args:
        results_file: Path to the sweep_results.json file
        output_path: Path to save the plot (defaults to same directory as results)
    """
    import matplotlib.pyplot as plt

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Convert string keys back to float
    results = {float(k): v for k, v in results.items()}

    if output_path is None:
        output_path = Path(results_file).parent / "validation_curves.png"

    plot_validation_curves(results, output_path)





if __name__ == "__main__":
    import argparse

    # Default checkpoint path on remote server
    DEFAULT_CHECKPOINT = "/workspace/assignment5-alignment/checkpoints/epoch_2"

    parser = argparse.ArgumentParser(description="GRPO Training with optional LR sweep")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--sweep", action="store_true", help="Run learning rate sweep")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of GRPO steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (for single run, sweet spot from sweep)")
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline",
                       choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                       help="Gradient accumulation steps. If not set, auto-computed to keep microbatch size = train_batch_size/128")
    parser.add_argument("--train_batch_size", type=int, default=256,
                       help="Number of rollout samples to train on per step (default: 256 = rollout_batch_size)")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1,
                       help="How many gradient passes over each rollout batch (>1 = off-policy reuse)")
    parser.add_argument("--debug", action="store_true", help="Enable timing instrumentation for performance debugging")
    parser.add_argument("--normalize_constant", type=float, default=None,
                       help="If set, use masked_normalize with this constant instead of masked_mean for per-token loss aggregation")
    parser.add_argument("--no_std_normalization", action="store_true",
                       help="Disable std normalization in group reward normalization (use_std_normalization=False)")
    parser.add_argument("--lrs", type=float, nargs="+", default=None, help="Learning rates for sweep (overrides built-in list)")

    args = parser.parse_args()

    # Auto-compute grad_accum_steps to keep microbatch size constant at train_batch_size/128
    BASE_MICROBATCH = 4  # fits on H100 with 1.5B model
    if args.grad_accum_steps is None:
        args.grad_accum_steps = max(1, args.train_batch_size // BASE_MICROBATCH)

    if args.sweep:
        # Learning rate sweep - skip 1e-6 (already done)
        sweep_output_dir = args.output_dir if args.output_dir else "lr_sweep_results"
        learning_rates = args.lrs if args.lrs else [5e-6, 1e-5, 5e-5, 1e-4]
        logger.info(f"Running learning rate sweep with rates: {learning_rates}")

        results = run_learning_rate_sweep(
            checkpoint_path=args.checkpoint,
            learning_rates=learning_rates,
            n_grpo_steps=args.n_steps,
            loss_type=args.loss_type,
            output_dir=sweep_output_dir,
            debug=args.debug,
            gradient_accumulation_steps=args.grad_accum_steps
        )

        # Print final summary
        print("\n" + "="*60)
        print("FINAL LEARNING RATE SWEEP RESULTS")
        print("="*60)
        for lr in learning_rates:
            result = results[lr]
            if result['diverged']:
                print(f"lr={lr:.1e}: DIVERGED")
            else:
                print(f"lr={lr:.1e}: final_accuracy={result['final_accuracy']:.4f}")
        print("="*60)
        print(f"\nResults saved to {sweep_output_dir}/")
        print(f"- sweep_results.json: Raw results")
        print(f"- validation_curves.png: Validation accuracy plot")

    else:
        # Single training run
        # Default output under grpo_comparison/<loss_type>
        base_output_dir = args.output_dir if args.output_dir else "grpo_comparison"
        suffix = args.loss_type
        if args.normalize_constant is not None:
            suffix += f"_norm{int(args.normalize_constant)}"
        if args.no_std_normalization:
            suffix += "_no_std"
        if args.train_batch_size != 256 or args.epochs_per_rollout_batch != 1:
            suffix += f"_tbs{args.train_batch_size}_e{args.epochs_per_rollout_batch}"
        single_output_dir = f"{base_output_dir}/{suffix}"
        lr_tag = f"{args.lr:.0e}".replace("+", "").replace("-0", "-")

        policy = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(TRAIN_DEVICE)
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=args.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        vllm_instance = init_vllm(args.checkpoint, seed=42, device=VLLM_DEVICE, gpu_memory_utilization=0.85)

        results = grpo_train(
            policy=policy,
            optimizer=optimizer,
            vllm_instance=vllm_instance,
            n_grpo_steps=args.n_steps,
            loss_type=args.loss_type,
            output_dir=single_output_dir,
            lr_tag=lr_tag,
            gradient_accumulation_steps=args.grad_accum_steps,
            normalize_constant=args.normalize_constant,
            use_std_normalization=not args.no_std_normalization,
            train_batch_size=args.train_batch_size,
            epochs_per_rollout_batch=args.epochs_per_rollout_batch,
            debug=args.debug
        )

        # Save eval history
        results_path = Path(single_output_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        results_file = results_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'loss_type': args.loss_type,
                'learning_rate': args.lr,
                'n_steps': args.n_steps,
                'eval_history': results['eval_history'],
                'final_accuracy': results['final_accuracy'],
                'diverged': results['diverged']
            }, f, indent=2)
        logger.info(f"Saved results to {results_file}")

        if results['diverged']:
            print(f"\nTraining DIVERGED")
        else:
            print(f"\nTraining completed. Final accuracy: {results['final_accuracy']:.4f}")
        print(f"Results saved to {single_output_dir}/")