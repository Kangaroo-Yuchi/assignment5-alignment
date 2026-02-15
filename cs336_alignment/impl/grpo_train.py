import torch
import logging
import json
import copy
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
from cs336_alignment.impl.util import get_response_log_probs, tokenize_prompt_and_output, \
    compute_group_normalized_rewards, grpo_microbatch_train_step
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
                train_batch_size: int = 256, # On-policy
                gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100
                clip_range = 0.2,
                loss_type: Literal[
                "no_baseline",
                "reinforce_with_baseline",
                "grpo_clip",
                ] = "reinforce_with_baseline",
                use_std_normalization: bool = True,
                ) -> dict:
    """
    Train policy using GRPO algorithm.

    Returns:
        dict containing:
            - 'eval_history': list of (step, accuracy, format_reward) tuples
            - 'final_accuracy': final validation accuracy
            - 'diverged': bool indicating if training diverged
    """
    eval_history = []
    diverged = False
    policy.train()  # Set to training mode
    if torch.cuda.is_available():
        policy = policy.to(TRAIN_DEVICE)
    ds = load_dataset("gsm8k", "main")
    prompt_template = load_r1_zero_prompt_template()
    valid_prompts = [prompt_template.format(question=example['question']) for example in ds['test']]
    valid_ground_truths = [example['answer'] for example in ds['test']]

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

    for i in range(n_grpo_steps):
        load_policy_into_vllm_instance(policy, vllm_instance)
        current_batch_prompts = formatted_prompts[i * n_prompts_per_rollout_batch: (i + 1) * n_prompts_per_rollout_batch]
        current_batch_answer = raw_answers[i * n_prompts_per_rollout_batch: (i + 1) * n_prompts_per_rollout_batch]
        repeated_batch_prompt = []
        repeated_ground_truths = []
        for j in range(n_prompts_per_rollout_batch):
            repeated_ground_truths+=[current_batch_answer[j]]* group_size
            repeated_batch_prompt+=[current_batch_prompts[j]]* group_size

        vllm_outputs = vllm_instance.generate(current_batch_prompts, sampling_params)
        rollout_responses = []
        for request_output in vllm_outputs:  # One per input prompt
            for completion in request_output.outputs:  # group_size completions
                rollout_responses.append(completion.text)
        advantage, raw_rewards, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, use_std_normalization)
        logger.info(f"Rollout batch {i}: mean_reward={metadata['mean_reward']:.4f}, std_reward={metadata['std_reward']:.4f}")
        advantage = advantage.unsqueeze(-1).to(TRAIN_DEVICE)
        raw_rewards = raw_rewards.unsqueeze(-1).to(TRAIN_DEVICE)

        tokenized = tokenize_prompt_and_output(repeated_batch_prompt, rollout_responses, tokenizer) #batch_size seq_len
        input_ids = tokenized['input_ids'].to(TRAIN_DEVICE) #batch_size seq_len
        labels = tokenized['labels'].to(TRAIN_DEVICE) #batch_size seq_len
        response_mask = tokenized['response_mask'].to(TRAIN_DEVICE)

        policy.eval()
        with torch.no_grad():
            old_log_probs = get_response_log_probs(policy, input_ids, labels, True)['log_probs']
            # old_log_probs is already on TRAIN_DEVICE since input_ids/labels are

        policy.train()
        for _ in range(epochs_per_rollout_batch):
            accumulated_loss = 0.0
            for idx, train_step in enumerate(range(0, train_batch_size, micro_train_batch_size)):
                new_log_prob = get_response_log_probs(policy, input_ids[train_step:train_step+micro_train_batch_size], labels[train_step:train_step+micro_train_batch_size], True)['log_probs']
                loss, loss_metadata = grpo_microbatch_train_step(new_log_prob, response_mask[train_step:train_step+micro_train_batch_size],
                                           gradient_accumulation_steps, loss_type, raw_rewards[train_step:train_step+micro_train_batch_size],
                                           advantage[train_step:train_step+micro_train_batch_size],
                                           old_log_probs[train_step:train_step+micro_train_batch_size], clip_range)
                accumulated_loss += loss.item()

                # Check for divergence (NaN or very large loss)
                if torch.isnan(loss) or torch.isinf(loss) or abs(loss.item()) > 1e6:
                    logger.warning(f"Training diverged at step {global_step}! Loss: {loss.item()}")
                    diverged = True
                    return {
                        'eval_history': eval_history,
                        'final_accuracy': eval_history[-1][1] if eval_history else 0.0,
                        'diverged': True
                    }

                if (idx + 1) % gradient_accumulation_steps == 0:
                    clip_grad_norm_(policy.parameters(), 1.0)
                    global_step += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.info(f"Step {global_step}: loss={accumulated_loss:.4f}")
                    accumulated_loss = 0.0
                    if global_step % EVALUATION_STEP == 0:
                        logger.info(f"Running evaluation at step {global_step}...")
                        load_policy_into_vllm_instance(policy, vllm_instance)
                        eval_metrics = evaluate_vllm(vllm_instance, r1_zero_reward_fn, valid_prompts, valid_ground_truths, EVAL_SAMPLING_PARAM, step_number=global_step)
                        logger.info(f"Evaluation results - accuracy: {eval_metrics['accuracy']:.4f}, format_reward: {eval_metrics['format_reward']:.4f}")
                        eval_history.append((global_step, eval_metrics['accuracy'], eval_metrics['format_reward']))

    # Final evaluation
    logger.info("Running final evaluation...")
    load_policy_into_vllm_instance(policy, vllm_instance)
    final_metrics = evaluate_vllm(vllm_instance, r1_zero_reward_fn, valid_prompts, valid_ground_truths, EVAL_SAMPLING_PARAM, step_number=global_step)
    eval_history.append((global_step, final_metrics['accuracy'], final_metrics['format_reward']))
    logger.info(f"Final evaluation - accuracy: {final_metrics['accuracy']:.4f}, format_reward: {final_metrics['format_reward']:.4f}")

    return {
        'eval_history': eval_history,
        'final_accuracy': final_metrics['accuracy'],
        'diverged': False
    }


def run_learning_rate_sweep(
    checkpoint_path: str,
    learning_rates: list[float],
    n_grpo_steps: int = 200,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    output_dir: str = "lr_sweep_results",
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
        **grpo_kwargs: Additional arguments to pass to grpo_train

    Returns:
        dict mapping learning_rate -> results dict
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for lr in learning_rates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting sweep with learning_rate={lr}")
        logger.info(f"{'='*60}\n")

        # Load fresh model for each learning rate
        policy = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        # Initialize vLLM instance
        vllm_instance = init_vllm(checkpoint_path, seed=42, device=VLLM_DEVICE, gpu_memory_utilization=0.85)

        try:
            results = grpo_train(
                policy=policy,
                optimizer=optimizer,
                vllm_instance=vllm_instance,
                n_grpo_steps=n_grpo_steps,
                loss_type=loss_type,
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

    parser = argparse.ArgumentParser(description="GRPO Training with optional LR sweep")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sweep", action="store_true", help="Run learning rate sweep")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of GRPO steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (for single run)")
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline",
                       choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--output_dir", type=str, default="lr_sweep_results", help="Output directory for sweep")

    args = parser.parse_args()

    if args.sweep:
        # Learning rate sweep
        learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
        logger.info(f"Running learning rate sweep with rates: {learning_rates}")

        results = run_learning_rate_sweep(
            checkpoint_path=args.checkpoint,
            learning_rates=learning_rates,
            n_grpo_steps=args.n_steps,
            loss_type=args.loss_type,
            output_dir=args.output_dir
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
        print(f"\nResults saved to {args.output_dir}/")
        print(f"- sweep_results.json: Raw results")
        print(f"- validation_curves.png: Validation accuracy plot")

    else:
        # Single training run
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
            loss_type=args.loss_type
        )

        if results['diverged']:
            print(f"\nTraining DIVERGED")
        else:
            print(f"\nTraining completed. Final accuracy: {results['final_accuracy']:.4f}")

