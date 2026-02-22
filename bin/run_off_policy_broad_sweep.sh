#!/bin/bash
# Broad sweep: epochs_per_rollout_batch x train_batch_size, 40 GRPO steps.
# Fixes rollout_batch_size=256, lr=1e-5, loss=reinforce_with_baseline.
# grad_accum_steps is auto-computed to keep microbatch=2 constant.
#
# Experiment log / rationale:
#   train_batch_size: {128, 256, 512}
#     - 128 = half the rollout; each step trains on only half the data (more noise, faster per step)
#     - 256 = on-policy default (all rollout data, one pass)
#     - 512 = requires 2 rollout batches worth (cycles data, more off-policy at higher epoch)
#   epochs_per_rollout_batch: {1, 2, 4}
#     - 1 = fully on-policy (baseline)
#     - 2 = moderate reuse, common in PPO
#     - 4 = aggressive reuse; tests staleness tolerance
#
# Memory: microbatch = train_batch_size // grad_accum_steps = 2 (fixed by auto-compute).
# Wall-clock: fewer rollouts per gradient step -> faster for high-epoch configs.

BASE=/workspace/assignment5-alignment
CHECKPOINT=$BASE/checkpoints/epoch_2
OUTPUT=$BASE/grpo_comparison/off_policy_sweep

run() {
  local tbs=$1
  local epochs=$2
  echo "=== tbs=${tbs} epochs=${epochs} ==="
  uv run python cs336_alignment/impl/grpo_train.py \
    --checkpoint $CHECKPOINT \
    --loss_type reinforce_with_baseline \
    --n_steps 40 \
    --lr 1e-5 \
    --train_batch_size $tbs \
    --epochs_per_rollout_batch $epochs \
    --output_dir $OUTPUT
}

# 3x3 grid: tbs in {128, 256, 512} x epochs in {1, 2, 4}
run 128 1
run 128 2
run 128 4
run 256 1
run 256 2
run 256 4
run 512 1
run 512 2
run 512 4