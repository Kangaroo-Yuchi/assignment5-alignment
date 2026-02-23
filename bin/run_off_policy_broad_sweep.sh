#!/bin/bash
# Broad sweep: epochs_per_rollout_batch x train_batch_size, 40 GRPO steps.
# Fixes rollout_batch_size=256, lr=1e-5, loss=grpo_clip (off-policy).
# grad_accum_steps is auto-computed to keep microbatch=4 constant.
#
# Experiment log / rationale:
#   train_batch_size: {64, 128, 256}  -- must be <= rollout_batch_size=256
#     - 64  = 4 optimizer steps per epoch, high gradient frequency
#     - 128 = 2 optimizer steps per epoch, moderate
#     - 256 = 1 optimizer step per epoch (same as original on-policy when epochs=1)
#   epochs_per_rollout_batch: {1, 2, 4}
#     - 1 = one full pass over rollout data
#     - 2 = two full passes (moderate off-policy reuse)
#     - 4 = four full passes (aggressive reuse; tests clip tolerance)
#
# Memory: microbatch = train_batch_size // grad_accum_steps = 4 (fixed by auto-compute).

BASE=/workspace/assignment5-alignment
CHECKPOINT=$BASE/checkpoints/epoch_2
OUTPUT=$BASE/grpo_comparison/off_policy_sweep

run() {
  local tbs=$1
  local epochs=$2
  echo "=== tbs=${tbs} epochs=${epochs} ==="
  uv run python cs336_alignment/impl/grpo_train.py \
    --checkpoint $CHECKPOINT \
    --loss_type grpo_clip \
    --n_steps 40 \
    --lr 1e-5 \
    --train_batch_size $tbs \
    --epochs_per_rollout_batch $epochs \
    --output_dir $OUTPUT
}

# 3x3 grid: tbs in {64, 128, 256} x epochs in {1, 2, 4}
run 64  1
run 64  2
run 64  4
run 128 1
run 128 2
run 128 4
run 256 1
run 256 2
run 256 4