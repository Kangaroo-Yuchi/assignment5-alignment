#!/bin/bash
# Broad sweep: epochs_per_rollout_batch x train_batch_size, 40 GRPO steps.
# Fixes rollout_batch_size=256, loss=grpo_clip (off-policy).
# grad_accum_steps is auto-computed to keep microbatch=4 constant.
#
# Experiment log / rationale:
#   train_batch_size: {128, 256}  -- must be <= rollout_batch_size=256
#     - 128 = 2 optimizer steps per epoch, moderate
#     - 256 = 1 optimizer step per epoch (same as original on-policy when epochs=1)
#   (tbs=64 excluded: 4 gradient steps per epoch causes reward collapse with stale old_log_probs)
#   epochs_per_rollout_batch: {1, 2, 4}
#     - 1 = one full pass over rollout data
#     - 2 = two full passes (moderate off-policy reuse)
#     - 4 = four full passes (aggressive reuse; tests clip tolerance)
#   lr: scaled down by epochs to keep effective update magnitude constant
#     - epochs=1 -> lr=1e-5
#     - epochs=2 -> lr=5e-6
#     - epochs=4 -> lr=2.5e-6
#
# Memory: microbatch = train_batch_size // grad_accum_steps = 4 (fixed by auto-compute).

BASE=/workspace/assignment5-alignment
CHECKPOINT=$BASE/checkpoints/epoch_2
OUTPUT=$BASE/grpo_comparison/off_policy_sweep

run() {
  local tbs=$1
  local epochs=$2
  local lr=$3
  echo "=== tbs=${tbs} epochs=${epochs} lr=${lr} ==="
  uv run python cs336_alignment/impl/grpo_train.py \
    --checkpoint $CHECKPOINT \
    --loss_type grpo_clip \
    --n_steps 40 \
    --lr $lr \
    --train_batch_size $tbs \
    --epochs_per_rollout_batch $epochs \
    --output_dir $OUTPUT
}

# tbs=256: 1 optimizer step per epoch
#run 256 1 1e-5
run 256 2 5e-6
run 128 2 5e-6
run 256 4 2.5e-6

# tbs=128: 2 optimizer steps per epoch
run 128 1 1e-5
run 128 4 2.5e-6