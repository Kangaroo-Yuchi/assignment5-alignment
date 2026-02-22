#!/bin/bash
# Focused sweep: 200 GRPO steps for the best configs identified in the broad sweep,
# plus the on-policy baseline. Edit the run() calls below after reviewing broad sweep results.
#
# Default: run all plausible candidates. Remove any that diverged or were clearly worse.

BASE=/workspace/assignment5-alignment
CHECKPOINT=$BASE/checkpoints/epoch_2
OUTPUT=$BASE/grpo_comparison/off_policy_focused

run() {
  local tbs=$1
  local epochs=$2
  echo "=== tbs=${tbs} epochs=${epochs} ==="
  uv run python cs336_alignment/impl/grpo_train.py \
    --checkpoint $CHECKPOINT \
    --loss_type reinforce_with_baseline \
    --n_steps 200 \
    --lr 1e-5 \
    --train_batch_size $tbs \
    --epochs_per_rollout_batch $epochs \
    --output_dir $OUTPUT
}

# On-policy baseline (epochs=1, tbs=256)
run 256 1

# Best candidates from broad sweep (edit as needed):
run 256 2
run 256 4
run 128 2