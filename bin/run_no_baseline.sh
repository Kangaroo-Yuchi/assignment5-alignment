#!/bin/bash
uv run python cs336_alignment/impl/grpo_train.py \
  --checkpoint /workspace/assignment5-alignment/checkpoints/epoch_2 \
  --loss_type no_baseline \
  --n_steps 100 \
  --lr 1e-5 \
  --grad_accum_steps 64 \
  --output_dir /workspace/assignment5-alignment/grpo_comparison