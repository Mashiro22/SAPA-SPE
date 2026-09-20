#!/usr/bin/env bash
set -euo pipefail

python -m training.test_survival \
  --model_type sapa_spe \
  --checkpoint_path "${CHECKPOINT:?set CHECKPOINT to s_checkpoint.pth}" \
  --data_source "${DATA_SOURCE:?set DATA_SOURCE to a feats_h5 directory}" \
  --split_dir "${SPLIT_DIR:?set SPLIT_DIR relative to ./splits}" \
  --split_names test \
  --task "${TASK:-STAD_survival}" \
  --target_col "${TARGET_COL:-dss_survival_days}" \
  --results_dir "${RESULTS_DIR:-./results_eval}" \
  --loss_fn capped_cox \
  --num_patches -1 \
  --device "${DEVICE:-cuda:0}"
