#!/usr/bin/env bash
set -euo pipefail

python -m training.main_survival \
  --model_type sapa_spe \
  --data_source "${DATA_SOURCE:?set DATA_SOURCE to a feats_h5 directory}" \
  --split_dir "${SPLIT_DIR:?set SPLIT_DIR relative to ./splits}" \
  --task "${TASK:-STAD_survival}" \
  --target_col "${TARGET_COL:-dss_survival_days}" \
  --results_dir "${RESULTS_DIR:-./results}" \
  --loss_fn capped_cox \
  --cox_margin 1.2 \
  --num_patches -1 \
  --device "${DEVICE:-cuda:0}"
