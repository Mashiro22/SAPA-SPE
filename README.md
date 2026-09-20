# SAPA-SPE

This implementation is derived from OTSurv and retains its
CC BY-NC-SA 4.0 license. See `LICENSE.md`.

Upstream OTSurv implementation: https://github.com/Y-Research-SBU/OTSurv

Reference implementation of **SAPA-SPE**, a spatially enriched optimal-transport MIL model for whole-slide-image survival prediction.

Repository: https://github.com/Mashiro22/SAPA-SPE

This public-release candidate contains the core model and execution path only:

- OTSurv optimal-transport aggregation;
- slide-adaptive prototype attention (SAPA);
- spatial prototype encoding (SPE);
- Cox, capped-Cox, ranking, and discrete-time survival losses;
- training, checkpoint selection, and held-out inference entry points.

Experimental Local/Global Transformer, MAMMOTH, PTCMIL, and other exploratory variants are intentionally excluded.

## Relationship to OTSurv

Where SAPA-SPE does not require a methodological change, this release follows
the official OTSurv implementation: the learnable patch encoder, OT/Sinkhorn
aggregation, optimizer and scheduler defaults, 20-epoch training default,
loss-based early stopping, split reader, C-index computation, and result
serialization retain the upstream behavior. The intentional extensions are
SAPA, SPE and coordinate propagation, capped-Cox support, the unified
`--num_patches` interface, diagnostic exports, and fault-tolerant checkpoint
and parallel-evaluation safeguards.

## Method overview

For a slide with patch features `X` and patch coordinates `C`, the implementation applies

```text
X -> patch encoder -> OT-induced prototypes
  -> SPE spatial descriptors -> enriched prototypes
  -> SAPA attention and calibration -> scalar risk
```

The public model name is `sapa_spe`. The corresponding class is `OTSurvPGSpatialPantherConcat` in `mil_models/model_otsurv.py`.

## Repository layout

```text
mil_models/
  model_otsurv.py              # OTSurv, SAPA, and SPE
  model_factory.py             # public model registry
  otsurv_component/            # OT and Sinkhorn implementation
training/
  main_survival.py             # training/validation/test workflow
  test_survival.py             # checkpoint-only inference
  engine.py                    # optimization and evaluation loops
wsi_datasets/
  wsi_survival.py              # feature/coordinate loading
utils/                          # losses, split handling, checkpoints
scripts/                        # reproducible command templates
```

## Environment

The reference OTSurv environment uses Python 3.9.19 and the package versions
listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA runtime when the generic command above is unsuitable.

## Data format

Each slide is represented by one `.h5` file under a directory named `feats_h5`. The file must contain:

- `features`: `[N, D]` patch features (the paper uses UNI features with `D=1024`);
- `coords`: `[N, 2]` patch coordinates.

The split CSV files must be placed below `splits/` and contain the columns expected by `WSI_OTSurv_Dataset`, including `case_id`, `slide_id`, the selected survival-time column, and its matching censorship column. Patient-level splitting must be performed before training.

For strict reproduction of the OTSurv evaluation setup, use the official
OTSurv five-fold CSV files. In those files, `val.csv` and `test.csv` are the
same held-out fold; therefore, `test` should not be described as an additional
independent cohort.

Neither WSI data, extracted features, cohort labels, nor pretrained foundation-model weights are distributed in this folder.

## Unified patch-count interface

Training and inference use the same option:

```text
--num_patches N
```

Its default is `-1`, which keeps every available patch in every split. A positive value deterministically subsamples at most `N` patches per slide using `--bag_sample_strategy random` or `spatial_uniform`.

There are no separate public `train_bag_size`, `val_bag_size`, or `test_bag_size` options. This prevents an accidental train/inference mismatch.

## Training

Run from the repository root so that the package imports and relative split paths resolve correctly:

```bash
python -m training.main_survival \
  --model_type sapa_spe \
  --data_source /path/to/features/feats_h5 \
  --split_dir survival/TCGA_STAD_overall_survival_k=0 \
  --split_names train,val,test \
  --task STAD_survival \
  --target_col dss_survival_days \
  --results_dir ./results \
  --loss_fn capped_cox \
  --cox_margin 1.2 \
  --early_stopping 1 \
  --es_metric loss \
  --num_patches -1 \
  --device cuda:0
```

Adjust the target and censorship column names to match the cohort CSV. The censorship column is derived from the prefix of `--target_col` by the existing data pipeline.

## Inference

Use the same `--num_patches` value used for training. With the default `-1`, both stages use the complete bag.

```bash
python -m training.test_survival \
  --model_type sapa_spe \
  --checkpoint_path /path/to/s_checkpoint.pth \
  --data_source /path/to/features/feats_h5 \
  --split_dir survival/TCGA_STAD_overall_survival_k=0 \
  --split_names test \
  --task STAD_survival \
  --target_col dss_survival_days \
  --results_dir ./results_eval \
  --loss_fn capped_cox \
  --num_patches -1 \
  --device cuda:0
```

## Reproducibility notes

- Keep patient splits, extracted features, seeds, loss, and checkpoint-selection protocol fixed when comparing models.
- `--num_patches -1` is the release default for both training and inference.
- `--patch_encoder_type original` is the reproduced OTSurv/SAPA-SPE setting;
  `random_orthogonal` is retained as an optional experimental encoder.
- SPE requires coordinates aligned one-to-one with patch features.
- The dominant support reaches 70% cumulative OT mass and is capped at 1024 patches for local spatial statistics by default.
- The implementation is intended for Linux-based training; `training/engine.py` uses `fcntl` for checkpoint locking.

## Checks

After installing the dependencies, run:

```bash
python -m unittest discover -s tests -v
```

The tests enforce the unified patch-count interface and run a small SAPA-SPE forward/backward pass.

## Attribution and release notes

This code builds on OTSurv (MICCAI 2025; arXiv:2506.20741). Please cite the
upstream work as well as SAPA-SPE when using this implementation. Add the final
SAPA-SPE bibliographic entry when it becomes available. Pretrained checkpoints
may be released separately only when their
underlying feature and data licenses permit redistribution.

Do not publish private paths, cohort files, credentials, or restricted
foundation-model weights.

Only load feature files and checkpoints from trusted sources: PyTorch and
pickle-based artifacts may execute code while being deserialized.
