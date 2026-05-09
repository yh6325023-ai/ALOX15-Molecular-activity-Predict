# LX-ALOX15 Model Usage Guide

This document explains how to run:

1. the 16-model benchmark (scaffold split),
2. final single-model training (recommended family: `extratrees`),
3. external library screening and active-compound SDF export.

Current repository scope is classification-only.

## 1) Environment

Run commands from the repository root:

```bash
cd LX-ALOX15
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Run the 16-model benchmark

Command:

```bash
python run_benchmark.py
```

What it does:

- runs a fixed 16-model classification benchmark,
- uses strict Murcko scaffold split and 5-fold scaffold CV,
- keeps benchmark settings comparable across model families.

Main outputs:

- `results/benchmark/classification_benchmark_summary.csv`
- `results/benchmark/figures/classification_top30_roc_auc_oof_vs_test.png`
- `results/benchmark/figures/classification_top30_pr_auc_oof_vs_test.png`
- `results/benchmark/figures/classification_overlay_oof_roc_pr.png`
- `results/benchmark/figures/classification_overlay_test_roc_pr.png`

## 3) Run final single-model training

Recommended family:

- `extratrees`

Command:

```bash
python run_final_model.py --family extratrees
```

What it does:

- trains one selected model family in final mode,
- applies scaffold-aware split/CV,
- saves final model and metadata for reuse.

Main outputs:

- `models/final_ensemble_cls.joblib`
- `models/final_training_metadata_cls.joblib`
- `results/figures/final/cls/final_scaffold_cv_roc.png`
- `results/figures/final/cls/final_scaffold_cv_pr.png`

## 4) Screen external library and export SDF

Supported input formats:

- `.csv` (SMILES column auto-detected),
- `.sdf`.

Example:

```bash
python screen_and_export_sdf.py \
  --input "your_library.csv" \
  --output-sdf "results/active_compounds.sdf" \
  --output-csv "results/predictions.csv" \
  --threshold 0.5
```

Outputs:

- prediction table CSV,
- active-only SDF for docking.

Notes:

- SDF export preserves compound identifiers when available.
- CSV loading handles common encodings (`utf-8`, `utf-8-sig`, `gb18030`, `gbk`).

## 5) Metric Interpretation

Main reported metrics:

- ROC-AUC (OOF and Test),
- PR-AUC (OOF and Test),
- F1 (OOF and Test).

Model selection recommendation:

- primary: OOF ROC-AUC,
- secondary: test ROC-AUC / PR-AUC and stability patterns.

## 6) Troubleshooting

### Q1) Why does benchmark run many models?

`run_benchmark.py` is intentionally a multi-model benchmark entry.

### Q2) Why does final training run one model only?

`run_final_model.py` is intentionally a single-model finalization entry.

### Q3) How to filter more strictly for active compounds?

Increase threshold, for example:

```bash
--threshold 0.6
```

### Q4) Path-related export issues?

Prefer ASCII output paths (for example under `results/`) if your system has encoding/path compatibility limits.

## 7) Recommended Reproducible Workflow

1. Run benchmark: `python run_benchmark.py`.
2. Select top family by OOF ROC-AUC and consistency.
3. Run final training: `python run_final_model.py --family extratrees`.
4. Screen external library and export active SDF for docking.

