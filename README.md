# LX-ALOX15

Machine learning pipeline for ALOX15 inhibitor discovery, including:

- scaffold-split multi-model benchmarking,
- final single-model optimization,
- external compound library screening and SDF export for downstream docking.

## Repository Contents

This repository includes the components typically expected in an open-source ML project:

- **Code**: training, benchmarking, plotting, and screening scripts.
- **Raw data**: supervised dataset and pretraining corpus under `data/`.
- **Model artifacts**: saved models and metadata under `models/`.
- **Usage documentation**: step-by-step guides in this README and `MODEL_USAGE.md`.

## Project Structure

```text
LX-ALOX15/
├─ data/
│  ├─ raw/                  # Main supervised dataset (e.g., main.csv)
│  └─ pretrain/             # Pretraining / auxiliary data (e.g., zinc.smi)
├─ src/                     # Core modules: features, splits, training, plotting
├─ models/                  # Saved model artifacts (generated and/or provided)
├─ results/                 # Benchmark/final/screening outputs
├─ run_benchmark.py         # 16-model benchmark entry
├─ run_final_model.py       # Final single-model training entry
├─ screen_and_export_sdf.py # External library screening + SDF export
├─ benchmark.py             # Classification benchmark utilities
├─ train.py                 # Training entry helpers
├─ requirements.txt         # Python dependencies
├─ MODEL_USAGE.md           # Detailed usage & troubleshooting
└─ PROJECT_STRUCTURE.md     # Folder-by-folder explanation
```

For a more detailed breakdown, see `PROJECT_STRUCTURE.md`.

## Environment Setup

### 1) Clone repository

```bash
git clone <your-repo-url>
cd LX-ALOX15
```

### 2) Create environment and install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

Run all commands from the repository root.

### A. Run 16-model benchmark (scaffold split)

```bash
python run_benchmark.py
```

Main outputs:

- `results/benchmark/classification_benchmark_summary.csv`
- `results/benchmark/figures/classification_top30_roc_auc_oof_vs_test.png`
- `results/benchmark/figures/classification_top30_pr_auc_oof_vs_test.png`
- `results/benchmark/figures/classification_overlay_oof_roc_pr.png`
- `results/benchmark/figures/classification_overlay_test_roc_pr.png`

### B. Run final single-model training

Recommended model family: `extratrees`.

```bash
python run_final_model.py --family extratrees
```

Main outputs:

- `models/final_ensemble_cls.joblib`
- `models/final_training_metadata_cls.joblib`
- `results/figures/final/cls/final_scaffold_cv_roc.png`
- `results/figures/final/cls/final_scaffold_cv_pr.png`

### C. Screen an external compound library and export active SDF

Example:

```bash
python screen_and_export_sdf.py \
  --input "your_library.csv" \
  --output-sdf "results/active_compounds.sdf" \
  --output-csv "results/predictions.csv" \
  --threshold 0.5
```

Outputs:

- full prediction table (CSV),
- active-only SDF file for docking.

## Data and Model Notes

- `data/raw/main.csv` is used as the main supervised dataset.
- `data/pretrain/zinc.smi` is used for pretraining/representation learning.
- `models/` stores generated final artifacts (model + metadata).
- `results/` stores benchmark tables, figures, and screening exports.

## Metrics Reported

Primary classification metrics:

- ROC-AUC (OOF and Test),
- PR-AUC (OOF and Test),
- F1 (OOF and Test).

## Reproducibility Workflow

1. Run `python run_benchmark.py` for broad model comparison.
2. Select top family (typically by OOF ROC-AUC).
3. Run `python run_final_model.py --family extratrees`.
4. Screen external compounds and export active SDF for docking.

## Documentation

- `MODEL_USAGE.md`: detailed commands, options, and troubleshooting.
- `PROJECT_STRUCTURE.md`: detailed folder and module responsibilities.

## License

This project is released under the MIT License. See `LICENSE`.

