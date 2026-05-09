# LX-ALOX15 Project Structure

This document describes the purpose of each folder and key files.

## Top-Level Layout

- `src/`  
  Core Python modules for feature engineering, splitting, model training, benchmarking, and plotting.

- `data/`  
  Input datasets used for supervised training and pretraining.

- `models/`  
  Saved model artifacts and metadata (generated after final training).

- `results/`  
  Generated benchmark summaries, figures, and screening outputs.

- `.vscode/`  
  Local IDE settings (editor configuration only).

## Data Folder

- `data/raw/`  
  Main supervised dataset files (for example: `main.csv`).

- `data/pretrain/`  
  Auxiliary/unlabeled data used by representation learning (for example: `zinc.smi`).

## Source Modules (`src/`)

- `src/features.py`  
  Molecular feature construction (Morgan + MACCS + pretraining representation assembly).

- `src/scaffold_split.py`  
  Scaffold-aware splitting utilities for holdout and cross-validation.

- `src/final_module.py`  
  Final model training, CV, metric aggregation, and artifact saving.

- `src/benchmark_data.py`  
  Shared feature/data preparation for benchmark workflows.

- `src/plots_cv.py`  
  Cross-validation visualization utilities (ROC/PR and related plots).

- `src/plot_classification_comparison.py` and `src/plot_classification_overlay_curves.py`  
  Benchmark comparison plots and overlay curve generation.

- `src/pretrain_module.py`  
  Pretraining and representation transform utilities.

- `src/ensemble.py`, `src/dataio.py`  
  Supporting modules for ensemble prediction and data I/O.

## Entry Scripts

- `run_benchmark.py`  
  Runs the multi-model benchmark stage (classification benchmark).

- `run_final_model.py`  
  Runs final single-model training and exports model artifacts.

- `screen_and_export_sdf.py`  
  Screens external compounds and exports active predictions to SDF/CSV.

- `benchmark.py`  
  Classification benchmark entry and plotting utilities.

- `train.py` and `pretrain.py`  
  Supporting train/pretrain entry logic.

## Results Folder

- `results/benchmark/`  
  Benchmark-stage outputs only (tables and figures), for example:
  - `classification_benchmark_summary.csv`
  - `Supplementary_Table_ML_Classification_AUC_F1.csv`

- `results/figures/`  
  Generated visualizations from benchmark/final runs.

- `results/` root  
  Screening exports and additional generated outputs (for example prediction CSV and active-compound SDF).

## Stage Separation (Reporting Guidance)

- **Benchmark stage (multi-model):** `run_benchmark.py` -> `results/benchmark/`
- **Final stage (single-model):** `run_final_model.py` -> `models/` and `results/figures/final/`

For manuscripts or reports, do not mix benchmark comparison metrics with final optimization results.
