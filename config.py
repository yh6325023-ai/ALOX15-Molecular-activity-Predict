# -*- coding: utf-8 -*-
"""Central hyperparameters and paths for ALOX15 classification pipeline."""
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (relative to project root = directory containing this file)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PRETRAIN_DIR = PROJECT_ROOT / "data" / "pretrain"

# User may put the pre-train data under different folder names.
# Put your `.smi` file(s) here (whitespace separated or tab separated):
#   smiles  pic50
# or
#   smiles  ...  pic50
DATA_PRETRAINED_DIR_1 = PROJECT_ROOT / "data" / "pretrained"
DATA_PRETRAINED_DIR_2 = PROJECT_ROOT / "data" / "pretreined"

# Main labelled set: place your ChEMBL export here (must include SMILES + PIC50)
MAIN_DATA_CSV = DATA_RAW_DIR / "main.csv"

# Optional larger table for pre-training (same columns: Smiles, PIC50).
# If this path does not exist, pre-training is skipped.
PRETRAIN_DATA_CSV = DATA_PRETRAIN_DIR / "pretrain.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Subfolders for artifacts/plots
FIGURES_PRETRAIN_DIR = FIGURES_DIR / "pretrain"
FIGURES_FINAL_DIR = FIGURES_DIR / "final"
FIGURES_BENCHMARK_DIR = FIGURES_DIR / "benchmark"

# Saved artifacts
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BASE_MODEL_PATH = MODELS_DIR / "base_lgbm.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_lgbm.joblib"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.joblib"

# Separate artifacts for pre-training (transfer module)
PRETRAIN_PREPROCESSOR_PATH = MODELS_DIR / "pretrain_preprocessor.joblib"
PRETRAIN_BASE_MODEL_PATH = MODELS_DIR / "transfer_base_model.joblib"
PRETRAIN_TRAINING_METADATA_PATH = MODELS_DIR / "pretrain_training_metadata.joblib"
PRETRAIN_REPR_MODEL_PATH = MODELS_DIR / "pretrain_repr_model.joblib"

# Final model artifacts
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
FINAL_TRAINING_METADATA_PATH = MODELS_DIR / "final_training_metadata.joblib"

# -----------------------------------------------------------------------------
# Column names (input file may use alternate spellings; see dataio.normalize_df)
# -----------------------------------------------------------------------------
SMILES_COL = "Smiles"
TARGET_COL = "PIC50"

# -----------------------------------------------------------------------------
# Split ratios (where each script reads from — not all scripts use the same split)
# -----------------------------------------------------------------------------
# • pipeline.py / legacy supervised pretrain-style runs: TRAIN_FRACTION of rows → train;
#   the remaining 20% is split 50/50 → validation 10% + test 10%.  Ratio train:val:test = 8:1:1.
# • train.py final supervised (final_module): scaffold/cluster/random GROUP holdout uses
#   FINAL_HOLDOUT_TEST_FRAC — e.g. 0.10 ⇒ ~90% train_dev : ~10% test (exact counts depend on groups).
# -----------------------------------------------------------------------------
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Duplicates: aggregate multiple measurements per SMILES
# -----------------------------------------------------------------------------
DROP_DUPLICATE_SMILES = True
DUPLICATE_AGG = "mean"  # mean of PIC50 for same canonical SMILES

# -----------------------------------------------------------------------------
# Morgan fingerprint (ECFP-style)
# -----------------------------------------------------------------------------
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048

# -----------------------------------------------------------------------------
# Feature post-processing
# -----------------------------------------------------------------------------
# Remove near-constant columns after imputation
VARIANCE_THRESHOLD = 0.8 * (1.0 - 0.8)  # sklearn default-style: 1e-8 often used; keep prior project scale

# GradientBoostingClassifier params (classification ``gbr`` only; no GBR regressor in final_module)
GBR_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.75,
    "random_state": RANDOM_STATE,
}

# ExtraTrees classifier/shared tree settings used by classification pipeline.
EXTRATREES_PARAMS = {
    "n_estimators": 2000,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# -----------------------------------------------------------------------------
# Pre-train data (smi format) and parsing behavior
# -----------------------------------------------------------------------------
# If PRETRAIN_SMI_PATH is None, we will auto-search for a `.smi` file under
# the directories below.
PRETRAIN_SMI_PATH = None  # e.g. DATA_PRETRAINED_DIR_1 / "pretrained.smi"

PRETRAIN_SMI_SEARCH_DIRS = [DATA_PRETRAINED_DIR_1, DATA_PRETRAINED_DIR_2, DATA_PRETRAIN_DIR]

# If set, randomly reservoir-sample this many valid SMILES from the huge file.
# Set None to use all rows (not recommended for first run with 3M rows).
PRETRAIN_MAX_ROWS = None  # None = use all rows (e.g., 300+万) for full pretraining
PRETRAIN_SAMPLING_SEED = RANDOM_STATE

# Unsupervised representation learning config
PRETRAIN_REPR_DIM = 128
PRETRAIN_IPCA_BATCH_SIZE = 4096
FINAL_USE_CONCAT_RAW_FEATURES = True  # raw selected cols + repr
# If True: load models/pretrain_preprocessor.joblib (IPCA repr, pretrain-fitted imputer/scaler). If False: no repr;
# imputer + VarianceThreshold + StandardScaler are fit on train_dev only, then transform all rows.
FINAL_USE_PRETRAIN_ARTIFACTS = True  # keep pretrain representation in classification pipeline

# After building X (raw+repr or no-pretrain path): clip extreme values per column using train_dev quantiles.
FINAL_WINSORIZE_TRAIN_DEV = True
FINAL_WINSORIZE_LOW_Q = 0.005
FINAL_WINSORIZE_HIGH_Q = 0.995

# ---------------------------------------------------------------------------
# Cross validation / ensembling (main supervised training)
# ---------------------------------------------------------------------------
# "scaffold": Bemis–Murcko group holdout + GroupKFold — strict new-scaffold extrapolation.
# "cluster": Morgan + Butina clusters, whole-cluster holdout + GroupKFold — NOT i.i.d. random; usually easier than
#   pure Murcko, often higher R2, still structure-grouped (tune FINAL_CLUSTER_BUTINA_DIST if too few/many clusters).
# "random": shuffle molecule holdout + KFold — optimistic result; not grouped extrapolation.
FINAL_CV_PROTOCOL = "scaffold"  # scaffold | cluster | random
# Grouped holdout (scaffold/cluster): greedy assignment until ~this fraction of *molecules* is in test.
# Lower ⇒ more data for train_dev (better use of labels) but smaller/noisier holdout metrics.
# 0.08 ≈ 92:8 train:test (nominal); avoid <0.05 on small sets (test R2 becomes unstable).
FINAL_HOLDOUT_TEST_FRAC = 0.10  # keep overall ≈8:1:1 style split from pipeline
# Butina Tanimoto distance threshold (larger ⇒ fewer, larger clusters). If GroupKFold errors (too few groups), lower this.
FINAL_CLUSTER_BUTINA_DIST = 0.38

FINAL_CV_FOLDS = 5  # 5-fold on train_dev to match requested 5-fold CV
FINAL_CV_SHUFFLE = True
FINAL_CV_RANDOM_STATE = RANDOM_STATE
# Classification ensemble on holdout test:
# True = weighted mean by per-fold validation ROC-AUC; False = simple mean
CLASSIFICATION_ENSEMBLE_WEIGHTED = True
# If True, map decision_function scores to probabilities using sigmoid when predict_proba is unavailable.
CLASSIFICATION_USE_SIGMOID_FOR_DECISION = False

# Save one model per fold and build an ensemble predictor (mean of fold preds)
FINAL_ENSEMBLE_PATH = MODELS_DIR / "final_ensemble.joblib"
FINAL_FOLD_MODELS_DIR = MODELS_DIR / "fold_models"

# ---------------------------------------------------------------------------
# Nested hyperparameter tuning (reviewer requirement)
# ---------------------------------------------------------------------------
NESTED_TUNING_ENABLED = True  # final single-model run: enable nested tuning
NESTED_INNER_FOLDS = 4
NESTED_N_ITER = 60  # 比默认 40 略大，搜索更充分
NESTED_RANDOM_STATE = RANDOM_STATE

# Scoring metric for nested selection
NESTED_CLASSIFICATION_METRIC = "roc_auc"  # roc_auc | average_precision

# Candidate families supported in strict training
# Classification benchmark families (fixed 16-model comparison set)
CLASSIFICATION_CANDIDATE_FAMILIES = [
    "gaussiannb",
    "bernoullinb",
    "logreg",
    "ridgecls",
    "sgd",
    "passiveaggr",
    "knn",
    "svc",
    "linearsvc",
    "lda",
    "bagging",
    "adaboost",
    "gbr",
    "histgb",
    "randomforest",
    "extratrees",
]

# ---------------------------------------------------------------------------
# Classification mode (ROC/PR)
# ---------------------------------------------------------------------------
TASK_TYPE = "classification"  # classification-only pipeline
CLASSIFICATION_THRESHOLD = 5.2

# Classifier choice for classification mode（最终单模型：ExtraTrees）
FINAL_CLASSIFIER_KIND = "extratrees"
FINAL_HISTGB_CLASSIFIER_PARAMS = {
    "max_iter": 800,
    "learning_rate": 0.03,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
}
RF_CLASSIFIER_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 24,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
# Native LightGBM binary (see src/final_module._LGBMBinaryClassifier; avoids sklearn wrapper issues)
LGBM_CLASSIFIER_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_samples": 20,
    "random_state": RANDOM_STATE,
}
# Native lgb.train only: decay learning_rate each boosting round (sklearn HistGB/GBR have no train-time LR schedule).
LGBM_DYNAMIC_LEARNING_RATE = True
LGBM_LR_DECAY = 0.995  # lr_t = max(LGBM_LR_FLOOR, lr_0 * LGBM_LR_DECAY**t)
LGBM_LR_FLOOR = 0.01

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
FIGURE_DPI = 300

# Ensure directories exist when importing config (safe for fresh clones)
for _d in (
    DATA_RAW_DIR,
    DATA_PRETRAIN_DIR,
    DATA_PRETRAINED_DIR_1,
    DATA_PRETRAINED_DIR_2,
    MODELS_DIR,
    FIGURES_DIR,
    FIGURES_PRETRAIN_DIR,
    FIGURES_FINAL_DIR,
    FIGURES_BENCHMARK_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
