# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, ParameterSampler
from sklearn.base import clone

import config as cfg
from src.dataio import aggregate_duplicates, load_activity_table
from src.ensemble import MeanEnsembleRegressor
from src.features import build_feature_matrix
from src.plots_cv import compute_fold_pr, compute_fold_roc, plot_cv_pr, plot_cv_roc
from src.scaffold_split import (
    build_butina_cluster_group_ids,
    murcko_scaffold_smiles,
    nested_inner_fold_ids,
    random_molecule_holdout_split,
    random_molecule_kfold_split,
    scaffold_holdout_split,
    scaffold_kfold_split,
)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _positive_class_score(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return positive-class scores.
    - If predict_proba exists, use calibrated class-1 probability directly.
    - Else if decision_function exists, map score via sigmoid.
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.reshape(-1)
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
        if bool(getattr(cfg, "CLASSIFICATION_USE_SIGMOID_FOR_DECISION", True)):
            return _sigmoid(s)
        return s
    # Fallback for estimators without proba/decision_function
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


def _metrics_cls(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    yhat = (s >= float(threshold)).astype(int)
    return {
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, yhat)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "accuracy": float(accuracy_score(y, yhat)),
    }


def _sanitize_sklearn_forest_params(params: dict) -> dict:
    """
    sklearn constraint: max_samples is only valid when bootstrap=True for forest ensembles
    (e.g., ExtraTrees/RandomForest).
    """
    p = dict(params)
    # Accept either spelling defensively; sklearn uses `max_samples`
    if "max_sample" in p and "max_samples" not in p:
        p["max_samples"] = p.pop("max_sample")

    if (not bool(p.get("bootstrap", False))) and ("max_samples" in p) and (p.get("max_samples") is not None):
        # Avoid ValueError: max_samples cannot be set if bootstrap=False
        # Pop entirely (some sklearn versions still error if attribute is present)
        p.pop("max_samples", None)
    return p


def _split_8_1_1(X: pd.DataFrame, y: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=random_state,
        shuffle=True,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def _build_regressor(kind: str, override_params: dict | None = None):
    """Final training regression: ExtraTrees only."""
    override_params = override_params or {}
    if kind == "extratrees":
        params = _sanitize_sklearn_forest_params({**cfg.EXTRATREES_PARAMS, **override_params})
        return ExtraTreesRegressor(**params)
    raise ValueError(f"Unknown regressor kind: {kind} (final regression only supports 'extratrees')")


def _regression_supervised_estimator(kind: str, override_params: dict | None, *, n_features: int) -> Any:
    """
    Optional per-fold SelectKBest + regressor (nested / outer train only).
    """
    override_params = override_params or {}
    fs = str(getattr(cfg, "REGRESSION_FEATURE_SELECTION", "none")).lower()
    k_cfg = int(getattr(cfg, "REGRESSION_FEATURE_SELECTION_K", 1500))
    k = max(1, min(k_cfg, int(n_features)))

    base = _build_regressor(kind, override_params=override_params)

    if fs in ("none", ""):
        return base
    if fs == "f_regression":
        score_fn = f_regression
    elif fs == "mutual_info":
        score_fn = mutual_info_regression
    else:
        raise ValueError(f"Unknown REGRESSION_FEATURE_SELECTION: {fs!r}")
    return Pipeline(
        [
            ("select", SelectKBest(score_func=score_fn, k=k)),
            ("est", base),
        ]
    )


class _LGBMBinaryClassifier:
    """Binary classifier via native `lgb.train` (skips LGBMClassifier/sklearn compatibility issues)."""

    def __init__(self, **params: Any):
        self._params = dict(params)
        self._booster = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _LGBMBinaryClassifier:
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        X_arr = np.asarray(X, dtype=np.float64, order="C")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        train_set = lgb.Dataset(X_arr, label=y_arr)
        n_est = int(self._params.get("n_estimators", 500))
        lr0 = float(self._params.get("learning_rate", 0.05))
        p: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": int(self._params.get("num_leaves", 31)),
            "learning_rate": lr0,
            "feature_fraction": float(self._params.get("colsample_bytree", 0.8)),
            "bagging_fraction": float(self._params.get("subsample", 0.8)),
            "bagging_freq": 1,
            "min_child_samples": int(self._params.get("min_child_samples", 20)),
            "seed": int(self._params.get("random_state", cfg.RANDOM_STATE)),
            "feature_pre_filter": False,
        }
        md = self._params.get("max_depth", -1)
        if md is not None and int(md) > 0:
            p["max_depth"] = int(md)

        callbacks: list[Any] = []
        if bool(getattr(cfg, "LGBM_DYNAMIC_LEARNING_RATE", False)):
            decay = float(getattr(cfg, "LGBM_LR_DECAY", 0.995))
            floor = float(getattr(cfg, "LGBM_LR_FLOOR", 0.01))
            callbacks.append(
                lgb.reset_parameter(
                    learning_rate=lambda env: max(floor, lr0 * (decay ** int(env.iteration))),
                )
            )

        if callbacks:
            self._booster = lgb.train(p, train_set, num_boost_round=n_est, callbacks=callbacks)
        else:
            self._booster = lgb.train(p, train_set, num_boost_round=n_est)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape (n_samples, 2) for classes 0 and 1."""
        p1 = self._booster.predict(np.asarray(X, dtype=np.float64, order="C"))
        p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
        return np.column_stack([1.0 - p1, p1])


def _build_classifier(kind: str, override_params: dict | None = None):
    override_params = override_params or {}
    if kind == "histgb":
        params = {**cfg.FINAL_HISTGB_CLASSIFIER_PARAMS, **override_params}
        return HistGradientBoostingClassifier(**params)
    if kind == "randomforest":
        params = _sanitize_sklearn_forest_params({**cfg.RF_CLASSIFIER_PARAMS, **override_params})
        return RandomForestClassifier(**params)
    if kind == "gbr":
        params = {**cfg.GBR_PARAMS, **override_params}
        return GradientBoostingClassifier(**params)
    if kind == "extratrees":
        params = _sanitize_sklearn_forest_params({**cfg.EXTRATREES_PARAMS, **override_params})
        return ExtraTreesClassifier(**params)
    if kind == "bagging":
        base = DecisionTreeClassifier(random_state=cfg.RANDOM_STATE)
        params = {
            "n_estimators": 500,
            "random_state": cfg.RANDOM_STATE,
            "n_jobs": -1,
        }
        params.update(override_params)
        return BaggingClassifier(estimator=base, **params)
    if kind == "adaboost":
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "random_state": cfg.RANDOM_STATE,
        }
        params.update(override_params)
        return AdaBoostClassifier(**params)
    if kind == "logreg":
        params = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": cfg.RANDOM_STATE,
        }
        params.update(override_params)
        return LogisticRegression(**params)
    if kind == "ridgecls":
        params = {"alpha": 1.0, "random_state": cfg.RANDOM_STATE}
        params.update(override_params)
        return RidgeClassifier(**params)
    if kind == "sgd":
        params = {"loss": "log_loss", "alpha": 1e-4, "max_iter": 2000, "random_state": cfg.RANDOM_STATE}
        params.update(override_params)
        return SGDClassifier(**params)
    if kind == "passiveaggr":
        params = {"C": 1.0, "max_iter": 2000, "random_state": cfg.RANDOM_STATE}
        params.update(override_params)
        return PassiveAggressiveClassifier(**params)
    if kind == "knn":
        params = {"n_neighbors": 15, "weights": "distance"}
        params.update(override_params)
        return KNeighborsClassifier(**params)
    if kind == "svc":
        params = {"C": 2.0, "gamma": "scale", "probability": True, "random_state": cfg.RANDOM_STATE}
        params.update(override_params)
        return SVC(**params)
    if kind == "linearsvc":
        params = {"C": 1.0, "max_iter": 5000, "random_state": cfg.RANDOM_STATE}
        params.update(override_params)
        return LinearSVC(**params)
    if kind == "gaussiannb":
        params = {}
        params.update(override_params)
        return GaussianNB(**params)
    if kind == "bernoullinb":
        params = {"alpha": 1.0}
        params.update(override_params)
        return BernoulliNB(**params)
    if kind == "lda":
        params = {}
        params.update(override_params)
        return LinearDiscriminantAnalysis(**params)
    if kind == "qda":
        params = {}
        params.update(override_params)
        return QuadraticDiscriminantAnalysis(**params)
    if kind == "lgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed; `pip install lightgbm` or drop lgbm from candidates")
        params = {**cfg.LGBM_CLASSIFIER_PARAMS, **override_params}
        return _LGBMBinaryClassifier(**params)
    raise ValueError(f"Unknown classifier kind: {kind}")

def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6), dpi=cfg.FIGURE_DPI, facecolor="white")
    sns.scatterplot(x=y_true, y=y_pred, edgecolor="w", linewidth=0.4, alpha=0.75)
    lims = [min(float(np.min(y_true)), float(np.min(y_pred))), max(float(np.max(y_true)), float(np.max(y_pred)))]
    plt.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
    plt.xlabel("Observed pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    res = y_pred - y_true
    plt.figure(figsize=(6, 4), dpi=cfg.FIGURE_DPI, facecolor="white")
    sns.scatterplot(x=y_true, y=res, alpha=0.7, edgecolor="w", linewidth=0.3)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    plt.xlabel("Observed pIC50")
    plt.ylabel("Residual (pred − true)")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


@dataclass(frozen=True)
class FinalPreparedData:
    """
    One PaDEL + preprocessor pass; shared by multiple candidate families (train.py).

    ``scaffolds``: Murcko SMILES per row (logging / overlap). ``cv_groups``: ids for group holdout
    and GroupKFold (Murcko, Butina cluster, or Murcko when protocol is random).
    """

    X_all: pd.DataFrame
    y_all_cont: pd.Series
    scaffolds: list[str]
    cv_groups: list[str]
    train_dev_idx: np.ndarray
    test_idx: np.ndarray
    fold_id: np.ndarray


def _compute_supervised_cv_splits(
    *,
    smiles_list: list[str],
    y_for_split: np.ndarray,
    log: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Murcko scaffolds + train/test/fold assignment (protocol from cfg.FINAL_CV_PROTOCOL)."""
    scaffolds = [murcko_scaffold_smiles(s) for s in smiles_list]
    cv_proto = str(getattr(cfg, "FINAL_CV_PROTOCOL", "scaffold")).lower().strip()
    test_frac = float(getattr(cfg, "FINAL_HOLDOUT_TEST_FRAC", 0.10))
    n_all = len(y_for_split)
    rs_cv = int(cfg.FINAL_CV_RANDOM_STATE)
    log(
        f"Holdout design: FINAL_HOLDOUT_TEST_FRAC={test_frac} → nominal train_dev:test "
        f"≈ {1.0 - test_frac:.0%}:{test_frac:.0%} (grouped; molecule counts follow)."
    )

    if cv_proto == "scaffold":
        cv_groups = list(scaffolds)
        train_dev_idx, test_idx = scaffold_holdout_split(
            scaffolds=cv_groups,
            y=y_for_split,
            test_frac=test_frac,
            classification=False,
        )
        log(
            f"CV protocol=scaffold — holdout: train_dev={len(train_dev_idx)}, test={len(test_idx)}; "
            f"train_dev scaffolds={len({cv_groups[i] for i in train_dev_idx})}"
        )
        fold_id = scaffold_kfold_split(
            scaffolds=cv_groups,
            y=y_for_split,
            train_dev_idx=train_dev_idx,
            n_splits=int(cfg.FINAL_CV_FOLDS),
            classification=False,
        )
    elif cv_proto == "cluster":
        dist_t = float(getattr(cfg, "FINAL_CLUSTER_BUTINA_DIST", 0.38))
        log(
            f"CV protocol=cluster — Morgan(r={cfg.MORGAN_RADIUS}, nBits={cfg.MORGAN_N_BITS}) "
            f"+ Butina (Tanimoto distance cutoff={dist_t}). Whole clusters held out (not i.i.d. random)."
        )
        cv_groups = build_butina_cluster_group_ids(
            smiles_list,
            radius=int(cfg.MORGAN_RADIUS),
            n_bits=int(cfg.MORGAN_N_BITS),
            dist_thresh=dist_t,
        )
        n_clust = len({cv_groups[i] for i in range(n_all)})
        log(f"Butina: {n_clust} clusters over {n_all} molecules.")
        train_dev_idx, test_idx = scaffold_holdout_split(
            scaffolds=cv_groups,
            y=y_for_split,
            test_frac=test_frac,
            classification=False,
        )
        log(
            f"Cluster holdout — train_dev={len(train_dev_idx)}, test={len(test_idx)}; "
            f"train_dev clusters={len({cv_groups[i] for i in train_dev_idx})}"
        )
        fold_id = scaffold_kfold_split(
            scaffolds=cv_groups,
            y=y_for_split,
            train_dev_idx=train_dev_idx,
            n_splits=int(cfg.FINAL_CV_FOLDS),
            classification=False,
        )
    elif cv_proto == "random":
        log(
            "CV protocol=random — molecule-level shuffle holdout + KFold (NOT scaffold extrapolation). "
            "Train/test may share Bemis–Murcko scaffolds; test R2 is often much higher than scaffold "
            "holdout and must not be reported as scaffold generalization."
        )
        train_dev_idx, test_idx = random_molecule_holdout_split(
            n_samples=n_all,
            test_frac=test_frac,
            random_state=rs_cv,
        )
        log(
            f"Random holdout — train_dev: {len(train_dev_idx)}, test: {len(test_idx)}; "
            f"distinct scaffolds in test that also appear in train_dev: "
            f"(see overlap; many overlaps typically explain higher R2)"
        )
        fold_id = random_molecule_kfold_split(
            n_samples=n_all,
            train_dev_idx=train_dev_idx,
            n_splits=int(cfg.FINAL_CV_FOLDS),
            random_state=rs_cv,
        )
        te_sc = {scaffolds[i] for i in test_idx}
        tr_sc = {scaffolds[i] for i in train_dev_idx}
        overlap = len(te_sc & tr_sc)
        log(
            f"Scaffold overlap (random protocol): {overlap}/{len(te_sc)} distinct test scaffolds "
            f"also appear in train_dev — overlap drives optimistic R2 vs scaffold holdout."
        )
        cv_groups = list(scaffolds)
    else:
        raise ValueError(
            f"Unknown FINAL_CV_PROTOCOL={cv_proto!r}; use 'scaffold', 'cluster', or 'random'."
        )

    if cv_proto == "scaffold":
        log(
            "Leakage guard: nested tuning uses inner CV on outer train only; scaffold holdout excludes "
            "test scaffolds from train_dev; GroupKFold keeps each scaffold in one outer fold."
        )
    elif cv_proto == "cluster":
        log(
            "Leakage guard: nested tuning uses inner GroupKFold on outer train only; cluster holdout "
            "excludes entire Butina clusters from train_dev; easier than Murcko-only but still grouped."
        )
    else:
        log(
            "Leakage guard: nested tuning uses inner CV on outer train only; random protocol does NOT "
            "remove scaffold overlap between train_dev and test — interpret test metrics accordingly."
        )

    return train_dev_idx, test_idx, fold_id, cv_groups, scaffolds


def _coerce_features_for_sklearn(X: np.ndarray, *, log: Callable[[str], None]) -> np.ndarray:
    """
    PaDEL/RDKit can emit inf or overflow; sklearn SimpleImputer rejects non-finite input.
    Map non-finite → NaN so mean imputation (on train_dev) can run.
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    if np.isfinite(X).all():
        return X
    n_bad = int(np.size(X) - np.isfinite(X).sum())
    log(f"Sanitizing features: {n_bad} non-finite entries → NaN (mean-imputed on train_dev).")
    out = np.where(np.isfinite(X), X, np.nan)
    return out


def _winsorize_by_train_dev_quantiles(
    X: pd.DataFrame,
    train_dev_idx: np.ndarray,
    *,
    low_q: float,
    high_q: float,
    log: Callable[[str], None],
) -> pd.DataFrame:
    """Clip each column to [q_low, q_high] from train_dev only (test uses same bounds — no y leakage)."""
    if low_q >= high_q:
        raise ValueError("FINAL_WINSORIZE_* quantiles must satisfy low_q < high_q.")
    arr = X.values.astype(np.float64, copy=True)
    ref_all = arr[train_dev_idx]
    n_changed = 0
    for j in range(arr.shape[1]):
        c = ref_all[:, j]
        c = c[np.isfinite(c)]
        if c.size < 10:
            continue
        lo = float(np.quantile(c, low_q))
        hi = float(np.quantile(c, high_q))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            continue
        before = arr[:, j].copy()
        arr[:, j] = np.clip(arr[:, j], lo, hi)
        n_changed += int(np.nansum(np.abs(before - arr[:, j]) > 1e-15))
    if n_changed:
        log(
            f"Winsorize: clipped extremes using train_dev quantiles "
            f"{low_q:.3f}–{high_q:.3f} ({n_changed} cells changed)."
        )
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def build_shared_supervised_data(*, log=_log) -> FinalPreparedData:
    """
    Build feature matrix (incl. PaDEL) once and CV splits. Safe to reuse for several
    model families so benchmark-style “one feature table, many models” applies here too.
    """
    use_pre = bool(getattr(cfg, "FINAL_USE_PRETRAIN_ARTIFACTS", True))

    log(f"Loading main CSV: {cfg.MAIN_DATA_CSV}")
    df_main = load_activity_table(cfg.MAIN_DATA_CSV, cfg.SMILES_COL, cfg.TARGET_COL)
    log(f"Main table loaded: n={len(df_main)} (after dropping empty SMILES / non-numeric targets)")
    if cfg.DROP_DUPLICATE_SMILES:
        before = len(df_main)
        df_main = aggregate_duplicates(df_main, cfg.SMILES_COL, cfg.TARGET_COL, cfg.DUPLICATE_AGG)
        log(
            f"Duplicates aggregated: how={cfg.DUPLICATE_AGG} "
            f"→ n={len(df_main)} (dropped {before - len(df_main)})"
        )

    # RDKit parseability diagnostics (before feature extraction)
    invalid_examples: list[tuple[int, str]] = []
    n_valid = 0
    for i, raw in enumerate(df_main[cfg.SMILES_COL].astype(str).tolist()):
        sm = raw.strip() if raw is not None else ""
        if not sm:
            if len(invalid_examples) < 20:
                invalid_examples.append((i, sm))
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            if len(invalid_examples) < 20:
                invalid_examples.append((i, sm))
            continue
        n_valid += 1
    n_total = len(df_main)
    n_invalid = n_total - n_valid
    log(f"RDKit-parseable SMILES: {n_valid}/{n_total} (invalid/empty dropped later: {n_invalid})")
    if invalid_examples:
        shown = "; ".join([f"row{i}={s!r}" for i, s in invalid_examples[:20]])
        log(f"Invalid SMILES examples (first {min(20, len(invalid_examples))}): {shown}")

    if use_pre:
        if not cfg.PRETRAIN_PREPROCESSOR_PATH.is_file():
            raise FileNotFoundError(f"Missing pretrain preprocessor: {cfg.PRETRAIN_PREPROCESSOR_PATH}")
        log("Loading unsupervised pretrain artifacts …")
        preprocessor = joblib.load(cfg.PRETRAIN_PREPROCESSOR_PATH)
        log(
            f"Pretrain features: scaler+IPCA from {cfg.PRETRAIN_PREPROCESSOR_PATH.name}; "
            f"FINAL_USE_CONCAT_RAW_FEATURES={cfg.FINAL_USE_CONCAT_RAW_FEATURES} "
            f"({'raw selected cols + repr' if cfg.FINAL_USE_CONCAT_RAW_FEATURES else 'repr only'})."
        )
        master_cols = preprocessor["master_columns"]
    else:
        preprocessor = None
        master_cols = None
        log(
            "FINAL_USE_PRETRAIN_ARTIFACTS=False — no IPCA representation; "
            "imputer + VarianceThreshold + StandardScaler fit on train_dev only, then applied to all rows."
        )

    X_main = build_feature_matrix(
        df_main[cfg.SMILES_COL],
        morgan_radius=cfg.MORGAN_RADIUS,
        morgan_n_bits=cfg.MORGAN_N_BITS,
        master_columns=master_cols,
        log=log,
    )
    y_main = df_main.loc[X_main.index, cfg.TARGET_COL].astype(float)
    if X_main.empty or len(y_main) < 5:
        raise RuntimeError("Main dataset yielded too few valid molecules after feature extraction.")

    smiles_list = (
        df_main.loc[X_main.index, cfg.SMILES_COL]
        .astype(str)
        .str.strip()
        .reset_index(drop=True)
        .tolist()
    )
    y_arr = y_main.reset_index(drop=True).values.astype(np.float64)

    train_dev_idx, test_idx, fold_id, cv_groups, scaffolds = _compute_supervised_cv_splits(
        smiles_list=smiles_list,
        y_for_split=y_arr,
        log=log,
    )

    if use_pre:
        assert preprocessor is not None
        X_main_clean = pd.DataFrame(
            _coerce_features_for_sklearn(X_main.values, log=log),
            index=X_main.index,
            columns=X_main.columns,
        )
        X_imp = preprocessor["imputer"].transform(X_main_clean)
        X_sel = preprocessor["variance_selector"].transform(X_imp)
        X_scaled = preprocessor["scaler"].transform(X_sel)
        X_repr = preprocessor["projector"].transform(X_scaled)
        X_repr_df = pd.DataFrame(
            X_repr,
            index=X_main.index,
            columns=[f"repr_{i}" for i in range(X_repr.shape[1])],
        )
        if cfg.FINAL_USE_CONCAT_RAW_FEATURES:
            X_final_df = pd.concat(
                [
                    pd.DataFrame(
                        X_sel,
                        index=X_main.index,
                        columns=list(preprocessor["selected_columns"]),
                    ),
                    X_repr_df,
                ],
                axis=1,
            )
        else:
            X_final_df = X_repr_df
        X_all = X_final_df.reset_index(drop=True)
    else:
        X_np = _coerce_features_for_sklearn(X_main.values, log=log)
        # Feature cleanup before fitting imputer/selector/scaler
        # 1) drop columns with too many NaNs (e.g., >30% missing in train_dev)
        nan_frac = np.mean(pd.isna(X_main).values, axis=0)
        drop_nan_cols = np.where(nan_frac > 0.30)[0].tolist()
        if drop_nan_cols:
            keep_mask = np.ones(X_main.shape[1], dtype=bool)
            keep_mask[drop_nan_cols] = False
            X_main = pd.DataFrame(X_main.values[:, keep_mask], index=X_main.index, columns=[c for c, k in zip(X_main.columns, keep_mask) if k])
            X_np = _coerce_features_for_sklearn(X_main.values, log=log)
            log(f"Feature cleanup: dropped {len(drop_nan_cols)} columns with >30% missing")
        # 2) simple imputer on remaining
        imp = SimpleImputer(strategy="mean")
        imp.fit(X_np[train_dev_idx])
        X_imp = imp.transform(X_np)
        vt = VarianceThreshold(threshold=float(cfg.VARIANCE_THRESHOLD))
        vt.fit(X_imp[train_dev_idx])
        X_sel = vt.transform(X_imp)
        mask = vt.get_support()
        selected_names = [str(c) for c, ok in zip(X_main.columns, mask) if ok]
        if not selected_names:
            raise RuntimeError(
                "VarianceThreshold removed all features — lower cfg.VARIANCE_THRESHOLD or check inputs."
            )
        sc = StandardScaler()
        sc.fit(X_sel[train_dev_idx])
        X_fin = sc.transform(X_sel)
        X_all = pd.DataFrame(X_fin, columns=selected_names)

    # Final cleanup: inf → NaN, then mean-impute on train_dev (covers IPCA / PaDEL edge cases).
    X_mat = _coerce_features_for_sklearn(X_all.values, log=log)
    if np.isnan(X_mat).any():
        log("Post-process: mean-imputing remaining NaNs using train_dev statistics only.")
        post_imp = SimpleImputer(strategy="mean")
        post_imp.fit(X_mat[train_dev_idx])
        X_mat = post_imp.transform(X_mat)
    X_all = pd.DataFrame(X_mat, columns=X_all.columns)

    if bool(getattr(cfg, "FINAL_WINSORIZE_TRAIN_DEV", False)):
        X_all = _winsorize_by_train_dev_quantiles(
            X_all,
            train_dev_idx,
            low_q=float(getattr(cfg, "FINAL_WINSORIZE_LOW_Q", 0.005)),
            high_q=float(getattr(cfg, "FINAL_WINSORIZE_HIGH_Q", 0.995)),
            log=log,
        )

    y_all_cont = pd.Series(y_arr)

    pos_thr = float(cfg.CLASSIFICATION_THRESHOLD)
    n_pos = int((y_all_cont >= pos_thr).sum())
    log(
        f"Shared supervised table ready: n={len(X_all)}, features={X_all.shape[1]}; "
        f"classification positives @ {pos_thr} would be {n_pos}/{len(y_all_cont)} (for reference)."
    )
    return FinalPreparedData(
        X_all=X_all,
        y_all_cont=y_all_cont,
        scaffolds=scaffolds,
        cv_groups=cv_groups,
        train_dev_idx=train_dev_idx,
        test_idx=test_idx,
        fold_id=fold_id,
    )


def run_final_training(
    task_type: str | None = None,
    model_family: str | None = None,
    *,
    prepared_data: FinalPreparedData | None = None,
) -> dict:
    t0 = time.time()
    sns.set_style("white")

    if task_type is None:
        task_type = "classification"
    if task_type != "classification":
        raise ValueError("run_final_training() supports classification only.")

    classification_mode = True
    task_tag = "cls"

    if prepared_data is None:
        prepared_data = build_shared_supervised_data(log=_log)
    else:
        _log("Reusing shared feature matrix & CV splits (PaDEL + preprocessor run once for all candidate families).")

    X_all = prepared_data.X_all
    y_all_cont = prepared_data.y_all_cont
    scaffolds = prepared_data.scaffolds
    cv_groups = prepared_data.cv_groups
    train_dev_idx = prepared_data.train_dev_idx
    test_idx = prepared_data.test_idx
    fold_id = prepared_data.fold_id
    _cv_proto = str(getattr(cfg, "FINAL_CV_PROTOCOL", "scaffold")).lower().strip()
    _cv_tag = {
        "scaffold": "scaffold-CV",
        "cluster": "cluster-CV",
        "random": "random-CV",
    }.get(_cv_proto, _cv_proto + "-CV")

    if classification_mode:
        y_all = (y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int)
        final_kind = model_family or cfg.FINAL_CLASSIFIER_KIND
        if final_kind == "lgbm" and lgb is None:
            raise ImportError("lightgbm is not installed; install it or remove 'lgbm' from classification candidates.")
        _log(
            f"Task=classification (threshold={cfg.CLASSIFICATION_THRESHOLD}); positives={int(y_all.sum())}/{len(y_all)}"
        )
        _log(f"Training final classifier with {cfg.FINAL_CV_FOLDS}-fold {_cv_proto} CV: {final_kind}")
    else:
        y_all = y_all_cont
        final_kind = model_family or cfg.FINAL_REGRESSOR_KIND
        _supported_final_reg = ("extratrees",)
        if final_kind not in _supported_final_reg:
            raise ValueError(
                f"Final regression supports only ExtraTrees ({_supported_final_reg}). "
                f"Got {final_kind!r} — use regression_benchmark or regression_benchmark_lazy_pool for other regressors."
            )
        _log(f"Training final regressor with {cfg.FINAL_CV_FOLDS}-fold {_cv_proto} CV: {final_kind}")

    def _metrics_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    oof_pred = np.full(shape=(len(X_all),), fill_value=np.nan, dtype=float)
    fold_models = []
    roc_folds = []
    pr_folds = []
    fold_weights = []

    cfg.FINAL_FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fold_models_dir = cfg.FINAL_FOLD_MODELS_DIR / task_tag
    fold_models_dir.mkdir(parents=True, exist_ok=True)
    requested_splits = int(cfg.FINAL_CV_FOLDS)
    present_folds = sorted({int(f) for f in fold_id[train_dev_idx].tolist() if int(f) >= 0})
    n_splits = len(present_folds)
    if n_splits < 2:
        raise RuntimeError(
            f"Outer CV produced too few non-empty folds: {present_folds} (requested={requested_splits})."
        )
    if n_splits != requested_splits:
        _log(
            f"Warning: outer CV has {n_splits}/{requested_splits} non-empty folds "
            f"(fold ids present: {present_folds})."
        )

    # CV loop over folds
    for fold in present_folds:
        va_idx = np.where(fold_id == int(fold))[0]
        if len(va_idx) == 0:
            _log(f"Scaffold fold {int(fold)+1:02d} is empty; skipping.")
            continue
        tr_idx = train_dev_idx[fold_id[train_dev_idx] != int(fold)]

        X_tr = X_all.iloc[tr_idx].values
        X_va = X_all.iloc[va_idx].values

        if classification_mode:
            y_tr = y_all.values[tr_idx]
            y_va = y_all.values[va_idx]

            # Nested hyperparameter tuning (inner scaffold CV); maximize cfg.NESTED_CLASSIFICATION_METRIC
            best_params = None
            # Families that use nested inner-CV tuning for hyperparameters
            _cls_tune_families = (
                "histgb",
                "gbr",
                "extratrees",
                "randomforest",
                "lgbm",
                "bagging",
                "svc",
            )
            if cfg.NESTED_TUNING_ENABLED and final_kind in _cls_tune_families:
                outer_train_global_idx = np.array(tr_idx, dtype=int)
                outer_train_groups = [cv_groups[i] for i in outer_train_global_idx.tolist()]
                outer_train_y = y_tr

                n_subset = len(outer_train_global_idx)
                inner_fold_id = nested_inner_fold_ids(
                    protocol=_cv_proto,
                    scaffolds_subset=outer_train_groups,
                    y_subset=outer_train_y,
                    n_splits=int(cfg.NESTED_INNER_FOLDS),
                    random_state=int(cfg.NESTED_RANDOM_STATE),
                    classification=True,
                )

                if final_kind == "histgb":
                    param_dist = {
                        "max_depth": [3, 4, 5, 6, 8],
                        "learning_rate": [0.01, 0.03, 0.05],
                        "max_iter": [300, 600, 900],
                        "min_samples_leaf": [10, 20, 50],
                        "l2_regularization": [0.0, 0.1, 1.0],
                    }
                elif final_kind == "gbr":
                    param_dist = {
                        "n_estimators": [400, 800, 1200],
                        "learning_rate": [0.03, 0.05, 0.08],
                        "max_depth": [3, 4, 5],
                        "min_samples_split": [5, 10, 20],
                        "min_samples_leaf": [3, 5, 10],
                        "subsample": [0.7, 0.85, 1.0],
                    }
                elif final_kind == "extratrees":
                    # ExtraTrees tends to overfit on scaffold CV when leaves are too small
                    # or when feature subsampling is too aggressive. Use a "stability-first"
                    # search space to reduce fold-to-fold variance.
                    param_dist = {
                        "n_estimators": [800, 1200, 2000],
                        "max_depth": [None, 25, 40],
                        "min_samples_leaf": [2, 4, 6, 10],
                        "max_features": ["sqrt"],
                    }
                elif final_kind == "randomforest":
                    param_dist = {
                        "n_estimators": [400, 800, 1200],
                        "max_depth": [None, 15, 25, 40],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", 0.5, 0.7],
                    }
                elif final_kind == "bagging":
                    # Bagging over decision trees: tune ensemble size and subsampling.
                    param_dist = {
                        "n_estimators": [200, 400, 800],
                        "max_samples": [0.5, 0.7, 0.9, 1.0],
                        "max_features": [0.5, 0.7, 1.0],
                    }
                elif final_kind == "svc":
                    # RBF SVC: tune C/gamma and class_weight for stability on scaffold splits.
                    # Note: probability=True is fixed in _build_classifier for `svc`.
                    param_dist = {
                        "C": np.logspace(-2.0, 3.0, 8).tolist(),
                        "gamma": ["scale", "auto"] + np.logspace(-4.0, 1.0, 7).tolist(),
                        "class_weight": ["balanced", None],
                    }
                else:
                    param_dist = {
                        "n_estimators": [400, 800, 1200],
                        "learning_rate": [0.03, 0.05, 0.08],
                        "max_depth": [6, 8, 12, -1],
                        "num_leaves": [31, 63, 127],
                        "subsample": [0.75, 0.85, 1.0],
                        "colsample_bytree": [0.75, 0.85, 1.0],
                        "min_child_samples": [10, 20, 40],
                    }

                candidates = list(
                    ParameterSampler(
                        param_distributions=param_dist,
                        n_iter=int(cfg.NESTED_N_ITER),
                        random_state=int(cfg.NESTED_RANDOM_STATE),
                    )
                )

                best_score = -np.inf
                for cand in candidates:
                    inner_scores = []
                    for inner_fold in range(int(cfg.NESTED_INNER_FOLDS)):
                        inner_va_pos = np.where(inner_fold_id == inner_fold)[0]
                        inner_tr_pos = np.where(inner_fold_id != inner_fold)[0]
                        if len(inner_va_pos) == 0:
                            continue

                        X_inner_tr = X_tr[inner_tr_pos]
                        X_inner_va = X_tr[inner_va_pos]
                        y_inner_tr = y_tr[inner_tr_pos]
                        y_inner_va = y_tr[inner_va_pos]

                        if len(np.unique(y_inner_va)) < 2:
                            inner_scores.append(np.nan)
                            continue

                        inner_model = _build_classifier(final_kind, override_params=cand)
                        inner_model.fit(X_inner_tr, y_inner_tr)
                        inner_pred = _positive_class_score(inner_model, X_inner_va)

                        metric = cfg.NESTED_CLASSIFICATION_METRIC
                        if metric == "roc_auc":
                            _, _, fold_auc = compute_fold_roc(y_inner_va, inner_pred)
                            inner_scores.append(fold_auc)
                        elif metric == "average_precision":
                            _, _, fold_ap = compute_fold_pr(y_inner_va, inner_pred)
                            inner_scores.append(fold_ap)
                        else:
                            raise ValueError(f"Unknown NESTED_CLASSIFICATION_METRIC: {metric}")

                    inner_scores = [s for s in inner_scores if not np.isnan(s)]
                    if not inner_scores:
                        continue
                    score = float(np.mean(inner_scores))
                    if score > best_score:
                        best_score = score
                        best_params = cand

                if best_params is None:
                    _log(f"Nested tuning produced no valid candidates; using default params for fold {fold+1:02d}.")
                else:
                    _log(f"Fold {fold+1:02d} nested best params: {best_params} (score={best_score:.4f})")

            model = _build_classifier(final_kind, override_params=best_params)
            model.fit(X_tr, y_tr)
            pred_va = _positive_class_score(model, X_va)
            oof_pred[va_idx] = pred_va

            fold_roc = compute_fold_roc(y_va, pred_va)
            roc_folds.append(fold_roc)
            pr_folds.append(compute_fold_pr(y_va, pred_va))
            fold_weights.append(float(fold_roc[2]))
        else:
            y_tr = y_all.values[tr_idx]

            best_params = None
            # Nested tuning: ExtraTrees only in final regression (no GBR / HistGB).
            if cfg.NESTED_TUNING_ENABLED and final_kind == "extratrees":
                outer_train_global_idx = np.array(tr_idx, dtype=int)
                outer_train_groups = [cv_groups[i] for i in outer_train_global_idx.tolist()]
                outer_train_y = y_tr

                n_subset = len(outer_train_global_idx)
                inner_fold_id = nested_inner_fold_ids(
                    protocol=_cv_proto,
                    scaffolds_subset=outer_train_groups,
                    y_subset=outer_train_y,
                    n_splits=int(cfg.NESTED_INNER_FOLDS),
                    random_state=int(cfg.NESTED_RANDOM_STATE),
                    classification=False,
                )

                # ExtraTrees has no learning_rate; bootstrap+max_samples can reduce variance / improve test R2.
                param_dist = {
                    "n_estimators": [1200, 2000, 2500],
                    "max_depth": [15, 25, 35, None],
                    "min_samples_split": [5, 10, 20],
                    "min_samples_leaf": [4, 6, 10, 20],
                    "max_features": ["sqrt", 0.2, 0.35, 0.5],
                    "bootstrap": [True],
                    "max_samples": [0.6, 0.75, 0.9, 1.0],
                }
                inner_kind = "extratrees"

                candidates = list(
                    ParameterSampler(
                        param_distributions=param_dist,
                        n_iter=int(cfg.NESTED_N_ITER),
                        random_state=int(cfg.NESTED_RANDOM_STATE),
                    )
                )

                best_score = np.inf
                metric = cfg.NESTED_REGRESSION_METRIC

                for cand in candidates:
                    inner_scores = []
                    for inner_fold in range(int(cfg.NESTED_INNER_FOLDS)):
                        inner_va_pos = np.where(inner_fold_id == inner_fold)[0]
                        inner_tr_pos = np.where(inner_fold_id != inner_fold)[0]
                        if len(inner_va_pos) == 0:
                            continue
                        X_inner_tr = X_tr[inner_tr_pos]
                        X_inner_va = X_tr[inner_va_pos]
                        y_inner_tr = y_tr[inner_tr_pos]
                        y_inner_va = y_tr[inner_va_pos]

                        inner_model = _regression_supervised_estimator(
                            inner_kind, cand, n_features=X_tr.shape[1]
                        )
                        inner_model.fit(X_inner_tr, y_inner_tr)
                        inner_pred = inner_model.predict(X_inner_va)

                        rmse = float(np.sqrt(mean_squared_error(y_inner_va, inner_pred)))
                        mae = float(mean_absolute_error(y_inner_va, inner_pred))
                        r2 = float(r2_score(y_inner_va, inner_pred))

                        if metric == "rmse":
                            inner_scores.append(rmse)
                        elif metric == "mae":
                            inner_scores.append(mae)
                        elif metric == "r2":
                            inner_scores.append(-r2)  # minimize negative R2
                        else:
                            raise ValueError(f"Unknown NESTED_REGRESSION_METRIC: {metric}")

                    if not inner_scores:
                        continue
                    mean_inner = float(np.mean(inner_scores))

                    if mean_inner < best_score:
                        best_score = mean_inner
                        best_params = cand

                if best_params is None:
                    _log(
                        f"Nested tuning produced no valid candidates; using default params for fold {int(fold)+1:02d}."
                    )
                else:
                    _log(
                        f"Fold {int(fold)+1:02d} nested best params: {best_params} (inner={best_score:.4f})"
                    )

            model = _regression_supervised_estimator(
                final_kind, best_params, n_features=X_tr.shape[1]
            )
            model.fit(X_tr, y_tr)
            pred_va = model.predict(X_va)
            oof_pred[va_idx] = pred_va

        fold_models.append(model)
        joblib.dump(model, fold_models_dir / f"fold_{int(fold)+1:02d}.joblib")
        _log(f"Scaffold fold {int(fold)+1:02d}/{requested_splits} done")

    if np.isnan(oof_pred[train_dev_idx]).any():
        raise RuntimeError("OOF predictions contain NaNs on train_dev — scaffold CV did not cover all samples.")

    # OOF metrics and plots
    if classification_mode:
        oof_roc = compute_fold_roc(y_all.values[train_dev_idx], oof_pred[train_dev_idx])
        oof_pr = compute_fold_pr(y_all.values[train_dev_idx], oof_pred[train_dev_idx])
        # Choose decision threshold based on out-of-fold predictions on train_dev only (no leakage).
        # ROC-AUC / PR-AUC are threshold-independent, but F1/MCC/BACC depend on the threshold.
        y_oof = y_all.values[train_dev_idx]
        s_oof = oof_pred[train_dev_idx]
        thresholds = np.linspace(0.01, 0.99, 99)
        f1s = []
        mccs = []
        for t in thresholds:
            m_t = _metrics_cls(y_oof, s_oof, threshold=float(t))
            f1s.append(m_t["f1"])
            mccs.append(m_t["mcc"])
        f1s = np.asarray(f1s, dtype=float)
        mccs = np.asarray(mccs, dtype=float)
        best_idx_f1 = int(np.nanargmax(f1s))
        best_idx_mcc = int(np.nanargmax(mccs))
        best_thr_f1 = float(thresholds[best_idx_f1])
        best_thr_mcc = float(thresholds[best_idx_mcc])

        # Primary threshold: maximize F1
        m_oof_cls = _metrics_cls(y_oof, s_oof, threshold=best_thr_f1)
        _log(
            f"[Final][OOF {_cv_tag}] ROC-AUC={oof_roc[2]:.4f} PR-AUC(AP)={oof_pr[2]:.4f} "
            f"thr(F1)={best_thr_f1:.3f} F1={m_oof_cls['f1']:.4f} "
            f"BACC={m_oof_cls['balanced_acc']:.4f} MCC={m_oof_cls['mcc']:.4f} "
            f"thr(MCC)={best_thr_mcc:.3f}"
        )

        fig_dir = cfg.FIGURES_FINAL_DIR
        fold_title = (
            f"Scaffold CV — {n_splits} non-empty folds "
            f"(FINAL_CV_FOLDS={requested_splits})"
        )
        # One model family per run; not the multi-family bar chart in figures/final/.
        fam_title = f"{fold_title} — model={final_kind}"
        plot_cv_roc(
            roc_folds,
            fig_dir / task_tag / "final_scaffold_cv_roc.png",
            f"{fam_title} — ROC",
            dpi=cfg.FIGURE_DPI,
        )
        baseline = float(np.mean(y_all.values[train_dev_idx]))
        plot_cv_pr(
            pr_folds,
            baseline,
            fig_dir / task_tag / "final_scaffold_cv_pr.png",
            f"{fam_title} — PR",
            dpi=cfg.FIGURE_DPI,
        )
    else:
        m_oof = _metrics_reg(y_all.values[train_dev_idx], oof_pred[train_dev_idx])
        _log(
            f"[Final][OOF {_cv_tag}] RMSE={m_oof['rmse']:.4f} MAE={m_oof['mae']:.4f} R2={m_oof['r2']:.4f}"
        )
        _plot_scatter(
            y_all.values[train_dev_idx],
            oof_pred[train_dev_idx],
            cfg.FIGURES_FINAL_DIR / task_tag / "final_scaffold_oof_pred_vs_true.png",
            f"Final scaffold OOF ({n_splits} folds, FINAL_CV_FOLDS={requested_splits})",
        )
        _plot_residuals(
            y_all.values[train_dev_idx],
            oof_pred[train_dev_idx],
            cfg.FIGURES_FINAL_DIR / task_tag / "final_scaffold_oof_residuals.png",
            f"Final scaffold OOF residuals ({n_splits} folds, FINAL_CV_FOLDS={requested_splits})",
        )

    # Ensemble evaluation on scaffold holdout test (only final evaluation).
    ensemble = MeanEnsembleRegressor(
        models=fold_models,
        use_predict_proba=classification_mode,
        weights=(
            fold_weights
            if (classification_mode and bool(getattr(cfg, "CLASSIFICATION_ENSEMBLE_WEIGHTED", True)))
            else None
        ),
    )
    X_te = X_all.iloc[test_idx].values
    if classification_mode:
        y_te = y_all.values[test_idx]
    else:
        y_te = y_all.values[test_idx]

    te_pred = ensemble.predict(X_te)
    if classification_mode:
        te_roc = compute_fold_roc(y_te, te_pred)
        te_pr = compute_fold_pr(y_te, te_pred)
        m_te_cls = _metrics_cls(y_te, te_pred, threshold=best_thr_f1)
        _log(
            f"[Final][Test {_cv_tag}] ROC-AUC={te_roc[2]:.4f} PR-AUC(AP)={te_pr[2]:.4f} "
            f"thr(F1)={best_thr_f1:.3f} F1={m_te_cls['f1']:.4f} "
            f"BACC={m_te_cls['balanced_acc']:.4f} MCC={m_te_cls['mcc']:.4f}"
        )
        # Per-family arrays for multi-model ROC/PR overlay plots (train.py).
        _ov_dir = cfg.RESULTS_DIR / "classification_candidate_predictions"
        _ov_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            _ov_dir / f"{final_kind}.npz",
            y_test=np.asarray(y_te, dtype=np.float64),
            score_test=np.asarray(te_pred, dtype=np.float64),
            y_oof=np.asarray(y_all.values[train_dev_idx], dtype=np.float64),
            score_oof=np.asarray(oof_pred[train_dev_idx], dtype=np.float64),
        )
    else:
        m_te = _metrics_reg(y_te, te_pred)
        _log(f"[Final][Test {_cv_tag}] RMSE={m_te['rmse']:.4f} MAE={m_te['mae']:.4f} R2={m_te['r2']:.4f}")

    # Save artifacts (fold models + ensemble)
    ensemble_path = cfg.FINAL_ENSEMBLE_PATH.with_name(f"{cfg.FINAL_ENSEMBLE_PATH.stem}_{task_tag}.joblib")

    joblib.dump(ensemble, ensemble_path)
    joblib.dump(
        {
            "task_type": task_type,
            "final_cv_protocol": _cv_proto,
            "metrics_holdout": (
                {"test": {"roc_auc": float(te_roc[2]), "pr_auc": float(te_pr[2]), **m_te_cls}}
                if classification_mode
                else {"test": {k: m_te[k] for k in ("rmse", "mae", "r2")}}
            ),
            "metrics_oof": (
                {"roc_auc": float(oof_roc[2]), "pr_auc": float(oof_pr[2]), **m_oof_cls}
                if classification_mode
                else {k: m_oof[k] for k in ("rmse", "mae", "r2")}
            ),
            "threshold_selection": (
                {"thr_best_f1": best_thr_f1, "thr_best_mcc": best_thr_mcc, "threshold_used_for_metrics": best_thr_f1}
                if classification_mode
                else None
            ),
            "n_samples": int(len(X_all)),
            "n_folds_requested": int(cfg.FINAL_CV_FOLDS),
            "n_folds_effective": int(n_splits),
            "fold_metrics": (
                {
                    "roc_auc_by_fold": [float(x[2]) for x in roc_folds],
                    "pr_auc_by_fold": [float(x[2]) for x in pr_folds],
                }
                if classification_mode
                else None
            ),
            "split": {"train_dev": int(len(train_dev_idx)), "test": int(len(test_idx))},
            "seconds": float(time.time() - t0),
            "leakage_prevention": {
                "feature_pipeline": (
                    "transform_only; fitted on unsupervised pretrain (not on main labels)"
                    if bool(getattr(cfg, "FINAL_USE_PRETRAIN_ARTIFACTS", True))
                    else "imputer+variance+StandardScaler fit on train_dev only; no IPCA repr"
                ),
                "holdout_protocol": (
                    "scaffold-group test excluded from train_dev; outer CV = GroupKFold by scaffold"
                    if _cv_proto == "scaffold"
                    else (
                        "Butina cluster holdout; outer CV = GroupKFold by cluster (not i.i.d. random)"
                        if _cv_proto == "cluster"
                        else "random molecule holdout; outer CV = shuffled KFold; train/test may share scaffolds"
                    )
                ),
                "nested_tuning": "hyperparameters chosen on inner CV folds of outer train only",
                "test_metrics": "evaluated once on holdout; not used for tuning",
                "regression_fold_feature_screening": (
                    f"{getattr(cfg, 'REGRESSION_FEATURE_SELECTION', 'none')} (fit on each outer train fold only)"
                    if not classification_mode
                    else "n/a"
                ),
            },
        },
        cfg.FINAL_TRAINING_METADATA_PATH.with_name(f"{cfg.FINAL_TRAINING_METADATA_PATH.stem}_{task_tag}.joblib"),
    )

    _log(f"Final artifacts saved under: {cfg.MODELS_DIR}")
    _log(f"Final finished in {time.time() - t0:.1f}s")

    return {
        "task_type": task_type,
        "model_family": final_kind,
        "metrics_oof": (
            {"roc_auc": float(oof_roc[2]), "pr_auc": float(oof_pr[2]), **m_oof_cls}
            if classification_mode
            else {k: m_oof[k] for k in ("rmse", "mae", "r2")}
        ),
        "metrics_test": (
            {"roc_auc": float(te_roc[2]), "pr_auc": float(te_pr[2]), **m_te_cls}
            if classification_mode
            else {k: m_te[k] for k in ("rmse", "mae", "r2")}
        ),
        "seconds": float(time.time() - t0),
    }


if __name__ == "__main__":
    run_final_training()

