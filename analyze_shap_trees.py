#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TreeSHAP-based post-hoc explanations for tree classifiers (default: ExtraTrees).

Complements ``analyze_feature_evidence.py``:
  - Global attribution: summary (beeswarm) of SHAP values on a train_dev sample.
  - Consistency: cross-fold stability of *ordered* mean(|SHAP|) feature rankings
    computed on each outer-fold validation set (same splits as final pipeline).
  - Dose--response style: SHAP dependence plots for top features (SHAP vs feature value).

This does **not** supply biological ground truth; it strengthens the *model-based*
evidence package requested when discussing feature analyses in manuscripts.
"""

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg
from src.publication_figures import apply_journal_style, double_column_figsize
from analyze_feature_evidence import (
    _mean_pairwise_jaccard,
    _spearman_rank_corr_matrix,
    _try_load_fold_models,
)
from src.final_module import _build_classifier, build_shared_supervised_data

try:
    import shap
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: shap. Install with:\n"
        "  pip install \"shap>=0.43\"\n"
        f"Original error: {e}"
    ) from e


def _log(msg):
    print(msg, flush=True)


def _positive_class_shap(sv):
    """Normalize TreeExplainer output to (n_samples, n_features) for the positive class."""
    if isinstance(sv, list):
        if len(sv) == 2:
            return np.asarray(sv[1], dtype=np.float64)
        return np.asarray(sv[-1], dtype=np.float64)
    arr = np.asarray(sv, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    return arr


def _subsample_rows(X, y, max_rows, random_state):
    n = X.shape[0]
    if n <= max_rows:
        return X, y, np.arange(n, dtype=int)
    rng = np.random.RandomState(int(random_state))
    idx = rng.choice(n, size=int(max_rows), replace=False)
    return X[idx], y[idx], idx


def main():
    parser = argparse.ArgumentParser(description="TreeSHAP: summary, dependence, cross-fold stability.")
    parser.add_argument("--family", type=str, default="extratrees", help="Tree classifier family (default: extratrees).")
    parser.add_argument("--refit", action="store_true", help="Refit each outer fold (ignore saved fold models).")
    parser.add_argument("--max-explain-rows", type=int, default=1200, help="Max train_dev rows for global SHAP plot.")
    parser.add_argument("--dep-top", type=int, default=3, help="Number of top features for dependence subplots.")
    parser.add_argument(
        "--summary-max-display",
        type=int,
        default=25,
        help="Top-K features in beeswarm summary (fewer rows if overlap persists).",
    )
    parser.add_argument(
        "--summary-inches-per-feature",
        type=float,
        default=0.38,
        help="Vertical figure height budget per ranked feature (in); larger = more space for y-axis names.",
    )
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Do not load IPCA pretrain artifacts (fit imputer/VT/scaler on train_dev only).",
    )
    args = parser.parse_args()

    family = str(args.family).strip().lower()
    out_dir = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.TASK_TYPE = "classification"
    if bool(getattr(args, "no_pretrain", False)):
        cfg.FINAL_USE_PRETRAIN_ARTIFACTS = False
        _log("[shap] --no-pretrain: FINAL_USE_PRETRAIN_ARTIFACTS=False")
    _log("[shap] Building shared supervised matrix …")
    prepared = build_shared_supervised_data(log=_log)
    X_all = prepared.X_all
    feature_names = list(X_all.columns)
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    fold_id = np.asarray(prepared.fold_id, dtype=int)
    present_folds = sorted({int(f) for f in fold_id[train_dev_idx].tolist() if int(f) >= 0})
    if len(present_folds) < 2:
        raise RuntimeError("Need at least 2 CV folds on train_dev.")

    X_td = X_all.iloc[train_dev_idx].values.astype(np.float64, copy=False)
    y_td = y[train_dev_idx]
    fold_td = fold_id[train_dev_idx]
    n_features = X_td.shape[1]

    fold_models = None if args.refit else _try_load_fold_models(present_folds, n_features)
    rows_mean_abs = []

    for fi, fold in enumerate(present_folds):
        va_mask = fold_td == int(fold)
        tr_mask = fold_td != int(fold)
        X_tr, y_tr = X_td[tr_mask], y_td[tr_mask]
        X_va, y_va = X_td[va_mask], y_td[va_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            _log(f"[shap] Skip fold {fold}: single class in train or val.")
            continue

        if fold_models is not None and fi < len(fold_models):
            model = fold_models[fi]
        else:
            model = _build_classifier(family, override_params=None)
            model.fit(X_tr, y_tr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_va, check_additivity=False)
        phi = _positive_class_shap(sv)
        mean_abs = np.mean(np.abs(phi), axis=0)
        rows_mean_abs.append(mean_abs)

    if len(rows_mean_abs) < 2:
        raise RuntimeError("Not enough fold-level SHAP profiles.")

    shap_mat = np.vstack(rows_mean_abs)
    mean_profile = np.mean(shap_mat, axis=0)
    std_profile = np.std(shap_mat, axis=0)

    corr = _spearman_rank_corr_matrix(shap_mat)
    fold_labels = ["Fold %d" % (i + 1) for i in range(len(rows_mean_abs))]
    apply_journal_style()
    wh = float(cfg.FIGURE_WIDTH_SINGLE_COL_IN)
    plt.figure(figsize=(wh, wh), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        xticklabels=fold_labels,
        yticklabels=fold_labels,
        square=True,
        cbar_kws={"label": "\u03c1"},
    )
    plt.xlabel("Fold")
    plt.ylabel("Fold")
    plt.tight_layout()
    p_rank = out_dir / "shap_fold_meanabs_rank_correlation.png"
    plt.savefig(p_rank, bbox_inches="tight")
    plt.close()

    jstats = _mean_pairwise_jaccard(shap_mat, (20, 50, 100))
    with open(out_dir / "shap_stability.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_folds": int(len(rows_mean_abs)),
                "n_features": int(n_features),
                "family": family,
                "mean_offdiag_spearman": float(np.mean(corr[np.triu_indices(len(corr), k=1)])),
                **jstats,
                "note": "Per-fold vector = mean(abs(SHAP)) on that fold's validation molecules; "
                "sklearn TreeExplainer; positive-class SHAP channel.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    order = np.argsort(-mean_profile)
    topn = min(100, len(order))
    csv_rows = []
    for rank, j in enumerate(order[:topn], start=1):
        j = int(j)
        csv_rows.append(
            {
                "rank": rank,
                "feature": feature_names[j],
                "mean_abs_shap_across_folds": float(mean_profile[j]),
                "std_mean_abs_shap_across_folds": float(std_profile[j]),
            }
        )
    pd.DataFrame(csv_rows).to_csv(out_dir / "shap_top_mean_abs_summary.csv", index=False)

    # --- Global summary on full train_dev (refit) + subsample for speed ---
    _log("[shap] Refitting on full train_dev for global summary / dependence …")
    full_model = _build_classifier(family, override_params=None)
    full_model.fit(X_td, y_td)
    X_sub, y_sub, _ = _subsample_rows(X_td, y_td, int(args.max_explain_rows), cfg.RANDOM_STATE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(full_model)
        sv_full = explainer.shap_values(X_sub, check_additivity=False)
    phi_sub = _positive_class_shap(sv_full)

    # Legacy summary_plot: height must scale with max_display or y-labels overlap (common SHAP issue).
    X_df = pd.DataFrame(X_sub, columns=feature_names)
    md = min(int(args.summary_max_display), int(phi_sub.shape[1]))
    w_sum = float(cfg.FIGURE_WIDTH_DOUBLE_COL_IN)
    pad_y = 2.35
    h_sum = min(11.2, max(5.4, float(args.summary_inches_per_feature) * float(md) + pad_y))
    # plot_size sets figsize inside SHAP; avoid a duplicate empty plt.figure.
    shap.summary_plot(
        phi_sub,
        X_df,
        max_display=md,
        show=False,
        plot_size=(w_sum, h_sum),
    )
    _log("[shap] Summary beeswarm size: %.2f x %.2f in, max_display=%d" % (w_sum, h_sum, md))
    fig = plt.gcf()
    left_m = min(0.36, 0.15 + 0.0048 * float(md))
    for ax in fig.axes:
        if len(ax.get_yticklabels()) > 6:
            ax.tick_params(axis="y", which="major", labelsize=int(cfg.FIGURE_FONT_PT), pad=4)
            for t in ax.get_yticklabels():
                t.set_ha("right")
        ax.tick_params(axis="x", which="major", labelsize=int(cfg.FIGURE_TICK_PT))
    for ax in fig.axes:
        xl = (ax.get_xlabel() or "").lower()
        yl = (ax.get_ylabel() or "").lower()
        if "feature value" in xl:
            ax.set_xlabel("")
        elif "shap" in xl and "impact" in xl:
            ax.set_xlabel("SHAP value")
        if "feature value" in yl:
            ax.set_ylabel("")
    fig.subplots_adjust(left=left_m, right=0.97, top=0.96, bottom=0.10)
    p_sum = out_dir / "shap_summary_beeswarm.png"
    plt.savefig(p_sum, bbox_inches="tight")
    plt.close()

    # Dependence plots for top features (SHAP vs feature value)
    dep_k = min(int(args.dep_top), len(order))
    w_d, h_d = double_column_figsize(3.55)
    fig_w = min(float(cfg.FIGURE_WIDTH_DOUBLE_COL_IN), 2.15 * float(dep_k))
    fig, axes = plt.subplots(1, dep_k, figsize=(fig_w, h_d), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    if dep_k == 1:
        axes = [axes]
    for ax, j in zip(axes, order[:dep_k]):
        j = int(j)
        xv = X_sub[:, j]
        yv = phi_sub[:, j]
        ax.scatter(xv, yv, s=8, alpha=0.35, c="#2c7fb8", edgecolors="none")
        ax.set_xlabel(feature_names[j])
        ax.set_ylabel("SHAP value")
    fig.tight_layout()
    p_dep = out_dir / "shap_dependence_top_features.png"
    fig.savefig(p_dep, bbox_inches="tight")
    plt.close("all")

    _log("[shap] Done. Outputs under: %s" % out_dir)
    _log("  - %s" % p_rank.name)
    _log("  - %s" % p_sum.name)
    _log("  - %s" % p_dep.name)
    _log("  - shap_stability.json")
    _log("  - shap_top_mean_abs_summary.csv")


if __name__ == "__main__":
    main()
