#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-hoc evidence for *model-based* feature relevance (not chemical causality).

Addresses common reviewer concerns:
  (1) Consistency: stability of ranked importances across scaffold CV folds.
  (2) Dose-response style behavior: partial dependence of predicted positive-class
      probability on selected high-importance input dimensions.
  (3) Global allocation: cumulative Gini mass over *all* descriptors (how much
      of total importance sits in the top-ranked fraction of features).

Notes
-----
- IPCA components here are *compressed unsupervised coordinates*, not validated
  mechanistic descriptors. PDPs over repr_* show how the *supervised* model uses
  those coordinates, not ground-truth pathway importance.
- Target prediction metrics (ROC-AUC / PR-AUC) remain the only fully calibrated
  notion of "accuracy"; feature analyses are supportive and explicitly bounded.

Outputs (default): results/figures/feature_evidence/
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay

import config as cfg
from src.final_module import _build_classifier, build_shared_supervised_data
from src.publication_figures import apply_journal_style, single_column_figsize


def _log(msg):
    print(msg, flush=True)


def _topk_indices(imp, k):
    k = min(int(k), len(imp))
    order = np.argsort(-np.asarray(imp, dtype=float))
    return set(int(order[i]) for i in range(k))


def _mean_pairwise_jaccard(mat, ks):
    """mat: (n_folds, n_features) importance matrix."""
    n = mat.shape[0]
    out = {}
    for k in ks:
        jac = []
        for i in range(n):
            for j in range(i + 1, n):
                a = _topk_indices(mat[i], k)
                b = _topk_indices(mat[j], k)
                inter = len(a & b)
                union = len(a | b) or 1
                jac.append(inter / union)
        out[f"jaccard_top{k}_mean"] = float(np.mean(jac)) if jac else float("nan")
    return out


def _spearman_rank_corr_matrix(mat):
    """Pairwise Spearman between rows of mat (each row = one fold vector)."""
    ranks = pd.DataFrame(mat).rank(axis=1, method="average").to_numpy(dtype=float)
    return np.asarray(np.corrcoef(ranks), dtype=float)


def _try_load_fold_models(present_folds, n_features):
    """Load fold models in the same order as ``present_folds`` (matches final_module naming)."""
    d = cfg.MODELS_DIR / "fold_models" / "cls"
    if not d.is_dir():
        return None
    models = []
    for fold in present_folds:
        p = d / f"fold_{int(fold) + 1:02d}.joblib"
        if not p.is_file():
            _log(f"[feature_evidence] Missing expected fold artifact: {p}")
            return None
        m = joblib.load(p)
        n_in = getattr(m, "n_features_in_", None)
        if n_in is not None and int(n_in) != int(n_features):
            _log(f"[feature_evidence] Skip loaded fold model {p.name}: n_features_in={n_in} != {n_features}")
            return None
        if not hasattr(m, "feature_importances_"):
            _log(f"[feature_evidence] Skip loaded fold model {p.name}: no feature_importances_")
            return None
        models.append(m)
    return models if models else None


def main():
    parser = argparse.ArgumentParser(description="Feature importance consistency + PDP dose-response style plots.")
    parser.add_argument("--family", type=str, default="extratrees", help="Classifier family (default: extratrees).")
    parser.add_argument("--top-k-report", type=int, default=100, help="Rows in CSV summary.")
    parser.add_argument("--pdp-top", type=int, default=8, help="Number of top features for partial dependence grid.")
    parser.add_argument("--refit", action="store_true", help="Refit per fold with cfg defaults (ignore saved fold models).")
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
        _log("[feature_evidence] --no-pretrain: FINAL_USE_PRETRAIN_ARTIFACTS=False")
    _log("[feature_evidence] Building shared supervised matrix (same pipeline as final training) …")
    prepared = build_shared_supervised_data(log=_log)
    X_all = prepared.X_all
    feature_names = list(X_all.columns)
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    fold_id = np.asarray(prepared.fold_id, dtype=int)
    present_folds = sorted({int(f) for f in fold_id[train_dev_idx].tolist() if int(f) >= 0})
    if len(present_folds) < 2:
        raise RuntimeError("Need at least 2 CV folds on train_dev for stability analysis.")

    X_td = X_all.iloc[train_dev_idx].values
    y_td = y[train_dev_idx]
    fold_td = fold_id[train_dev_idx]

    n_features = X_td.shape[1]
    imp_rows = []

    loaded = None if args.refit else _try_load_fold_models(present_folds, n_features)
    if loaded is not None and len(loaded) == len(present_folds):
        _log(f"[feature_evidence] Using {len(loaded)} saved fold models from {cfg.MODELS_DIR / 'fold_models' / 'cls'}")
        for m in loaded:
            imp_rows.append(np.asarray(m.feature_importances_, dtype=float))
    else:
        if loaded is not None:
            _log("[feature_evidence] Saved fold models missing or mismatch; refitting per outer fold …")
        else:
            _log("[feature_evidence] No saved fold models; refitting per outer fold (defaults from config) …")
        for fold in present_folds:
            va_mask = fold_td == int(fold)
            tr_mask = fold_td != int(fold)
            X_tr, y_tr = X_td[tr_mask], y_td[tr_mask]
            X_va, y_va = X_td[va_mask], y_td[va_mask]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                _log(f"[feature_evidence] Warning: fold {fold} skipped (single class in train or val).")
                continue
            model = _build_classifier(family, override_params=None)
            model.fit(X_tr, y_tr)
            imp_rows.append(np.asarray(model.feature_importances_, dtype=float))

    if len(imp_rows) < 2:
        raise RuntimeError("Not enough fold importance vectors — check CV folds / class balance.")

    imp_mat = np.vstack(imp_rows)
    mean_imp = np.mean(imp_mat, axis=0)
    std_imp = np.std(imp_mat, axis=0)

    # --- Consistency: Spearman correlation of importance ranks across folds ---
    corr = _spearman_rank_corr_matrix(imp_mat)
    fold_labels = [f"Fold {f+1}" for f in range(len(imp_rows))]
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
    p_corr = out_dir / "feature_importance_fold_rank_correlation.png"
    plt.savefig(p_corr, bbox_inches="tight")
    plt.close()

    jstats = _mean_pairwise_jaccard(imp_mat, (20, 50, 100))
    with open(out_dir / "feature_importance_stability.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_folds": int(len(imp_rows)),
                "n_features": int(n_features),
                "family": family,
                "mean_offdiag_spearman": float(np.mean(corr[np.triu_indices(len(corr), k=1)])),
                **jstats,
                "note": "Importances are forest mean decrease in impurity; ranks compared across folds.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --- CSV: top features by mean importance ---
    order = np.argsort(-mean_imp)
    topn = min(int(args.top_k_report), len(order))
    rows = []
    for rank, j in enumerate(order[:topn], start=1):
        j = int(j)
        rows.append(
            {
                "rank": rank,
                "feature": feature_names[j],
                "mean_importance": float(mean_imp[j]),
                "std_importance_across_folds": float(std_imp[j]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "feature_importance_top_summary.csv", index=False)

    # --- All features: cumulative importance (global "where is the mass?") ---
    dec = np.sort(np.asarray(mean_imp, dtype=float))[::-1]
    csum = np.cumsum(dec)
    tot_imp = float(csum[-1]) if csum[-1] > 0 else 1.0
    cum_frac = csum / tot_imp
    ranks_all = np.arange(1, n_features + 1, dtype=int)
    rank50 = int(np.searchsorted(cum_frac, 0.5, side="left")) + 1
    rank90 = int(np.searchsorted(cum_frac, 0.9, side="left")) + 1
    apply_journal_style()
    w_c, h_c = single_column_figsize(2.35)
    fig_c, ax_c = plt.subplots(figsize=(w_c, h_c), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    ax_c.plot(ranks_all, cum_frac, color="#2c7fb8", linewidth=1.15)
    ax_c.axhline(0.5, color="#aaaaaa", linestyle=":", linewidth=0.75)
    ax_c.axhline(0.9, color="#aaaaaa", linestyle=":", linewidth=0.75)
    ax_c.set_xlim(1, int(n_features))
    ax_c.set_ylim(0.0, 1.02)
    ax_c.set_xlabel("Feature rank (mean Gini, descending)")
    ax_c.set_ylabel("Cumulative fraction of\ntotal importance")
    ax_c.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.55)
    ax_c.set_axisbelow(True)
    fig_c.tight_layout()
    p_cum = out_dir / "feature_importance_cumulative_gini.png"
    fig_c.savefig(p_cum, bbox_inches="tight", dpi=int(cfg.FIGURE_DPI))
    plt.close(fig_c)
    _log(
        "[feature_evidence] Cumulative Gini: top %d / %d features reach 50%% mass; "
        "top %d reach 90%%."
        % (rank50, n_features, rank90)
    )

    # --- Dose-response style: partial dependence on full train_dev refit ---
    _log("[feature_evidence] Refitting single model on full train_dev for partial dependence …")
    X_df = X_all.iloc[train_dev_idx]
    full_model = _build_classifier(family, override_params=None)
    full_model.fit(X_df, y_td)
    top_pdp = min(int(args.pdp_top), n_features)
    top_idx = [int(j) for j in np.argsort(-mean_imp)[:top_pdp]]

    # PartialDependenceDisplay: ``kind`` requires newer sklearn; keep compatible.
    # Pass a DataFrame so every subplot is labeled with morgan_*/maccs_*/repr_* names.
    _pdp_kw = dict(
        estimator=full_model,
        X=X_df,
        features=top_idx,
        grid_resolution=25,
        target=1,
        random_state=int(cfg.RANDOM_STATE),
    )
    try:
        disp = PartialDependenceDisplay.from_estimator(**_pdp_kw, kind="average")
    except TypeError:
        disp = PartialDependenceDisplay.from_estimator(**_pdp_kw)
    max_w = float(cfg.FIGURE_WIDTH_DOUBLE_COL_IN)
    _ag = disp.axes_
    if isinstance(_ag, tuple) and len(_ag) > 0 and not isinstance(_ag[0], plt.Axes):
        # Some sklearn versions: tuple of rows, each row a tuple/ndarray of Axes
        _n_rows = len(_ag)
        _n_cols = len(_ag[0]) if _n_rows else 1
    else:
        _arr = np.asarray(_ag)
        if _arr.ndim == 2:
            _n_rows, _n_cols = int(_arr.shape[0]), int(_arr.shape[1])
        elif _arr.ndim == 1 and _arr.size > 0:
            _n_ax = int(_arr.size)
            _n_cols = min(3, max(1, _n_ax))
            _n_rows = int(np.ceil(float(_n_ax) / float(_n_cols)))
        else:
            _n_cols = min(3, max(1, int(np.ceil(np.sqrt(float(top_pdp))))))
            _n_rows = int(np.ceil(float(top_pdp) / float(_n_cols)))
    # Width: cap at journal double-column; height: enough per row so titles do not
    # overlap the previous row's x-axis (do not use max_w*0.52 — that squashes 3-row grids).
    _w_per_col = 2.05
    _h_per_row = 1.95
    calc_w = min(_w_per_col * _n_cols + 0.85, max_w)
    calc_h = _h_per_row * _n_rows + 0.85
    _fig = disp.figure_
    _fig.set_size_inches(calc_w, calc_h)
    # Manual spacing: tight_layout + set_title often collides row titles with the
    # x-axis of the row above; subplots_adjust(hspace=...) is more predictable.
    _fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.09, hspace=0.58, wspace=0.30)

    # sklearn only labels the bottom-most panel in each column on the x axis; label
    # every PDP with morgan_*/maccs_*/repr_* using axes-relative text above each panel.
    _fs_title = max(5, int(cfg.FIGURE_TICK_PT) - 1)
    _axes_flat = np.ravel(np.asarray(disp.axes_))
    for k, j in enumerate(top_idx):
        if k >= len(_axes_flat):
            break
        ax = _axes_flat[k]
        if ax is None:
            continue
        ax.set_title("")
        if ax.get_xlabel() and ax.get_xlabel() not in ("", "Partial dependence"):
            ax.set_xlabel("")
        ax.text(
            0.5,
            1.05,
            feature_names[int(j)],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=_fs_title,
            clip_on=False,
        )
    for k in range(len(top_idx), len(_axes_flat)):
        _ax_extra = _axes_flat[k]
        if _ax_extra is not None:
            _ax_extra.set_visible(False)
    p_pdp = out_dir / "partial_dependence_top_features.png"
    disp.figure_.savefig(p_pdp, bbox_inches="tight", dpi=int(cfg.FIGURE_DPI))
    plt.close("all")

    _log(f"[feature_evidence] Done. Figures and tables under: {out_dir}")
    _log(f"  - {p_corr.name}")
    _log(f"  - {p_cum.name}")
    _log(f"  - {p_pdp.name}")
    _log("  - feature_importance_stability.json")
    _log("  - feature_importance_top_summary.csv")


if __name__ == "__main__":
    main()
