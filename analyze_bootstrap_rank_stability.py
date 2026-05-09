#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap stability of *ordered* feature-importance rankings (ExtraTrees).

Addresses reviewer requests for robustness of a ranked feature set under data
resampling: pairwise Spearman correlation of importance ranks across bootstrap
replicates, plus mean Kendall's tau (optional scipy), Top-K frequency, and a
compact heatmap of rank-correlations among the first N bootstraps.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

import config as cfg
from src.final_module import build_shared_supervised_data
from src.publication_figures import apply_journal_style

try:
    from scipy.stats import kendalltau
except Exception:  # pragma: no cover
    kendalltau = None


def _log(msg):
    print(msg, flush=True)


def _pairwise_kendall(mat):
    """Mean Kendall tau over upper triangle of row-wise rank vectors."""
    if kendalltau is None:
        return float("nan")
    n = mat.shape[0]
    ranks = pd.DataFrame(mat).rank(axis=1, method="average").to_numpy(dtype=float)
    taus = []
    for i in range(n):
        for j in range(i + 1, n):
            t, _ = kendalltau(ranks[i], ranks[j])
            if not np.isnan(t):
                taus.append(float(t))
    return float(np.mean(taus)) if taus else float("nan")


def _spearman_corr_matrix(mat):
    ranks = pd.DataFrame(mat).rank(axis=1, method="average").to_numpy(dtype=float)
    return np.asarray(np.corrcoef(ranks), dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap rank stability for forest importances.")
    parser.add_argument("--n-bootstrap", type=int, default=40, help="Number of bootstrap resamples.")
    parser.add_argument("--n-estimators", type=int, default=400, help="Trees per fit (lower = faster).")
    parser.add_argument("--heatmap-subset", type=int, default=12, help="First K bootstraps in heatmap.")
    parser.add_argument("--top-k-freq", type=int, default=30, help="Report frequency in top-K.")
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Do not load IPCA pretrain artifacts (fit imputer/VT/scaler on train_dev only).",
    )
    args = parser.parse_args()

    out_dir = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.TASK_TYPE = "classification"
    if bool(getattr(args, "no_pretrain", False)):
        cfg.FINAL_USE_PRETRAIN_ARTIFACTS = False
        _log("[bootstrap] --no-pretrain: FINAL_USE_PRETRAIN_ARTIFACTS=False")
    _log("[bootstrap] Loading supervised data …")
    prepared = build_shared_supervised_data(log=_log)
    X_all = prepared.X_all
    names = list(X_all.columns)
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    X_td = X_all.iloc[train_dev_idx].values.astype(np.float64, copy=False)
    y_td = y[train_dev_idx]

    if len(np.unique(y_td)) < 2:
        raise RuntimeError("train_dev is single-class.")

    n_samples = X_td.shape[0]
    n_features = X_td.shape[1]
    B = int(args.n_bootstrap)
    rs_base = int(cfg.RANDOM_STATE)

    et_params = dict(cfg.EXTRATREES_PARAMS)
    et_params["n_estimators"] = int(args.n_estimators)

    imp_mat = np.zeros((B, n_features), dtype=np.float64)
    oob_aucs = []

    idx_all = np.arange(n_samples, dtype=int)
    for b in range(B):
        rng = np.random.RandomState(rs_base + b)
        boot_idx = resample(
            idx_all,
            replace=True,
            n_samples=n_samples,
            stratify=y_td,
            random_state=rng,
        )
        X_b = X_td[boot_idx]
        y_b = y_td[boot_idx]
        ep = dict(et_params)
        ep["random_state"] = int(rs_base + b)
        clf = ExtraTreesClassifier(**ep)
        clf.fit(X_b, y_b)
        imp_mat[b, :] = np.asarray(clf.feature_importances_, dtype=np.float64)
        # OOB-like sanity: holdout mask (samples not in bootstrap — approximate)
        in_boot = np.zeros(n_samples, dtype=bool)
        in_boot[boot_idx] = True
        oob_mask = ~in_boot
        if oob_mask.sum() >= 10 and len(np.unique(y_td[oob_mask])) >= 2:
            s = clf.predict_proba(X_td[oob_mask])[:, 1]
            oob_aucs.append(float(roc_auc_score(y_td[oob_mask], s)))

        if (b + 1) % max(1, B // 10) == 0 or (b + 1) == B:
            _log("[bootstrap]  %d / %d" % (b + 1, B))

    corr = _spearman_corr_matrix(imp_mat)
    off = corr[np.triu_indices(B, k=1)]
    mean_off_spearman = float(np.mean(off)) if len(off) else float("nan")
    mean_kendall = _pairwise_kendall(imp_mat)

    k_sub = min(int(args.heatmap_subset), B)
    corr_sub = corr[:k_sub, :k_sub]
    apply_journal_style()
    wh = float(cfg.FIGURE_WIDTH_SINGLE_COL_IN)
    plt.figure(figsize=(wh, wh), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    sns.heatmap(
        corr_sub,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        square=True,
        xticklabels=["B%d" % (i + 1) for i in range(k_sub)],
        yticklabels=["B%d" % (i + 1) for i in range(k_sub)],
        cbar_kws={"label": "\u03c1"},
    )
    plt.xlabel("Replicate")
    plt.ylabel("Replicate")
    plt.tight_layout()
    p_heat = out_dir / "bootstrap_importance_rank_correlation.png"
    plt.savefig(p_heat, bbox_inches="tight")
    plt.close()

    topk = int(args.top_k_freq)
    freq = np.zeros(n_features, dtype=int)
    for b in range(B):
        order = np.argsort(-imp_mat[b])
        freq[order[:topk]] += 1
    freq_order = np.argsort(-freq)
    rows = []
    for r, j in enumerate(freq_order[: min(80, n_features)], start=1):
        j = int(j)
        rows.append(
            {
                "rank_by_frequency": r,
                "feature": names[j],
                "times_in_top%d" % topk: int(freq[j]),
                "fraction": float(freq[j] / B),
                "mean_importance_over_bootstraps": float(np.mean(imp_mat[:, j])),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap_top_feature_frequency.csv", index=False)

    summary = {
        "n_bootstrap": B,
        "n_estimators_per_fit": int(args.n_estimators),
        "n_samples_train_dev": int(n_samples),
        "n_features": int(n_features),
        "mean_offdiag_spearman_rho": mean_off_spearman,
        "mean_pairwise_kendall_tau": mean_kendall,
        "oob_auc_mean_approx": float(np.mean(oob_aucs)) if oob_aucs else None,
        "note": "Bootstrap on train_dev with stratified resampling; importances = ExtraTrees Gini. "
        "High Spearman/Kendall => ordered ranking stable under row resampling (not biological proof).",
    }
    with open(out_dir / "bootstrap_rank_stability.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _log("[bootstrap] Done. %s" % p_heat.name)
    _log("  mean off-diag Spearman (ranks) = %.4f" % mean_off_spearman)
    if kendalltau is not None:
        _log("  mean pairwise Kendall tau = %.4f" % mean_kendall)
    else:
        _log("  (install scipy for Kendall tau)")


if __name__ == "__main__":
    main()
