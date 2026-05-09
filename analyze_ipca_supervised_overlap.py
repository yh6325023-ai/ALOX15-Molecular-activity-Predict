#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Align unsupervised IPCA inputs with supervised forest importances.

IPCA components are defined on the *scaled* variance-selected fingerprint block
used during pretraining. Supervised X_all concatenates the *raw* same block
(with column names in ``selected_columns``) plus ``repr_*`` coordinates.

We aggregate absolute loadings across the first K IPCA components to score
each pretraining input dimension, then measure overlap (Jaccard) between
Top-N loading-driven features and Top-N supervised importances restricted to
the same column namespace (excluding repr_*).
"""

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from sklearn.ensemble import ExtraTreesClassifier

from src.final_module import build_shared_supervised_data
from src.publication_figures import apply_journal_style


def _log(msg):
    print(msg, flush=True)


def _jaccard_top(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b) or 1
    return inter / union


def main():
    parser = argparse.ArgumentParser(description="IPCA loadings vs supervised importance overlap.")
    parser.add_argument("--n-pc", type=int, default=15, help="Sum |loadings| over first n IPCA components.")
    parser.add_argument("--top-n", type=int, default=50, help="Top-N sets for Jaccard.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Trees for supervised reference fit.")
    args = parser.parse_args()

    out_dir = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.PRETRAIN_PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError("Missing %s — run pretraining first." % cfg.PRETRAIN_PREPROCESSOR_PATH)

    pre = joblib.load(cfg.PRETRAIN_PREPROCESSOR_PATH)
    selected = list(pre["selected_columns"])
    ipca = pre["projector"]
    comp = np.asarray(ipca.components_, dtype=float)
    n_pc = min(int(args.n_pc), comp.shape[0])
    loading_scores = np.sum(np.abs(comp[:n_pc, :]), axis=0)
    if loading_scores.shape[0] != len(selected):
        raise RuntimeError("IPCA components width != len(selected_columns)")

    loading_rank = np.argsort(-loading_scores)
    top_n = int(args.top_n)
    top_loading = set(selected[int(loading_rank[i])] for i in range(min(top_n, len(selected))))

    cfg.TASK_TYPE = "classification"
    _log("[ipca_overlap] Loading supervised data …")
    prepared = build_shared_supervised_data(log=_log)
    X_all = prepared.X_all
    names = list(X_all.columns)
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    X_td = X_all.iloc[train_dev_idx].values.astype(np.float64, copy=False)
    y_td = y[train_dev_idx]

    et_params = dict(cfg.EXTRATREES_PARAMS)
    et_params["n_estimators"] = int(args.n_estimators)
    clf = ExtraTreesClassifier(**et_params)
    clf.fit(X_td, y_td)
    imp = np.asarray(clf.feature_importances_, dtype=float)
    order_sup = np.argsort(-imp)
    raw_name_set = set(selected)
    sup_candidates = []
    for j in order_sup:
        j = int(j)
        fn = names[j]
        if fn.startswith("repr_"):
            continue
        if fn not in raw_name_set:
            continue
        sup_candidates.append(fn)
        if len(sup_candidates) >= top_n:
            break
    top_supervised = set(sup_candidates)

    jac = _jaccard_top(top_loading, top_supervised)

    rows = []
    for i in range(min(60, len(loading_rank))):
        idx = int(loading_rank[i])
        fn = selected[idx]
        rows.append(
            {
                "rank": i + 1,
                "feature": fn,
                "ipca_loading_score": float(loading_scores[idx]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "ipca_loading_top_features.csv", index=False)

    overlap = sorted(top_loading & top_supervised)
    pd.DataFrame({"feature": overlap}).to_csv(out_dir / "ipca_supervised_top_overlap.csv", index=False)

    apply_journal_style()
    w_bar = float(cfg.FIGURE_WIDTH_SINGLE_COL_IN)
    h_bar = max(2.9, 0.22 * min(28, max(1, len(overlap))))
    fig, ax = plt.subplots(figsize=(w_bar, h_bar), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    if overlap:
        sel_index = {n: i for i, n in enumerate(selected)}
        feat_index = {n: i for i, n in enumerate(names)}
        w_ipca = np.array([float(loading_scores[sel_index[fn]]) for fn in overlap], dtype=float)
        w_gini = np.array([float(imp[feat_index[fn]]) for fn in overlap], dtype=float)
        # Comparable lengths within the overlap set (not comparable across studies).
        w_ipca_n = w_ipca / (float(np.max(w_ipca)) or 1.0)
        w_gini_n = w_gini / (float(np.max(w_gini)) or 1.0)
        y = np.arange(len(overlap))
        h = 0.34
        ax.barh(y - 0.175, w_ipca_n, height=h, color="#6baed6", label="IPCA agg. |loading| (norm.)")
        ax.barh(y + 0.175, w_gini_n, height=h, color="#fd8d3c", label="Supervised Gini (norm.)")
        ax.set_yticks(y)
        ax.set_yticklabels(overlap, fontsize=int(cfg.FIGURE_TICK_PT))
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score within overlap set (max = 1)")
        ax.legend(loc="lower right", fontsize=max(5, int(cfg.FIGURE_TICK_PT) - 1), framealpha=0.92)
        ax.grid(axis="x", linestyle=":", linewidth=0.55, alpha=0.55)
        ax.set_axisbelow(True)
        fig.text(
            0.5,
            0.01,
            "Top-%d Jaccard = %.3f (%d overlapping names)"
            % (top_n, float(jac), len(overlap)),
            ha="center",
            fontsize=max(5, int(cfg.FIGURE_TICK_PT) - 1),
            color="#2c3e50",
        )
    else:
        ax.text(0.5, 0.5, "No name overlap in Top-N\n(check feature naming)", ha="center", va="center")
        ax.axis("off")
    plt.tight_layout(rect=[0, 0.06, 1, 0.99])
    p_fig = out_dir / "ipca_supervised_top_overlap.png"
    plt.savefig(p_fig, bbox_inches="tight")
    plt.close()

    summary = {
        "n_pc_for_loading_sum": int(n_pc),
        "top_n": int(top_n),
        "jaccard_topN": float(jac),
        "n_overlap_features": int(len(overlap)),
        "n_selected_pretrain_columns": int(len(selected)),
        "note": "Overlap is heuristic: multicollinearity splits supervised credit; "
        "IPCA loadings live in scaled pretrain space, supervised uses raw+repr concat.",
    }
    with open(out_dir / "ipca_supervised_overlap.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _log("[ipca_overlap] Jaccard(top-%d) = %.4f  | overlap count = %d" % (top_n, jac, len(overlap)))
    _log("[ipca_overlap] Saved %s" % p_fig.name)


if __name__ == "__main__":
    main()
