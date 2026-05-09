#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantitative dose--response style evidence for discrete/continuous input dims:

  (1) Per outer-fold validation set: Spearman rho between feature values x_j and
      SHAP values phi_j (same samples). Aggregated mean +/- std across folds.
  (2) Direction consistency: sign of mean SHAP for feature j on each fold's
      validation set; report counts (e.g., 5/5 positive).

Outputs support reviewer-facing language such as "association between feature
magnitude and marginal contribution is stable and directionally consistent",
without claiming biological causality.
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as cfg
from analyze_feature_evidence import _try_load_fold_models
from analyze_shap_trees import _positive_class_shap
from src.final_module import _build_classifier, build_shared_supervised_data

try:
    import shap
except ImportError as e:  # pragma: no cover
    raise SystemExit('Install shap: pip install "shap>=0.43"\n%s' % e) from e


def _log(msg):
    print(msg, flush=True)


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def main():
    parser = argparse.ArgumentParser(description="SHAP vs feature value: Spearman + fold-wise direction.")
    parser.add_argument("--family", type=str, default="extratrees")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--top-k", type=int, default=20, help="Top features by mean(|SHAP|) across folds.")
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Do not load IPCA pretrain artifacts.",
    )
    args = parser.parse_args()

    family = str(args.family).strip().lower()
    top_k = int(args.top_k)
    out_dir = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.TASK_TYPE = "classification"
    if bool(getattr(args, "no_pretrain", False)):
        cfg.FINAL_USE_PRETRAIN_ARTIFACTS = False
        _log("[shap_q] --no-pretrain: FINAL_USE_PRETRAIN_ARTIFACTS=False")

    prepared = build_shared_supervised_data(log=_log)
    X_all = prepared.X_all
    feature_names = list(X_all.columns)
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    fold_id = np.asarray(prepared.fold_id, dtype=int)
    present_folds = sorted({int(f) for f in fold_id[train_dev_idx].tolist() if int(f) >= 0})

    X_td = X_all.iloc[train_dev_idx].values.astype(np.float64, copy=False)
    y_td = y[train_dev_idx]
    fold_td = fold_id[train_dev_idx]
    n_features = X_td.shape[1]

    fold_models = None if args.refit else _try_load_fold_models(present_folds, n_features)

    fold_mean_abs = []
    fold_rhos = []  # list of arrays (n_features,) per fold
    fold_signs = []  # list of arrays sign(mean shap) per feature, -1/0/1

    for fi, fold in enumerate(present_folds):
        va_mask = fold_td == int(fold)
        tr_mask = fold_td != int(fold)
        X_tr, y_tr = X_td[tr_mask], y_td[tr_mask]
        X_va, y_va = X_td[va_mask], y_td[va_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
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

        ma = np.mean(np.abs(phi), axis=0)
        fold_mean_abs.append(ma)

        rho_row = np.full(n_features, np.nan, dtype=np.float64)
        for j in range(n_features):
            rho_row[j], _ = _safe_spearman(X_va[:, j], phi[:, j])
        fold_rhos.append(rho_row)

        ms = np.mean(phi, axis=0)
        sgn = np.sign(ms)
        sgn[np.abs(ms) < 1e-10] = 0
        fold_signs.append(sgn)

    if len(fold_mean_abs) < 2:
        raise RuntimeError("Need >=2 valid folds for SHAP quantitative analysis.")

    mean_abs_profile = np.mean(np.vstack(fold_mean_abs), axis=0)
    order = np.argsort(-mean_abs_profile)
    top_idx = [int(order[i]) for i in range(min(top_k, len(order)))]

    rho_mat = np.vstack(fold_rhos)
    sign_mat = np.vstack(fold_signs)
    n_folds_eff = rho_mat.shape[0]

    rows = []
    for rank, j in enumerate(top_idx, start=1):
        rhos_j = rho_mat[:, j]
        valid = rhos_j[~np.isnan(rhos_j)]
        mean_rho = float(np.mean(valid)) if len(valid) else float("nan")
        std_rho = float(np.std(valid)) if len(valid) > 1 else 0.0

        signs_j = sign_mat[:, j]
        n_pos = int(np.sum(signs_j > 0))
        n_neg = int(np.sum(signs_j < 0))
        n_zero = int(np.sum(signs_j == 0))

        if n_pos > n_neg and n_pos >= n_folds_eff - 1:
            dir_label = "stable_positive_class_push"
        elif n_neg > n_pos and n_neg >= n_folds_eff - 1:
            dir_label = "stable_negative_class_push"
        elif n_pos > n_neg:
            dir_label = "mostly_positive"
        elif n_neg > n_pos:
            dir_label = "mostly_negative"
        else:
            dir_label = "mixed"

        if not np.isnan(mean_rho):
            assoc = "positive_assoc" if mean_rho > 0.05 else ("negative_assoc" if mean_rho < -0.05 else "weak")
            trend = "higher_x_higher_shap" if mean_rho > 0.05 else ("higher_x_lower_shap" if mean_rho < -0.05 else "flat")
        else:
            assoc = "undefined"
            trend = "undefined"

        rows.append(
            {
                "rank_by_mean_abs_shap": rank,
                "feature": feature_names[j],
                "mean_spearman_feature_vs_shap": mean_rho,
                "std_spearman_across_folds": std_rho,
                "feature_shap_trend": trend,
                "association_label": assoc,
                "folds_positive_mean_shap": n_pos,
                "folds_negative_mean_shap": n_neg,
                "folds_near_zero_mean_shap": n_zero,
                "direction_consistency_label": dir_label,
            }
        )

    df = pd.DataFrame(rows)
    p_csv = out_dir / ("shap_dose_response_spearman_top%d.csv" % top_k)
    df.to_csv(p_csv, index=False)
    p_csv_s4 = out_dir / ("TableS4_shap_dose_response_spearman_top%d.csv" % top_k)
    df.to_csv(p_csv_s4, index=False)

    summary = {
        "top_k": int(top_k),
        "n_folds": int(n_folds_eff),
        "n_features": int(n_features),
        "stable_direction_count": int(
            df["direction_consistency_label"].isin(["stable_positive_class_push", "stable_negative_class_push"]).sum()
        ),
        "note": "Spearman(x_j, phi_j) on each fold's validation molecules; "
        "direction uses sign of mean SHAP per fold. Model-dependent; not causal.",
    }
    with open(out_dir / "shap_quantitative_dose_response.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "TableS4_shap_quantitative_dose_response.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _log("[shap_q] Wrote %s and %s" % (p_csv.name, p_csv_s4.name))
    _log("[shap_q] stable-direction features among top-%d: %d / %d" % (top_k, summary["stable_direction_count"], top_k))


if __name__ == "__main__":
    main()
