#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch reviewer-facing concerns about collinearity vs functional impact:

  (1) Permutation importance (mean ROC-AUC drop on validation when column is
      shuffled) for the global Top-N features by mean Gini — avoids ~hours of
      runtime on thousands of bits while focusing on plausibly relevant dims.
  (2) Feature ablation: drop globally Top-10 / Top-20 by mean Gini, retrain
      ExtraTrees with same scaffold folds — compare OOF ROC-AUC.

Interpret permutation + ablation together with Gini/SHAP as complementary;
high fingerprint collinearity means no single bit is ``unique''.
"""

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

import config as cfg
from analyze_feature_evidence import _try_load_fold_models
from src.final_module import _build_classifier, build_shared_supervised_data
from src.publication_figures import apply_journal_style, single_column_figsize


def _log(msg):
    print(msg, flush=True)


def oof_and_fold_val_aucs_extratrees(X, y, fold_td, present_folds, et_params, random_state):
    """Return pooled OOF ROC-AUC and one validation ROC-AUC per outer fold (same fits)."""
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_val_aucs = []
    for fold in present_folds:
        tr_m = fold_td != int(fold)
        va_m = fold_td == int(fold)
        if len(np.unique(y[tr_m])) < 2 or len(np.unique(y[va_m])) < 2:
            return float("nan"), None
        clf = ExtraTreesClassifier(**et_params)
        clf.set_params(random_state=int(random_state) + int(fold))
        clf.fit(X[tr_m], y[tr_m])
        p_va = clf.predict_proba(X[va_m])[:, 1]
        oof[va_m] = p_va
        fold_val_aucs.append(float(roc_auc_score(y[va_m], p_va)))
    if np.isnan(oof).any():
        return float("nan"), None
    return float(roc_auc_score(y, oof)), fold_val_aucs


def _per_fold_delta_mean_std(base_folds, ab_folds):
    if base_folds is None or ab_folds is None or len(base_folds) != len(ab_folds):
        return None, None, None
    deltas = [float(b) - float(a) for b, a in zip(base_folds, ab_folds)]
    m = float(np.mean(deltas))
    s = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    return deltas, m, s


def _mean_sd_fold_val_aucs(fold_aucs):
    if fold_aucs is None or len(fold_aucs) == 0:
        return None, None
    a = np.asarray(fold_aucs, dtype=np.float64)
    if len(a) < 2:
        return float(np.mean(a)), 0.0
    return float(np.mean(a)), float(np.std(a, ddof=1))


def _save_ablation_cv_figure(
    out_dir,
    fold_baseline,
    fold_minus10,
    fold_minus20,
    oof_base,
    oof_m10,
    oof_m20,
):
    """
    In silico feature ablation: mean +/- s.d. of validation-set AUROC per outer fold
    (scaffold-grouped CV). Pooled out-of-fold AUROC annotated for alignment with OOF text.
    """
    apply_journal_style()
    fname = "feature_ablation_scaffold_cv_val_rocauc.png"
    m0, s0 = _mean_sd_fold_val_aucs(fold_baseline)
    m10, s10 = _mean_sd_fold_val_aucs(fold_minus10)
    m20, s20 = _mean_sd_fold_val_aucs(fold_minus20)
    if m0 is None:
        return None

    means = np.array([m0, m10, m20], dtype=np.float64)
    stds = np.array([s0, s10, s20], dtype=np.float64)
    x = np.arange(3)
    nfold = int(len(fold_baseline))
    labels = ["Full", "Remove\ntop 10", "Remove\ntop 20"]

    w, base_h = single_column_figsize()
    # Extra vertical space for wrapped x-label + footnote (avoid tight bbox crop).
    fig_h = float(base_h) + 1.25
    fig, ax = plt.subplots(figsize=(w, fig_h), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    ax.plot(x, means, color="#7f8c8d", lw=1.0, alpha=0.88, zorder=1, ls="-")
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        ms=5.5,
        capsize=3,
        color="#1b4f72",
        ecolor="#2c3e50",
        elinewidth=1.0,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
        clip_on=False,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=int(cfg.FIGURE_TICK_PT) - 1, rotation=0, linespacing=0.95)
    ax.set_xlabel(
        "Descriptor ablation (ExtraTrees)\n"
        "Columns removed: global Top-10 / Top-20 by mean Gini rank",
        fontsize=int(cfg.FIGURE_TICK_PT),
        labelpad=6,
    )
    ax.set_ylabel(
        "Validation AUROC\n(mean +/- s.d.; n = %d outer folds)" % nfold,
        fontsize=int(cfg.FIGURE_FONT_PT),
        labelpad=5,
    )
    ax.tick_params(axis="y", labelsize=int(cfg.FIGURE_TICK_PT))
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.55)
    ax.set_axisbelow(True)
    ax.margins(x=0.06, y=0.1)

    fig.text(
        0.5,
        0.03,
        "Pooled OOF AUROC (concat. val. preds.): %.3f → %.3f → %.3f"
        % (float(oof_base), float(oof_m10), float(oof_m20)),
        ha="center",
        fontsize=6.5,
        color="#2c3e50",
    )
    fig.subplots_adjust(bottom=0.42, left=0.18, right=0.96, top=0.92)
    fig.savefig(
        out_dir / fname,
        bbox_inches="tight",
        pad_inches=0.22,
        dpi=int(cfg.FIGURE_DPI),
    )
    plt.close(fig)

    legacy = out_dir / "ablation_top_features_oof_bars.png"
    if legacy.is_file():
        try:
            legacy.unlink()
        except OSError:
            pass

    return fname


def _perm_drop_for_indices(model, X_va, y_va, indices, n_repeats, rng):
    """Mean ROC-AUC decrease when each column j is permuted (validation only)."""
    X_va = np.asarray(X_va, dtype=np.float64, order="C")
    y_va = np.asarray(y_va).reshape(-1)
    base = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    out = {}
    for j in indices:
        j = int(j)
        col = X_va[:, j].copy()
        drops = []
        for _ in range(int(n_repeats)):
            Xp = X_va.copy()
            Xp[:, j] = rng.permutation(col)
            drops.append(base - roc_auc_score(y_va, model.predict_proba(Xp)[:, 1]))
        out[j] = float(np.mean(drops))
    return out


def main():
    parser = argparse.ArgumentParser(description="Permutation importance + top-K ablation OOF AUC.")
    parser.add_argument("--n-repeats-perm", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument(
        "--perm-features-top",
        type=int,
        default=250,
        help="Compute permutation importance only for this many highest-Gini features (speed).",
    )
    parser.add_argument("--no-pretrain", action="store_true")
    args = parser.parse_args()

    out_dir = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.TASK_TYPE = "classification"
    if bool(getattr(args, "no_pretrain", False)):
        cfg.FINAL_USE_PRETRAIN_ARTIFACTS = False

    prepared = build_shared_supervised_data(log=_log)
    X_all = prepared.X_all
    names = list(X_all.columns)
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    fold_id = np.asarray(prepared.fold_id, dtype=int)
    X_td = X_all.iloc[train_dev_idx].values.astype(np.float64, copy=False)
    y_td = y[train_dev_idx]
    fold_td = fold_id[train_dev_idx]
    present_folds = sorted({int(f) for f in fold_td.tolist() if int(f) >= 0})
    n_features = X_td.shape[1]

    et_params = dict(cfg.EXTRATREES_PARAMS)
    et_params["n_estimators"] = int(args.n_estimators)
    rs = int(cfg.RANDOM_STATE)

    fold_models = _try_load_fold_models(present_folds, n_features)
    models_list = []
    gini_rows = []

    for fi, fold in enumerate(present_folds):
        tr_m = fold_td != int(fold)
        va_m = fold_td == int(fold)
        X_tr, y_tr = X_td[tr_m], y_td[tr_m]
        X_va, y_va = X_td[va_m], y_td[va_m]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            raise RuntimeError("Fold %s invalid for binary classification." % fold)

        if fold_models is not None and fi < len(fold_models):
            model = fold_models[fi]
        else:
            model = _build_classifier("extratrees", override_params=None)
            model.set_params(**et_params)
            model.fit(X_tr, y_tr)

        models_list.append((model, X_va, y_va))
        gini_rows.append(np.asarray(model.feature_importances_, dtype=np.float64))

    gini_mat = np.vstack(gini_rows)
    mean_gini = np.mean(gini_mat, axis=0)

    n_perm_scope = min(int(args.perm_features_top), n_features)
    candidate_idx = list(np.argsort(-mean_gini)[:n_perm_scope])

    perm_mat = np.full((len(models_list), n_features), np.nan, dtype=np.float64)
    for fi, ((model, X_va, y_va), fold) in enumerate(zip(models_list, present_folds)):
        rng = np.random.RandomState(rs + fi * 7919)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            drops = _perm_drop_for_indices(
                model, X_va, y_va, candidate_idx, int(args.n_repeats_perm), rng
            )
        for j, v in drops.items():
            perm_mat[fi, int(j)] = v
        _log("[perm_ablate] Fold %s permutation done (%d features)" % (int(fold) + 1, len(candidate_idx)))

    mean_perm = np.full(n_features, np.nan, dtype=np.float64)
    std_perm = np.full(n_features, np.nan, dtype=np.float64)
    for j in candidate_idx:
        mean_perm[int(j)] = float(np.nanmean(perm_mat[:, int(j)]))
        std_perm[int(j)] = float(np.nanstd(perm_mat[:, int(j)]))

    mask = np.isfinite(mean_perm) & np.isfinite(mean_gini)
    rho_pg, p_pg = spearmanr(mean_gini[mask], mean_perm[mask])

    order_perm = np.argsort(-np.nan_to_num(mean_perm, nan=-1.0))
    perm_csv = []
    for j in order_perm:
        j = int(j)
        if np.isnan(mean_perm[j]):
            continue
        perm_csv.append(
            {
                "rank_by_mean_perm_importance": len(perm_csv) + 1,
                "feature": names[j],
                "mean_perm_importance_delta_auc": float(mean_perm[j]),
                "std_across_folds": float(std_perm[j]),
                "mean_gini_importance": float(mean_gini[j]),
            }
        )
        if len(perm_csv) >= 150:
            break

    df_perm = pd.DataFrame(perm_csv)
    df_perm.to_csv(out_dir / "permutation_importance_top_summary.csv", index=False)
    # SI archive alias (same contents as permutation_importance_top_summary.csv)
    df_perm.to_csv(out_dir / "TableS3_permutation_importance_top_summary.csv", index=False)

    perm_vs_gini_payload = {
        "spearman_mean_gini_vs_mean_perm": float(rho_pg),
        "p_value_approx": float(p_pg) if p_pg == p_pg else None,
        "n_repeats_perm": int(args.n_repeats_perm),
        "n_folds": int(len(models_list)),
        "n_features_perm_computed": int(mask.sum()),
        "perm_scope_top_by_gini": int(n_perm_scope),
        "note": "Permutation = mean ROC-AUC drop on validation when column shuffled; "
        "computed for Top-N by mean Gini across folds for tractability.",
    }
    with open(out_dir / "permutation_vs_gini.json", "w", encoding="utf-8") as f:
        json.dump(perm_vs_gini_payload, f, ensure_ascii=False, indent=2)
    with open(out_dir / "TableS3_permutation_vs_gini.json", "w", encoding="utf-8") as f:
        json.dump(perm_vs_gini_payload, f, ensure_ascii=False, indent=2)

    _log("[perm_ablate] Spearman(Gini, permutation delta), scoped=%d feats: %.4f" % (int(mask.sum()), float(rho_pg)))

    order_gini = np.argsort(-mean_gini)
    baseline_auc, fold_auc_baseline = oof_and_fold_val_aucs_extratrees(
        X_td, y_td, fold_td, present_folds, et_params, rs
    )

    def auc_after_dropping_top_k(k_drop):
        if k_drop <= 0:
            return baseline_auc, fold_auc_baseline
        drop_idx = set(int(order_gini[i]) for i in range(min(k_drop, n_features)))
        keep = np.array([i not in drop_idx for i in range(n_features)], dtype=bool)
        if keep.sum() < 5:
            return float("nan"), None
        X_ab = X_td[:, keep]
        return oof_and_fold_val_aucs_extratrees(
            X_ab, y_td, fold_td, present_folds, et_params, rs + 9000
        )

    auc_minus10, fold_auc_minus10 = auc_after_dropping_top_k(10)
    auc_minus20, fold_auc_minus20 = auc_after_dropping_top_k(20)

    d10_list, d10_mean, d10_std = _per_fold_delta_mean_std(fold_auc_baseline, fold_auc_minus10)
    d20_list, d20_mean, d20_std = _per_fold_delta_mean_std(fold_auc_baseline, fold_auc_minus20)

    ablation = {
        "baseline_oof_roc_auc": float(baseline_auc),
        "drop_top10_features_oof_roc_auc": float(auc_minus10),
        "drop_top20_features_oof_roc_auc": float(auc_minus20),
        "delta_auc_top10": float(baseline_auc - auc_minus10) if auc_minus10 == auc_minus10 else None,
        "delta_auc_top20": float(baseline_auc - auc_minus20) if auc_minus20 == auc_minus20 else None,
        "fold_validation_auc_baseline": fold_auc_baseline,
        "fold_validation_auc_minus_top10": fold_auc_minus10,
        "fold_validation_auc_minus_top20": fold_auc_minus20,
        "delta_fold_validation_auc_top10_vs_full": d10_list,
        "delta_fold_validation_auc_top20_vs_full": d20_list,
        "mean_delta_fold_validation_auc_top10": d10_mean,
        "std_delta_fold_validation_auc_top10": d10_std,
        "mean_delta_fold_validation_auc_top20": d20_mean,
        "std_delta_fold_validation_auc_top20": d20_std,
        "delta_fold_validation_note": "Per-fold delta = fold_val_AUC(full) − fold_val_AUC(ablated); "
        "OOF AUC is the pooled score over concatenated OOF predictions (not the mean of fold val AUCs).",
        "top10_feature_names": [names[int(order_gini[i])] for i in range(min(10, n_features))],
        "top20_feature_names": [names[int(order_gini[i])] for i in range(min(20, n_features))],
        "n_estimators_ablation": int(args.n_estimators),
        "note": "Top-K by mean Gini across folds; same ExtraTrees+CV protocol.",
    }
    fig_name = _save_ablation_cv_figure(
        out_dir,
        fold_auc_baseline,
        fold_auc_minus10,
        fold_auc_minus20,
        baseline_auc,
        auc_minus10,
        auc_minus20,
    )
    if fig_name:
        ablation["figure_file_scaffold_cv_ablation"] = fig_name

    with open(out_dir / "ablation_top_features_oof.json", "w", encoding="utf-8") as f:
        json.dump(ablation, f, ensure_ascii=False, indent=2)

    _log(
        "[perm_ablate] Baseline OOF AUC=%.4f  -Top10=%.4f  -Top20=%.4f"
        % (baseline_auc, auc_minus10, auc_minus20)
    )
    if d10_mean == d10_mean and d10_std == d10_std:
        _log(
            "[perm_ablate] Per-fold val-AUC drop (full - ablated): Top10 mean+/-sd=%.4f+/-%.4f  Top20=%.4f+/-%.4f"
            % (d10_mean, d10_std, d20_mean, d20_std)
        )


if __name__ == "__main__":
    main()
