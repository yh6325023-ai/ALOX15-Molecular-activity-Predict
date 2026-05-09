#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Screen an external compound database and export predicted actives to one SDF.

Supports:
- CSV input (must contain a SMILES column, auto-detected)
- SDF input

Uses the current final classification ensemble:
- models/final_ensemble_cls.joblib
- models/final_training_metadata_cls.joblib (for default threshold)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

import config as cfg
from src.features import build_feature_matrix


def _log(msg: str) -> None:
    print(msg, flush=True)


def _coerce_features_for_sklearn(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="C")
    return np.where(np.isfinite(X), X, np.nan)


def _detect_smiles_column(columns: list[str]) -> str:
    aliases = ["smiles", "smile", "canonical_smiles", "structure"]
    low_map = {c.lower(): c for c in columns}
    for a in aliases:
        if a in low_map:
            return low_map[a]
    raise ValueError(
        "Could not detect SMILES column. Please include one of: "
        "Smiles/smiles/smile/canonical_smiles/structure"
    )


def _detect_name_column(columns: list[str]) -> str | None:
    aliases = [
        "name",
        "compound_name",
        "compound",
        "molecule_name",
        "mol_name",
        "id",
        "compound_id",
        "cid",
        "编号",
        "名称",
    ]
    low_map = {c.lower(): c for c in columns}
    for a in aliases:
        if a in low_map:
            return low_map[a]
    return None


def _read_csv_robust(path: Path) -> pd.DataFrame:
    # Try common encodings on Chinese Windows exports first.
    tried = []
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "ansi", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            tried.append(f"{enc}: {exc.__class__.__name__}")
    raise UnicodeError(
        "Failed to read CSV with attempted encodings: " + ", ".join(tried)
    )


def _read_input(path: Path) -> Tuple[pd.DataFrame, list[Chem.Mol], pd.Series]:
    ext = path.suffix.lower()
    if ext == ".csv":
        df = _read_csv_robust(path)
        smi_col = _detect_smiles_column(list(df.columns))
        smiles = df[smi_col].astype(str).str.strip()
        smiles = smiles.replace({"nan": "", "NaN": "", "None": ""})
        mols = []
        for sm in smiles.tolist():
            mol = Chem.MolFromSmiles(sm) if sm and sm.lower() != "nan" else None
            mols.append(mol)
        return df, mols, smiles

    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False)
        mols = [m for m in suppl]
        smiles = pd.Series(
            [Chem.MolToSmiles(m) if m is not None else "" for m in mols], dtype=str
        )
        # lightweight table for reporting
        df = pd.DataFrame({"Smiles": smiles})
        return df, mols, smiles

    raise ValueError(f"Unsupported input format: {ext}. Use .csv or .sdf")


def _build_inference_features(smiles: pd.Series) -> pd.DataFrame:
    use_pre = bool(getattr(cfg, "FINAL_USE_PRETRAIN_ARTIFACTS", True))
    if not use_pre:
        raise RuntimeError(
            "FINAL_USE_PRETRAIN_ARTIFACTS is False in current config. "
            "This screening script currently supports the pretrain-artifact "
            "pipeline used by your final classifier."
        )

    if not cfg.PRETRAIN_PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(f"Missing pretrain preprocessor: {cfg.PRETRAIN_PREPROCESSOR_PATH}")

    pre = joblib.load(cfg.PRETRAIN_PREPROCESSOR_PATH)
    master_cols = pre["master_columns"]

    X_raw = build_feature_matrix(
        smiles,
        morgan_radius=cfg.MORGAN_RADIUS,
        morgan_n_bits=cfg.MORGAN_N_BITS,
        master_columns=master_cols,
        log=_log,
    )
    if X_raw.empty:
        return X_raw

    X_imp = pre["imputer"].transform(X_raw)
    X_sel = pre["variance_selector"].transform(X_imp)
    X_scaled = pre["scaler"].transform(X_sel)
    X_repr = pre["projector"].transform(X_scaled)
    X_repr_df = pd.DataFrame(
        X_repr,
        index=X_raw.index,
        columns=[f"repr_{i}" for i in range(X_repr.shape[1])],
    )

    if bool(getattr(cfg, "FINAL_USE_CONCAT_RAW_FEATURES", True)):
        X_final = pd.concat(
            [
                pd.DataFrame(
                    X_sel,
                    index=X_raw.index,
                    columns=list(pre["selected_columns"]),
                ),
                X_repr_df,
            ],
            axis=1,
        )
    else:
        X_final = X_repr_df

    # Same sanitation style as training
    X_mat = _coerce_features_for_sklearn(X_final.values)
    if np.isnan(X_mat).any():
        col_means = np.nanmean(X_mat, axis=0)
        inds = np.where(np.isnan(X_mat))
        X_mat[inds] = np.take(col_means, inds[1])
    return pd.DataFrame(X_mat, index=X_final.index, columns=X_final.columns)


def _default_threshold() -> float:
    meta_path = cfg.FINAL_TRAINING_METADATA_PATH.with_name(
        f"{cfg.FINAL_TRAINING_METADATA_PATH.stem}_cls.joblib"
    )
    if not meta_path.is_file():
        return 0.5
    meta = joblib.load(meta_path)
    ts = meta.get("threshold_selection") or {}
    thr = ts.get("threshold_used_for_metrics")
    if thr is None:
        thr = ts.get("thr_best_f1")
    return float(thr) if thr is not None else 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen compounds and export predicted actives as one SDF.")
    parser.add_argument("--input", required=True, help="Input file path (.csv or .sdf)")
    parser.add_argument(
        "--output-sdf",
        default=str(cfg.RESULTS_DIR / "screening_active_compounds.sdf"),
        help="Output SDF path for predicted actives",
    )
    parser.add_argument(
        "--output-csv",
        default=str(cfg.RESULTS_DIR / "screening_predictions.csv"),
        help="Output CSV path with all predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Active threshold on predicted score (default: from final metadata or 0.5)",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_sdf = Path(args.output_sdf).expanduser().resolve()
    out_csv = Path(args.output_csv).expanduser().resolve()
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ensemble_path = cfg.FINAL_ENSEMBLE_PATH.with_name(f"{cfg.FINAL_ENSEMBLE_PATH.stem}_cls.joblib")
    if not ensemble_path.is_file():
        raise FileNotFoundError(f"Missing final classification ensemble: {ensemble_path}")

    threshold = float(args.threshold) if args.threshold is not None else _default_threshold()
    _log(f"Using model: {ensemble_path}")
    _log(f"Using active threshold: {threshold:.4f}")

    df_raw, mols, smiles = _read_input(in_path)
    _log(f"Loaded {len(smiles)} molecules from: {in_path}")

    X = _build_inference_features(smiles)
    if X.empty:
        raise RuntimeError("No valid molecules after feature construction.")

    model = joblib.load(ensemble_path)
    scores = np.asarray(model.predict(X.values), dtype=float).reshape(-1)
    labels = (scores >= threshold).astype(int)

    # Join predictions back to original order
    pred_df = pd.DataFrame(
        {
            "row_index": X.index.to_numpy(),
            "score_active": scores,
            "pred_active": labels,
        }
    )
    out_table = df_raw.copy()
    out_table["score_active"] = np.nan
    out_table["pred_active"] = 0
    for _, r in pred_df.iterrows():
        i = int(r["row_index"])
        out_table.at[i, "score_active"] = float(r["score_active"])
        out_table.at[i, "pred_active"] = int(r["pred_active"])

    out_table.to_csv(out_csv, index=False)
    _log(f"Saved all predictions CSV: {out_csv}")

    # Export actives to a single SDF
    try:
        writer = Chem.SDWriter(str(out_sdf))
    except OSError:
        # RDKit on some Windows setups cannot open non-ASCII output paths directly.
        tmp_sdf = out_sdf.with_name("screening_active_compounds_tmp_ascii.sdf")
        writer = Chem.SDWriter(str(tmp_sdf))
    n_active = 0
    for _, r in pred_df.iterrows():
        i = int(r["row_index"])
        if int(r["pred_active"]) != 1:
            continue
        name_col = _detect_name_column(list(df_raw.columns))
        raw_name = None
        if name_col is not None and i in df_raw.index:
            try:
                raw_name = str(df_raw.at[i, name_col]).strip()
            except Exception:
                raw_name = None
        if not raw_name or raw_name.lower() in ("nan", "none", ""):
            raw_name = f"compound_{i}"

        mol = mols[i]
        if mol is None:
            sm = str(smiles.iloc[i]).strip()
            mol = Chem.MolFromSmiles(sm) if sm else None
            if mol is None:
                continue
        mol = Chem.Mol(mol)  # copy
        # Set SDF title line to preserve compound identity in docking tools.
        mol.SetProp("_Name", raw_name)
        if name_col is not None:
            mol.SetProp("SourceNameColumn", name_col)
        mol.SetProp("CompoundName", raw_name)
        mol.SetProp("PredScoreActive", f"{float(r['score_active']):.6f}")
        mol.SetProp("PredActive", "1")
        mol.SetProp("SourceRowIndex", str(i))
        if i in out_table.index:
            # Keep all original fields as SDF properties when possible.
            for c in out_table.columns:
                if c in ("score_active", "pred_active"):
                    continue
                try:
                    v = out_table.at[i, c]
                    if pd.isna(v):
                        continue
                    s = str(v).strip()
                    if s and s.lower() not in ("nan", "none"):
                        mol.SetProp(str(c), s)
                except Exception:
                    continue
        writer.write(mol)
        n_active += 1
    writer.close()
    if "tmp_sdf" in locals():
        shutil.move(str(tmp_sdf), str(out_sdf))

    _log(f"Exported predicted actives to SDF: {out_sdf}")
    _log(f"Active molecules: {n_active}")


if __name__ == "__main__":
    main()

