# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Iterable, List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator


def _morgan_row(mol: Chem.Mol, morgan_gen) -> np.ndarray:
    """Compute one Morgan bit-vector using a pre-built generator."""
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _maccs_row(mol: Chem.Mol) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(
    smiles: pd.Series,
    *,
    morgan_radius: int,
    morgan_n_bits: int,
    master_columns: Iterable[str] | None = None,
    log: Callable[..., None] | None = None,
) -> pd.DataFrame:
    """Morgan + MACCS features (PaDEL disabled) → numeric ``DataFrame``."""
    log = log or (lambda *a, **k: None)

    indices: List = []
    smiles_ok: List[str] = []
    morgan_rows: List[np.ndarray] = []
    maccs_rows: List[np.ndarray] = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=morgan_radius,
        fpSize=morgan_n_bits,
    )

    for idx, raw in smiles.items():
        sm = str(raw).strip() if raw is not None else ""
        if not sm:
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        indices.append(idx)
        smiles_ok.append(sm)
        morgan_rows.append(_morgan_row(mol, morgan_gen))
        maccs_rows.append(_maccs_row(mol))

    if not indices:
        return pd.DataFrame()

    morgan_mat = np.vstack(morgan_rows)
    maccs_mat = np.vstack(maccs_rows)

    morgan_df = pd.DataFrame(
        morgan_mat, index=indices, columns=[f"morgan_{j}" for j in range(morgan_n_bits)]
    )
    maccs_df = pd.DataFrame(
        maccs_mat,
        index=indices,
        columns=[f"maccs_{j}" for j in range(maccs_mat.shape[1])],
    )

    # PaDEL support has been removed for this project cleanup.
    feats = pd.concat([morgan_df, maccs_df], axis=1)
    if master_columns is not None:
        feats = feats.reindex(columns=list(master_columns), fill_value=np.nan)
    return feats


def union_columns(a: pd.DataFrame, b: pd.DataFrame | None) -> list[str]:
    if b is None or b.empty:
        return sorted(a.columns)
    return sorted(set(a.columns) | set(b.columns))
