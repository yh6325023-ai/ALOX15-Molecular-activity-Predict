# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, KFold, train_test_split


def murcko_scaffold_smiles(smiles: str) -> str:
    """
    Return Bemis-Murcko scaffold SMILES for a given input SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Use the raw string to keep it unique-ish per invalid molecule.
        return f"INVALID_{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        # Fallback: for unusual molecules where Murcko returns empty.
        return f"EMPTY_{smiles}"
    return scaffold


def group_indices_by_scaffold(
    scaffolds: Sequence[str],
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for pos, sc in enumerate(scaffolds):
        groups.setdefault(sc, []).append(pos)
    return groups


def scaffold_holdout_split(
    scaffolds: Sequence[str],
    y: np.ndarray | None,
    test_frac: float,
    classification: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group-wise holdout split using scaffolds.
    Returns: (train_dev_idx, test_idx) as positions in [0..n-1]

    Any scaffold assigned to test never appears in train_dev (no scaffold leakage).
    """
    n = len(scaffolds)
    if n == 0:
        raise ValueError("Empty scaffolds list.")

    groups = group_indices_by_scaffold(scaffolds)
    group_items = [(sc, idxs) for sc, idxs in groups.items()]
    # Sort by group size desc (deterministic tie-break by scaffold string)
    group_items.sort(key=lambda t: (-len(t[1]), str(t[0])))

    test_target = int(round(test_frac * n))
    test_target = max(1, min(n - 1, test_target))

    test_idx: List[int] = []
    chosen = set()
    cur = 0
    for sc, idxs in group_items:
        if cur >= test_target:
            break
        test_idx.extend(idxs)
        chosen.add(sc)
        cur = len(test_idx)

    test_idx_arr = np.array(sorted(test_idx), dtype=int)
    all_idx = np.arange(n, dtype=int)
    train_dev_idx = np.setdiff1d(all_idx, test_idx_arr, assume_unique=False)
    return train_dev_idx, test_idx_arr


def scaffold_kfold_split(
    scaffolds: Sequence[str],
    y: np.ndarray,
    train_dev_idx: np.ndarray,
    n_splits: int,
    classification: bool,
) -> np.ndarray:
    """
    Assign each sample position in the full array to a fold id [0..n_splits-1],
    using scaffold grouping, only for indices in train_dev_idx.

    Uses sklearn GroupKFold: each Bemis–Murcko scaffold stays in one fold; folds are
    as balanced as possible in **number of groups** (not the old greedy classifier
    heuristic, which often left most folds empty when n_splits was large).

    Returns fold_id array of length n with -1 for test/unused samples.

    Requires: at least ``n_splits`` distinct scaffolds among train_dev samples.
    """
    del classification  # same split for regression / classification
    n = len(scaffolds)
    fold_id = np.full(shape=(n,), fill_value=-1, dtype=int)
    if n_splits < 2:
        raise ValueError("n_splits must be >=2")

    train_dev_idx = np.asarray(train_dev_idx, dtype=int)
    groups_train = np.array([scaffolds[i] for i in train_dev_idx], dtype=object)
    group_ids, _ = pd.factorize(groups_train, sort=True)
    n_groups = int(np.unique(group_ids).size)
    if n_groups < n_splits:
        raise ValueError(
            f"GroupKFold needs at least n_splits distinct scaffolds in train_dev; "
            f"got {n_groups} groups and n_splits={n_splits}. "
            f"Lower FINAL_CV_FOLDS or use more diverse scaffolds."
        )

    X_dummy = np.zeros((len(train_dev_idx), 1))
    y_sub = y[train_dev_idx]
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, test_rel) in enumerate(gkf.split(X_dummy, y_sub, groups=group_ids)):
        fold_id[train_dev_idx[test_rel]] = int(fold)

    return fold_id


def random_molecule_holdout_split(
    n_samples: int,
    test_frac: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Molecule-level random holdout (not scaffold-grouped). Train/test may share scaffolds.
    """
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for train/test split.")
    all_idx = np.arange(n_samples, dtype=int)
    tr, te = train_test_split(
        all_idx,
        test_size=float(test_frac),
        random_state=int(random_state),
        shuffle=True,
    )
    return np.sort(np.asarray(tr, dtype=int)), np.sort(np.asarray(te, dtype=int))


def random_molecule_kfold_split(
    n_samples: int,
    train_dev_idx: np.ndarray,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """
    Same shape contract as ``scaffold_kfold_split``: ``fold_id`` length ``n_samples``,
    ``-1`` on test indices, ``0..n_splits-1`` on train_dev rows.
    """
    fold_id = np.full(n_samples, -1, dtype=int)
    train_dev_idx = np.asarray(train_dev_idx, dtype=int)
    if len(train_dev_idx) < n_splits:
        raise ValueError(
            f"KFold needs at least n_splits={n_splits} train_dev samples; got {len(train_dev_idx)}."
        )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    X_dummy = np.zeros((len(train_dev_idx), 1))
    for fold, (_, test_rel) in enumerate(kf.split(X_dummy)):
        fold_id[train_dev_idx[test_rel]] = int(fold)
    return fold_id


def nested_inner_fold_ids(
    *,
    protocol: str,
    scaffolds_subset: Sequence[str],
    y_subset: np.ndarray,
    n_splits: int,
    random_state: int,
    classification: bool,
) -> np.ndarray:
    """
    Per-outer-train nested CV fold assignment, length ``len(scaffolds_subset)``.

    ``protocol`` matches ``cfg.FINAL_CV_PROTOCOL``: ``scaffold`` / ``cluster`` use
    GroupKFold on group ids; ``random`` uses shuffled KFold on row order.
    """
    del classification  # scaffold path shares the same split for cls/reg
    n_subset = len(scaffolds_subset)
    if n_subset < int(n_splits):
        raise ValueError(
            f"Nested inner CV needs at least n_splits={n_splits} samples in outer train; got {n_subset}."
        )
    p = str(protocol).lower().strip()
    if p in ("scaffold", "cluster"):
        return scaffold_kfold_split(
            scaffolds=list(scaffolds_subset),
            y=np.asarray(y_subset),
            train_dev_idx=np.arange(n_subset, dtype=int),
            n_splits=int(n_splits),
            classification=False,
        )
    if p == "random":
        fold_id = np.full(n_subset, -1, dtype=int)
        kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
        X_dummy = np.zeros((n_subset, 1))
        for fold, (_, test_rel) in enumerate(kf.split(X_dummy)):
            fold_id[test_rel] = int(fold)
        return fold_id
    raise ValueError(f"Unknown FINAL_CV_PROTOCOL {protocol!r} (expected 'scaffold', 'cluster', or 'random').")


def build_butina_cluster_group_ids(
    smiles_list: Sequence[str],
    *,
    radius: int,
    n_bits: int,
    dist_thresh: float,
) -> list[str]:
    """
    Morgan FP + Butina clustering: one group id per molecule for group holdout / GroupKFold.
    Not i.i.d. random splitting: whole clusters are held out together (similar chemistry stays together).
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )

    n = len(smiles_list)
    fps: list = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s).strip())
        if m is None:
            fps.append(None)
        else:
            fps.append(morgan_gen.GetFingerprint(m))

    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    fp_list = [fps[i] for i in valid_idx]
    if len(fp_list) < 3:
        raise ValueError("Too few valid fingerprints for Butina clustering.")

    nfps = len(fp_list)
    dists: list[float] = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1.0 - float(x) for x in sims])

    clusters = Butina.ClusterData(dists, nfps, float(dist_thresh), isDistData=True)

    out = [f"singleton_invalid_{i}" for i in range(n)]
    for ci, cluster in enumerate(clusters):
        for local_j in cluster:
            gi = valid_idx[int(local_j)]
            out[gi] = f"butina_{ci}"
    return out

