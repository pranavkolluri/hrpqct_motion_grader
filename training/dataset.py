from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import torchvision.transforms.functional as TF
import random
from typing import Dict, Optional, List
from collections import defaultdict, Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def apply_sobel(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude of a (1, H, W) float tensor. Returns [0,1] normalized."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    x4d = x.unsqueeze(0)
    gx = F.conv2d(x4d, kx, padding=1)
    gy = F.conv2d(x4d, ky, padding=1)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2)).squeeze(0)
    vmin, vmax = mag.min(), mag.max()
    if vmax > vmin:
        mag = (mag - vmin) / (vmax - vmin)
    return mag


class HRPQCTDatasetV2(Dataset):
    """
    HR-pQCT slice dataset.

    Discovers .npy files under dataset_dir/**/scan_id/*.npy where scan_id
    matches a key in scores.xlsx 'Slice' column (scan folder names like
    's00000362_m00001201').

    Supports class grouping, majority-class downsampling, Sobel edge filter,
    bone-type filtering, and grade filtering for cascade training.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        excel_filename: str = 'scores.xlsx',
        input_size: Optional[int] = None,
        augment: bool = False,
        p_flip: float = 0.25,
        p_rot: float = 0.25,
        group_map=None,
        downsample_majority: bool = True,
        max_per_class: Optional[float] = 0.5,
        seed: int = 42,
        bone_type: Optional[str] = None,
        mmap_mode: Optional[str] = None if os.name == 'nt' else 'r',
        edge_filter: bool = False,
        grade_filter: Optional[List[int]] = None,
    ):
        self.root = Path(dataset_dir)
        self.mmap_mode = mmap_mode

        xls = self.root / excel_filename
        if not xls.exists():
            raise FileNotFoundError(f"{xls} not found")
        df = pd.read_excel(xls)
        df.columns = [c.strip() for c in df.columns]
        self.lookup = {str(k).strip().lower(): int(v) for k, v in zip(df['Slice'], df['Score'])}

        bone_lookup: dict = {}
        if bone_type is not None:
            if 'BoneType' not in df.columns:
                raise ValueError("scores.xlsx must have a 'BoneType' column to filter by bone type")
            bone_lookup = {str(k).strip().lower(): str(v).strip().lower()
                           for k, v in zip(df['Slice'], df['BoneType'])}

        self.paths: list[Path] = []
        self.labels_raw: list[int] = []
        self.scans: list[str] = []

        for p in self.root.rglob('*.npy'):
            scan_id = p.parent.name.lower()
            if scan_id not in self.lookup:
                continue
            if bone_type is not None and bone_lookup.get(scan_id, '') != bone_type.lower():
                continue
            raw_grade = self.lookup[scan_id]
            if grade_filter is not None and raw_grade not in grade_filter:
                continue
            self.paths.append(p)
            self.labels_raw.append(raw_grade)
            self.scans.append(scan_id)

        self.class_map = groups_to_class_map(group_map)
        if self.class_map is None:
            uniq = sorted(set(self.labels_raw))
            base = min(uniq) if uniq else 0
            self.labels = [y - base for y in self.labels_raw]
        else:
            def map_y(y):
                return self.class_map.get(y, self.class_map.get(int(y), max(self.class_map.values())))
            self.labels = [int(map_y(y)) for y in self.labels_raw]

        if downsample_majority:
            rng = np.random.default_rng(seed)
            class_indices: dict = defaultdict(list)
            for idx, label in enumerate(self.labels):
                class_indices[label].append(idx)
            counts = [len(v) for v in class_indices.values()]
            if max_per_class is not None and max_per_class > 0:
                target = int(max_per_class * min(counts))
                for label, idxs in class_indices.items():
                    if len(idxs) > target:
                        class_indices[label] = list(rng.choice(idxs, target, replace=False))
            else:
                legacy_max = int(sorted(counts)[-2] * 1.5) if len(counts) > 1 else counts[0]
                for label, idxs in class_indices.items():
                    if len(idxs) > legacy_max:
                        class_indices[label] = list(rng.choice(idxs, legacy_max, replace=False))

            selected = []
            for idxs in class_indices.values():
                selected.extend(idxs)
            rng.shuffle(selected)
            self.paths = [self.paths[i] for i in selected]
            self.labels = [self.labels[i] for i in selected]
            self.labels_raw = [self.labels_raw[i] for i in selected]
            self.scans = [self.scans[i] for i in selected]
            print("Class distribution after downsampling:", {k: len(v) for k, v in class_indices.items()})

        self.num_classes = int(max(self.labels) + 1) if self.labels else 0
        self.input_size = input_size
        self.augment = augment
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.edge_filter = edge_filter

    def __len__(self):
        return len(self.paths)

    def _apply_aug(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_rot:
            k = random.randint(0, 3)
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
        if random.random() < self.p_flip:
            dim = 1 if random.random() < 0.5 else 2
            x = torch.flip(x, dims=(dim,))
        return x

    def __getitem__(self, i):
        p = self.paths[i]
        y = self.labels[i]
        arr = np.load(p, mmap_mode=self.mmap_mode)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            x = torch.from_numpy(arr.copy()).unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            x = torch.from_numpy(arr.copy())
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} for {p}")
        if self.input_size is not None:
            x = TF.resize(x, [self.input_size, self.input_size], antialias=True)
        if self.augment:
            x = self._apply_aug(x)
        if self.edge_filter:
            x = apply_sobel(x)
        return x, torch.tensor(y, dtype=torch.long)


def split_by_scan(dataset: HRPQCTDatasetV2, train=0.7, val=0.15, test=0.15, seed: int = 42):
    """Split dataset into train/val/test at the scan level to prevent leakage."""
    import math
    import random as _r
    rng = _r.Random(seed)

    per_scan: dict = defaultdict(list)
    for idx, scan_id in enumerate(dataset.scans):
        per_scan[scan_id].append(idx)

    scan_ids = list(per_scan.keys())
    if not scan_ids:
        raise ValueError("No scans found. Check dataset_dir and scores.xlsx names.")

    rng.shuffle(scan_ids)
    n = len(scan_ids)
    n_train = max(1, math.floor(train * n))
    n_val = max(0, math.floor(val * n))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - (1 if n > 1 else 0))

    train_scans = set(scan_ids[:n_train])
    val_scans = set(scan_ids[n_train:n_train + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for sid, idxs in per_scan.items():
        if sid in train_scans:
            train_idx.extend(idxs)
        elif sid in val_scans:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    if not train_idx:
        for pool in (val_idx, test_idx):
            if pool:
                train_idx.extend(pool)
                pool.clear()
                break

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def kfold_scan_splits(dataset: HRPQCTDatasetV2, n_splits: int = 5, val_frac: float = 0.15, seed: int = 42):
    """
    Stratified k-fold split at scan level.
    Yields (train_subset, val_subset, test_subset, fold_idx, test_scan_ids) for each fold.
    Every scan appears in exactly one test fold.
    """
    import random as _r
    rng = _r.Random(seed)

    per_scan: dict = defaultdict(list)
    for idx, scan_id in enumerate(dataset.scans):
        per_scan[scan_id].append(idx)

    scan_class = {
        sid: Counter(dataset.labels[i] for i in idxs).most_common(1)[0][0]
        for sid, idxs in per_scan.items()
    }

    class_scans: dict = defaultdict(list)
    for sid, cls in scan_class.items():
        class_scans[cls].append(sid)
    for scans in class_scans.values():
        rng.shuffle(scans)

    folds: list[list] = [[] for _ in range(n_splits)]
    for scans in class_scans.values():
        for i, sid in enumerate(scans):
            folds[i % n_splits].append(sid)

    for fold_idx in range(n_splits):
        test_scans = set(folds[fold_idx])
        remaining = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]
        rng2 = _r.Random(seed + fold_idx)
        rng2.shuffle(remaining)
        n_val = max(1, int(val_frac * len(remaining)))
        val_scans = set(remaining[:n_val])
        train_scans = set(remaining[n_val:])

        train_idx = [i for sid, idxs in per_scan.items() if sid in train_scans for i in idxs]
        val_idx = [i for sid, idxs in per_scan.items() if sid in val_scans for i in idxs]
        test_idx = [i for sid, idxs in per_scan.items() if sid in test_scans for i in idxs]

        yield Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx), fold_idx, sorted(test_scans)


def groups_to_class_map(groups) -> Optional[Dict[int, int]]:
    if groups is None:
        return None
    cmap: Dict[int, int] = {}
    for new_id, group in enumerate(groups):
        for old in group:
            try:
                cmap[int(old)] = int(new_id)
            except Exception:
                pass
    return cmap
