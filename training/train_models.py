"""
Train motion grade models (non-cascade, general-purpose).

Supports single model for all data, or separate models per bone type.
For the cascade pipeline use train_cascade.py instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in [_ROOT, _THIS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trainer import ensemble_trainer, kfold_ensemble_trainer

DEFAULT_GROUP_MAP = [[1, 2], [3, 4, 5]]


def _parse_group_map(value: str):
    if value.lower() == "none":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid --group_map: {value!r}")


def _train(dataset_dir, bone_type, num_models, num_epochs, batch_size, group_map, arch, kfold, n_folds, edge_filter, grade_filter):
    tag = f"Bone: {bone_type or 'all'} | K-fold: {kfold} | Arch: {arch} | Groups: {group_map}"
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    if kfold:
        kfold_ensemble_trainer(
            num_models=num_models, dataset_dir=str(dataset_dir), n_splits=n_folds,
            num_epochs=num_epochs, batch_size=batch_size, group_map=group_map,
            max_per_class=2.0, bone_type=bone_type, arch=arch,
            edge_filter=edge_filter, grade_filter=grade_filter,
        )
    else:
        from trainer import ensemble_trainer as _et
        _et(
            num_models=num_models, dataset_dir=str(dataset_dir),
            num_epochs=num_epochs, batch_size=batch_size, group_map=group_map,
            max_per_class=2.0, bone_type=bone_type, arch=arch,
            edge_filter=edge_filter, grade_filter=grade_filter,
        )


def main():
    parser = argparse.ArgumentParser(description="Train motion grade models.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--strategy", choices=["single", "per_bone"], default="per_bone")
    parser.add_argument("--bone_type", choices=["Tibia", "Radius"], default=None)
    parser.add_argument("--arch", choices=["custom", "efficientnet", "convnext"], default="custom")
    parser.add_argument("--kfold", action="store_true")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--group_map", type=_parse_group_map, default=DEFAULT_GROUP_MAP, metavar="JSON_OR_NONE")
    parser.add_argument("--edge_filter", action="store_true")
    parser.add_argument("--grade_filter", type=lambda s: json.loads(s), default=None, metavar="JSON_LIST")

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA — training on CPU")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    bones = [args.bone_type] if args.bone_type else (["Tibia", "Radius"] if args.strategy == "per_bone" else [None])
    for bone in bones:
        _train(dataset_dir, bone, args.num_models, args.num_epochs, args.batch_size,
               args.group_map, args.arch, args.kfold, args.n_folds,
               args.edge_filter, args.grade_filter)


if __name__ == "__main__":
    main()
