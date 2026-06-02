"""
Train (or fine-tune) a two-stage cascade classifier for motion grading.

Stage 1: Grade 1 vs Rest       (all scans)
Stage 2: Grade 2 vs Grade 3-5  (grade 2-5 scans only)

After training, runs cascade_kfold_eval for a full no-leakage 3-class report.

Usage — full training from preprocessed dataset:
  python training/train_cascade.py --dataset_dir F:/data/training_dataset --bone_type Tibia

Usage — build dataset then train:
  python training/train_cascade.py --dataset_dir F:/data/training_dataset
    --build_dataset --input_csv path/to/labeled.csv

Usage — fine-tune existing models with additional data:
  python training/train_cascade.py --dataset_dir F:/data/training_dataset
    --finetune --stage1_pretrained models/tibia/stage1 --stage2_pretrained models/tibia/stage2
"""

from __future__ import annotations

import argparse
import json as _json
import os
import sys
from pathlib import Path

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_DATA = os.path.join(_ROOT, 'data')
for _p in [_ROOT, _THIS, _DATA]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trainer import kfold_ensemble_trainer, kfold_finetune_trainer, cascade_kfold_eval


def _latest_kfold_dir(models_root: Path, bone_tag: str) -> Path:
    """Return most-recently modified kfold dir matching bone_tag."""
    candidates = [d for d in models_root.iterdir()
                  if d.is_dir() and bone_tag in d.name.lower() and 'kfold' in d.name]
    if not candidates:
        raise FileNotFoundError(f"No kfold dirs found in {models_root} matching '{bone_tag}'")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune cascade motion-grade classifier."
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--bone_type", choices=["Tibia", "Radius"], default="Tibia")
    parser.add_argument("--arch", choices=["custom", "efficientnet", "convnext"], default="efficientnet")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--edge_filter", action="store_true")

    # Epoch counts
    parser.add_argument("--num_epochs_s1", type=int, default=15, help="Stage 1 training epochs")
    parser.add_argument("--num_epochs_s2", type=int, default=20, help="Stage 2 training epochs")

    # Dataset build
    parser.add_argument("--build_dataset", action="store_true",
                        help="Build .npy dataset from ISQs before training")
    parser.add_argument("--input_csv", help="Labeled CSV for --build_dataset")
    parser.add_argument("--isq_root", default=r"Y:\data")
    parser.add_argument("--num_workers", type=int, default=4)

    # Fine-tuning
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune existing models instead of training from scratch")
    parser.add_argument("--stage1_pretrained", default=None,
                        help="Path to existing stage1 kfold dir (required for --finetune)")
    parser.add_argument("--stage2_pretrained", default=None,
                        help="Path to existing stage2 kfold dir (required for --finetune)")
    parser.add_argument("--num_head_epochs_s1", type=int, default=5,
                        help="Stage 1 head-only epochs during fine-tuning")
    parser.add_argument("--num_head_epochs_s2", type=int, default=5,
                        help="Stage 2 head-only epochs during fine-tuning")

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA — training on CPU")

    # ── Optionally build dataset ──────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    scores_path = dataset_dir / 'scores.xlsx'

    if args.build_dataset:
        if not args.input_csv:
            parser.error("--build_dataset requires --input_csv")
        from prepare_dataset import main as _prep_main
        import sys as _sys
        _sys.argv = [
            'prepare_dataset.py',
            '--dataset_dir', str(dataset_dir),
            '--input_csv', args.input_csv,
            '--isq_root', args.isq_root,
            '--num_workers', str(args.num_workers),
        ]
        _prep_main()
    elif not scores_path.exists():
        print(f"Warning: {scores_path} not found. Use --build_dataset to create it first.")

    models_root = Path(_ROOT) / 'models' / 'ensemble'
    bone_tag = args.bone_type.lower()

    if args.finetune:
        if not args.stage1_pretrained or not args.stage2_pretrained:
            parser.error("--finetune requires --stage1_pretrained and --stage2_pretrained")

        print("\n" + "=" * 70)
        print("STAGE 1 — Fine-tune: Grade 1 vs Rest")
        print("=" * 70)
        kfold_finetune_trainer(
            num_models=args.num_models,
            dataset_dir=str(dataset_dir),
            pretrained_kfold_dir=args.stage1_pretrained,
            n_splits=args.n_folds,
            num_head_epochs=args.num_head_epochs_s1,
            num_full_epochs=args.num_epochs_s1,
            batch_size=args.batch_size,
            group_map=[[1], [2, 3, 4, 5]],
            max_per_class=2.0,
            bone_type=args.bone_type,
            arch=args.arch,
            edge_filter=args.edge_filter,
            grade_filter=None,
        )

        s1_dir = _latest_kfold_dir(models_root, bone_tag)
        print(f"\nStage 1 fine-tuned models: {s1_dir}")

        print("\n" + "=" * 70)
        print("STAGE 2 — Fine-tune: Grade 2 vs Grade 3-5")
        print("=" * 70)
        kfold_finetune_trainer(
            num_models=args.num_models,
            dataset_dir=str(dataset_dir),
            pretrained_kfold_dir=args.stage2_pretrained,
            n_splits=args.n_folds,
            num_head_epochs=args.num_head_epochs_s2,
            num_full_epochs=args.num_epochs_s2,
            batch_size=args.batch_size,
            group_map=[[2], [3, 4, 5]],
            max_per_class=2.0,
            bone_type=args.bone_type,
            arch=args.arch,
            edge_filter=args.edge_filter,
            grade_filter=[2, 3, 4, 5],
        )

    else:
        print("\n" + "=" * 70)
        print("STAGE 1 — Train: Grade 1 vs Rest")
        print("=" * 70)
        kfold_ensemble_trainer(
            num_models=args.num_models,
            dataset_dir=str(dataset_dir),
            n_splits=args.n_folds,
            num_epochs=args.num_epochs_s1,
            batch_size=args.batch_size,
            group_map=[[1], [2, 3, 4, 5]],
            max_per_class=2.0,
            bone_type=args.bone_type,
            arch=args.arch,
            edge_filter=args.edge_filter,
            grade_filter=None,
        )

        s1_dir = _latest_kfold_dir(models_root, bone_tag)
        print(f"\nStage 1 saved to: {s1_dir}")

        print("\n" + "=" * 70)
        print("STAGE 2 — Train: Grade 2 vs Grade 3-5")
        print("=" * 70)
        kfold_ensemble_trainer(
            num_models=args.num_models,
            dataset_dir=str(dataset_dir),
            n_splits=args.n_folds,
            num_epochs=args.num_epochs_s2,
            batch_size=args.batch_size,
            group_map=[[2], [3, 4, 5]],
            max_per_class=2.0,
            bone_type=args.bone_type,
            arch=args.arch,
            edge_filter=args.edge_filter,
            grade_filter=[2, 3, 4, 5],
        )

    s2_dir = _latest_kfold_dir(models_root, bone_tag)
    print(f"\nStage 2 saved to: {s2_dir}")

    # Re-find stage1 as the second-most-recent (stage2 was just added)
    all_kfold = sorted(
        [d for d in models_root.iterdir() if d.is_dir() and bone_tag in d.name.lower() and 'kfold' in d.name],
        key=lambda d: d.stat().st_mtime,
    )
    s1_dir = all_kfold[-2] if len(all_kfold) >= 2 else all_kfold[-1]

    print("\n" + "=" * 70)
    print("CASCADE EVALUATION")
    print("=" * 70)
    cascade_kfold_eval(
        dataset_dir=str(dataset_dir),
        stage1_kfold_dir=str(s1_dir),
        stage2_kfold_dir=str(s2_dir),
        num_models=args.num_models,
    )


if __name__ == "__main__":
    main()
