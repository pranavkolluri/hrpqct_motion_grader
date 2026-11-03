"""
Training, evaluation, and fine-tuning for motion grade cascade classifier.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in [_ROOT, _THIS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import build_model, set_finetune_mode
from dataset import HRPQCTDatasetV2, split_by_scan, kfold_scan_splits, HRPQCTBagDataset, kfold_bag_splits


def compute_class_weights(dataset):
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        labels = dataset.labels
    labels = np.array(labels)
    num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    return torch.FloatTensor(weights / weights.sum())


def train_model(model, train_loader, criterion, optimizer, val_loader=None,
                num_epochs=5, model_idx=None, ensemble_dir=None, wandb_group=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    metrics_path = None
    if ensemble_dir is not None:
        fname = f"model_{model_idx + 1}_metrics.csv" if model_idx is not None else "metrics.csv"
        metrics_path = os.path.join(ensemble_dir, fname)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_accs, train_losses, val_accs = [], [], []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch + 1} — stopping early")
                if best_state is not None:
                    model.load_state_dict(best_state)
                return

            running_loss += loss.item()
            probs = F.softmax(outputs.float(), dim=1)
            _, predicted = torch.max(probs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': running_loss / (i + 1), 'acc': 100 * correct_train / total_train})

        train_acc = 100 * correct_train / total_train
        train_accs.append(train_acc)
        train_losses.append(running_loss / len(train_loader))
        row = {"phase": wandb_group or "train", "epoch": epoch,
               "train_loss": running_loss / len(train_loader), "train_accuracy": train_acc}

        if val_loader:
            val_acc = evaluate_model(model, val_loader, device)
            model.train()
            val_accs.append(val_acc)
            row["val_accuracy"] = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if metrics_path is not None:
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["phase", "epoch", "train_loss", "train_accuracy", "val_accuracy"])
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Finished Training — restored epoch {best_epoch + 1} (best val_acc={best_val_acc:.2f}%)')
    else:
        print('Finished Training')

    if model_idx is not None and ensemble_dir is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(train_accs, label='Train')
        if val_accs:
            ax1.plot(val_accs, label='Val')
        ax1.set(xlabel='Epoch', ylabel='Accuracy (%)', title=f'Model {model_idx + 1} Accuracy')
        ax1.legend()
        ax2.plot(train_losses, label='Train Loss')
        ax2.set(xlabel='Epoch', ylabel='Loss', title=f'Model {model_idx + 1} Loss')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ensemble_dir, f'model_{model_idx + 1}_curves.png'))
        plt.close()


def evaluate_model(model, data_loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
    return 100 * correct / total


def _make_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kw),
        torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kw),
        torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kw),
    )


def simple_trainer(train_dataset, val_dataset, test_dataset,
                   num_epochs=5, batch_size=16,
                   model_idx=None, ensemble_dir=None, wandb_group=None, arch='custom'):
    """Train one model from scratch and return (model, test_acc, val_acc)."""
    train_loader, val_loader, test_loader = _make_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    class_weights = compute_class_weights(train_dataset)
    num_classes = len(class_weights)
    model = build_model(num_classes, arch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    lr = 1e-4 if arch in ('efficientnet', 'convnext') else 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"  Class weights: {class_weights.numpy()}")
    train_model(model, train_loader, criterion, optimizer, val_loader,
                num_epochs, model_idx, ensemble_dir, wandb_group)

    test_acc = evaluate_model(model, test_loader, device)
    val_acc = evaluate_model(model, val_loader, device)
    print(f'  Test: {test_acc:.2f}%  Val: {val_acc:.2f}%')
    return model, test_acc, val_acc


def finetune_simple(train_dataset, val_dataset, test_dataset,
                    pretrained_path: str,
                    arch: str = 'efficientnet',
                    num_head_epochs: int = 5,
                    num_full_epochs: int = 10,
                    batch_size: int = 16,
                    model_idx=None, ensemble_dir=None, wandb_group=None):
    """
    Two-phase fine-tuning starting from pretrained_path weights.

    Phase 1: freeze backbone, train classifier head at 1e-3 for num_head_epochs.
    Phase 2: unfreeze all, train end-to-end at 1e-5 for num_full_epochs.
    """
    train_loader, val_loader, test_loader = _make_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    class_weights = compute_class_weights(train_dataset)
    num_classes = len(class_weights)
    model = build_model(num_classes, arch)
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if num_head_epochs > 0:
        set_finetune_mode(model, arch, 'head')
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(head_params, lr=1e-3, weight_decay=1e-4)
        print(f"  Phase 1 (head only): {num_head_epochs} epochs @ lr=1e-3")
        train_model(model, train_loader, criterion, optimizer, val_loader,
                    num_head_epochs, model_idx, ensemble_dir, wandb_group and f"{wandb_group}_head")

    if num_full_epochs > 0:
        set_finetune_mode(model, arch, 'full')
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        print(f"  Phase 2 (full model): {num_full_epochs} epochs @ lr=1e-5")
        train_model(model, train_loader, criterion, optimizer, val_loader,
                    num_full_epochs, model_idx, ensemble_dir, wandb_group and f"{wandb_group}_full")

    test_acc = evaluate_model(model, test_loader, device)
    val_acc = evaluate_model(model, val_loader, device)
    print(f'  Test: {test_acc:.2f}%  Val: {val_acc:.2f}%')
    return model, test_acc, val_acc


def _scan_level_report(models, test_dataset, full_dataset, fold_dir, fold_idx, n_splits, test_scan_ids, device):
    """Compute scan-level predictions by averaging slice prob vectors per scan."""
    scan_slice_probs: dict = {}
    scan_true_label: dict = {}

    for m in models:
        m.eval()
        m.to(device)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            model_probs = [torch.softmax(m(inputs), dim=1).cpu().numpy()[0] for m in models]
            sid = full_dataset.scans[test_dataset.indices[i]]
            scan_slice_probs.setdefault(sid, []).append(np.mean(model_probs, axis=0))
            scan_true_label[sid] = labels.item()

    y_true, y_pred = [], []
    for sid, probs in scan_slice_probs.items():
        y_true.append(scan_true_label[sid])
        y_pred.append(int(np.argmax(np.mean(probs, axis=0))))

    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nFold {fold_idx + 1} scan-level report:\n{report}\n{cm}")

    with open(os.path.join(fold_dir, 'scan_level_report.txt'), 'w') as f:
        f.write(f"Fold {fold_idx + 1}/{n_splits}\n")
        f.write(f"Test scans ({len(test_scan_ids)}): {test_scan_ids}\n\n")
        f.write(report)
        f.write('\nConfusion matrix:\n')
        f.write(str(cm) + '\n')

    return y_true, y_pred


def kfold_ensemble_trainer(
    num_models, dataset_dir, n_splits=5, num_epochs=5, batch_size=16,
    group_map=None, max_per_class=2, bone_type=None, arch='custom',
    edge_filter=False, edge_blend=0.5, edge_legacy=False, grade_filter=None,
):
    """K-fold cross-validation ensemble training. Saves fold models + aggregate report."""
    dataset = HRPQCTDatasetV2(
        dataset_dir, group_map=group_map, downsample_majority=True,
        max_per_class=max_per_class, seed=42, bone_type=bone_type,
        edge_filter=edge_filter, edge_blend=edge_blend, edge_legacy=edge_legacy,
        grade_filter=grade_filter,
    )

    run_label = os.path.basename(dataset_dir) + (f"_{bone_type.lower()}" if bone_type else "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kfold_dir = os.path.join(_ROOT, 'models', 'ensemble', f"{run_label}_{timestamp}_kfold{n_splits}")
    os.makedirs(kfold_dir, exist_ok=True)

    with open(os.path.join(kfold_dir, 'config.json'), 'w') as f:
        json.dump({'arch': arch, 'bone_type': bone_type, 'group_map': group_map,
                   'n_splits': n_splits, 'edge_filter': edge_filter,
                   'edge_blend': edge_blend, 'edge_legacy': edge_legacy,
                   'grade_filter': grade_filter}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_y_true, all_y_pred = [], []

    for train_ds, val_ds, test_ds, fold_idx, test_scan_ids in kfold_scan_splits(dataset, n_splits, seed=42):
        print(f"\n{'=' * 70}\nFold {fold_idx + 1}/{n_splits}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n{'=' * 70}")

        fold_dir = os.path.join(kfold_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, 'config.json'), 'w') as f:
            json.dump({'arch': arch, 'bone_type': bone_type, 'group_map': group_map}, f, indent=2)
        with open(os.path.join(fold_dir, 'test_scans.txt'), 'w') as f:
            f.writelines(f"{sid}\n" for sid in test_scan_ids)

        fold_models = []
        for i in range(num_models):
            print(f'  Training model {i + 1}/{num_models}')
            model, test_acc, val_acc = simple_trainer(
                train_ds, val_ds, test_ds, num_epochs, batch_size,
                model_idx=i, ensemble_dir=fold_dir,
                wandb_group=f"{run_label}_fold{fold_idx}", arch=arch,
            )
            model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            fold_models.append(model)

        for i, m in enumerate(fold_models):
            torch.save(m.state_dict(), os.path.join(fold_dir, f'model_{i}.pt'))

        yt, yp = _scan_level_report(fold_models, test_ds, dataset, fold_dir, fold_idx, n_splits, test_scan_ids, device)
        all_y_true.extend(yt)
        all_y_pred.extend(yp)

        for m in fold_models:
            m.cpu()
        torch.cuda.empty_cache()
        gc.collect()

    agg_report = classification_report(all_y_true, all_y_pred)
    agg_cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n{'=' * 70}\nK-fold aggregate ({n_splits} folds):\n{agg_report}\n{agg_cm}")

    with open(os.path.join(kfold_dir, 'aggregate_report.txt'), 'w') as f:
        f.write(f"{n_splits}-fold cross-validation aggregate scan-level report\n\n")
        f.write(agg_report)
        f.write('\nConfusion matrix:\n')
        f.write(str(agg_cm) + '\n')
    np.save(os.path.join(kfold_dir, 'confusion_matrix_aggregate.npy'), agg_cm)

    return kfold_dir


def kfold_finetune_trainer(
    num_models, dataset_dir, pretrained_kfold_dir: str,
    n_splits=5, num_head_epochs=5, num_full_epochs=10, batch_size=16,
    group_map=None, max_per_class=2, bone_type=None, arch='efficientnet',
    edge_filter=False, edge_blend=0.5, edge_legacy=False, grade_filter=None,
):
    """
    Fine-tune an existing k-fold ensemble on new/expanded data.

    Each fold's model(s) are loaded from pretrained_kfold_dir/fold_<n>/model_<i>.pt
    and fine-tuned using two-phase training (head → full).
    """
    dataset = HRPQCTDatasetV2(
        dataset_dir, group_map=group_map, downsample_majority=True,
        max_per_class=max_per_class, seed=42, bone_type=bone_type,
        edge_filter=edge_filter, edge_blend=edge_blend, edge_legacy=edge_legacy,
        grade_filter=grade_filter,
    )

    run_label = os.path.basename(dataset_dir) + (f"_{bone_type.lower()}" if bone_type else "") + "_ft"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kfold_dir = os.path.join(_ROOT, 'models', 'ensemble', f"{run_label}_{timestamp}_kfold{n_splits}")
    os.makedirs(kfold_dir, exist_ok=True)

    with open(os.path.join(kfold_dir, 'config.json'), 'w') as f:
        json.dump({
            'arch': arch, 'bone_type': bone_type, 'group_map': group_map,
            'n_splits': n_splits, 'edge_filter': edge_filter,
            'edge_blend': edge_blend, 'edge_legacy': edge_legacy,
            'grade_filter': grade_filter,
            'finetuned_from': str(pretrained_kfold_dir),
        }, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_y_true, all_y_pred = [], []
    pretrained_dir = Path(pretrained_kfold_dir)

    for train_ds, val_ds, test_ds, fold_idx, test_scan_ids in kfold_scan_splits(dataset, n_splits, seed=42):
        print(f"\n{'=' * 70}\nFold {fold_idx + 1}/{n_splits} [FINETUNE]  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n{'=' * 70}")

        fold_dir = os.path.join(kfold_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, 'config.json'), 'w') as f:
            json.dump({'arch': arch, 'bone_type': bone_type, 'group_map': group_map}, f, indent=2)
        with open(os.path.join(fold_dir, 'test_scans.txt'), 'w') as f:
            f.writelines(f"{sid}\n" for sid in test_scan_ids)

        fold_models = []
        for i in range(num_models):
            pretrained_path = pretrained_dir / f'fold_{fold_idx}' / f'model_{i}.pt'
            if not pretrained_path.exists():
                # Fall back to fold 0 if the specific fold model is missing
                pretrained_path = pretrained_dir / 'fold_0' / f'model_{i}.pt'
                print(f"  Warning: using fold_0/model_{i}.pt as pretrained fallback")
            print(f'  Fine-tuning model {i + 1}/{num_models} from {pretrained_path.name}')
            model, test_acc, val_acc = finetune_simple(
                train_ds, val_ds, test_ds,
                pretrained_path=str(pretrained_path),
                arch=arch,
                num_head_epochs=num_head_epochs,
                num_full_epochs=num_full_epochs,
                batch_size=batch_size,
                model_idx=i, ensemble_dir=fold_dir,
                wandb_group=f"{run_label}_fold{fold_idx}",
            )
            model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            fold_models.append(model)

        for i, m in enumerate(fold_models):
            torch.save(m.state_dict(), os.path.join(fold_dir, f'model_{i}.pt'))

        yt, yp = _scan_level_report(fold_models, test_ds, dataset, fold_dir, fold_idx, n_splits, test_scan_ids, device)
        all_y_true.extend(yt)
        all_y_pred.extend(yp)

        for m in fold_models:
            m.cpu()
        torch.cuda.empty_cache()
        gc.collect()

    agg_report = classification_report(all_y_true, all_y_pred)
    agg_cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\nK-fold finetune aggregate:\n{agg_report}\n{agg_cm}")

    with open(os.path.join(kfold_dir, 'aggregate_report.txt'), 'w') as f:
        f.write(f"Fine-tuned {n_splits}-fold aggregate\n\n{agg_report}\nConfusion matrix:\n{agg_cm}\n")
    np.save(os.path.join(kfold_dir, 'confusion_matrix_aggregate.npy'), agg_cm)

    return kfold_dir


# ── Attention-MIL training (2.5D bags, learned slice aggregation) ─────────────
#
# Each training/eval step processes one scan's whole bag of instances at once
# (batch_size=1 at the DataLoader level, since bags vary in slice count N) —
# gradients are accumulated over `accum_steps` scans before each optimizer
# step to approximate a larger effective batch size.

def evaluate_mil_model(model, bag_loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for bag, label in bag_loader:
            bag, label = bag.squeeze(0).to(device), label.to(device)
            logits, _ = model(bag)
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            total += 1
            correct += (pred == label.long()).sum().item()
    return 100 * correct / total


def train_mil_model(model, train_loader, criterion, optimizer, val_loader=None,
                    num_epochs=5, model_idx=None, ensemble_dir=None, wandb_group=None,
                    accum_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    metrics_path = None
    if ensemble_dir is not None:
        fname = f"model_{model_idx + 1}_metrics.csv" if model_idx is not None else "metrics.csv"
        metrics_path = os.path.join(ensemble_dir, fname)

    train_accs, train_losses, val_accs = [], [], []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='scan')
        for i, (bag, label) in enumerate(pbar):
            bag, label = bag.squeeze(0).to(device), label.to(device)

            logits, _ = model(bag)
            loss = criterion(logits, label.long())

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch + 1} — stopping early")
                optimizer.zero_grad()
                if best_state is not None:
                    model.load_state_dict(best_state)
                return

            (loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            total_train += 1
            correct_train += (pred == label.long()).sum().item()
            pbar.set_postfix({'loss': running_loss / (i + 1), 'acc': 100 * correct_train / total_train})

        train_acc = 100 * correct_train / total_train
        train_accs.append(train_acc)
        train_losses.append(running_loss / len(train_loader))
        row = {"phase": wandb_group or "train", "epoch": epoch,
               "train_loss": running_loss / len(train_loader), "train_accuracy": train_acc}

        if val_loader:
            val_acc = evaluate_mil_model(model, val_loader, device)
            model.train()
            val_accs.append(val_acc)
            row["val_accuracy"] = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if metrics_path is not None:
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["phase", "epoch", "train_loss", "train_accuracy", "val_accuracy"])
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Finished Training — restored epoch {best_epoch + 1} (best val_acc={best_val_acc:.2f}%)')
    else:
        print('Finished Training')

    if model_idx is not None and ensemble_dir is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(train_accs, label='Train')
        if val_accs:
            ax1.plot(val_accs, label='Val')
        ax1.set(xlabel='Epoch', ylabel='Accuracy (%)', title=f'Model {model_idx + 1} Accuracy')
        ax1.legend()
        ax2.plot(train_losses, label='Train Loss')
        ax2.set(xlabel='Epoch', ylabel='Loss', title=f'Model {model_idx + 1} Loss')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ensemble_dir, f'model_{model_idx + 1}_curves.png'))
        plt.close()


def _make_bag_loaders(train_dataset, val_dataset, test_dataset):
    kw = dict(num_workers=2, pin_memory=True)
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kw),
        torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kw),
        torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kw),
    )


def simple_mil_trainer(train_dataset, val_dataset, test_dataset,
                       num_epochs=5, model_idx=None, ensemble_dir=None, wandb_group=None,
                       arch='efficientnet', in_channels=3, accum_steps=4):
    """Train one attention-MIL model from scratch and return (model, test_acc, val_acc)."""
    train_loader, val_loader, test_loader = _make_bag_loaders(train_dataset, val_dataset, test_dataset)

    class_weights = compute_class_weights(train_dataset)
    num_classes = len(class_weights)
    model = build_model(num_classes, arch, in_channels=in_channels, mil=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"  Class weights: {class_weights.numpy()}")
    train_mil_model(model, train_loader, criterion, optimizer, val_loader,
                    num_epochs, model_idx, ensemble_dir, wandb_group, accum_steps=accum_steps)

    test_acc = evaluate_mil_model(model, test_loader, device)
    val_acc = evaluate_mil_model(model, val_loader, device)
    print(f'  Test: {test_acc:.2f}%  Val: {val_acc:.2f}%')
    return model, test_acc, val_acc


def _mil_scan_level_report(models, test_dataset, fold_dir, fold_idx, n_splits, test_scan_ids, device):
    """Each test item is already one whole scan, so no per-slice aggregation is needed."""
    for m in models:
        m.eval()
        m.to(device)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for bag, label in test_loader:
            bag = bag.squeeze(0).to(device)
            model_probs = [torch.softmax(m(bag)[0], dim=1).cpu().numpy()[0] for m in models]
            y_true.append(label.item())
            y_pred.append(int(np.argmax(np.mean(model_probs, axis=0))))

    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nFold {fold_idx + 1} scan-level report:\n{report}\n{cm}")

    with open(os.path.join(fold_dir, 'scan_level_report.txt'), 'w') as f:
        f.write(f"Fold {fold_idx + 1}/{n_splits}\n")
        f.write(f"Test scans ({len(test_scan_ids)}): {test_scan_ids}\n\n")
        f.write(report)
        f.write('\nConfusion matrix:\n')
        f.write(str(cm) + '\n')

    return y_true, y_pred


def kfold_mil_ensemble_trainer(
    num_models, dataset_dir, n_splits=5, num_epochs=5,
    group_map=None, max_per_class=2, bone_type=None, arch='efficientnet',
    edge_filter=False, edge_blend=0.5, z_context=3, mil_input_size=224,
    accum_steps=4, grade_filter=None,
):
    """K-fold cross-validation for attention-MIL models: 2.5D z-context bags
    aggregated via learned attention instead of mean-pooled per-slice predictions."""
    dataset = HRPQCTBagDataset(
        dataset_dir, group_map=group_map, downsample_majority=True,
        max_per_class=max_per_class, seed=42, bone_type=bone_type,
        edge_filter=edge_filter, edge_blend=edge_blend, edge_legacy=False,
        grade_filter=grade_filter, z_context=z_context, input_size=mil_input_size,
        augment=True,
    )

    run_label = os.path.basename(dataset_dir) + (f"_{bone_type.lower()}" if bone_type else "") + "_mil"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kfold_dir = os.path.join(_ROOT, 'models', 'ensemble', f"{run_label}_{timestamp}_kfold{n_splits}")
    os.makedirs(kfold_dir, exist_ok=True)

    with open(os.path.join(kfold_dir, 'config.json'), 'w') as f:
        json.dump({'arch': arch, 'bone_type': bone_type, 'group_map': group_map,
                   'n_splits': n_splits, 'edge_filter': edge_filter, 'edge_blend': edge_blend,
                   'aggregation': 'attention', 'z_context': z_context, 'mil_input_size': mil_input_size,
                   'grade_filter': grade_filter}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_y_true, all_y_pred = [], []

    for train_ds, val_ds, test_ds, fold_idx, test_scan_ids in kfold_bag_splits(dataset, n_splits, seed=42):
        print(f"\n{'=' * 70}\nFold {fold_idx + 1}/{n_splits}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n{'=' * 70}")

        fold_dir = os.path.join(kfold_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, 'config.json'), 'w') as f:
            json.dump({'arch': arch, 'bone_type': bone_type, 'group_map': group_map,
                       'aggregation': 'attention', 'z_context': z_context}, f, indent=2)
        with open(os.path.join(fold_dir, 'test_scans.txt'), 'w') as f:
            f.writelines(f"{sid}\n" for sid in test_scan_ids)

        fold_models = []
        for i in range(num_models):
            print(f'  Training model {i + 1}/{num_models}')
            model, test_acc, val_acc = simple_mil_trainer(
                train_ds, val_ds, test_ds, num_epochs,
                model_idx=i, ensemble_dir=fold_dir,
                wandb_group=f"{run_label}_fold{fold_idx}", arch=arch,
                in_channels=z_context, accum_steps=accum_steps,
            )
            model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            fold_models.append(model)

        for i, m in enumerate(fold_models):
            torch.save(m.state_dict(), os.path.join(fold_dir, f'model_{i}.pt'))

        yt, yp = _mil_scan_level_report(fold_models, test_ds, fold_dir, fold_idx, n_splits, test_scan_ids, device)
        all_y_true.extend(yt)
        all_y_pred.extend(yp)

        for m in fold_models:
            m.cpu()
        torch.cuda.empty_cache()
        gc.collect()

    agg_report = classification_report(all_y_true, all_y_pred)
    agg_cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n{'=' * 70}\nK-fold aggregate ({n_splits} folds):\n{agg_report}\n{agg_cm}")

    with open(os.path.join(kfold_dir, 'aggregate_report.txt'), 'w') as f:
        f.write(f"{n_splits}-fold cross-validation aggregate scan-level report\n\n")
        f.write(agg_report)
        f.write('\nConfusion matrix:\n')
        f.write(str(agg_cm) + '\n')
    np.save(os.path.join(kfold_dir, 'confusion_matrix_aggregate.npy'), agg_cm)

    return kfold_dir


def cascade_kfold_eval(
    dataset_dir: str,
    stage1_kfold_dir: str,
    stage2_kfold_dir: str,
    num_models: int = 1,
    save_report: bool = True,
):
    """
    Evaluate cascade classifier across full dataset using held-out fold models.

    For each scan:
      - Stage 1 model from its test fold predicts Grade 1 vs Rest
      - If Rest, Stage 2 model from its test fold predicts Grade 2 vs Grade 3-5
    Final 3-class label: 0=Grade1, 1=Grade2, 2=Grade3-5.
    No data leakage — each scan is evaluated by a model that never saw it.

    Dispatches to the attention-MIL bag-based eval when the stage1 run was
    trained with aggregation='attention'; otherwise uses the original
    per-slice mean-pooling eval.
    """
    s1_dir = Path(stage1_kfold_dir)
    s2_dir = Path(stage2_kfold_dir)

    with open(s1_dir / 'config.json') as f:
        s1_cfg = json.load(f)
    with open(s2_dir / 'config.json') as f:
        s2_cfg = json.load(f)

    if s1_cfg.get('aggregation', 'mean') == 'attention':
        return _cascade_kfold_eval_mil(dataset_dir, s1_dir, s2_dir, s1_cfg, s2_cfg, num_models, save_report)
    return _cascade_kfold_eval_mean(dataset_dir, s1_dir, s2_dir, s1_cfg, s2_cfg, num_models, save_report)


def _cascade_kfold_eval_mean(
    dataset_dir: str, s1_dir: Path, s2_dir: Path, s1_cfg: dict, s2_cfg: dict,
    num_models: int = 1, save_report: bool = True,
):
    """Original per-slice mean-pooling cascade eval."""
    from dataset import apply_sobel

    arch = s1_cfg.get('arch', 'custom')
    edge_filter = s1_cfg.get('edge_filter', False)
    # Old config.json files predate blend/legacy — default to legacy=True so
    # those already-trained models keep being evaluated the way they were trained.
    edge_blend = s1_cfg.get('edge_blend', 0.5)
    edge_legacy = s1_cfg.get('edge_legacy', True)
    n_splits = s1_cfg.get('n_splits', 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build scan→fold maps
    def build_scan_fold_map(kfold_dir: Path) -> dict:
        scan_fold: dict = {}
        for fold_idx in range(n_splits):
            txt = kfold_dir / f'fold_{fold_idx}' / 'test_scans.txt'
            if txt.exists():
                for line in txt.read_text().splitlines():
                    line = line.strip()
                    if line:
                        scan_fold[line] = fold_idx
        return scan_fold

    s1_scan_fold = build_scan_fold_map(s1_dir)
    s2_scan_fold = build_scan_fold_map(s2_dir)

    # Load all fold models
    def load_fold_models(kfold_dir: Path, num_classes: int) -> dict:
        fold_models_map: dict = {}
        for fold_idx in range(n_splits):
            fold_dir = kfold_dir / f'fold_{fold_idx}'
            models = []
            for i in range(num_models):
                m = build_model(num_classes, arch)
                m.load_state_dict(torch.load(fold_dir / f'model_{i}.pt', map_location=device))
                m.to(device).eval()
                models.append(m)
            fold_models_map[fold_idx] = models
        return fold_models_map

    s1_models = load_fold_models(s1_dir, num_classes=2)
    s2_models = load_fold_models(s2_dir, num_classes=2)

    # Load full stage1 dataset to enumerate all scans
    s1_full_group_map = s1_cfg.get('group_map', [[1], [2, 3, 4, 5]])
    bone_type = s1_cfg.get('bone_type')
    full_ds = HRPQCTDatasetV2(
        dataset_dir, group_map=s1_full_group_map, downsample_majority=False,
        bone_type=bone_type, edge_filter=False,  # apply manually below
    )

    # Group slices by scan
    per_scan: dict = {}
    for idx, scan_id in enumerate(full_ds.scans):
        per_scan.setdefault(scan_id, []).append(idx)

    scan_label = {}
    for scan_id, idxs in per_scan.items():
        raw = full_ds.labels_raw[idxs[0]]
        # 3-class ground truth: 0=grade1, 1=grade2, 2=grade3-5
        if raw == 1:
            scan_label[scan_id] = 0
        elif raw == 2:
            scan_label[scan_id] = 1
        else:
            scan_label[scan_id] = 2

    def predict_scan_with_models(models, slice_indices, apply_edge):
        probs_list = []
        with torch.no_grad():
            for idx in slice_indices:
                x, _ = full_ds[idx]
                if apply_edge:
                    x = apply_sobel(x, blend=edge_blend, legacy=edge_legacy)
                x = x.unsqueeze(0).to(device)
                ensemble_probs = torch.stack([torch.softmax(m(x), dim=1) for m in models]).mean(0)
                probs_list.append(ensemble_probs.cpu().numpy()[0])
        avg_probs = np.mean(probs_list, axis=0)
        return int(np.argmax(avg_probs)), float(avg_probs.max())

    y_true, y_pred_cascade = [], []

    for scan_id, slice_idxs in per_scan.items():
        true_label = scan_label[scan_id]

        s1_fold = s1_scan_fold.get(scan_id, 0)
        s1_pred, _ = predict_scan_with_models(s1_models[s1_fold], slice_idxs, edge_filter)

        if s1_pred == 0:
            final_pred = 0  # Grade 1
        else:
            s2_fold = s2_scan_fold.get(scan_id, 0)
            s2_pred, _ = predict_scan_with_models(s2_models[s2_fold], slice_idxs, edge_filter)
            final_pred = 1 if s2_pred == 0 else 2  # Grade 2 or Grade 3-5

        y_true.append(true_label)
        y_pred_cascade.append(final_pred)

    report = classification_report(y_true, y_pred_cascade,
                                   target_names=["Grade 1", "Grade 2", "Grade 3-5"])
    cm = confusion_matrix(y_true, y_pred_cascade)
    print(f"\n{'=' * 70}\nCASCADE EVALUATION\n{'=' * 70}")
    print(report)
    print("Confusion matrix:\n", cm)

    if save_report:
        out_path = s1_dir / 'cascade_eval_report.txt'
        with open(out_path, 'w') as f:
            f.write("Cascade Classifier Evaluation (k-fold held-out)\n")
            f.write(f"Stage1 dir: {s1_dir}\nStage2 dir: {s2_dir}\n\n")
            f.write(report)
            f.write(f'\nConfusion matrix:\n{cm}\n')
        print(f"Report saved to {out_path}")

    return y_true, y_pred_cascade


def _cascade_kfold_eval_mil(
    dataset_dir: str, s1_dir: Path, s2_dir: Path, s1_cfg: dict, s2_cfg: dict,
    num_models: int = 1, save_report: bool = True,
):
    """Attention-MIL cascade eval — each dataset item is already one whole
    scan's bag, so no per-slice fan-out/averaging is needed per stage."""
    arch = s1_cfg.get('arch', 'efficientnet')
    edge_filter = s1_cfg.get('edge_filter', False)
    edge_blend = s1_cfg.get('edge_blend', 0.5)
    z_context = s1_cfg.get('z_context', 3)
    mil_input_size = s1_cfg.get('mil_input_size', 224)
    n_splits = s1_cfg.get('n_splits', 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_scan_fold_map(kfold_dir: Path) -> dict:
        scan_fold: dict = {}
        for fold_idx in range(n_splits):
            txt = kfold_dir / f'fold_{fold_idx}' / 'test_scans.txt'
            if txt.exists():
                for line in txt.read_text().splitlines():
                    line = line.strip()
                    if line:
                        scan_fold[line] = fold_idx
        return scan_fold

    s1_scan_fold = build_scan_fold_map(s1_dir)
    s2_scan_fold = build_scan_fold_map(s2_dir)

    def load_fold_models(kfold_dir: Path, num_classes: int) -> dict:
        fold_models_map: dict = {}
        for fold_idx in range(n_splits):
            fold_dir = kfold_dir / f'fold_{fold_idx}'
            models = []
            for i in range(num_models):
                m = build_model(num_classes, arch, in_channels=z_context, mil=True)
                m.load_state_dict(torch.load(fold_dir / f'model_{i}.pt', map_location=device))
                m.to(device).eval()
                models.append(m)
            fold_models_map[fold_idx] = models
        return fold_models_map

    s1_models = load_fold_models(s1_dir, num_classes=2)
    s2_models = load_fold_models(s2_dir, num_classes=2)

    s1_full_group_map = s1_cfg.get('group_map', [[1], [2, 3, 4, 5]])
    bone_type = s1_cfg.get('bone_type')
    full_ds = HRPQCTBagDataset(
        dataset_dir, group_map=s1_full_group_map, downsample_majority=False,
        bone_type=bone_type, edge_filter=edge_filter, edge_blend=edge_blend,
        z_context=z_context, input_size=mil_input_size,
    )

    scan_to_idx = {sid: i for i, sid in enumerate(full_ds.scans)}
    scan_label = {}
    for sid, i in scan_to_idx.items():
        raw = full_ds.labels_raw[i]
        if raw == 1:
            scan_label[sid] = 0
        elif raw == 2:
            scan_label[sid] = 1
        else:
            scan_label[sid] = 2

    def predict_scan_with_models(models, scan_id):
        bag, _ = full_ds[scan_to_idx[scan_id]]
        bag = bag.to(device)
        with torch.no_grad():
            probs = torch.stack([torch.softmax(m(bag)[0], dim=1) for m in models]).mean(0)
        probs = probs.cpu().numpy()[0]
        return int(np.argmax(probs)), float(probs.max())

    y_true, y_pred_cascade = [], []

    for scan_id in full_ds.scans:
        true_label = scan_label[scan_id]

        s1_fold = s1_scan_fold.get(scan_id, 0)
        s1_pred, _ = predict_scan_with_models(s1_models[s1_fold], scan_id)

        if s1_pred == 0:
            final_pred = 0
        else:
            s2_fold = s2_scan_fold.get(scan_id, 0)
            s2_pred, _ = predict_scan_with_models(s2_models[s2_fold], scan_id)
            final_pred = 1 if s2_pred == 0 else 2

        y_true.append(true_label)
        y_pred_cascade.append(final_pred)

    report = classification_report(y_true, y_pred_cascade,
                                   target_names=["Grade 1", "Grade 2", "Grade 3-5"])
    cm = confusion_matrix(y_true, y_pred_cascade)
    print(f"\n{'=' * 70}\nCASCADE EVALUATION (attention-MIL)\n{'=' * 70}")
    print(report)
    print("Confusion matrix:\n", cm)

    if save_report:
        out_path = s1_dir / 'cascade_eval_report.txt'
        with open(out_path, 'w') as f:
            f.write("Cascade Classifier Evaluation (k-fold held-out, attention-MIL)\n")
            f.write(f"Stage1 dir: {s1_dir}\nStage2 dir: {s2_dir}\n\n")
            f.write(report)
            f.write(f'\nConfusion matrix:\n{cm}\n')
        print(f"Report saved to {out_path}")

    return y_true, y_pred_cascade
