"""
Cascade inference engine for motion grading.

Two modes:
  Normal:  reads ISQ files from Y:\\data\\<sample>\\<measurement>\\  on-the-fly
  Debug:   reads pre-processed .npy files from a local dataset directory

Stage 1 predicts Grade 1 vs Rest.
Stage 2 (applied only when Stage 1 says Rest) predicts Grade 2 vs Grade 3-5.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAINING = os.path.join(_ROOT, 'training')
_DATA = os.path.join(_ROOT, 'data')
for _p in [_ROOT, _TRAINING, _DATA]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from model import build_model
from dataset import apply_sobel
from bone_cropper import crop_slice_per_bone
from isq_reader import read_isq
from csv_path_resolver import first_isq


def _load_models(stage_dir: str | Path, num_classes: int, arch: str, device: torch.device,
                 in_channels: int = 1, mil: bool = False) -> list:
    """Load all model_*.pt from a stage dir.

    Handles three layouts:
      stage_dir/fold_*/model_*.pt          (config points directly at kfold dir)
      stage_dir/<timestamp>/fold_*/model_*.pt  (timestamped dir dropped inside stage_dir)
      stage_dir/model_*.pt                 (flat ensemble_trainer output)
    """
    stage_dir = Path(stage_dir)
    files = sorted(stage_dir.glob("fold_*/model_*.pt"))
    if not files:
        files = sorted(stage_dir.glob("*/fold_*/model_*.pt"))
    if not files:
        files = sorted(stage_dir.glob("model_*.pt"))
    if not files:
        raise FileNotFoundError(f"No model files found in {stage_dir}")

    models = []
    for path in files:
        m = build_model(num_classes, arch, in_channels=in_channels, mil=mil)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device).eval()
        models.append(m)
    return models


def _infer_slices(models: list, slices_float: list[np.ndarray], edge_filter: bool,
                  edge_blend: float, edge_legacy: bool, device: torch.device) -> tuple[int, float]:
    """
    Run ensemble inference on a list of pre-normalized float32 (H, W) numpy arrays.
    Returns (predicted_class, confidence) where confidence is the average winning-class probability.
    """
    all_slice_probs = []

    with torch.no_grad():
        for arr in slices_float:
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
            if edge_filter:
                x = apply_sobel(x.squeeze(0), blend=edge_blend, legacy=edge_legacy).unsqueeze(0)  # keep (1,1,H,W)

            ensemble_probs = torch.stack(
                [F.softmax(m(x), dim=1) for m in models]
            ).mean(0).cpu().numpy()[0]  # (num_classes,)
            all_slice_probs.append(ensemble_probs)

    avg_probs = np.mean(all_slice_probs, axis=0)
    pred = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred])
    return pred, confidence


def _build_bag(slices_float: list[np.ndarray], z_context: int, input_size: Optional[int]) -> torch.Tensor:
    """Stack each slice with its z-neighbors into (N, z_context, H, W), edge-replicated
    at the volume ends — mirrors HRPQCTBagDataset's instance construction."""
    n = len(slices_float)
    half = z_context // 2
    instances = [
        np.stack([slices_float[min(max(pos + off, 0), n - 1)] for off in range(-half, half + 1)], axis=0)
        for pos in range(n)
    ]
    bag = torch.from_numpy(np.stack(instances, axis=0).astype(np.float32))  # (N, z_context, H, W)
    if input_size is not None:
        n_, c_, h_, w_ = bag.shape
        bag = TF.resize(bag.reshape(n_ * c_, h_, w_), [input_size, input_size],
                        antialias=True).reshape(n_, c_, input_size, input_size)
    return bag


def _infer_slices_mil(models: list, slices_float: list[np.ndarray], edge_filter: bool,
                      edge_blend: float, z_context: int, mil_input_size: Optional[int],
                      device: torch.device) -> tuple[int, float]:
    """Attention-MIL inference: one bag forward pass per model instead of
    per-slice predictions averaged after the fact."""
    bag = _build_bag(slices_float, z_context, mil_input_size).to(device)
    if edge_filter:
        n_, c_, h_, w_ = bag.shape
        bag = apply_sobel(bag.reshape(n_ * c_, h_, w_), blend=edge_blend).reshape(n_, c_, h_, w_)

    with torch.no_grad():
        ensemble_probs = torch.stack(
            [F.softmax(m(bag)[0], dim=1) for m in models]
        ).mean(0).cpu().numpy()[0]

    pred = int(np.argmax(ensemble_probs))
    confidence = float(ensemble_probs[pred])
    return pred, confidence


def _load_npy_slices(dataset_dir: Path, sample_num: int, meas_num: int) -> list[np.ndarray]:
    """Load pre-processed .npy slices from dataset_dir/s<sample>_m<meas>/."""
    scan_folder = dataset_dir / f"s{sample_num:08d}_m{meas_num:08d}"
    if not scan_folder.exists():
        raise FileNotFoundError(f"Debug dataset folder not found: {scan_folder}")
    files = sorted(scan_folder.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {scan_folder}")
    return [np.load(str(p)).astype(np.float32) for p in files]


def _load_isq_slices(isq_root: Path, sample_num: int, meas_num: int, crop_size: int) -> list[np.ndarray]:
    """Read ISQ file, crop each slice, normalize to [0,1]. Returns list of float32 arrays."""
    scan_dir = isq_root / f"{sample_num:08d}" / f"{meas_num:08d}"
    isq_path = first_isq(scan_dir)
    if isq_path is None:
        raise FileNotFoundError(f"No ISQ file found at {scan_dir}")

    volume = read_isq(isq_path, verbose=False)  # (z, y, x) uint16

    slices = []
    for slice_arr in volume:
        cropped = crop_slice_per_bone(slice_arr, crop_size=crop_size, threshold_method="percentile", percentile=85.0)
        slc = cropped.astype(np.float32)
        vmin, vmax = slc.min(), slc.max()
        if vmax > vmin:
            slc = (slc - vmin) / (vmax - vmin)
        slices.append(slc)
    return slices


class CascadePredictor:
    """
    Loads both cascade stages for one or both bone types and runs inference.
    Call predict() for on-the-fly ISQ inference or predict_debug() for pre-processed .npy.
    """

    def __init__(self, arch: Optional[str] = None, edge_filter: Optional[bool] = None,
                edge_blend: Optional[float] = None, edge_legacy: Optional[bool] = None,
                aggregation: Optional[str] = None, z_context: Optional[int] = None,
                mil_input_size: Optional[int] = None):
        self.arch = arch or config.ARCH
        self.edge_filter = edge_filter if edge_filter is not None else config.EDGE_FILTER
        self.edge_blend = edge_blend if edge_blend is not None else config.EDGE_BLEND
        self.edge_legacy = edge_legacy if edge_legacy is not None else config.EDGE_LEGACY
        self.aggregation = aggregation or config.AGGREGATION
        self.z_context = z_context if z_context is not None else config.Z_CONTEXT
        self.mil_input_size = mil_input_size if mil_input_size is not None else config.MIL_INPUT_SIZE
        self.device = torch.device(config.INFERENCE_DEVICE)
        self._models: dict = {}  # bone_type -> {'stage1': [...], 'stage2': [...]}
        self._loaded_bones: set = set()

    def load_bone(self, bone_type: str) -> None:
        """Load stage1 and stage2 models for a bone type (lazy, call before predict)."""
        if bone_type in self._loaded_bones:
            return
        bone_cfg = config.MODELS.get(bone_type)
        if bone_cfg is None:
            raise ValueError(f"No model config for bone type: {bone_type!r}")

        s1_dir = Path(bone_cfg['stage1'])
        s2_dir = Path(bone_cfg['stage2'])

        if not s1_dir.exists() or not any(s1_dir.rglob("model_*.pt")):
            raise FileNotFoundError(f"Stage 1 models not found at {s1_dir}")
        if not s2_dir.exists() or not any(s2_dir.rglob("model_*.pt")):
            raise FileNotFoundError(f"Stage 2 models not found at {s2_dir}")

        mil = self.aggregation == "attention"
        in_channels = self.z_context if mil else 1

        print(f"  Loading {bone_type} stage 1 and 2 from {s1_dir} and {s2_dir}...")
        self._models[bone_type] = {
            'stage1': _load_models(s1_dir, num_classes=2, arch=self.arch, device=self.device,
                                   in_channels=in_channels, mil=mil),
            'stage2': _load_models(s2_dir, num_classes=2, arch=self.arch, device=self.device,
                                   in_channels=in_channels, mil=mil),
        }
        self._loaded_bones.add(bone_type)

    def _predict_slices(self, bone_type: str, slices: list[np.ndarray]) -> tuple[int, float, str]:
        """
        Run cascade inference on pre-loaded slices.
        Returns (bin: 0/1/2, confidence: float, label: str).
        """
        models = self._models[bone_type]

        if self.aggregation == "attention":
            s1_pred, s1_conf = _infer_slices_mil(models['stage1'], slices, self.edge_filter,
                                                 self.edge_blend, self.z_context, self.mil_input_size, self.device)
        else:
            s1_pred, s1_conf = _infer_slices(models['stage1'], slices, self.edge_filter,
                                              self.edge_blend, self.edge_legacy, self.device)

        if s1_pred == 0:
            return 0, s1_conf, config.GRADE_BIN_LABELS[0]

        if self.aggregation == "attention":
            s2_pred, s2_conf = _infer_slices_mil(models['stage2'], slices, self.edge_filter,
                                                 self.edge_blend, self.z_context, self.mil_input_size, self.device)
        else:
            s2_pred, s2_conf = _infer_slices(models['stage2'], slices, self.edge_filter,
                                              self.edge_blend, self.edge_legacy, self.device)
        combined_conf = (s1_conf + s2_conf) / 2
        grade_bin = 1 if s2_pred == 0 else 2
        return grade_bin, combined_conf, config.GRADE_BIN_LABELS[grade_bin]

    def predict(self, sample_num: int, meas_num: int, bone_type: str,
                isq_root: Optional[str] = None, crop_size: Optional[int] = None) -> dict:
        """Live ISQ inference. Returns result dict."""
        self.load_bone(bone_type)
        isq_root_path = Path(isq_root or config.ISQ_ROOT)
        crop = crop_size or config.CROP_SIZE

        slices = _load_isq_slices(isq_root_path, sample_num, meas_num, crop)
        grade_bin, confidence, label = self._predict_slices(bone_type, slices)

        return {
            'grade_bin': grade_bin,
            'grade_label': label,
            'confidence': confidence,
            'num_slices': len(slices),
        }

    def predict_debug(self, sample_num: int, meas_num: int, bone_type: str,
                      dataset_dir: Optional[str] = None) -> dict:
        """Debug mode: read pre-processed .npy slices instead of ISQ."""
        self.load_bone(bone_type)
        ds_dir = Path(dataset_dir or config.DEBUG_DATASET_DIR)

        slices = _load_npy_slices(ds_dir, sample_num, meas_num)
        grade_bin, confidence, label = self._predict_slices(bone_type, slices)

        return {
            'grade_bin': grade_bin,
            'grade_label': label,
            'confidence': confidence,
            'num_slices': len(slices),
        }
