"""Per-slice bone detection and cropping for HR-pQCT scans."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage


def detect_bone_centroid(
    slice_array: np.ndarray,
    threshold_method: str = "percentile",
    percentile: float = 85.0,
    min_bone_size: int = 100,
) -> Tuple[int, int]:
    """
    Detect bone centroid using high-intensity thresholding.
    When multiple bones are present, selects the largest by area.
    Returns (center_y, center_x).
    """
    img = slice_array.astype(np.float32)
    threshold = np.percentile(img, percentile)
    bone_mask = img > threshold

    labeled, num_features = ndimage.label(bone_mask)
    if num_features == 0:
        return img.shape[0] // 2, img.shape[1] // 2

    component_sizes = np.bincount(labeled.ravel())
    valid_labels = [i for i in range(1, len(component_sizes)) if component_sizes[i] >= min_bone_size]

    if not valid_labels:
        return img.shape[0] // 2, img.shape[1] // 2

    largest_label = max(valid_labels, key=lambda l: component_sizes[l])
    bone_region = labeled == largest_label
    y_coords, x_coords = np.where(bone_region)

    if len(y_coords) == 0:
        return img.shape[0] // 2, img.shape[1] // 2

    return int(np.mean(y_coords)), int(np.mean(x_coords))


def crop_slice_around_centroid(
    slice_array: np.ndarray, center_y: int, center_x: int, crop_size: int
) -> np.ndarray:
    """Crop a 2D slice to crop_size x crop_size centered on (center_y, center_x). Pads if near edge."""
    height, width = slice_array.shape
    half = crop_size // 2

    y_start, y_end = center_y - half, center_y + half
    x_start, x_end = center_x - half, center_x + half

    img_y0 = max(0, y_start)
    img_y1 = min(height, y_end)
    img_x0 = max(0, x_start)
    img_x1 = min(width, x_end)

    cropped = slice_array[img_y0:img_y1, img_x0:img_x1]

    pad_y0 = img_y0 - y_start
    pad_y1 = y_end - img_y1
    pad_x0 = img_x0 - x_start
    pad_x1 = x_end - img_x1

    if pad_y0 > 0 or pad_y1 > 0 or pad_x0 > 0 or pad_x1 > 0:
        cropped = np.pad(cropped, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant", constant_values=0)

    return cropped[:crop_size, :crop_size]


def crop_slice_per_bone(
    slice_array: np.ndarray,
    crop_size: int = 1024,
    threshold_method: str = "percentile",
    percentile: float = 85.0,
    min_bone_size: int = 100,
) -> np.ndarray:
    """Detect bone on a 2D slice and crop around its centroid. Returns (crop_size, crop_size) array."""
    cy, cx = detect_bone_centroid(slice_array, threshold_method, percentile, min_bone_size)
    return crop_slice_around_centroid(slice_array, cy, cx, crop_size)
