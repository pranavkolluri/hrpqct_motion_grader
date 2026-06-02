"""ISQ reader for Scanco XCT2 systems using itk-ioscanco."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import itk
except ImportError:
    raise ImportError("itk-ioscanco not installed. Run: pip install itk-ioscanco")


def read_isq(isq_path: str | Path, slices: Optional[slice] = None, verbose: bool = False) -> np.ndarray:
    """
    Read an ISQ file and return volumetric data as (num_slices, height, width).

    Args:
        isq_path: Path to .ISQ or .ISQ;* file.
        slices: Optional slice object to read only a subset of z-slices.
        verbose: Print shape/dtype info.

    Returns:
        uint16 numpy array of shape (z, y, x).
    """
    isq_path = Path(isq_path)
    if not isq_path.exists():
        raise FileNotFoundError(f"ISQ file not found: {isq_path}")

    try:
        reader = itk.ImageFileReader.New(FileName=str(isq_path))
        reader.Update()
        image = reader.GetOutput()
        volume = itk.array_from_image(image)  # (z, y, x)

        if verbose:
            print(f"ISQ shape: {volume.shape}, dtype: {volume.dtype}")

        if slices is not None:
            volume = volume[slices]

        return volume

    except Exception as e:
        raise RuntimeError(f"Failed to read ISQ {isq_path}: {e}") from e


def get_isq_info(isq_path: str | Path) -> dict:
    isq_path = Path(isq_path)
    try:
        reader = itk.ImageFileReader.New(FileName=str(isq_path))
        reader.Update()
        image = reader.GetOutput()
        volume = itk.array_from_image(image)
        return {
            "shape": volume.shape,
            "dtype": str(volume.dtype),
            "spacing": tuple(image.GetSpacing()),
            "origin": tuple(image.GetOrigin()),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to read ISQ metadata from {isq_path}: {e}") from e
