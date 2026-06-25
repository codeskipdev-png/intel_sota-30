"""Affine score calibration used by exported conv1d ONNX bundles."""

from __future__ import annotations

import numpy as np


def apply_affine_calibration(
    scores: np.ndarray,
    *,
    scale: float,
    shift: float,
) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float) * scale + shift, 0.0, 1.0)
