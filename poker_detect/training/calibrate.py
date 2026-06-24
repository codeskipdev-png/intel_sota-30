"""Post-hoc score calibration aligned with subnet reward (FPR-constrained)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from poker44.score.scoring import reward


def _ap_score(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def fit_affine_reward_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_fpr: float = 0.05,
    scale_range: Tuple[float, float] = (1.0, 20.0),
    shift_range: Tuple[float, float] = (-6.0, 1.0),
    n_scale: int = 120,
    n_shift: int = 120,
    min_ap_ratio: float | None = None,
) -> tuple[float, float, float]:
    """
    Find ``clip(scale * score + shift, 0, 1)`` maximizing reward with ``fpr <= max_fpr``.

    When ``min_ap_ratio`` is set, reject candidates whose AP falls below ``raw_ap * ratio``.
    Returns identity (1, 0) if no candidate improves reward under constraints.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    raw_reward, _ = reward(scores, labels)
    raw_ap = _ap_score(scores, labels)
    best_reward = float(raw_reward)
    best_scale = 1.0
    best_shift = 0.0

    for scale in np.linspace(scale_range[0], scale_range[1], n_scale):
        for shift in np.linspace(shift_range[0], shift_range[1], n_shift):
            calibrated = np.clip(scores * scale + shift, 0.0, 1.0)
            rew, metrics = reward(calibrated, labels)
            if float(metrics["fpr"]) > max_fpr:
                continue
            if min_ap_ratio is not None and raw_ap > 0.0:
                cal_ap = _ap_score(calibrated, labels)
                if cal_ap < raw_ap * float(min_ap_ratio):
                    continue
            reward_val = float(rew)
            if reward_val > best_reward + 1e-9:
                best_reward = reward_val
                best_scale = float(scale)
                best_shift = float(shift)

    return best_scale, best_shift, best_reward


def fit_conservative_blend_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_fpr: float = 0.03,
    max_scale: float = 2.5,
    max_shift: float = 0.18,
    min_ap_ratio: float = 0.95,
    n_scale: int = 40,
    n_shift: int = 60,
) -> tuple[float, float, float]:
    """
    Mild val-only calibration for blended scores: small scale, small positive shift.

    Prefers shift over steep scaling to preserve ranking/AP on unseen dates.
    """
    return fit_affine_reward_calibration(
        scores,
        labels,
        max_fpr=max_fpr,
        scale_range=(1.0, float(max_scale)),
        shift_range=(0.0, float(max_shift)),
        n_scale=n_scale,
        n_shift=n_shift,
        min_ap_ratio=min_ap_ratio,
    )


def apply_affine_calibration(
    scores: np.ndarray,
    *,
    scale: float,
    shift: float,
) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float) * scale + shift, 0.0, 1.0)
