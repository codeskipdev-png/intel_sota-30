"""Subnet-aligned reward metrics for conv1d evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from poker44.score.scoring import reward


def subnet_reward_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    _, res = reward(
        np.asarray(y_pred, dtype=float),
        np.asarray(y_true, dtype=float),
    )
    return {k: float(v) for k, v in res.items()}
