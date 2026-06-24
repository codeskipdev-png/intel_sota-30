"""Feature spec v3: deduplicated hand stats + mean/std chunk aggregation only."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from poker_detect.features.extractor import hand_feature_vector as hand_feature_vector_v2

FEATURE_SPEC_VERSION = 3

# Subset of v2 hand dims (see ``extractor.hand_feature_vector`` ordering).
_V2_HAND_SLIM_INDICES = (0, 1, 2, 3, 4, 7, 9, 10, 12, 15, 16, 18, 19, 22)

HAND_FEATURE_DIM = len(_V2_HAND_SLIM_INDICES)
KEY_HAND_FEATURE_DIMS = (3, 4, 9)  # raise, fold, amt_sd in slim space

CHUNK_FEATURE_DIM = HAND_FEATURE_DIM * 2 + 1
CHUNK_DISPERSION_DIM = len(KEY_HAND_FEATURE_DIMS) * 2 + 2
TREE_CHUNK_FEATURE_DIM = CHUNK_FEATURE_DIM + CHUNK_DISPERSION_DIM


def hand_feature_vector(hand: Dict[str, Any]) -> np.ndarray:
    """14-dim hand vector: action ratios + street/showdown/player/action + betting spread."""
    full = hand_feature_vector_v2(hand)
    return full[list(_V2_HAND_SLIM_INDICES)].astype(np.float32)


def chunk_feature_vector(hands: List[Dict[str, Any]]) -> np.ndarray:
    """29-dim chunk vector: mean + std per slim hand dim + normalized chunk size."""
    if not hands:
        return np.zeros(CHUNK_FEATURE_DIM, dtype=np.float32)

    mat = np.stack([hand_feature_vector(h) for h in hands], axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    n_norm = np.array([min(1.0, max(0.0, len(hands) / 120.0))], dtype=np.float32)
    out = np.concatenate([mean, std, n_norm]).astype(np.float32)
    assert out.shape[0] == CHUNK_FEATURE_DIM
    return out


def chunk_dispersion_vector(hands: List[Dict[str, Any]]) -> np.ndarray:
    """8-dim dispersion on raise/fold/amt_sd + global cross-hand variance."""
    if not hands:
        return np.zeros(CHUNK_DISPERSION_DIM, dtype=np.float32)

    mat = np.stack([hand_feature_vector(h) for h in hands], axis=0)
    count = float(len(hands))
    parts: list[float] = []

    for dim_idx in KEY_HAND_FEATURE_DIMS:
        vals = mat[:, dim_idx]
        parts.append(float(vals.mean()))
        parts.append(float((vals > 0.35).sum()) / count)

    centered = mat - mat.mean(axis=0, keepdims=True)
    per_dim_std = centered.std(axis=0)
    parts.append(float(per_dim_std.mean()))
    parts.append(float(per_dim_std.max()))

    out = np.asarray(parts, dtype=np.float32)
    assert out.shape[0] == CHUNK_DISPERSION_DIM
    return out


def tree_chunk_feature_vector(hands: List[Dict[str, Any]]) -> np.ndarray:
    """37-dim tabular chunk vector for classical models."""
    return np.concatenate(
        [chunk_feature_vector(hands), chunk_dispersion_vector(hands)]
    ).astype(np.float32)
