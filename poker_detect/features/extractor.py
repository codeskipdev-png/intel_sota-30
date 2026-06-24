"""Fixed-size feature vectors from sanitized Poker44 hand JSON (miner-visible schema)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

FEATURE_SPEC_VERSION = 2
HAND_FEATURE_DIM = 23
EPS = 1e-10

_MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold")

CHUNK_FEATURE_DIM = HAND_FEATURE_DIM * 4 + 1

# Hand dims used for chunk dispersion (sync with features/batch.py).
KEY_HAND_FEATURE_DIMS = (3, 4, 5, 7, 15)  # raise, fold, aggression, street_depth, amt_sd
CHUNK_DISPERSION_DIM = len(KEY_HAND_FEATURE_DIMS) * 2 + 2
TREE_CHUNK_FEATURE_DIM = CHUNK_FEATURE_DIM + CHUNK_DISPERSION_DIM


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _mean_std_max_norm(vals: List[float], *, max_scale: float) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float64)
    peak = max(float(arr.max()), EPS)
    return (
        _clamp01(float(arr.mean()) / peak),
        _clamp01(float(arr.std()) / peak),
        _clamp01(float(arr.max()) / max_scale),
    )


def hand_feature_vector(hand: Dict[str, Any]) -> np.ndarray:
    """
    Map one sanitized hand dict to a fixed-length float vector.

    Raw inputs mirror the reference heuristic miner (``neurons/miner.py``) so an
    MLP can learn combining weights instead of hardcoded coefficients. Betting
    size / pot trajectories are included as extra signal.

    Must stay in sync with training export and ONNX miner inference.
    """
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

    action_counts = Counter(
        str(a.get("action_type") or "other").strip().lower()
        for a in actions
        if isinstance(a, dict)
    )
    meaningful = max(
        1,
        sum(action_counts.get(kind, 0) for kind in _MEANINGFUL_ACTIONS),
    )

    def m_ratio(kind: str) -> float:
        return action_counts.get(kind, 0) / meaningful

    call_ratio = m_ratio("call")
    check_ratio = m_ratio("check")
    bet_ratio = m_ratio("bet")
    raise_ratio = m_ratio("raise")
    fold_ratio = m_ratio("fold")

    street_depth = _clamp01(len(streets) / 3.0)
    streets_norm = _clamp01(len(streets) / 4.0)
    showdown_flag = 1.0 if outcome.get("showdown") else 0.0

    player_count_signal = 0.0
    if players:
        player_count_signal = _clamp01((6 - min(len(players), 6)) / 4.0)
    player_count_norm = _clamp01(min(len(players), 8) / 8.0)

    action_count_norm = _clamp01(len(actions) / 20.0)
    streets_from_actions = {
        str(a.get("street") or "").strip().lower()
        for a in actions
        if isinstance(a, dict) and a.get("street")
    }
    unique_streets_norm = _clamp01(len(streets_from_actions) / 4.0)

    amts: List[float] = []
    pots_before: List[float] = []
    pots_after: List[float] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        amts.append(float(action.get("normalized_amount_bb") or 0.0))
        pots_before.append(float(action.get("pot_before") or 0.0))
        pots_after.append(float(action.get("pot_after") or 0.0))

    amt_mu, amt_sd, amt_mx = _mean_std_max_norm(amts, max_scale=80.0)
    pot_mu, pot_sd, pot_mx = _mean_std_max_norm(pots_after, max_scale=200.0)
    pot_before_mu, pot_before_sd, _ = _mean_std_max_norm(pots_before, max_scale=200.0)

    vec: List[float] = [
        call_ratio,
        check_ratio,
        bet_ratio,
        raise_ratio,
        fold_ratio,
        _clamp01(bet_ratio + raise_ratio),
        _clamp01(call_ratio + check_ratio),
        street_depth,
        streets_norm,
        showdown_flag,
        player_count_signal,
        player_count_norm,
        action_count_norm,
        unique_streets_norm,
        amt_mu,
        amt_sd,
        amt_mx,
        pot_mu,
        pot_sd,
        pot_mx,
        pot_before_mu,
        pot_before_sd,
        m_ratio("all_in") if "all_in" in action_counts else 0.0,
    ]

    assert len(vec) == HAND_FEATURE_DIM
    return np.asarray(vec, dtype=np.float32)


def chunk_feature_vector(hands: List[Dict[str, Any]]) -> np.ndarray:
    """Aggregate hand vectors: mean, std, min, max per dim + normalized chunk size."""
    if not hands:
        return np.zeros(CHUNK_FEATURE_DIM, dtype=np.float32)

    mat = np.stack([hand_feature_vector(h) for h in hands], axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    vmin = mat.min(axis=0)
    vmax = mat.max(axis=0)
    n_norm = np.array([_clamp01(len(hands) / 120.0)], dtype=np.float32)

    out = np.concatenate([mean, std, vmin, vmax, n_norm]).astype(np.float32)
    assert out.shape[0] == CHUNK_FEATURE_DIM
    return out


def chunk_dispersion_vector(hands: List[Dict[str, Any]]) -> np.ndarray:
    """
    Chunk-level dispersion (numpy mirror of ``batch_chunk_dispersion_tensor``).

    Captures heterogeneity within a chunk: per-key means, fraction of high-aggression
    hands, and cross-hand variance — same signal the attention MIL head uses.
    """
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
    """Chunk stats + dispersion for classical classifiers (aligned with neural chunk head)."""
    return np.concatenate(
        [chunk_feature_vector(hands), chunk_dispersion_vector(hands)]
    ).astype(np.float32)
