"""Chunk-level bot detection for HTTP servers and local scripts.

Loads exported ``model.onnx`` (+ optional ``tree_ensemble.joblib`` sidecar) and scores
each chunk independently. Predictions follow subnet scoring: ``round(risk_score)``.

Environment:

- ``POKER44_ONNX_MODEL_PATH`` — path to ``model.onnx`` (default: ``poker_detect/dist/model.onnx``)
- ``POKER44_ONNX_PREPROCESS_PATH`` — optional ``*.preprocess.json``
- ``POKER44_TREE_BUNDLE_PATH`` — optional ``tree_ensemble.joblib`` (default: next to ONNX)

Requires: ``pip install -e ".[detect]"`` or at least ``onnxruntime`` (+ ``joblib``/``xgboost`` if blended).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

_scorer = None
_scorer_lock = threading.Lock()


def _get_scorer():
    global _scorer
    if _scorer is not None:
        return _scorer
    with _scorer_lock:
        if _scorer is None:
            from poker_detect.inference.onnx_scorer import load_detection_scorer_from_env

            _scorer = load_detection_scorer_from_env()
    return _scorer


def _sanitize_chunk(chunk: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Match validator miner-visible hand schema before feature extraction."""
    from poker44.validator.sanitization import sanitize_hand_for_miner

    return [
        sanitize_hand_for_miner(h) if isinstance(h, dict) else sanitize_hand_for_miner({})
        for h in chunk
    ]


def _score_summary(scores: List[float]) -> str:
    if not scores:
        return "n=0"
    arr = np.asarray(scores, dtype=float)
    return (
        f"n={len(scores)} min={float(arr.min()):.4f} max={float(arr.max()):.4f} "
        f"mean={float(arr.mean()):.4f} ge_0.5={int(np.sum(arr >= 0.5))} "
        f"pred_bot={int(np.sum(np.round(arr).astype(int)))}"
    )


def sanitize_chunks(chunks: List[List[dict[str, Any]]]) -> List[List[dict[str, Any]]]:
    """Sanitize all hands in each chunk (validator-compatible schema)."""
    return [_sanitize_chunk(chunk) for chunk in (chunks or [])]


def score_summary(scores: List[float]) -> str:
    """Human-readable summary of chunk risk scores (for logging / debugging)."""
    return _score_summary(scores)


def post_process_request(
    risk_scores: List[float],
    *,
    bot_fraction: float = 0.3,
) -> Tuple[List[float], List[bool]]:
    """
    Per-request post-processing: label the top ``round(n * bot_fraction)`` chunks
    by raw score as bot, then renormalize so bot scores are > 0.5 and human
    scores are < 0.5 (rank preserved within each group).
    """
    n = len(risk_scores)
    if n == 0:
        return [], []

    raw = np.asarray(risk_scores, dtype=np.float64)
    k = min(n, max(0, int(round(n * bot_fraction))))

    order = np.argsort(-raw, kind="stable")
    is_bot = np.zeros(n, dtype=bool)
    if k > 0:
        is_bot[order[:k]] = True

    out = np.empty(n, dtype=np.float64)

    bot_idx = np.flatnonzero(is_bot)
    if bot_idx.size == 1:
        out[bot_idx[0]] = max(0.51, min(0.99, float(raw[bot_idx[0]])))
    elif bot_idx.size > 1:
        ranked = bot_idx[np.argsort(-raw[bot_idx], kind="stable")]
        for rank, i in enumerate(ranked):
            t = rank / (ranked.size - 1)
            out[i] = 0.51 + t * 0.48

    human_idx = np.flatnonzero(~is_bot)
    if human_idx.size == 1:
        out[human_idx[0]] = min(0.49, max(0.01, float(raw[human_idx[0]])))
    elif human_idx.size > 0:
        ranked = human_idx[np.argsort(-raw[human_idx], kind="stable")]
        for rank, i in enumerate(ranked):
            t = rank / (ranked.size - 1)
            out[i] = 0.01 + (1.0 - t) * 0.48

    predictions = [bool(b) for b in is_bot]
    return out.astype(float).tolist(), predictions


def detect_bots(
    chunks: List[List[dict[str, Any]]],
    *,
    sanitize: bool = True,
    post_process: bool = True,
    bot_fraction: float = 0.3,
) -> Tuple[List[float], List[bool]]:
    """
    Score each chunk (list of hand dicts).

    When ``sanitize=True`` (default), hands are passed through
    ``sanitize_hand_for_miner`` so features match training and the live validator.

    When ``post_process=True`` (default), applies ``post_process_request``:
    top ``round(n * bot_fraction)`` chunks by raw score are labeled bot; scores are
    renormalized so bots are > 0.5 and humans are < 0.5.

    Returns:
        ``risk_scores`` — float per chunk (post-processed when enabled).
        ``predictions`` — ``True`` for bot chunks after post-processing.
    """
    chunk_list = chunks or []
    if not chunk_list:
        return [], []

    scorer = _get_scorer()
    risk_scores: list[float] = []
    for chunk in chunk_list:
        hands = _sanitize_chunk(chunk) if sanitize else [h or {} for h in chunk]
        risk_scores.append(float(scorer.score_chunk(hands)))
    if post_process:
        return post_process_request(risk_scores, bot_fraction=bot_fraction)
    predictions = [bool(round(score)) for score in risk_scores]
    return risk_scores, predictions


def reset_scorer() -> None:
    """Clear cached scorer (for tests or hot reload)."""
    global _scorer
    with _scorer_lock:
        _scorer = None


if __name__ == "__main__":
    import json

    from poker44.score.scoring import reward

    path = Path(
        os.environ.get(
            "POKER44_BENCHMARK_JSON",
            "C:/Users/admin/Documents/workspace/poker/bt_tool/dataset_maker/benchmark_out/benchmark_2026-06-23.json",
        )
    )
    if not path.exists():
        raise SystemExit(f"benchmark not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sub_data in data["data"]["chunks"]:
        chunks = sub_data["chunks"]
        ground_truth = sub_data["groundTruth"]

        print("=" * 60)
        print("WITHOUT post_process")
        raw_scores, raw_preds = detect_bots(chunks, post_process=False)
        print(f"scores: {_score_summary(raw_scores)}")
        print(f"scores: {raw_scores}")
        raw_rew, raw_metrics = reward(
            np.asarray(raw_scores, dtype=float), np.asarray(ground_truth)
        )
        print(f"reward={raw_rew:.4f} fpr={raw_metrics['fpr']:.4f} recall={raw_metrics['bot_recall']:.4f}")
        print(f"predictions={raw_preds}")
        print(f"sum bots={sum(raw_preds)}")

        print("-" * 60)
        print("WITH post_process")
        pp_scores, pp_preds = detect_bots(chunks, post_process=True)
        print(f"scores: {_score_summary(pp_scores)}")
        print(f"scores: {pp_scores}")
        pp_rew, pp_metrics = reward(
            np.asarray(pp_scores, dtype=float), np.asarray(ground_truth)
        )
        print(f"reward={pp_rew:.4f} fpr={pp_metrics['fpr']:.4f} recall={pp_metrics['bot_recall']:.4f}")
        print(f"predictions={pp_preds}")
        print(f"sum bots={sum(pp_preds)}")

        print(f"groundTruth={ground_truth}")
        print(f"sum groundTruth bots={sum(ground_truth)}")
