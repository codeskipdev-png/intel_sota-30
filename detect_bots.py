"""Chunk-level bot detection via 1D-CNN action-sequence model.

Model path (first match wins):

- ``POKER44_CONV1D_ONNX_MODEL_PATH`` — exported ``conv1d_model.onnx`` (deployment)
- ``POKER44_CONV1D_MODEL_PATH`` — ``conv1d_model.pt`` or ``.onnx``
- ``poker_detect/dist/conv1d_model.onnx`` if present, else training ``conv1d_model.pt``

Deploy::

    python -m poker_detect.export.cli_export_conv1d --bundle-path /path/to/conv1d_model.pt --dist
    cp .env.example .env
    ./scripts/miner/run/run_app.sh

Eval::

    python detect_bots.py --eval
    python detect_bots.py --eval --val
"""

from __future__ import annotations

import argparse
import json
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


def _load_scorer():
    from poker_detect.inference.conv1d_scorer import load_conv1d_scorer_from_env

    return load_conv1d_scorer_from_env()


def _get_scorer():
    global _scorer
    if _scorer is not None:
        return _scorer
    with _scorer_lock:
        if _scorer is None:
            _scorer = _load_scorer()
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
    bot_fraction: float = 0.2,
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
    bot_fraction: float = 0.2,
) -> Tuple[List[float], List[bool]]:
    """
    Score each chunk (list of hand dicts) with the conv1d model.

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


def _print_reward_block(
    title: str,
    scores: List[float],
    ground_truth: List[float | int | bool],
) -> None:
    from poker44.score.scoring import reward

    print(title)
    print(f"scores: {_score_summary(scores)}")
    rew, metrics = reward(np.asarray(scores, dtype=float), np.asarray(ground_truth))
    print(
        f"reward={rew:.4f} fpr={metrics['fpr']:.4f} "
        f"recall={metrics['bot_recall']:.4f} ap={metrics['ap_score']:.4f}"
    )
    preds = [bool(round(s)) for s in scores]
    print(f"predictions={preds}")
    print(f"sum bots={sum(preds)}")


def _run_benchmark_demo(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sub_data in data["data"]["chunks"]:
        chunks = sub_data["chunks"]
        ground_truth = sub_data["groundTruth"]

        print("=" * 60)
        raw_scores, _ = detect_bots(chunks, post_process=False)
        _print_reward_block("WITHOUT post_process", raw_scores, ground_truth)
        print(f"scores: {raw_scores}")

        print("-" * 60)
        pp_scores, _ = detect_bots(chunks, post_process=True)
        _print_reward_block("WITH post_process", pp_scores, ground_truth)
        print(f"scores: {pp_scores}")

        print(f"groundTruth={ground_truth}")
        print(f"sum groundTruth bots={sum(ground_truth)}")


def _run_eval(args: argparse.Namespace) -> None:
    from poker_detect.training.defaults import (
        DEFAULT_BENCHMARK_DIR,
        TEST_END_DATE,
        TEST_START_DATE,
        VAL_END_DATE,
        VAL_START_DATE,
    )
    from poker_detect.training.eval_conv1d_loop import evaluate_conv1d

    start_date = args.start_date or TEST_START_DATE
    end_date = args.end_date or TEST_END_DATE
    if args.val:
        start_date = VAL_START_DATE
        end_date = VAL_END_DATE

    data_dir = args.data_dir or DEFAULT_BENCHMARK_DIR

    reset_scorer()
    result = evaluate_conv1d(
        data_dir,
        args.model_path,
        start_date,
        end_date,
        device=args.device,
    )
    print(json.dumps({"ok": True, **result}, indent=2))


def main(argv: list[str] | None = None) -> None:
    from poker_detect.training.defaults import DEFAULT_BENCHMARK_DIR, TEST_END_DATE

    p = argparse.ArgumentParser(description="Poker44 conv1d chunk bot detection.")
    p.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate saved conv1d model on benchmark date range (default: test split)",
    )
    p.add_argument("--val", action="store_true", help="With --eval: use val split dates")
    p.add_argument("--device", type=str, default="cpu", help="PyTorch device for --eval")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override conv1d_model.pt path for --eval (inference uses env / dist ONNX)",
    )
    p.add_argument("--start_date", type=str, default="2026-06-24")
    p.add_argument("--end_date", type=str, default="2026-06-24")
    p.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path(
            os.environ.get(
                "POKER44_BENCHMARK_JSON",
                str(DEFAULT_BENCHMARK_DIR / f"benchmark_{TEST_END_DATE}.json"),
            )
        ),
        help="Single benchmark JSON for demo mode (default when --eval not set)",
    )
    args = p.parse_args(argv)

    if args.eval or args.val:
        _run_eval(args)
        return

    benchmark = args.benchmark_json
    if not benchmark.exists():
        raise SystemExit(
            f"benchmark not found: {benchmark}\n"
            "Use --eval for date-range eval, or set POKER44_BENCHMARK_JSON."
        )
    _run_benchmark_demo(benchmark)


if __name__ == "__main__":
    main()
