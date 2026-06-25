"""Evaluate conv1d chunk classifier on benchmark JSON chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from poker_detect.inference.conv1d_scorer import (
    Conv1DChunkScorer,
    load_conv1d_scorer_from_env,
    resolve_conv1d_model_path,
)
from poker_detect.inference.onnx_conv1d_scorer import OnnxConv1DChunkScorer
from poker_detect.training.jsonl_loader import ChunkBenchmarkDataset, label_counts_chunk
from poker_detect.training.window_metrics import subnet_reward_metrics


def _load_scorer(model_path: Path | None, *, device: str = "cpu"):
    if model_path is None:
        return load_conv1d_scorer_from_env(device=device)

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"conv1d model not found: {path}")
    if path.suffix.lower() == ".onnx":
        return OnnxConv1DChunkScorer(path)
    return Conv1DChunkScorer(path, device=device)


def evaluate_conv1d(
    data_dir: Path,
    model_path: Path | None,
    start_date: str,
    end_date: str,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    resolved = (
        Path(model_path).expanduser().resolve()
        if model_path is not None
        else resolve_conv1d_model_path()
    )
    scorer = _load_scorer(model_path, device=device)

    test_ds = ChunkBenchmarkDataset(data_dir, start_date, end_date)
    n0, n1 = label_counts_chunk(test_ds)
    print(f"test chunks={len(test_ds)} human={n0} bot={n1}")

    scores = [float(scorer.score_chunk(chunk)) for chunk in test_ds.chunks]
    y_true = np.asarray(test_ds.labels, dtype=float)
    y_score = np.asarray(scores, dtype=float)

    reward_metrics = subnet_reward_metrics(y_score, y_true)
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    except ValueError:
        auroc = 0.0

    print(
        f"scorer=conv1d test_ap={ap:.4f} test_auroc={auroc:.4f} "
        f"test_reward={reward_metrics['reward']:.4f} "
        f"test_fpr={reward_metrics['fpr']:.4f} "
        f"test_bot_recall={reward_metrics['bot_recall']:.4f}"
    )

    train_meta: dict[str, Any] = {}
    if resolved.suffix.lower() == ".onnx":
        pre_path = resolved.with_suffix(".preprocess.json")
        if pre_path.exists():
            train_meta = json.loads(pre_path.read_text(encoding="utf-8"))

    return {
        "scorer": "conv1d",
        "model_path": str(resolved),
        "average_precision": float(ap),
        "auroc": float(auroc),
        "n_chunks": len(test_ds),
        "start_date": start_date,
        "end_date": end_date,
        "train_metrics": train_meta.get("train_metrics"),
        "val_metrics": train_meta.get("val_metrics"),
        **reward_metrics,
    }
