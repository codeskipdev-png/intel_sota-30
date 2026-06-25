"""Runtime scorer for 1D-CNN action-sequence MIL model (PyTorch or exported ONNX)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Protocol

from poker_detect.training.conv1d_model import load_conv1d_bundle, score_chunk_from_conv1d_bundle
from poker_detect.training.defaults import DEFAULT_CONV1D_MODEL, DEFAULT_CONV1D_ONNX


class Conv1DChunkScorerProtocol(Protocol):
    def score_chunk(self, hands: List[dict[str, Any]]) -> float: ...


class Conv1DChunkScorer:
    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        self._bundle = load_conv1d_bundle(Path(model_path).expanduser().resolve())
        self._device = device

    def score_chunk(self, hands: List[dict[str, Any]]) -> float:
        return score_chunk_from_conv1d_bundle(self._bundle, hands, device=self._device)


def resolve_conv1d_model_path() -> Path:
    onnx_env = os.environ.get("POKER44_CONV1D_ONNX_MODEL_PATH", "").strip()
    if onnx_env:
        return Path(onnx_env).expanduser().resolve()

    model_env = os.environ.get("POKER44_CONV1D_MODEL_PATH", "").strip()
    if model_env:
        return Path(model_env).expanduser().resolve()

    if DEFAULT_CONV1D_ONNX.exists():
        return DEFAULT_CONV1D_ONNX
    return DEFAULT_CONV1D_MODEL


def load_conv1d_scorer_from_env(*, device: str = "cpu") -> Conv1DChunkScorerProtocol:
    model_path = resolve_conv1d_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Conv1D model not found at {model_path}. "
            "Export a bundle with: python -m poker_detect.export.cli_export_conv1d "
            "--bundle-path /path/to/conv1d_model.pt --dist"
        )
    if model_path.suffix.lower() == ".onnx":
        from poker_detect.inference.onnx_conv1d_scorer import OnnxConv1DChunkScorer

        return OnnxConv1DChunkScorer(model_path)
    return Conv1DChunkScorer(model_path, device=device)
