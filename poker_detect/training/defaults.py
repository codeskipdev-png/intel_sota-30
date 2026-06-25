"""Defaults for detect_bots conv1d inference and benchmark evaluation."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BENCHMARK_DIR = Path(
    "C:/Users/admin/Documents/workspace/poker/bt_tool/dataset_maker/benchmark_out"
)

DEFAULT_CONV1D_ONNX = _REPO_ROOT / "poker_detect/dist/conv1d_model.onnx"
DEFAULT_CONV1D_MODEL = _REPO_ROOT / "poker_detect/dist/conv1d_model.pt"

VAL_START_DATE = "2026-06-18"
VAL_END_DATE = "2026-06-21"
TEST_START_DATE = "2026-06-22"
TEST_END_DATE = "2026-06-24"
