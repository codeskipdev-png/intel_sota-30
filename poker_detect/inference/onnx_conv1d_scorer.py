"""ONNXRuntime scorer for exported 1D-CNN action-sequence models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import numpy as np

from poker_detect.features.action_sequence import MAX_CHUNK_SEQ_LEN, chunk_action_index_sequence
from poker_detect.training.calibrate import apply_affine_calibration


def _load_preprocess(preprocess_path: Path) -> dict[str, Any]:
    if not preprocess_path.exists():
        return {}
    return json.loads(preprocess_path.read_text(encoding="utf-8"))


class OnnxConv1DChunkScorer:
    """Flat action-index sequence -> chunk bot risk score via exported ONNX."""

    def __init__(
        self,
        onnx_path: Path,
        *,
        preprocess_path: Path | None = None,
    ) -> None:
        import onnxruntime as ort

        onnx_path = Path(onnx_path).expanduser().resolve()
        if preprocess_path is None:
            preprocess_path = onnx_path.with_suffix(".preprocess.json")
        meta = _load_preprocess(Path(preprocess_path))

        if meta.get("model_type") not in (None, "conv1d"):
            raise ValueError(
                f"preprocess model_type={meta.get('model_type')!r} != conv1d for {onnx_path}"
            )

        self._max_seq_len = int(meta.get("max_seq_len", MAX_CHUNK_SEQ_LEN))
        self._calib_scale = float(meta.get("calib_scale", 1.0))
        self._calib_shift = float(meta.get("calib_shift", 0.0))
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

    def score_chunk(self, hands: List[dict[str, Any]]) -> float:
        if not hands:
            return 0.5
        ids = chunk_action_index_sequence(
            hands,
            sanitize=False,
            max_seq_len=self._max_seq_len,
        )
        (raw,) = self._session.run(
            None,
            {"action_ids": ids.reshape(1, -1).astype(np.int64)},
        )
        score = float(raw.reshape(-1)[0])
        calibrated = apply_affine_calibration(
            np.array([score]),
            scale=self._calib_scale,
            shift=self._calib_shift,
        )
        return float(calibrated[0])
