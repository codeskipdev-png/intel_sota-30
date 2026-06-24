"""ONNXRuntime chunk bot detection (+ optional classical tree blend)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from poker_detect.features.extractor import (
    FEATURE_SPEC_VERSION,
    HAND_FEATURE_DIM,
    TREE_CHUNK_FEATURE_DIM,
    hand_feature_vector,
)
from poker_detect.features.registry import resolve_tree_feature_spec, tree_feature_vector_for_hands
from poker_detect.training.calibrate import apply_affine_calibration
from poker_detect.training.tree_ensemble import (
    TREE_ENSEMBLE_FILENAME,
    blend_scores,
    predict_tree_probas,
)


class ChunkScorer(Protocol):
    def score_chunk(self, hands: List[Dict[str, Any]]) -> float: ...


def _load_preprocess(preprocess_path: Path) -> dict[str, Any]:
    if not preprocess_path.exists():
        return {}
    return json.loads(preprocess_path.read_text(encoding="utf-8"))


def _check_feature_spec(meta: dict[str, Any]) -> None:
    version = int(meta.get("feature_spec_version", FEATURE_SPEC_VERSION))
    if version != FEATURE_SPEC_VERSION:
        raise ValueError(
            f"preprocess feature_spec_version={version} != runtime {FEATURE_SPEC_VERSION}"
        )


class OnnxHandScorer:
    """Legacy hand-level MLP: mean sigmoid hand score per chunk."""

    def __init__(self, onnx_path: Path, preprocess_path: Optional[Path] = None):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if preprocess_path is None:
            preprocess_path = onnx_path.with_suffix(".preprocess.json")
        meta = _load_preprocess(Path(preprocess_path))
        _check_feature_spec(meta)
        self.hand_feature_dim = int(meta.get("hand_feature_dim", HAND_FEATURE_DIM))
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

    def score_hand(self, hand: Dict[str, Any]) -> float:
        x = hand_feature_vector(hand).astype(np.float32)
        if x.shape[0] != self.hand_feature_dim:
            raise ValueError(f"feature dim {x.shape[0]} != expected {self.hand_feature_dim}")
        (logit,) = self._session.run(None, {"hand_features": x.reshape(1, -1)})
        score = 1.0 / (1.0 + np.exp(-float(logit.reshape(-1)[0])))
        return float(max(0.0, min(1.0, score)))

    def score_chunk(self, hands: List[Dict[str, Any]]) -> float:
        if not hands:
            return 0.5
        scores = [self.score_hand(h or {}) for h in hands]
        return float(max(0.0, min(1.0, sum(scores) / len(scores))))


class OnnxChunkModelScorer:
    """Chunk ensemble / attention / hybrid exported as (features, mask) -> chunk_score."""

    def __init__(self, onnx_path: Path, preprocess_path: Optional[Path] = None):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if preprocess_path is None:
            preprocess_path = onnx_path.with_suffix(".preprocess.json")
        meta = _load_preprocess(Path(preprocess_path))
        _check_feature_spec(meta)
        self.hand_feature_dim = int(meta.get("hand_feature_dim", HAND_FEATURE_DIM))
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

    def score_chunk(self, hands: List[Dict[str, Any]]) -> float:
        from poker_detect.training.jsonl_loader import collate_hand_chunks

        if not hands:
            return 0.5
        features, mask, _labels = collate_hand_chunks([(hands, 0.0)])
        (score,) = self._session.run(
            None,
            {
                "features": features.numpy().astype(np.float32),
                "mask": mask.numpy().astype(np.float32),
            },
        )
        return float(max(0.0, min(1.0, float(score.reshape(-1)[0]))))


class OnnxBlendedChunkScorer:
    """
    ONNX neural chunk score + classical tree ensemble (from ``tree_ensemble.joblib``).

    Blend weights and calibration come from ``*.preprocess.json`` (val-tuned at train time).
    """

    def __init__(
        self,
        onnx_path: Path,
        *,
        preprocess_path: Optional[Path] = None,
        tree_bundle_path: Optional[Path] = None,
    ):
        import joblib

        onnx_path = Path(onnx_path)
        if preprocess_path is None:
            preprocess_path = onnx_path.with_suffix(".preprocess.json")
        meta = _load_preprocess(Path(preprocess_path))
        _check_feature_spec(meta)

        blend = meta.get("blend") or {}
        self._weights: dict[str, float] = {k: float(v) for k, v in blend.get("weights", {}).items()}
        if not self._weights:
            raise ValueError("blend weights missing in preprocess metadata")
        self._calib_scale = float(blend.get("calib_scale", 1.0))
        self._calib_shift = float(blend.get("calib_shift", 0.0))

        bundle_path = Path(
            tree_bundle_path
            or meta.get("tree_bundle_path")
            or (onnx_path.parent / TREE_ENSEMBLE_FILENAME)
        )
        if not bundle_path.exists():
            raise FileNotFoundError(f"tree ensemble bundle not found: {bundle_path}")
        bundle = joblib.load(bundle_path)
        self._train_pipelines = bundle.get("train_pipelines") or bundle.get("pipelines")
        if self._train_pipelines is None:
            raise ValueError(f"tree bundle missing pipelines: {bundle_path}")
        self._tree_feature_dim = int(
            bundle.get("tree_chunk_feature_dim", TREE_CHUNK_FEATURE_DIM)
        )
        self._tree_feature_spec = int(
            bundle.get(
                "tree_feature_spec_version",
                resolve_tree_feature_spec(
                    tree_chunk_feature_dim=self._tree_feature_dim
                ).version,
            )
        )

        self._neural = OnnxChunkModelScorer(onnx_path, preprocess_path=preprocess_path)

    def _tree_feature_vector(self, hands: List[Dict[str, Any]]) -> np.ndarray:
        return tree_feature_vector_for_hands(
            hands,
            feature_spec=self._tree_feature_spec,
            tree_chunk_feature_dim=self._tree_feature_dim,
        )

    def score_chunk(self, hands: List[Dict[str, Any]]) -> float:
        if not hands:
            return 0.5
        neural = self._neural.score_chunk(hands)
        feat = self._tree_feature_vector(hands).reshape(1, -1)
        if feat.shape[1] != self._tree_feature_dim:
            raise ValueError(
                f"tree feature dim {feat.shape[1]} != expected {self._tree_feature_dim}"
            )
        tree_probas = predict_tree_probas(self._train_pipelines, feat)
        blended = blend_scores(
            np.array([neural]),
            {k: v.reshape(1) for k, v in tree_probas.items()},
            self._weights,
        )
        calibrated = apply_affine_calibration(
            blended,
            scale=self._calib_scale,
            shift=self._calib_shift,
        )
        return float(calibrated[0])


def load_detection_scorer(
    onnx_path: Path | str,
    *,
    preprocess_path: Path | str | None = None,
    tree_bundle_path: Path | str | None = None,
) -> ChunkScorer:
    """
    Load the best available scorer for ``model.onnx`` + sidecar metadata.

    Uses tree blend when ``tree_ensemble.joblib`` and blend metadata are present.
    """
    onnx_path = Path(onnx_path).expanduser().resolve()
    pre_path = (
        Path(preprocess_path).expanduser().resolve()
        if preprocess_path
        else onnx_path.with_suffix(".preprocess.json")
    )
    meta = _load_preprocess(pre_path)
    model_type = str(meta.get("model_type", "hand"))
    tree_path = (
        Path(tree_bundle_path).expanduser().resolve()
        if tree_bundle_path
        else None
    )
    bundle_candidate = tree_path or Path(
        meta.get("tree_bundle_path") or (onnx_path.parent / TREE_ENSEMBLE_FILENAME)
    )
    has_blend = bool(meta.get("blend")) and bundle_candidate.exists()

    if has_blend and model_type != "hand":
        return OnnxBlendedChunkScorer(
            onnx_path,
            preprocess_path=pre_path,
            tree_bundle_path=bundle_candidate,
        )
    if model_type == "hand":
        return OnnxHandScorer(onnx_path, preprocess_path=pre_path)
    return OnnxChunkModelScorer(onnx_path, preprocess_path=pre_path)


def load_detection_scorer_from_env() -> ChunkScorer:
    """Resolve scorer paths from ``POKER44_ONNX_*`` environment variables."""
    onnx_env = os.environ.get("POKER44_ONNX_MODEL_PATH", "").strip()
    if not onnx_env:
        repo_root = Path(__file__).resolve().parents[2]
        onnx_path = repo_root / "poker_detect" / "dist" / "model.onnx"
    else:
        onnx_path = Path(onnx_env)
    pre_env = os.environ.get("POKER44_ONNX_PREPROCESS_PATH", "").strip()
    tree_env = os.environ.get("POKER44_TREE_BUNDLE_PATH", "").strip()
    return load_detection_scorer(
        onnx_path,
        preprocess_path=Path(pre_env) if pre_env else None,
        tree_bundle_path=Path(tree_env) if tree_env else None,
    )


OnnxChunkScorer = OnnxChunkModelScorer
