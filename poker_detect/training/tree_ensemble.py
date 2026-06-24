"""MLP + tree ensemble (XGBoost, RF, DT) on chunk feature vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from poker44.score.scoring import reward
from poker_detect.features.registry import (
    DEFAULT_TREE_FEATURE_SPEC,
    chunk_feature_matrix as registry_chunk_feature_matrix,
    get_tree_feature_spec,
    resolve_tree_feature_spec,
    tree_feature_vector_for_hands,
)
from poker_detect.training.calibrate import (
    apply_affine_calibration,
    fit_conservative_blend_calibration,
)

TREE_NAMES = ("xgb", "rf", "dt")
DEFAULT_MIN_MLP_WEIGHT = 0.75
TREE_ENSEMBLE_FILENAME = "tree_ensemble.joblib"
TREE_BLEND_META_FILENAME = "tree_blend_meta.json"


def build_tree_pipelines(*, seed: int = 42) -> dict[str, Pipeline]:
    """Regularized tree models for small-sample chunk classification (~300 chunks)."""
    return {
        "xgb": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "clf",
                    xgb.XGBClassifier(
                        n_estimators=40,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=5.0,
                        min_child_weight=8,
                        eval_metric="logloss",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=4,
                        min_samples_leaf=12,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "dt": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=3,
                        min_samples_leaf=15,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def chunk_feature_matrix(
    chunks: list,
    *,
    use_tree_features: bool = True,
    feature_spec: int = DEFAULT_TREE_FEATURE_SPEC,
) -> np.ndarray:
    del use_tree_features  # trees always use tree_chunk vector; kept for callers
    return registry_chunk_feature_matrix(chunks, feature_spec=feature_spec)


def predict_tree_probas(
    pipelines: Mapping[str, Pipeline],
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in TREE_NAMES:
        out[name] = pipelines[name].predict_proba(X)[:, 1]
    return out


def predict_tree_probas_ensemble(
    pipelines_list: list[dict[str, Pipeline]],
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    """Average tree probabilities across LODO fold models."""
    if not pipelines_list:
        raise ValueError("pipelines_list must not be empty")
    acc = {name: np.zeros(len(X), dtype=float) for name in TREE_NAMES}
    for pipelines in pipelines_list:
        probas = predict_tree_probas(pipelines, X)
        for name in TREE_NAMES:
            acc[name] += probas[name]
    for name in TREE_NAMES:
        acc[name] /= float(len(pipelines_list))
    return acc


def fit_lodo_tree_ensemble(
    chunks: list,
    labels: np.ndarray,
    chunk_dates: list[str],
    *,
    seed: int = 42,
    min_train_chunks: int = 20,
    feature_spec: int = DEFAULT_TREE_FEATURE_SPEC,
) -> tuple[list[dict[str, Pipeline]], dict[str, np.ndarray]]:
    """
    Leave-one-date-out tree training.

    Returns fold pipeline list and out-of-fold tree probabilities on train chunks.
    """
    n = len(chunks)
    if n != len(chunk_dates):
        raise ValueError("chunks and chunk_dates length mismatch")

    labels = np.asarray(labels, dtype=float)
    dates_arr = np.asarray(chunk_dates)
    unique_dates = sorted(set(chunk_dates))

    oof = {name: np.zeros(n, dtype=float) for name in TREE_NAMES}
    oof_count = np.zeros(n, dtype=int)
    fold_pipelines: list[dict[str, Pipeline]] = []

    for held_date in unique_dates:
        train_idx = np.where(dates_arr != held_date)[0]
        hold_idx = np.where(dates_arr == held_date)[0]
        if len(train_idx) < min_train_chunks or len(hold_idx) == 0:
            continue

        train_chunks = [chunks[int(i)] for i in train_idx]
        train_y = labels[train_idx]
        pipelines = fit_tree_ensemble_on_chunks(
            train_chunks, train_y, seed=seed, feature_spec=feature_spec
        )
        fold_pipelines.append(pipelines)

        hold_chunks = [chunks[int(i)] for i in hold_idx]
        X_hold = chunk_feature_matrix(hold_chunks, feature_spec=feature_spec)
        probas = predict_tree_probas(pipelines, X_hold)
        for name in TREE_NAMES:
            oof[name][hold_idx] += probas[name]
        oof_count[hold_idx] += 1

    filled = oof_count > 0
    for name in TREE_NAMES:
        oof[name][filled] /= oof_count[filled]

    if not fold_pipelines:
        full = fit_tree_ensemble_on_chunks(chunks, labels, seed=seed, feature_spec=feature_spec)
        fold_pipelines = [full]
        oof = predict_tree_probas(
            full, chunk_feature_matrix(chunks, feature_spec=feature_spec)
        )

    return fold_pipelines, oof


def blend_scores(
    mlp_scores: np.ndarray,
    tree_probas: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    total = sum(float(weights[k]) for k in weights)
    out = (float(weights["mlp"]) / total) * np.asarray(mlp_scores, dtype=float)
    for name in TREE_NAMES:
        out += (float(weights[name]) / total) * np.asarray(tree_probas[name], dtype=float)
    return np.clip(out, 0.0, 1.0)


def fit_blend_weights(
    mlp_scores: np.ndarray,
    tree_probas: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    max_fpr: float = 0.03,
    min_mlp_weight: float = DEFAULT_MIN_MLP_WEIGHT,
    step: float = 0.02,
) -> tuple[dict[str, float], dict[str, float]]:
    """Search blend weights maximizing reward with FPR <= max_fpr on val/train only."""
    labels = np.asarray(labels, dtype=float)
    mlp_scores = np.asarray(mlp_scores, dtype=float)
    best_reward = -1.0
    best_weights = {"mlp": min_mlp_weight, "xgb": 0.10, "rf": 0.10, "dt": 0.05}
    best_metrics: dict[str, float] = {}

    mlp_grid = np.arange(min_mlp_weight, 1.0 + step / 2, step)
    share_grid = np.arange(0.0, 1.0 + step / 2, step)

    for wm in mlp_grid:
        rem = 1.0 - float(wm)
        for wx in share_grid:
            for wr in share_grid:
                wd = rem - wx - wr
                if wd < -1e-9:
                    continue
                weights = {
                    "mlp": float(wm),
                    "xgb": float(wx),
                    "rf": float(wr),
                    "dt": float(max(0.0, wd)),
                }
                scores = blend_scores(mlp_scores, tree_probas, weights)
                rew, metrics = reward(scores, labels)
                fpr = float(metrics["fpr"])
                if fpr > max_fpr:
                    continue
                reward_val = float(rew)
                mlp_w = float(weights["mlp"])
                if reward_val > best_reward + 1e-9 or (
                    abs(reward_val - best_reward) <= 1e-9 and mlp_w > best_weights["mlp"]
                ):
                    best_reward = reward_val
                    best_weights = weights
                    best_metrics = {k: float(v) for k, v in metrics.items()}
                    best_metrics["reward"] = best_reward

    return best_weights, best_metrics


def classical_probas_for_dataset(
    pipelines: Mapping[str, Pipeline],
    dataset: Any,
    *,
    feature_spec: int = DEFAULT_TREE_FEATURE_SPEC,
) -> dict[str, np.ndarray]:
    """Tree probabilities for a ChunkBenchmarkDataset using pre-fit pipelines."""
    X = chunk_feature_matrix(dataset.chunks, feature_spec=feature_spec)
    return predict_tree_probas(pipelines, X)


def train_classical_on_dataset(
    train_ds: Any,
    *,
    seed: int = 42,
    feature_spec: int = DEFAULT_TREE_FEATURE_SPEC,
) -> dict[str, Pipeline]:
    """Fit XGB + RF + DT on the full train split only (independent of MLP)."""
    return fit_tree_ensemble_on_chunks(
        train_ds.chunks,
        np.asarray(train_ds.labels, dtype=float),
        seed=seed,
        feature_spec=feature_spec,
    )


def fit_combine_strategy(
    mlp_scores: np.ndarray,
    tree_probas: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    max_fpr: float = 0.05,
    min_mlp_weight: float = DEFAULT_MIN_MLP_WEIGHT,
    calib_max_fpr: float = 0.03,
    calib_max_scale: float = 2.5,
    calib_max_shift: float = 0.18,
    calib_min_ap_ratio: float = 0.95,
) -> tuple[dict[str, float], float, float, dict[str, float], dict[str, float]]:
    """
    Val-only: search blend weights, then conservative affine calibration on the blend.
    """
    weights, weight_metrics = fit_blend_weights(
        mlp_scores,
        tree_probas,
        labels,
        max_fpr=max_fpr,
        min_mlp_weight=min_mlp_weight,
    )
    blended = blend_scores(mlp_scores, tree_probas, weights)
    calib_scale, calib_shift, _ = fit_conservative_blend_calibration(
        blended,
        labels,
        max_fpr=calib_max_fpr,
        max_scale=calib_max_scale,
        max_shift=calib_max_shift,
        min_ap_ratio=calib_min_ap_ratio,
    )
    calib_metrics = metrics_for_blend(
        mlp_scores,
        tree_probas,
        labels,
        weights,
        calib_scale=calib_scale,
        calib_shift=calib_shift,
    )
    return weights, calib_scale, calib_shift, weight_metrics, calib_metrics


def metrics_for_blend(
    mlp_scores: np.ndarray,
    tree_probas: Mapping[str, np.ndarray],
    labels: np.ndarray,
    weights: Mapping[str, float],
    *,
    calib_scale: float = 1.0,
    calib_shift: float = 0.0,
) -> dict[str, float]:
    scores = blend_scores(mlp_scores, tree_probas, weights)
    scores = apply_affine_calibration(scores, scale=calib_scale, shift=calib_shift)
    _, metrics = reward(scores, np.asarray(labels, dtype=float))
    return {k: float(v) for k, v in metrics.items()}


def fit_tree_ensemble_on_chunks(
    train_chunks: list,
    train_labels: np.ndarray,
    *,
    seed: int = 42,
    feature_spec: int = DEFAULT_TREE_FEATURE_SPEC,
) -> dict[str, Pipeline]:
    X = chunk_feature_matrix(train_chunks, feature_spec=feature_spec)
    y = np.asarray(train_labels, dtype=float)
    pipelines = build_tree_pipelines(seed=seed)
    for pipe in pipelines.values():
        pipe.fit(X, y)
    return pipelines


def save_tree_ensemble_bundle(
    out_dir: Path,
    weights: Mapping[str, float],
    *,
    train_pipelines: Mapping[str, Pipeline] | None = None,
    pipelines: Mapping[str, Pipeline] | None = None,
    lodo_pipelines: list[dict[str, Pipeline]] | None = None,
    pipeline_mode: str = "full_train",
    blend_select_on: str = "val",
    max_fpr: float = 0.03,
    min_mlp_weight: float = DEFAULT_MIN_MLP_WEIGHT,
    calib_scale: float = 1.0,
    calib_shift: float = 0.0,
    calib_max_fpr: float = 0.03,
    calib_max_scale: float = 2.5,
    calib_max_shift: float = 0.18,
    calib_min_ap_ratio: float = 0.95,
    train_metrics: Mapping[str, float] | None = None,
    val_metrics: Mapping[str, float] | None = None,
    tree_feature_spec: int = DEFAULT_TREE_FEATURE_SPEC,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / TREE_ENSEMBLE_FILENAME
    spec = get_tree_feature_spec(tree_feature_spec)
    payload: dict[str, Any] = {
        "weights": dict(weights),
        "calib_scale": float(calib_scale),
        "calib_shift": float(calib_shift),
        "pipeline_mode": pipeline_mode,
        "tree_feature_spec_version": spec.version,
        "tree_chunk_feature_dim": spec.tree_chunk_feature_dim,
    }
    if train_pipelines is not None:
        payload["train_pipelines"] = dict(train_pipelines)
        payload["pipelines"] = dict(train_pipelines)
    elif pipelines is not None:
        payload["pipelines"] = dict(pipelines)
    if lodo_pipelines is not None:
        payload["lodo_pipelines"] = lodo_pipelines
    if "pipelines" not in payload and lodo_pipelines is not None and len(lodo_pipelines) == 1:
        payload["pipelines"] = dict(lodo_pipelines[0])
    joblib.dump(payload, bundle_path)
    meta = {
        "weights": {k: float(v) for k, v in weights.items()},
        "tree_names": list(TREE_NAMES),
        "chunk_feature_dim": spec.chunk_feature_dim,
        "tree_chunk_feature_dim": spec.tree_chunk_feature_dim,
        "tree_feature_spec_version": spec.version,
        "pipeline_mode": pipeline_mode,
        "lodo_folds": len(lodo_pipelines or []),
        "blend_select_on": blend_select_on,
        "max_fpr": float(max_fpr),
        "min_mlp_weight": float(min_mlp_weight),
        "calib_scale": float(calib_scale),
        "calib_shift": float(calib_shift),
        "calib_max_fpr": float(calib_max_fpr),
        "calib_max_scale": float(calib_max_scale),
        "calib_max_shift": float(calib_max_shift),
        "calib_min_ap_ratio": float(calib_min_ap_ratio),
        "train_metrics": dict(train_metrics or {}),
        "val_metrics": dict(val_metrics or {}),
    }
    (out_dir / TREE_BLEND_META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return bundle_path


def load_tree_ensemble_bundle(path: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    if path is None:
        if out_dir is None:
            raise ValueError("path or out_dir required")
        path = Path(out_dir) / TREE_ENSEMBLE_FILENAME
    bundle = joblib.load(path)
    return bundle


class MlpTreeEnsembleScorer:
    """Score chunks: blend neural MLP score with XGB + RF + DT on chunk features."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        tree_bundle_path: Path | None = None,
        device: str = "cpu",
    ):
        from poker_detect.training.train_loop import chunk_scores_from_model, load_model_checkpoint

        ckpt_path = Path(checkpoint_path)
        out_dir = ckpt_path.parent
        bundle_path = tree_bundle_path or (out_dir / TREE_ENSEMBLE_FILENAME)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"tree ensemble not found at {bundle_path}; run fit after training"
            )

        self.device = device
        self._model = load_model_checkpoint(ckpt_path, device=device)
        bundle = load_tree_ensemble_bundle(bundle_path)
        self._weights: dict[str, float] = bundle["weights"]
        self._blend_calib_scale = float(bundle.get("calib_scale", 1.0))
        self._blend_calib_shift = float(bundle.get("calib_shift", 0.0))
        self._train_pipelines: dict[str, Pipeline] | None = bundle.get("train_pipelines")
        self._pipelines: dict[str, Pipeline] | None = bundle.get("pipelines")
        self._fold_pipelines: list[dict[str, Pipeline]] | None = bundle.get("lodo_pipelines")
        if self._train_pipelines is None and self._pipelines is None and self._fold_pipelines is None:
            raise ValueError(f"tree bundle missing pipelines: {bundle_path}")
        self._tree_feature_spec = int(
            bundle.get(
                "tree_feature_spec_version",
                resolve_tree_feature_spec(
                    tree_chunk_feature_dim=bundle.get("tree_chunk_feature_dim")
                ).version,
            )
        )
        self._tree_feature_dim = int(
            bundle.get(
                "tree_chunk_feature_dim",
                get_tree_feature_spec(self._tree_feature_spec).tree_chunk_feature_dim,
            )
        )
        self._chunk_scores_from_model = chunk_scores_from_model
        self._collate = None

    def _tree_feature_vector(self, hands: list) -> np.ndarray:
        return tree_feature_vector_for_hands(
            hands,
            feature_spec=self._tree_feature_spec,
            tree_chunk_feature_dim=self._tree_feature_dim,
        )

    def _collate_chunks(self, hands: list):
        from poker_detect.training.jsonl_loader import collate_hand_chunks

        return collate_hand_chunks([(hands, 0.0)])

    def score_chunk(self, hands: list) -> float:
        if not hands:
            return 0.5

        import torch

        features, mask, _labels = self._collate_chunks(hands)
        features = features.to(self.device)
        mask = mask.to(self.device)
        with torch.no_grad():
            mlp = self._chunk_scores_from_model(self._model, features, mask)
        mlp_score = float(mlp[0].item())

        feat = self._tree_feature_vector(hands).reshape(1, -1)
        if self._train_pipelines is not None:
            tree_probas = predict_tree_probas(self._train_pipelines, feat)
        elif self._pipelines is not None:
            tree_probas = predict_tree_probas(self._pipelines, feat)
        else:
            tree_probas = predict_tree_probas_ensemble(self._fold_pipelines, feat)
        scores = blend_scores(
            np.array([mlp_score]),
            {k: v.reshape(1) for k, v in tree_probas.items()},
            self._weights,
        )
        return float(
            apply_affine_calibration(
                scores,
                scale=self._blend_calib_scale,
                shift=self._blend_calib_shift,
            )[0]
        )
