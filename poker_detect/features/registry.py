"""Tree/classical feature spec registry (v2 full vs v3 slim)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from poker_detect.features.extractor import (
    CHUNK_FEATURE_DIM as V2_CHUNK_FEATURE_DIM,
    TREE_CHUNK_FEATURE_DIM as V2_TREE_CHUNK_FEATURE_DIM,
    chunk_feature_vector as chunk_feature_vector_v2,
    tree_chunk_feature_vector as tree_chunk_feature_vector_v2,
)
from poker_detect.features import slim_v3

TreeVectorFn = Callable[[List[Dict[str, Any]]], np.ndarray]

DEFAULT_TREE_FEATURE_SPEC = 3


@dataclass(frozen=True)
class TreeFeatureSpec:
    version: int
    chunk_feature_dim: int
    tree_chunk_feature_dim: int
    chunk_feature_vector: TreeVectorFn
    tree_chunk_feature_vector: TreeVectorFn


_SPECS: dict[int, TreeFeatureSpec] = {
    2: TreeFeatureSpec(
        version=2,
        chunk_feature_dim=V2_CHUNK_FEATURE_DIM,
        tree_chunk_feature_dim=V2_TREE_CHUNK_FEATURE_DIM,
        chunk_feature_vector=chunk_feature_vector_v2,
        tree_chunk_feature_vector=tree_chunk_feature_vector_v2,
    ),
    3: TreeFeatureSpec(
        version=3,
        chunk_feature_dim=slim_v3.CHUNK_FEATURE_DIM,
        tree_chunk_feature_dim=slim_v3.TREE_CHUNK_FEATURE_DIM,
        chunk_feature_vector=slim_v3.chunk_feature_vector,
        tree_chunk_feature_vector=slim_v3.tree_chunk_feature_vector,
    ),
}


def get_tree_feature_spec(version: int = DEFAULT_TREE_FEATURE_SPEC) -> TreeFeatureSpec:
    try:
        return _SPECS[int(version)]
    except KeyError as exc:
        supported = ", ".join(str(v) for v in sorted(_SPECS))
        raise ValueError(f"unsupported tree feature spec {version!r}; use one of: {supported}") from exc


def resolve_tree_feature_spec(
    *,
    version: int | None = None,
    tree_chunk_feature_dim: int | None = None,
) -> TreeFeatureSpec:
    """Resolve spec from explicit version or stored bundle dimension."""
    if version is not None:
        return get_tree_feature_spec(version)
    if tree_chunk_feature_dim is not None:
        dim = int(tree_chunk_feature_dim)
        for spec in _SPECS.values():
            if spec.tree_chunk_feature_dim == dim:
                return spec
        raise ValueError(f"unknown tree_chunk_feature_dim={dim}")
    return get_tree_feature_spec(DEFAULT_TREE_FEATURE_SPEC)


def chunk_feature_matrix(chunks: list, *, feature_spec: int = DEFAULT_TREE_FEATURE_SPEC) -> np.ndarray:
    spec = get_tree_feature_spec(feature_spec)
    return np.stack([spec.tree_chunk_feature_vector(c) for c in chunks], axis=0)


def tree_feature_vector_for_hands(
    hands: List[Dict[str, Any]],
    *,
    feature_spec: int | None = None,
    tree_chunk_feature_dim: int | None = None,
) -> np.ndarray:
    spec = resolve_tree_feature_spec(
        version=feature_spec,
        tree_chunk_feature_dim=tree_chunk_feature_dim,
    )
    return spec.tree_chunk_feature_vector(hands)
