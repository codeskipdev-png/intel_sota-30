"""Action-only sequence encoding for 1D-CNN chunk classifiers.

A **chunk** is one flat timeline: all sanitized actions from all hands concatenated
in hand order (typical length ≈ 40 hands × 12 actions = 480 steps).

Only ``action_type`` is encoded (integer index → ``nn.Embedding`` at model time).
No amounts, pot sizes, streets, or seat features.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from poker44.validator.sanitization import sanitize_hand_for_miner

ACTION_TYPES: tuple[str, ...] = (
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "small_blind",
    "big_blind",
    "ante",
    "all_in",
    "other",
)
ACTION_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ACTION_TYPES)}

N_ACTION_TYPES = len(ACTION_TYPES)
PAD_IDX = N_ACTION_TYPES          # padding token
VOCAB_SIZE = N_ACTION_TYPES + 1   # action indices 0..9 + pad

SEQ_LEN = 12                      # actions per sanitized hand
HANDS_PER_CHUNK = 40
MAX_CHUNK_SEQ_LEN = HANDS_PER_CHUNK * SEQ_LEN  # 480


def action_type_index(action: Dict[str, Any]) -> int:
    """Map one action dict to an embedding index in ``[0, N_ACTION_TYPES)``."""
    atype = str(action.get("action_type") or "other").strip().lower()
    return int(ACTION_TYPE_TO_IDX.get(atype, ACTION_TYPE_TO_IDX["other"]))


def chunk_action_index_sequence(
    hands: List[Dict[str, Any]],
    *,
    sanitize: bool = True,
    max_seq_len: int = MAX_CHUNK_SEQ_LEN,
) -> np.ndarray:
    """Flatten all actions in a chunk → ``(max_seq_len,)`` int64 index vector.

    Unused tail positions are filled with ``PAD_IDX``.
    """
    indices: list[int] = []
    for hand in hands or []:
        h = sanitize_hand_for_miner(hand) if sanitize else hand
        for action in h.get("actions") or []:
            if isinstance(action, dict):
                indices.append(action_type_index(action))

    out = np.full(max_seq_len, PAD_IDX, dtype=np.int64)
    if indices:
        n = min(max_seq_len, len(indices))
        out[:n] = np.asarray(indices[:n], dtype=np.int64)
    return out
