"""1D-CNN chunk classifier over flat action-index sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from poker_detect.features.action_sequence import (
    MAX_CHUNK_SEQ_LEN,
    PAD_IDX,
    VOCAB_SIZE,
    chunk_action_index_sequence,
)

CONV1D_MODEL_FILENAME = "conv1d_model.pt"
CONV1D_META_FILENAME = "conv1d_train_meta.json"


class Conv1DActionMIL(nn.Module):
    """
    Flat chunk action sequence → embedding → 2× Conv1d → global max pool → MLP → sigmoid.

    Architecture (per user spec):
      - Embedding for action types only
      - Conv1d(channels=32, kernel=3, stride=1) + MaxPool1d(2)
      - Conv1d(channels=64, kernel=3, stride=1) + MaxPool1d(2)
      - Global max pool over time
      - Dense hidden + output with sigmoid
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        pad_idx: int = PAD_IDX,
        embed_dim: int = 16,
        max_seq_len: int = MAX_CHUNK_SEQ_LEN,
        conv_channels: tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        pool_size: int = 2,
        dense_hidden: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.conv_channels = conv_channels
        self.dense_hidden = dense_hidden

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        c1, c2 = conv_channels
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(embed_dim, c1, kernel_size=kernel_size, stride=1, padding=pad)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, stride=1, padding=pad)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c2, dense_hidden)
        self.fc_out = nn.Linear(dense_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[self.pad_idx].zero_()
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in (self.fc1, self.fc_out):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_logit(self, action_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        action_ids : (T,) or (B, T) long tensor of action indices (+ PAD_IDX).
        """
        if action_ids.dim() == 1:
            action_ids = action_ids.unsqueeze(0)
        x = self.embed(action_ids)              # (B, T, E)
        x = x.transpose(1, 2)                   # (B, E, T)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.max(dim=2).values                 # (B, C2) global max pool
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logit = self.fc_out(x).squeeze(-1)      # (B,)
        return logit.squeeze(0) if logit.shape[0] == 1 else logit

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability (scalar for single chunk)."""
        logit = self.forward_logit(action_ids)
        prob = torch.sigmoid(logit)
        return prob.squeeze() if prob.dim() > 0 and prob.numel() == 1 else prob


def save_conv1d_bundle(
    out_dir: Path,
    model: Conv1DActionMIL,
    *,
    calib_scale: float = 1.0,
    calib_shift: float = 0.0,
    val_metrics: dict | None = None,
    train_metrics: dict | None = None,
    extra: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / CONV1D_MODEL_FILENAME
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": model.vocab_size,
            "pad_idx": model.pad_idx,
            "embed_dim": model.embed_dim,
            "max_seq_len": model.max_seq_len,
            "kernel_size": model.kernel_size,
            "pool_size": model.pool_size,
            "conv_channels": list(model.conv_channels),
            "dense_hidden": model.dense_hidden,
            "calib_scale": float(calib_scale),
            "calib_shift": float(calib_shift),
            "model_type": "conv1d",
        },
        bundle_path,
    )
    meta: dict[str, Any] = {
        "model_type": "conv1d",
        "bundle_path": str(bundle_path.resolve()),
        "vocab_size": model.vocab_size,
        "embed_dim": model.embed_dim,
        "max_seq_len": model.max_seq_len,
        "kernel_size": model.kernel_size,
        "pool_size": model.pool_size,
        "conv_channels": list(model.conv_channels),
        "dense_hidden": model.dense_hidden,
        "calib_scale": float(calib_scale),
        "calib_shift": float(calib_shift),
        "train_metrics": dict(train_metrics or {}),
        "val_metrics": dict(val_metrics or {}),
    }
    if extra:
        meta.update(extra)
    (out_dir / CONV1D_META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return bundle_path


def load_conv1d_bundle(path: Path | str) -> dict[str, Any]:
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    if bundle.get("model_type") != "conv1d":
        raise ValueError(f"not a conv1d bundle: {path}")
    channels = bundle.get("conv_channels", [32, 64])
    model = Conv1DActionMIL(
        vocab_size=int(bundle.get("vocab_size", VOCAB_SIZE)),
        pad_idx=int(bundle.get("pad_idx", PAD_IDX)),
        embed_dim=int(bundle.get("embed_dim", 16)),
        max_seq_len=int(bundle.get("max_seq_len", MAX_CHUNK_SEQ_LEN)),
        conv_channels=(int(channels[0]), int(channels[1])),
        kernel_size=int(bundle.get("kernel_size", 3)),
        pool_size=int(bundle.get("pool_size", 2)),
        dense_hidden=int(bundle.get("dense_hidden", 64)),
    )
    model.load_state_dict(bundle["model_state"])
    model.eval()
    bundle["model"] = model
    return bundle


def score_chunk_with_conv1d(
    model: Conv1DActionMIL,
    hands: List[dict[str, Any]],
    *,
    sanitize: bool = True,
    calib_scale: float = 1.0,
    calib_shift: float = 0.0,
    device: str | torch.device = "cpu",
) -> float:
    if not hands:
        return 0.5
    ids = chunk_action_index_sequence(hands, sanitize=sanitize)
    t = torch.from_numpy(ids).long().to(device)
    model.eval()
    with torch.no_grad():
        prob = float(model(t).item())
    return float(np.clip(prob * calib_scale + calib_shift, 0.0, 1.0))


def score_chunk_from_conv1d_bundle(
    bundle: dict[str, Any],
    hands: List[dict[str, Any]],
    *,
    device: str | torch.device = "cpu",
) -> float:
    return score_chunk_with_conv1d(
        bundle["model"],
        hands,
        calib_scale=float(bundle.get("calib_scale", 1.0)),
        calib_shift=float(bundle.get("calib_shift", 0.0)),
        device=device,
    )
