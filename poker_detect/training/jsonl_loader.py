"""Load chunk records from JSONL (supports sharded exports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from poker_detect.features.extractor import chunk_feature_vector, hand_feature_vector

from datetime import date, timedelta


def dates_between(start_date: str, end_date: str) -> list[str]:
    """Return a list of 'YYYY-MM-DD' strings from start_date to end_date inclusive."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("start_date must be <= end_date")

    delta_days = (end - start).days
    return [(start + timedelta(days=d)).isoformat() for d in range(delta_days + 1)]


def jsonl_paths_for_split(data_dir: Path, split: str) -> List[Path]:
    stem = f"chunks.{split}"
    primary = data_dir / f"{stem}.jsonl"
    parts = sorted(data_dir.glob(f"{stem}.part*.jsonl"))
    paths: List[Path] = []
    if primary.exists():
        paths.append(primary)
    paths.extend(p for p in parts if p not in paths)
    if not paths:
        raise FileNotFoundError(f"No JSONL found for split={split} under {data_dir}")
    return paths

def json_paths_for_multiple_dates(data_dir: Path, start_date: str, end_date: str) -> List[Path]:
    return [data_dir / f"benchmark_{cur_day}.json" for cur_day in dates_between(start_date, end_date)]


def hand_fingerprint(h: Dict[str, Any]) -> str:
    return json.dumps(h, sort_keys=True, ensure_ascii=True)


def read_jsonl_records(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def chunk_fingerprint(hands: List[Dict[str, Any]]) -> str:
    return json.dumps(hands, sort_keys=True, ensure_ascii=True)


def read_json_hands(
    paths: List[Path],
    exclude_keys: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    hands: List[Dict[str, Any]] = []
    labels: List[float] = []
    seen: set[str] = set()
    blocked = exclude_keys or set()

    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sub_data in data["data"]["chunks"]:
            chunks = sub_data["chunks"]
            groundTruth = sub_data["groundTruth"]
            for chunk_idx, chunk in enumerate(chunks):
                label = float(groundTruth[chunk_idx])
                for hand in chunk:
                    key = hand_fingerprint(hand)
                    if key in blocked:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    hands.append(hand)
                    labels.append(label)

    return hands, labels


def read_json_chunks(
    paths: List[Path],
    exclude_keys: Optional[Set[str]] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[float]]:
    chunks, labels, _dates = read_json_chunks_with_dates(paths, exclude_keys=exclude_keys)
    return chunks, labels


def _date_from_benchmark_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("benchmark_"):
        return stem[len("benchmark_") :]
    return stem


def read_json_chunks_with_dates(
    paths: List[Path],
    exclude_keys: Optional[Set[str]] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[float], List[str]]:
    chunks: List[List[Dict[str, Any]]] = []
    labels: List[float] = []
    chunk_dates: List[str] = []
    seen: set[str] = set()
    blocked = exclude_keys or set()

    for path in paths:
        if not path.exists():
            continue
        source_date = _date_from_benchmark_path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sub_data in data["data"]["chunks"]:
            batch_chunks = sub_data["chunks"]
            groundTruth = sub_data["groundTruth"]
            for chunk_idx, chunk in enumerate(batch_chunks):
                key = chunk_fingerprint(chunk)
                if key in blocked or key in seen:
                    continue
                seen.add(key)
                chunks.append(chunk)
                labels.append(float(groundTruth[chunk_idx]))
                chunk_dates.append(source_date)

    return chunks, labels, chunk_dates


def collate_hand_chunks(
    batch: List[Tuple[List[Dict[str, Any]], float]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length hand lists to [batch, max_hands, feat_dim] + mask."""
    labels = torch.tensor([float(item[1]) for item in batch], dtype=torch.float32)
    max_hands = max(max(1, len(item[0])) for item in batch)
    feat_dim = hand_feature_vector(batch[0][0][0] if batch[0][0] else {}).shape[0]

    features = torch.zeros(len(batch), max_hands, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_hands, dtype=torch.float32)
    for batch_idx, (hands, _) in enumerate(batch):
        for hand_idx, hand in enumerate(hands):
            features[batch_idx, hand_idx] = torch.from_numpy(hand_feature_vector(hand))
            mask[batch_idx, hand_idx] = 1.0
        if not hands:
            mask[batch_idx, 0] = 1.0

    return features, mask, labels


class ChunkBenchmarkDataset(Dataset):
    """Benchmark JSON chunks with one label per chunk (matches subnet scoring unit)."""

    def __init__(
        self,
        data_dir: Path,
        start_date: str,
        end_date: str,
        *,
        exclude_chunk_keys: Optional[Set[str]] = None,
    ):
        paths = json_paths_for_multiple_dates(data_dir, start_date, end_date)
        self.chunks, self.labels, self.chunk_dates = read_json_chunks_with_dates(
            paths, exclude_keys=exclude_chunk_keys
        )
        print(f"chunks={len(self.chunks)}")

    def fingerprint_set(self) -> Set[str]:
        return {chunk_fingerprint(c) for c in self.chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[List[Dict[str, Any]], float]:
        return self.chunks[idx], self.labels[idx]


class ChunkJsonlDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        paths = jsonl_paths_for_split(data_dir, split)
        self.records = read_jsonl_records(paths)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        hands = rec.get("hands") or []
        x = chunk_feature_vector(hands)
        y = float(rec.get("label", 0))
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class HandJsonlDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        start_date: str,
        end_date: str,
        *,
        exclude_hand_keys: Optional[Set[str]] = None,
    ):
        paths = json_paths_for_multiple_dates(data_dir, start_date, end_date)
        self.hands, self.labels = read_json_hands(paths, exclude_keys=exclude_hand_keys)
        print(f"hands={len(self.hands)}")

    def fingerprint_set(self) -> Set[str]:
        return {hand_fingerprint(h) for h in self.hands}

    def __len__(self) -> int:
        return len(self.hands)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hand = self.hands[idx]
        x = hand_feature_vector(hand)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def label_counts(dataset: ChunkJsonlDataset) -> Tuple[int, int]:
    n0 = sum(1 for r in dataset.records if int(r.get("label", 0)) == 0)
    n1 = len(dataset.records) - n0
    return n0, n1

def label_counts_hand(dataset: HandJsonlDataset) -> Tuple[int, int]:
    n0 = sum(1 for label in dataset.labels if int(label) == 0)
    n1 = len(dataset.labels) - n0
    return n0, n1


def label_counts_chunk(dataset: ChunkBenchmarkDataset) -> Tuple[int, int]:
    n0 = sum(1 for label in dataset.labels if int(label) == 0)
    n1 = len(dataset.labels) - n0
    return n0, n1
