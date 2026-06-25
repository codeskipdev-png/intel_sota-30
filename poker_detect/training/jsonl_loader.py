"""Load benchmark JSON chunks for conv1d evaluation."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from torch.utils.data import Dataset


def dates_between(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be <= end_date")
    return [(start + timedelta(days=d)).isoformat() for d in range((end - start).days + 1)]


def json_paths_for_multiple_dates(data_dir: Path, start_date: str, end_date: str) -> List[Path]:
    return [data_dir / f"benchmark_{cur_day}.json" for cur_day in dates_between(start_date, end_date)]


def chunk_fingerprint(hands: List[Dict[str, Any]]) -> str:
    return json.dumps(hands, sort_keys=True, ensure_ascii=True)


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
            ground_truth = sub_data["groundTruth"]
            for chunk_idx, chunk in enumerate(batch_chunks):
                key = chunk_fingerprint(chunk)
                if key in blocked or key in seen:
                    continue
                seen.add(key)
                chunks.append(chunk)
                labels.append(float(ground_truth[chunk_idx]))
                chunk_dates.append(source_date)

    return chunks, labels, chunk_dates


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

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[List[Dict[str, Any]], float]:
        return self.chunks[idx], self.labels[idx]


def label_counts_chunk(dataset: ChunkBenchmarkDataset) -> Tuple[int, int]:
    n0 = sum(1 for label in dataset.labels if int(label) == 0)
    n1 = len(dataset.labels) - n0
    return n0, n1
