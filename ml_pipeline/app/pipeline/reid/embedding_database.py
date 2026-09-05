"""
reid/embedding_database.py

In-memory store of all known identities for the current video / stream /
camera. Designed so a future swap to Redis/Postgres/pgvector only requires
re-implementing this class's interface -- nothing else in the pipeline
should need to change.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PersonRecord:
    global_id: str
    embedding: np.ndarray            # smoothed, L2-normalized (D,)
    tracker_id: Optional[int]        # most recent ByteTrack id mapped to this person
    first_seen: float                # unix timestamp (or frame index, caller's choice)
    last_seen: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (latest)
    confidence: float
    camera_id: str = "default"
    obj_class: str = "person"        # "person" / "vehicle" -- keeps ID prefix consistent with class
    hits: int = 1                    # how many times this identity has been matched
    embedding_history_len: int = field(default=1, repr=False)
    feature_bank: List[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.feature_bank and self.embedding is not None:
            norm_emb = np.asarray(self.embedding, dtype=np.float32)
            n = np.linalg.norm(norm_emb)
            if n > 1e-8:
                norm_emb = norm_emb / n
            self.feature_bank = [norm_emb]

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class EmbeddingDatabase:
    """
    Thread-safe in-memory identity store.
    Multi-camera ready: pass `camera_id` to keep galleries separate, or leave
    the default to share one global gallery across cameras.
    """

    def __init__(self):
        self._records: Dict[str, PersonRecord] = {}
        # One independent counter per ID prefix, so Person_ and Vehicle_
        # sequences each start at 001 (Person_001..., Vehicle_001...).
        self._next_numeric_id: Dict[str, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # ID generation
    # ------------------------------------------------------------------ #
    def new_id(self, prefix: str = "Person_", padding: int = 3) -> str:
        with self._lock:
            next_num = self._next_numeric_id.get(prefix, 1)
            gid = f"{prefix}{str(next_num).zfill(padding)}"
            self._next_numeric_id[prefix] = next_num + 1
            return gid

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def add(self, record: PersonRecord) -> None:
        with self._lock:
            self._records[record.global_id] = record

    def get(self, global_id: str) -> Optional[PersonRecord]:
        with self._lock:
            return self._records.get(global_id)

    def remove(self, global_id: str) -> None:
        with self._lock:
            self._records.pop(global_id, None)

    def get_by_tracker_id(self, tracker_id: int, camera_id: str = "default", obj_class: Optional[str] = None) -> Optional[PersonRecord]:
        with self._lock:
            for rec in self._records.values():
                if rec.tracker_id == tracker_id and rec.camera_id == camera_id:
                    if obj_class is not None and rec.obj_class != obj_class:
                        continue
                    return rec
        return None

    def all_records(self, camera_id: Optional[str] = None) -> List[PersonRecord]:
        with self._lock:
            if camera_id is None:
                return list(self._records.values())
            return [r for r in self._records.values() if r.camera_id == camera_id]

    def gallery_arrays(self, camera_id: Optional[str] = None, obj_class: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
        """Returns (global_ids, embeddings[N, D]) for multi-feature similarity search."""
        with self._lock:
            records = self.all_records(camera_id)
            if obj_class is not None:
                records = [r for r in records if r.obj_class == obj_class]
            
            ids = []
            embs_list = []
            for r in records:
                bank = r.feature_bank if r.feature_bank else [r.embedding]
                for emb in bank:
                    ids.append(r.global_id)
                    embs_list.append(emb)
            
            if not ids:
                return [], np.zeros((0, 0), dtype=np.float32)
            embs = np.stack(embs_list).astype(np.float32)
            return ids, embs

    # ------------------------------------------------------------------ #
    # Updates
    # ------------------------------------------------------------------ #
    def smooth_update(
        self,
        global_id: str,
        new_embedding: np.ndarray,
        tracker_id: Optional[int],
        bbox: Tuple[float, float, float, float],
        confidence: float,
        timestamp: Optional[float] = None,
        ema_alpha: float = 0.2,
    ) -> PersonRecord:
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            rec = self._records.get(global_id)
            if rec is None:
                raise KeyError(f"Unknown global_id: {global_id}")

            new_emb = np.asarray(new_embedding, dtype=np.float32)
            norm = np.linalg.norm(new_emb)
            if norm > 1e-8:
                new_emb = new_emb / norm

            old = rec.embedding
            blended = (1.0 - ema_alpha) * old + ema_alpha * new_emb
            norm_b = np.linalg.norm(blended)
            if norm_b > 1e-8:
                blended = blended / norm_b

            rec.embedding = blended.astype(np.float32)

            # Store diverse feature keyframes in the feature bank (max 10)
            if not rec.feature_bank:
                rec.feature_bank = [blended.astype(np.float32)]
            else:
                sims = [float(np.dot(new_emb, f)) for f in rec.feature_bank]
                max_sim = max(sims) if sims else 0.0
                if max_sim < 0.88 and len(rec.feature_bank) < 10:
                    rec.feature_bank.append(new_emb.astype(np.float32))

            rec.tracker_id = tracker_id
            rec.bbox = bbox
            rec.confidence = confidence
            rec.last_seen = ts
            rec.hits += 1
            rec.embedding_history_len += 1
            return rec

    def purge_stale(self, ttl_seconds: float, now: Optional[float] = None) -> int:
        """Remove identities not seen for longer than ttl_seconds. Returns count removed."""
        now = now if now is not None else time.time()
        removed = 0
        with self._lock:
            stale = [gid for gid, r in self._records.items() if (now - r.last_seen) > ttl_seconds]
            for gid in stale:
                del self._records[gid]
                removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._records)
