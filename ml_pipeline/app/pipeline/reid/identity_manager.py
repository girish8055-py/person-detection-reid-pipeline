"""
reid/identity_manager.py

The brain of the ReID subsystem.

Given (tracker_id, embedding, bbox, confidence, timestamp) for a detection,
returns a STABLE global person id (e.g. "Person_001") that survives
ByteTrack id switches, occlusion, and re-entry into frame.

Resolution order for each incoming detection:
  1. Fast path: if this exact tracker_id is already mapped to a known
     identity, just update that identity (cheapest, most common case).
  2. Otherwise, this tracker_id is new (either a genuinely new person, or
     ByteTrack re-assigned an id for someone already known). Search the
     gallery by cosine similarity.
       - If the best match clears `similarity_threshold` (and, if enabled,
         passes spatio-temporal gating), re-attach to that identity.
       - Otherwise, mint a brand new global id.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

from .config import ReIDConfig, DEFAULT_CONFIG
from .embedding_database import EmbeddingDatabase, PersonRecord
from .similarity import best_match, cosine_similarity

logger = logging.getLogger(__name__)


class IdentityManager:
    def __init__(self, config: Optional[ReIDConfig] = None, db: Optional[EmbeddingDatabase] = None):
        self.cfg = config or DEFAULT_CONFIG
        self.db = db or EmbeddingDatabase()
        self._last_purge = time.time()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def resolve(
        self,
        tracker_id: int,
        embedding: np.ndarray,
        bbox: Tuple[float, float, float, float],
        confidence: float = 1.0,
        camera_id: str = "default",
        timestamp: Optional[float] = None,
    ) -> str:
        """
        Main entry point. Call this once per detection, per frame, after
        ByteTrack + crop + embedding extraction. Returns the stable
        global_id string to write into the JSON output.
        """
        ts = timestamp if timestamp is not None else time.time()
        self._maybe_purge_stale(ts)

        # 1. Fast path: tracker_id already known to us.
        existing = self.db.get_by_tracker_id(tracker_id, camera_id=camera_id)
        if existing is not None:
            self.db.smooth_update(
                existing.global_id, embedding, tracker_id, bbox, confidence, ts,
                ema_alpha=self.cfg.ema_alpha,
            )
            return existing.global_id

        # 2. New tracker_id -> search the gallery.
        gallery_ids, gallery_embs = self.db.gallery_arrays(camera_id=camera_id)
        candidate_id, score = best_match(
            embedding, gallery_ids, gallery_embs, self.cfg.similarity_threshold
        )
        logger.debug("Gallery search for tracker_id=%s: best_score=%.3f (threshold=%.2f)", tracker_id, score, self.cfg.similarity_threshold)

        if candidate_id is not None:
            candidate_rec = self.db.get(candidate_id)
            if self._passes_spatiotemporal_gate(candidate_rec, bbox, ts):
                self.db.smooth_update(
                    candidate_id, embedding, tracker_id, bbox, confidence, ts,
                    ema_alpha=self.cfg.ema_alpha,
                )
                logger.debug(
                    "Re-attached tracker_id=%s to existing %s (sim=%.3f)",
                    tracker_id, candidate_id, score,
                )
                return candidate_id
            else:
                logger.debug(
                    "Best match %s (sim=%.3f) rejected by spatio-temporal gate; "
                    "minting new identity for tracker_id=%s",
                    candidate_id, score, tracker_id,
                )

        # 3. No acceptable match -> brand new identity.
        new_id = self.db.new_id(prefix=self.cfg.id_prefix, padding=self.cfg.id_padding)
        record = PersonRecord(
            global_id=new_id,
            embedding=self._l2norm(embedding),
            tracker_id=tracker_id,
            first_seen=ts,
            last_seen=ts,
            bbox=bbox,
            confidence=confidence,
            camera_id=camera_id,
        )
        self.db.add(record)
        logger.debug("New identity %s created for tracker_id=%s", new_id, tracker_id)
        return new_id

    def resolve_batch(self, detections: list, camera_id: str = "default", timestamp: Optional[float] = None) -> list:
        """
        detections: list of dicts each with keys
            tracker_id, embedding, bbox, confidence (optional)
        Returns a list of global_id strings in the same order.
        """
        return [
            self.resolve(
                tracker_id=d["tracker_id"],
                embedding=d["embedding"],
                bbox=d["bbox"],
                confidence=d.get("confidence", 1.0),
                camera_id=camera_id,
                timestamp=timestamp,
            )
            for d in detections
        ]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = np.linalg.norm(x)
        return x / n if n > 1e-8 else x

    def _passes_spatiotemporal_gate(
        self, candidate: PersonRecord, bbox: Tuple[float, float, float, float], ts: float
    ) -> bool:
        if not self.cfg.use_spatiotemporal_gating or candidate is None:
            return True

        # Time gap check
        if (ts - candidate.last_seen) > self.cfg.max_time_gap_seconds:
            return False

        # Spatial jump check (rough sanity check, not a hard physics model --
        # a person can't teleport across the frame between re-detections)
        cx1, cy1 = candidate.center()
        x1, y1, x2, y2 = bbox
        cx2, cy2 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        if dist > self.cfg.max_center_jump_px:
            return False

        # Scale sanity check
        cand_area = max(candidate.area(), 1.0)
        new_area = max((x2 - x1) * (y2 - y1), 1.0)
        ratio = max(cand_area, new_area) / min(cand_area, new_area)
        if ratio > self.cfg.max_bbox_area_ratio:
            return False

        return True

    def _maybe_purge_stale(self, now: float, min_interval: float = 30.0) -> None:
        if self.cfg.identity_ttl_seconds is None:
            return
        if (now - self._last_purge) < min_interval:
            return
        removed = self.db.purge_stale(self.cfg.identity_ttl_seconds, now=now)
        self._last_purge = now
        if removed:
            logger.info("Purged %d stale identities (ttl=%ss).", removed, self.cfg.identity_ttl_seconds)