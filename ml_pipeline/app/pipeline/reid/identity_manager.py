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

from .config import ReIDConfig, DEFAULT_CONFIG, prefix_for_class
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
        obj_class: Optional[str] = None,
    ) -> str:
        """
        Main entry point. Call this once per detection, per frame, after
        ByteTrack + crop + embedding extraction. Returns the stable
        global_id string to write into the JSON output.

        obj_class ("person" / "vehicle") decides the ID prefix so the ID
        name always matches the class name (Person_001 / Vehicle_001) and
        identities are only ever matched within the same class.
        """
        ts = timestamp if timestamp is not None else time.time()
        self._maybe_purge_stale(ts)

        prefix = prefix_for_class(obj_class)

        # 1. Fast path: tracker_id already known to us (same class only).
        existing = self.db.get_by_tracker_id(tracker_id, camera_id=camera_id, obj_class=obj_class)
        if existing is None and obj_class == "vehicle":
            # Check if tracker_id was previously recorded as "person" before vehicle was confirmed
            prev_rec = self.db.get_by_tracker_id(tracker_id, camera_id=camera_id, obj_class=None)
            if prev_rec is not None and prev_rec.obj_class != "vehicle":
                prev_rec.obj_class = "vehicle"
                existing = prev_rec

        # STRICT CLASS BARRIER: reject fast-path match if classes differ.
        # This prevents a drone track from being silently reused as a vehicle
        # or vice-versa even if ByteTrack re-assigned the same tracker_id.
        if existing is not None and obj_class is not None:
            if existing.obj_class != obj_class:
                logger.warning(
                    "STRICT CLASS BARRIER: tracker_id=%s tried to reuse %s (class=%s) "
                    "but current detection is class=%s — minting new identity.",
                    tracker_id, existing.global_id, existing.obj_class, obj_class,
                )
                existing = None

        if existing is not None:
            self.db.smooth_update(
                existing.global_id, embedding, tracker_id, bbox, confidence, ts,
                ema_alpha=self.cfg.ema_alpha,
            )
            logger.debug(
                "Fast path: tracker_id=%s -> %s (hits=%d)",
                tracker_id, existing.global_id, existing.hits + 1,
            )
            return existing.global_id

        # 2. New tracker_id -> search the gallery (same class only).
        # gallery_arrays already filters by obj_class, but we double-check the
        # candidate's class after matching to enforce the hard class barrier.
        gallery_ids, gallery_embs = self.db.gallery_arrays(camera_id=camera_id, obj_class=obj_class)
        # Determine class-specific similarity threshold
        class_thresh_dict = getattr(self.cfg, "class_similarity_thresholds", {})
        thresh = class_thresh_dict.get(obj_class, self.cfg.similarity_threshold) if obj_class else self.cfg.similarity_threshold

        candidate_id, score = best_match(
            embedding, gallery_ids, gallery_embs, thresh
        )
        logger.debug("Gallery search for tracker_id=%s: best_score=%.3f (threshold=%.2f)", tracker_id, score, thresh)

        if candidate_id is not None:
            candidate_rec = self.db.get(candidate_id)
            # STRICT CLASS BARRIER on gallery re-attach: even if cosine similarity
            # cleared the threshold, never merge across different object classes.
            if candidate_rec is not None and obj_class is not None:
                if candidate_rec.obj_class != obj_class:
                    logger.warning(
                        "STRICT CLASS BARRIER (gallery): candidate %s is class=%s "
                        "but current detection is class=%s — refusing re-attach.",
                        candidate_id, candidate_rec.obj_class, obj_class,
                    )
                    candidate_id = None
                    candidate_rec = None
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

        # 3. No acceptable match -> brand new identity (prefix = class name).
        new_id = self.db.new_id(prefix=prefix, padding=self.cfg.id_padding)
        record = PersonRecord(
            global_id=new_id,
            embedding=self._l2norm(embedding),
            tracker_id=tracker_id,
            first_seen=ts,
            last_seen=ts,
            bbox=bbox,
            confidence=confidence,
            camera_id=camera_id,
            obj_class=obj_class or "person",
        )
        self.db.add(record)
        logger.debug("New identity %s created for tracker_id=%s", new_id, tracker_id)
        return new_id

    def resolve_batch(self, detections: list, camera_id: str = "default", timestamp: Optional[float] = None) -> list:
        """
        detections: list of dicts each with keys
            tracker_id, embedding, bbox, confidence (optional), class (optional)
        Returns a list of global_id strings in the same order.

        The optional per-detection "class" key ("person"/"vehicle") makes
        the minted ID match the class name and restricts matching to
        same-class identities.

        Two detections in the SAME frame must never share a global_id.
        If resolve() returns a global_id already assigned earlier in this
        batch, we fall through to minting a new identity instead.
        """
        used_ids = set()
        results = []
        for d in detections:
            gid = self.resolve(
                tracker_id=d["tracker_id"],
                embedding=d["embedding"],
                bbox=d["bbox"],
                confidence=d.get("confidence", 1.0),
                camera_id=camera_id,
                timestamp=timestamp,
                obj_class=d.get("class"),
            )
            # Enforce same-frame uniqueness: if this global_id was already
            # assigned to a different detection in this frame, the gallery
            # search produced a false merge.  Re-resolve with a temporary
            # threshold override (force new identity).
            if gid in used_ids:
                logger.warning(
                    "Same-frame uniqueness: tracker_id=%s resolved to %s "
                    "which was already used; minting new identity",
                    d["tracker_id"], gid,
                )
                new_id = self.db.new_id(
                    prefix=prefix_for_class(d.get("class")), padding=self.cfg.id_padding
                )
                record = PersonRecord(
                    global_id=new_id,
                    embedding=self._l2norm(d["embedding"]),
                    tracker_id=d["tracker_id"],
                    first_seen=timestamp if timestamp is not None else time.time(),
                    last_seen=timestamp if timestamp is not None else time.time(),
                    bbox=d["bbox"],
                    confidence=d.get("confidence", 1.0),
                    camera_id=camera_id,
                    obj_class=d.get("class") or "person",
                )
                self.db.add(record)
                gid = new_id
            used_ids.add(gid)
            results.append(gid)
        return results

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