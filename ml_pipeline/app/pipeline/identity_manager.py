"""
identity_manager.py

Maps temporary ByteTrack IDs to PERMANENT Person IDs using OSNet ReID
embeddings, so a person keeps the same identity even if ByteTrack's
track_id changes after an occlusion or re-entry.

Drop-in replacement for the previous counter-based version. Public method
name changed from get_person_id(track_id) -> resolve(...) because we now
need the actual person crop to compute an embedding, not just the track_id.
Update the one call site in detection_pipeline.py / person_manager.py
accordingly (see integration notes below the class).
"""

import time
from typing import List, Optional

from pipeline.reid.config import ReIDConfig, prefix_for_class
from pipeline.reid.reid_model import OSNetReIDExtractor
from pipeline.reid.identity_manager import IdentityManager as ReIDCore
from pipeline.reid.embedding_database import PersonRecord


class IdentityManager:

    def __init__(self, weights_path: str = "models/osnet_x1_0_imagenet.pth"):
        self.config = ReIDConfig(weights_path=weights_path)
        self.extractor = OSNetReIDExtractor(self.config)
        self.core = ReIDCore(self.config)

    def resolve(self, track_id, crop_bgr, bbox, confidence=1.0, timestamp=None, obj_class=None):
        """
        Returns the permanent ID (e.g. "Person_001" / "Vehicle_001")
        corresponding to a tracker ID, using ReID embeddings to survive
        track_id switches. The ID prefix matches obj_class.
        Args:
            track_id: ByteTrack's temporary id for this frame's detection.
            crop_bgr: cropped person image (numpy array, BGR, as from cv2).
            bbox: [x1, y1, x2, y2] in the original frame.
            confidence: detection confidence.
            timestamp: seconds (or unix time) for this detection. If your
                caller already computes entry/exit times, pass that same
                clock here so ReID's time-gap gating stays consistent with it.
            obj_class: "person" or "vehicle" -- decides the ID prefix.

        Returns:
            str: stable id matching the class name, e.g. "Person_001"
        """
        embedding = self.extractor.extract(crop_bgr)
        return self.core.resolve(
            tracker_id=track_id,
            embedding=embedding,
            bbox=tuple(bbox),
            confidence=confidence,
            timestamp=timestamp,
            obj_class=obj_class,
        )

    def resolve_batch(self, track_ids, crops_bgr, bboxes, confidences=None, timestamp=None, classes=None):
        """
        Batched version -- use this in your per-frame loop instead of calling
        resolve() once per person. Much faster on the RTX 3050 since all
        crops go through OSNet in a single forward pass.

        Enforces same-frame uniqueness: two detections in the same frame
        will never share the same global_id.

        Args:
            track_ids: list[int]
            crops_bgr: list[np.ndarray]
            bboxes: list[[x1,y1,x2,y2]]
            confidences: list[float] or None
            classes: list[str] or None -- detection class per crop
                ("person"/"vehicle"). The returned ID matches the class
                name (Person_001 / Vehicle_001) and matching never crosses
                classes.

        Returns:
            list[str]: person ids, same order/length as track_ids
        """
        if confidences is None:
            confidences = [1.0] * len(track_ids)
        if classes is None:
            classes = [None] * len(track_ids)

        embeddings = self.extractor.extract_batch(crops_bgr)

        used_ids = set()
        person_ids: List[str] = []
        for tid, emb, bbox, conf, obj_class in zip(track_ids, embeddings, bboxes, confidences, classes):
            gid = self.core.resolve(
                tracker_id=tid,
                embedding=emb,
                bbox=tuple(bbox),
                confidence=conf,
                timestamp=timestamp,
                obj_class=obj_class,
            )
            # If another detection in this same frame already got this
            # global_id, the gallery search produced a false merge.
            # Mint a brand-new identity instead.
            if gid in used_ids:
                new_id = self.core.db.new_id(
                    prefix=prefix_for_class(obj_class),
                    padding=self.core.cfg.id_padding,
                )
                record = PersonRecord(
                    global_id=new_id,
                    embedding=self.core._l2norm(emb),
                    tracker_id=tid,
                    first_seen=timestamp if timestamp is not None else time.time(),
                    last_seen=timestamp if timestamp is not None else time.time(),
                    bbox=tuple(bbox),
                    confidence=conf,
                    obj_class=obj_class or "person",
                )
                self.core.db.add(record)
                gid = new_id
            used_ids.add(gid)
            person_ids.append(gid)

        return person_ids