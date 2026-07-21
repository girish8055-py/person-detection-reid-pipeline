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

from pipeline.reid.config import ReIDConfig
from pipeline.reid.reid_model import OSNetReIDExtractor
from pipeline.reid.identity_manager import IdentityManager as ReIDCore


class IdentityManager:

    def __init__(self, weights_path: str = "models/osnet_x1_0_imagenet.pth"):
        self.config = ReIDConfig(weights_path=weights_path)
        self.extractor = OSNetReIDExtractor(self.config)
        self.core = ReIDCore(self.config)

    def resolve(self, track_id, crop_bgr, bbox, confidence=1.0, timestamp=None):
        """
        Returns the permanent person ID (e.g. "Person_001") corresponding
        to a tracker ID, using ReID embeddings to survive track_id switches.

        Args:
            track_id: ByteTrack's temporary id for this frame's detection.
            crop_bgr: cropped person image (numpy array, BGR, as from cv2).
            bbox: [x1, y1, x2, y2] in the original frame.
            confidence: detection confidence.
            timestamp: seconds (or unix time) for this detection. If your
                caller already computes entry/exit times, pass that same
                clock here so ReID's time-gap gating stays consistent with it.

        Returns:
            str: stable person id, e.g. "Person_001"
        """
        embedding = self.extractor.extract(crop_bgr)
        return self.core.resolve(
            tracker_id=track_id,
            embedding=embedding,
            bbox=tuple(bbox),
            confidence=confidence,
            timestamp=timestamp,
        )

    def resolve_batch(self, track_ids, crops_bgr, bboxes, confidences=None, timestamp=None):
        """
        Batched version -- use this in your per-frame loop instead of calling
        resolve() once per person. Much faster on the RTX 3050 since all
        crops go through OSNet in a single forward pass.

        Args:
            track_ids: list[int]
            crops_bgr: list[np.ndarray]
            bboxes: list[[x1,y1,x2,y2]]
            confidences: list[float] or None

        Returns:
            list[str]: person ids, same order/length as track_ids
        """
        if confidences is None:
            confidences = [1.0] * len(track_ids)

        embeddings = self.extractor.extract_batch(crops_bgr)

        person_ids = []
        for tid, emb, bbox, conf in zip(track_ids, embeddings, bboxes, confidences):
            pid = self.core.resolve(
                tracker_id=tid,
                embedding=emb,
                bbox=tuple(bbox),
                confidence=conf,
                timestamp=timestamp,
            )
            person_ids.append(pid)

        return person_ids