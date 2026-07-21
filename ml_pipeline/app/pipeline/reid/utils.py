"""
reid/utils.py

Small shared helpers used by pipeline.py / detector.py when wiring the ReID
subsystem in.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def crop_person(frame: np.ndarray, bbox: Tuple[float, float, float, float], pad_ratio: float = 0.0) -> np.ndarray:
    """
    Crop a person from a full frame given an (x1, y1, x2, y2) bbox.
    Clamps to frame bounds and optionally pads for a bit of context margin.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    if pad_ratio > 0:
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * pad_ratio
        x2 += bw * pad_ratio
        y1 -= bh * pad_ratio
        y2 += bh * pad_ratio

    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(x1 + 1, min(x2, w)))
    y2 = int(max(y1 + 1, min(y2, h)))

    return frame[y1:y2, x1:x2]


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return x, y, x + w, y + h


def frame_index_to_seconds(frame_idx: int, fps: float) -> float:
    if fps <= 0:
        return float(frame_idx)
    return frame_idx / fps
