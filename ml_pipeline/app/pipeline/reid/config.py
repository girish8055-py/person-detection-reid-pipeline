"""
reid/config.py

Central configuration for the ReID subsystem.
All tunables live here so nothing is hardcoded deeper in the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class ReIDConfig:
    # ---------------- Model ----------------
    model_name: str = "osnet_x1_0"
    # Path to a local .pth checkpoint. If None, the extractor will try to
    # download the official pretrained weights on first use.
    weights_path: str = "weights/osnet_x1_0_imagenet.pth"
    embedding_dim: int = 512
    input_size: Tuple[int, int] = (256, 128)  # (height, width), OSNet default

    # ---------------- Device ----------------
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True  # safe on RTX 3050, halves VRAM + speeds up inference

    # ---------------- Matching ----------------
# ---------------- Matching ----------------
    similarity_threshold: float = 0.4
    # Soft margin allows a small buffer below the threshold for gating.
    soft_margin: float = 0.05

    # ---------------- Embedding smoothing ----------------
    # new_embedding = (1 - ema_alpha) * old_embedding + ema_alpha * current_embedding
    ema_alpha: float = 0.4

    # ---------------- Spatio-temporal gating (optional, off by default) ----------------
    use_spatiotemporal_gating: bool = False
    max_time_gap_seconds: float = 8.0       # ignore candidates not seen recently
    max_center_jump_px: float = 500.0       # relaxed spatial continuity constraint
    max_bbox_area_ratio: float = 2.5        # ignore candidates with wildly different scale

    # ---------------- Identity bookkeeping ----------------
    id_prefix: str = "Person_"
    id_padding: int = 3  # produces Person_001, Person_002, ...

    # ---------------- Housekeeping ----------------
    # Drop identities not seen for this long (seconds) to bound memory growth
    # on long videos / live RTSP streams. Set to None to disable.
    identity_ttl_seconds: float = None


DEFAULT_CONFIG = ReIDConfig()


def prefix_for_class(obj_class: Optional[str]) -> str:
    """ID prefix matching the detection class: person -> 'Person_', vehicle -> 'Vehicle_'."""
    if not obj_class:
        return DEFAULT_CONFIG.id_prefix
    return f"{str(obj_class).strip().capitalize()}_"
