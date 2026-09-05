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
    model_name: str = "osnet_ain_x1_0"
    # Path to a local .pth checkpoint.
    weights_path: str = "models/osnet_ain_x1_0_imagenet.pth"
    embedding_dim: int = 512
    input_size: Tuple[int, int] = (256, 128)  # (height, width), OSNet default

    # ---------------- Device ----------------
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True  # safe on RTX 3050, halves VRAM + speeds up inference

    # ---------------- Matching ----------------
    # 0.28 with multi-embedding feature bank matching allows robust re-attachment
    # across angle/pose changes while preventing false merges.
    similarity_threshold: float = 0.28
    # Soft margin allows a small buffer below the threshold for gating.
    soft_margin: float = 0.05

    # ---------------- Embedding smoothing ----------------
    # new_embedding = (1 - ema_alpha) * old_embedding + ema_alpha * current_embedding
    ema_alpha: float = 0.35

    # ---------------- Spatio-temporal gating ----------------
    use_spatiotemporal_gating: bool = False
    max_time_gap_seconds: float = 30.0      # allow re-identification up to 30 seconds after disappearing
    max_center_jump_px: float = 1200.0      # flexible spatial distance limit for moving targets
    max_bbox_area_ratio: float = 5.0        # flexible scale change ratio for perspective changes

    # ---------------- Identity bookkeeping ----------------
    id_prefix: str = "Person_"
    id_padding: int = 3  # produces Person_001, Person_002, ...

    # ---------------- Housekeeping ----------------
    # Drop identities not seen for this long (seconds) to bound memory growth
    # on long videos / live RTSP streams. Set to None to disable.
    identity_ttl_seconds: float = None


DEFAULT_CONFIG = ReIDConfig()


def prefix_for_class(obj_class: Optional[str]) -> str:
    """Dynamic ID prefix matching ANY detection class (Person_, Vehicle_, Drone_, Boat_, etc.)."""
    if not obj_class:
        return DEFAULT_CONFIG.id_prefix
    clean_cls = str(obj_class).strip().replace(" ", "_").capitalize()
    return f"{clean_cls}_"
