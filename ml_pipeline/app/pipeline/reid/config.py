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
    model_name: str = "dinov2_vitb14"
    # Path to a local .safetensors or .pth checkpoint.
    weights_path: str = "models/dinov2_vitb14.safetensors"
    embedding_dim: int = 768
    input_size: Tuple[int, int] = (224, 224)  # (height, width), DINOv2 standard resolution

    # ---------------- Device ----------------
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True  # safe on RTX 3050, halves VRAM + speeds up inference

    # ---------------- Matching ----------------
    # 0.50 threshold with DINOv2 768-D feature bank prevents false merges between different targets.
    similarity_threshold: float = 0.50
    # Soft margin allows a small buffer below the threshold for gating.
    soft_margin: float = 0.05

    # ---------------- Embedding smoothing ----------------
    # new_embedding = (1 - ema_alpha) * old_embedding + ema_alpha * current_embedding
    ema_alpha: float = 0.35

    # ---------------- Spatio-temporal gating ----------------
    use_spatiotemporal_gating: bool = True
    max_time_gap_seconds: float = 60.0      # allow re-identification up to 1 minute after disappearing
    max_center_jump_px: float = 1000.0      # realistic spatial distance limit for moving targets
    max_bbox_area_ratio: float = 4.0        # scale change ratio limit

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
