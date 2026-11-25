"""
ml_pipeline/app/pipeline/reid/reid_model.py

Universal Multi-Class ReID Feature Extractor using ConvNeXt-Base (timm).
Replaces the previous DINOv2 backbone.

WHY ConvNeXt-Base over DINOv2 for drone/aerial surveillance:
  - DINOv2 learns patch-level scene texture features which are too similar
    across aerial objects (drone vs vehicle both appear as dark blobs).
  - ConvNeXt-Base was pretrained on ImageNet-21k with strong object-level
    class discrimination — it has seen thousands of distinct vehicles,
    aircraft, and mechanical objects, giving it far better class-separating
    embeddings in the 1024-D space.
  - Cosine distance between drone and vehicle embeddings is significantly
    higher with ConvNeXt-Base, making the threshold gate actually work.

Supports ALL custom YOLO classes dynamically (person, drone, vehicle, etc.)
Optimized for RTX 3060 Ti (FP16 autocast) & 8GB VRAM.
"""

import logging
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

logger = logging.getLogger(__name__)


class UniversalReIDExtractor:
    """
    Universal 1024-D ReID feature extractor using ConvNeXt-Base (timm).
    Works for ANY custom class detected by YOLO (person, drone, vehicle, boat, etc.)
    Loads pretrained ImageNet weights via timm — no local weight file required.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(
            "Initializing ConvNeXt-Base ReID Extractor on %s (timm pretrained)...",
            self.device,
        )

        # ConvNeXt-Base: num_classes=0 returns the 1024-D global average-pool
        # feature vector directly, without the classification head.
        self.model = timm.create_model(
            "convnext_base", pretrained=True, num_classes=0
        )
        try:
            self.model.to(self.device)
        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower() and self.device.type == "cuda":
                logger.warning(
                    "CUDA VRAM constrained by background services; falling back ConvNeXt ReID to CPU: %s",
                    oom_err,
                )
                print(f"[REID ENGINE] Note: CUDA VRAM is full; running ConvNeXt ReID smoothly on CPU.")
                self.device = torch.device("cpu")
                self.model.to(self.device)
            else:
                raise oom_err

        self.model.eval()
        self.embedding_dim = 1024  # ConvNeXt-Base GAP output dimension

        # ImageNet normalisation (same as DINOv2 / standard torchvision)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("ConvNeXt-Base ReID extractor ready. Embedding dim = %d", self.embedding_dim)

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Single-crop convenience wrapper -> (1024,) L2-normalised embedding."""
        return self.extract_batch([crop_bgr])[0]

    @torch.no_grad()
    def extract_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Input : list of BGR numpy image crops (any YOLO class).
        Output: (N, 1024) L2-normalised float32 embeddings.
        """
        if not crops_bgr:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        valid_idx, tensors = [], []
        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
                continue
            rgb = crop[:, :, ::-1]  # BGR -> RGB
            pil_img = Image.fromarray(np.ascontiguousarray(rgb))
            tensors.append(self.transform(pil_img))
            valid_idx.append(i)

        out = np.zeros((len(crops_bgr), self.embedding_dim), dtype=np.float32)
        if not tensors:
            return out

        batch = torch.stack(tensors).to(self.device)

        # FP16 inference for speed on Tensor Cores
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            features = self.model(batch)

        # L2 normalise so cosine similarity = dot product
        features = F.normalize(features.float(), p=2, dim=1).cpu().numpy()

        for pos, i in enumerate(valid_idx):
            out[i] = features[pos]

        return out
