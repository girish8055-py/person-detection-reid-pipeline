"""
ml_pipeline/app/pipeline/reid/reid_model.py

Universal Multi-Class ReID Feature Extractor using DINOv2 ViT-B/14.
Optimized for RTX 3050 Ti 8GB VRAM (FP16 autocast) & 128GB RAM.
Supports ALL custom classes dynamically.
"""

import logging
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from .dinov2_download import DINOV2_WEIGHTS_PATH, ensure_dinov2

logger = logging.getLogger(__name__)

class UniversalReIDExtractor:
    """
    Universal 768-D ReID feature extractor using Meta AI's DINOv2 ViT-B/14.
    Works for ANY custom class detected by YOLO (person, drone, vehicle, boat, bag, etc.)
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info("Initializing Universal DINOv2 ViT-B/14 ReID Extractor on %s...", self.device)

        # Load DINOv2 Base (86M params, ~450MB VRAM) from the local models/ folder;
        # source + weights are downloaded there on first run if missing.
        repo_dir, weights_path = ensure_dinov2(weights_path=weights_path or DINOV2_WEIGHTS_PATH)
        self.model = torch.hub.load(repo_dir, "dinov2_vitb14", source="local", pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        logger.info("Loaded DINOv2 weights from %s", weights_path)
        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = 768  # DINOv2 ViT-B/14 output dimension

        # Preprocessing transform optimized for DINOv2 patch size (14x14)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Single-crop convenience wrapper around extract_batch -> (768,) embedding."""
        return self.extract_batch([crop_bgr])[0]

    @torch.no_grad()
    def extract_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Input: List of BGR numpy image crops of ANY custom YOLO class.
        Output: (N, 768) L2-normalized float32 embeddings.
        """
        if not crops_bgr:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        valid_idx, tensors = [], []
        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
                continue
            rgb = crop[:, :, ::-1]  # Convert BGR (OpenCV) to RGB
            pil_img = Image.fromarray(np.ascontiguousarray(rgb))
            tensors.append(self.transform(pil_img))
            valid_idx.append(i)

        out = np.zeros((len(crops_bgr), self.embedding_dim), dtype=np.float32)
        if not tensors:
            return out

        batch = torch.stack(tensors).to(self.device)

        # Execute ultra-fast FP16 inference on RTX 3050 Ti Tensor Cores
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            features = self.model(batch)

        # L2 Normalize feature vectors
        features = F.normalize(features, p=2, dim=1).float().cpu().numpy()

        for pos, i in enumerate(valid_idx):
            out[i] = features[pos]

        return out
