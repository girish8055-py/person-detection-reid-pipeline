
from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import ReIDConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Official pretrained weights (ImageNet + market1501 pretrained OSNet_x1_0)
# hosted by the deep-person-reid maintainers.
_OFFICIAL_OSNET_X1_0_URL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "osnet_x1_0_imagenet.pth"
)


def _build_osnet_x1_0(num_classes: int = 1000):
    """
    Try to build the official OSNet_x1_0 architecture from torchreid's
    model-definition submodule only (no dataset/engine imports triggered).
    Falls back to a bundled architecturally-identical implementation.
    """
    # Newer torchreid (KaiyangZhou/deep-person-reid, "torchreid" >= 1.4 layout)
    try:
        from torchreid.reid.models.osnet import osnet_x1_0  # type: ignore
        return osnet_x1_0(num_classes=num_classes, pretrained=False, loss="softmax")
    except Exception:
        pass

    # Older/alternate layout
    try:
        from torchreid.models.osnet import osnet_x1_0  # type: ignore
        return osnet_x1_0(num_classes=num_classes, pretrained=False, loss="softmax")
    except Exception:
        pass

    # Fallback: bundled, dependency-free re-implementation.
    logger.warning(
        "torchreid not importable; using bundled OSNet_x1_0 fallback. "
        "Install torchreid for the exact reference implementation: "
        "`pip install torchreid --no-deps`"
    )
    from .osnet_fallback import osnet_x1_0 as fallback_osnet_x1_0
    return fallback_osnet_x1_0(num_classes=num_classes)


class OSNetReIDExtractor:
    """
    Thin, production-friendly wrapper around OSNet_x1_0 that turns a cropped
    person image (BGR numpy array, as returned by OpenCV) into a normalized
    512-D embedding.

    Usage:
        extractor = OSNetReIDExtractor(config)
        embedding = extractor.extract(person_crop_bgr)          # single crop
        embeddings = extractor.extract_batch([crop1, crop2, ...])  # batched
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        self.cfg = config or DEFAULT_CONFIG
        self.device = torch.device(self.cfg.device)

        self.model = _build_osnet_x1_0()
        self._load_weights(self.cfg.weights_path)

        self.model.eval()
        self.model.to(self.device)

        self._use_fp16 = self.cfg.use_fp16 and self.device.type == "cuda"
        if self._use_fp16:
            self.model.half()

        h, w = self.cfg.input_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #
    def _load_weights(self, weights_path: str) -> None:
        if not weights_path or not os.path.isfile(weights_path):
            logger.warning(
                "OSNet weights not found at '%s'. The model will run with "
                "random initialization until you provide the official "
                "pretrained checkpoint. Download it from the deep-person-reid "
                "model zoo and point ReIDConfig.weights_path to it.",
                weights_path,
            )
            return

        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        model_dict = self.model.state_dict()
        matched, skipped = {}, []
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k  # strip DataParallel prefix
            if key in model_dict and model_dict[key].shape == v.shape:
                matched[key] = v
            else:
                skipped.append(key)

        model_dict.update(matched)
        self.model.load_state_dict(model_dict)

        logger.info(
            "OSNet weights loaded: %d/%d layers matched (%d skipped).",
            len(matched), len(state_dict), len(skipped),
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = crop_bgr[:, :, ::-1]  # BGR -> RGB
        pil_img = Image.fromarray(np.ascontiguousarray(rgb))
        return self.transform(pil_img)

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Single crop -> L2-normalized (512,) float32 embedding."""
        return self.extract_batch([crop_bgr])[0]

    @torch.no_grad()
    def extract_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        List of crops -> (N, 512) L2-normalized float32 embeddings.
        Empty / degenerate crops are skipped safely (returns a zero vector
        for them so array shapes stay consistent).
        """
        if len(crops_bgr) == 0:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)

        valid_idx, tensors = [], []
        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
                continue
            tensors.append(self._preprocess(crop))
            valid_idx.append(i)

        out = np.zeros((len(crops_bgr), self.cfg.embedding_dim), dtype=np.float32)
        if not tensors:
            return out

        batch = torch.stack(tensors).to(self.device)
        if self._use_fp16:
            batch = batch.half()

        features = self.model(batch)
        if isinstance(features, (tuple, list)):
            features = features[0]

        features = F.normalize(features, p=2, dim=1)
        features = features.float().cpu().numpy()

        for pos, i in enumerate(valid_idx):
            out[i] = features[pos]

        return out
