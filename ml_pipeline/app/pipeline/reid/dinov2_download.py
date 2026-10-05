"""
reid/dinov2_download.py

Downloads the DINOv2 ViT-B/14 ReID backbone into the local `models/` folder
so the pipeline never depends on ~/.cache/torch/hub:

    models/dinov2/                       <- facebookresearch/dinov2 source (hubconf.py)
    models/dinov2_vitb14_pretrain.pth    <- pretrained LVD-142M weights

Run once manually (from ml_pipeline/app):
    python -m pipeline.reid.dinov2_download
UniversalReIDExtractor also calls ensure_dinov2() on startup, so missing
files are fetched automatically.
"""

import io
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile

import torch

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
DINOV2_REPO_DIR = os.path.join(MODELS_DIR, "dinov2")
DINOV2_WEIGHTS_PATH = os.path.join(MODELS_DIR, "dinov2_vitb14_pretrain.pth")

DINOV2_REPO_ZIP_URL = "https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip"
DINOV2_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"


def _download_repo(repo_dir: str):
    logger.info("Downloading DINOv2 source from %s -> %s", DINOV2_REPO_ZIP_URL, repo_dir)
    with urllib.request.urlopen(DINOV2_REPO_ZIP_URL, timeout=120) as resp:
        data = resp.read()

    with tempfile.TemporaryDirectory(dir=os.path.dirname(repo_dir)) as tmp:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(tmp)
        extracted = os.path.join(tmp, os.listdir(tmp)[0])  # dinov2-main/
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        shutil.move(extracted, repo_dir)


def _download_weights(weights_path: str):
    logger.info("Downloading DINOv2 weights from %s -> %s", DINOV2_WEIGHTS_URL, weights_path)
    tmp_path = weights_path + ".part"
    torch.hub.download_url_to_file(DINOV2_WEIGHTS_URL, tmp_path, progress=True)
    os.replace(tmp_path, weights_path)


def ensure_dinov2(repo_dir: str = DINOV2_REPO_DIR, weights_path: str = DINOV2_WEIGHTS_PATH):
    """Download the DINOv2 source and weights into models/ if they are missing."""
    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    if not os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
        _download_repo(repo_dir)

    if not os.path.isfile(weights_path):
        _download_weights(weights_path)

    return repo_dir, weights_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    repo, weights = ensure_dinov2()
    print(f"DINOv2 source : {repo}")
    print(f"DINOv2 weights: {weights} ({os.path.getsize(weights) / 1e6:.1f} MB)")
