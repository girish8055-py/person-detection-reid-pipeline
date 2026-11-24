"""
app.py

Production Worker Service for Person & Vehicle Detection + ReID Pipeline.
Processes incoming jobs from Test Server and Production Server queues.

Run:
    python app.py
"""

import logging
import os
import sys
import threading
import time

from pipeline.video_processor import VideoProcessor
from worker.worker import start_worker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables (.env file)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback env parser checking current dir, parent dir, and root project dir
    for candidate in [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(os.path.dirname(BASE_DIR), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), ".env"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
            break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Set OLLAMA_MODELS to point directly to local vlm_models folder inside the project
VLM_MODELS_PATH = os.path.join(BASE_DIR, "vlm_models")
if os.path.exists(VLM_MODELS_PATH):
    os.environ["OLLAMA_MODELS"] = VLM_MODELS_PATH

DEFAULT_MODEL = os.path.join(BASE_DIR, "downloaded_model", "best.pt")
if os.path.exists(DEFAULT_MODEL):
    MODEL_PATH = DEFAULT_MODEL
else:
    print(f"Model URL is not present and downloaded model is missing at: {DEFAULT_MODEL}")
    MODEL_PATH = DEFAULT_MODEL
REID_WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dinov2_vitb14_pretrain.pth"
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "upload")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Models load ONCE at startup, reused across every job.
processor = VideoProcessor(
    model_path=MODEL_PATH,
    reid_weights_path=REID_WEIGHTS_PATH,
    confidence=0.55,
    target_fps=2,
)

processing_lock = threading.Lock()  # ensures only one video processes at a time


if __name__ == "__main__":
    logger.info("Starting Automatic Multi-Server Pipeline Worker...")
    current_module = sys.modules[__name__]
    start_worker(current_module)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping Production Workers...")