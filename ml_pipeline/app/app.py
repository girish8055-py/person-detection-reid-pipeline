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
from worker.worker import worker_loop, TEST_BASE_URL, PROD_BASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "downloaded_model", "best.pt")
MODEL_PATH = DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else os.path.join(BASE_DIR, "models", "best.pt")
REID_WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "upload")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Models load ONCE at startup, reused across every job.
processor = VideoProcessor(
    model_path=MODEL_PATH,
    reid_weights_path=REID_WEIGHTS_PATH,
    confidence=0.25,
    target_fps=10,
)

processing_lock = threading.Lock()  # ensures only one video processes at a time


if __name__ == "__main__":
    logger.info("Starting Production Pipeline Workers...")

    current_module = sys.modules[__name__]

    # Start Test Server background worker
    test_thread = threading.Thread(
        target=worker_loop,
        args=(current_module, "Test Server", TEST_BASE_URL),
        daemon=True
    )
    test_thread.start()
    logger.info("Background Worker for Test Server Started")

    # Start Production Server background worker
    prod_thread = threading.Thread(
        target=worker_loop,
        args=(current_module, "Production Server", PROD_BASE_URL),
        daemon=True
    )
    prod_thread.start()
    logger.info("Background Worker for Production Server Started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping Production Workers...")