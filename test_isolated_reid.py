#!/usr/bin/env python3
"""
test_isolated_reid.py

Standalone, Isolated Video Testing Script for Multi-Class ReID Pipeline.

This script allows you to test YOLO detection + ByteTrack + DINOv2 ReID
on ANY video file independently.

Usage:
  python test_isolated_reid.py --video /path/to/video.mp4
  python test_isolated_reid.py --video /path/to/video.mp4 --yolo_weights ml_pipeline/app/models/best.pt
  python test_isolated_reid.py --video /path/to/video.mp4 --output output_annotated.mp4

Note: This script is COMPLETELY ISOLATED. If you delete or move this file,
the core production system (app.py / worker.py) will continue working with ZERO impact.
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure ml_pipeline/app is in Python path for standalone execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(CURRENT_DIR, "ml_pipeline", "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from pipeline.detector import ObjectDetector
    from pipeline.identity_manager import IdentityManager
    from pipeline.reid.config import ReIDConfig
except ImportError as err:
    print(f"[ERROR] Could not import pipeline modules: {err}")
    print("Ensure you are running this script from the repository root directory.")
    sys.exit(1)


def generate_color(id_str: str):
    """Generates a consistent, distinct BGR color for each target identity string."""
    hash_code = hash(id_str)
    r = (hash_code & 0xFF0000) >> 16
    g = (hash_code & 0x00FF00) >> 8
    b = hash_code & 0x0000FF
    return (b % 200 + 55, g % 200 + 55, r % 200 + 55)


def run_isolated_reid_test(
    video_path: str,
    yolo_weights: str,
    reid_weights: str,
    output_path: str,
    confidence: float,
    similarity_threshold: float,
    display: bool = False,
):
    print("=" * 75)
    print("🚀 STANDALONE ISOLATED MULTI-CLASS RE-ID TEST")
    print("=" * 75)
    print(f"📹 Input Video       : {video_path}")
    print(f"🎯 YOLO Weights      : {yolo_weights}")
    print(f"🧬 ReID Weights      : {reid_weights}")
    print(f"⚙️  Confidence       : {confidence}")
    print(f"📊 Similarity Cutoff : {similarity_threshold}")
    print(f"💾 Output Video      : {output_path}")
    print("=" * 75)

    if not os.path.isfile(video_path):
        print(f"❌ [ERROR] Video file not found: {video_path}")
        sys.exit(1)

    if not os.path.isfile(yolo_weights):
        print(f"⚠️  [WARNING] YOLO weights not found at '{yolo_weights}'. Falling back to 'yolov8n.pt'...")
        yolo_weights = "yolov8n.pt"

    # Initialize Detector
    print("\n[1/3] Loading Object Detector & ByteTrack...")
    detector = ObjectDetector(model_path=yolo_weights, confidence=confidence)

    # Initialize ReID Identity Manager
    print("\n[2/3] Loading DINOv2 ReID Engine...")
    cfg = ReIDConfig(weights_path=reid_weights, similarity_threshold=similarity_threshold)
    identity_mgr = IdentityManager(weights_path=reid_weights)
    identity_mgr.config.similarity_threshold = similarity_threshold
    identity_mgr.core.cfg.similarity_threshold = similarity_threshold

    # Open Video
    print("\n[3/3] Processing Video Frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ [ERROR] Failed to open video file: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    target_stats = {}
    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp = frame_idx / fps

            # 1. Detect & Track with YOLO + ByteTrack
            detections = detector.detect(frame)

            if detections:
                track_ids = [d["track_id"] for d in detections]
                bboxes = [d["bbox"] for d in detections]
                confidences = [d.get("confidence", 1.0) for d in detections]
                classes = [d.get("class", "object") for d in detections]

                # Crop bounding boxes for ReID
                crops = []
                for box in bboxes:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    crop = frame[y1:y2, x1:x2]
                    crops.append(crop)

                # 2. Batched ReID Feature Extraction & Identity Resolution
                reid_ids = identity_mgr.resolve_batch(
                    track_ids=track_ids,
                    crops_bgr=crops,
                    bboxes=bboxes,
                    confidences=confidences,
                    timestamp=timestamp,
                    classes=classes,
                )

                # 3. Draw Annotations on Frame
                for det, reid_id in zip(detections, reid_ids):
                    x1, y1, x2, y2 = det["bbox"]
                    det_cls = det.get("class", "object").capitalize()
                    color = generate_color(reid_id)

                    # Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label String: [Class_001 | Track: 12 | Conf: 0.89]
                    label = f"{reid_id} (Trk:{det['track_id']})"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    # Bookkeeping
                    if reid_id not in target_stats:
                        target_stats[reid_id] = {
                            "class": det_cls,
                            "first_seen": timestamp,
                            "last_seen": timestamp,
                            "hits": 0,
                        }
                    target_stats[reid_id]["last_seen"] = timestamp
                    target_stats[reid_id]["hits"] += 1

            # Save Annotated Frame
            out.write(frame)

            if display:
                cv2.imshow("Isolated ReID Test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] User interrupted video playback.")
                    break

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f" Frame {frame_idx}/{total_frames} ({progress:.1f}%) | Processing Speed: {current_fps:.1f} FPS", end="\r")

    finally:
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print("\n" + "=" * 75)
    print("✅ TEST COMPLETE")
    print("=" * 75)
    print(f"⏱️  Total Time       : {total_time:.2f} seconds")
    print(f"🎬 Frames Processed : {frame_idx}")
    print(f"⚡ Average Speed    : {avg_fps:.1f} FPS")
    print(f"💾 Output Saved To  : {output_path}")
    print("=" * 75)
    print("\n📊 UNIQUE TARGET IDENTITIES DETECTED:")
    print(f"{'Global ID':<18} | {'Class':<12} | {'Duration (s)':<12} | {'Total Frames':<12}")
    print("-" * 60)
    for gid, stats in sorted(target_stats.items()):
        duration = stats["last_seen"] - stats["first_seen"]
        print(f"{gid:<18} | {stats['class']:<12} | {duration:<12.1f} | {stats['hits']:<12}")
    print("-" * 60)


if __name__ == "__main__":
    DEFAULT_VIDEO = os.path.join(APP_DIR, "upload", "//home/aiml/Desktop/deployment_model/person-detection-reid-pipeline/3a19e7fc-4944-4479-87dd-7ea15389d04a.webm")
    DEFAULT_YOLO = os.path.join(APP_DIR, "downloaded_model", "Drone_Model_Latest_v3.pt")

    parser = argparse.ArgumentParser(description="Standalone Isolated Multi-Class ReID Video Test Script")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="Path to input video file")
    parser.add_argument("--yolo_weights", type=str, default=DEFAULT_YOLO, help="Path to YOLO weights (.pt)")
    parser.add_argument("--reid_weights", type=str, default="ml_pipeline/app/models/dinov2_vitb14.safetensors", help="Path to ReID weights (.safetensors/.pth)")
    parser.add_argument("--output", type=str, default="isolated_reid_output.mp4", help="Path to output annotated video file")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument("--similarity_threshold", type=float, default=0.35, help="ReID cosine similarity cutoff threshold")
    parser.add_argument("--display", action="store_true", help="Display video live while processing")

    args = parser.parse_args()

    run_isolated_reid_test(
        video_path=args.video,
        yolo_weights=args.yolo_weights,
        reid_weights=args.reid_weights,
        output_path=args.output,
        confidence=args.conf,
        similarity_threshold=args.similarity_threshold,
        display=args.display,
    )
