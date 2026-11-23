import os
import sys
import time
import subprocess
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def compute_frame_sharpness(gray):
    # Laplacian variance as a standard focus/sharpness metric
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_frame_stats(frame):
    # Brightness (mean) and Contrast (std dev)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    sharpness = compute_frame_sharpness(gray)
    return mean_val, std_val, sharpness

def main():
    video_path = '/home/aiml/Desktop/deployment_model/person-detection-reid-pipeline/DJI_0429.MP4'
    model_path = '/home/aiml/Desktop/deployment_model/person-detection-reid-pipeline/ml_pipeline/app/downloaded_model/best.pt'
    output_path = '/home/aiml/Desktop/deployment_model/person-detection-reid-pipeline/DJI_0429_detected.mp4'
    ffmpeg_path = '/home/aiml/.local/bin/ffmpeg'

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {width}x{height} @ {fps:.2f} fps, Total frames: {total_frames}")

    # Launch ffmpeg subprocess to encode with high quality H.264
    # Using crf 18 (visually lossless) and preset fast for speed and preservation of quality
    ffmpeg_cmd = [
        ffmpeg_path,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f"{width}x{height}",
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Metrics trackers
    in_brightness_list = []
    in_contrast_list = []
    in_sharpness_list = []

    out_brightness_list = []
    out_contrast_list = []
    out_sharpness_list = []

    detection_counts = {'drone': 0, 'person': 0, 'vehicle': 0}
    frames_with_detections = 0

    frame_idx = 0
    start_time = time.time()

    sample_interval = 15 # Sample quality metrics every 15 frames for speed and representation

    print("Starting processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Calculate input frame quality on sampled frames
        if frame_idx % sample_interval == 0:
            b_in, c_in, s_in = compute_frame_stats(frame)
            in_brightness_list.append(b_in)
            in_contrast_list.append(c_in)
            in_sharpness_list.append(s_in)

        # Run detection
        # Predict on GPU
        results = model.predict(frame, conf=0.25, device='cuda:0', verbose=False)
        det = results[0]
        boxes = det.boxes

        if len(boxes) > 0:
            frames_with_detections += 1
            for cls_id in boxes.cls.tolist():
                cls_name = model.names.get(int(cls_id), 'unknown')
                if cls_name in detection_counts:
                    detection_counts[cls_name] += 1

        annotated_frame = det.plot()

        if frame_idx % sample_interval == 0:
            b_out, c_out, s_out = compute_frame_stats(annotated_frame)
            out_brightness_list.append(b_out)
            out_contrast_list.append(c_out)
            out_sharpness_list.append(s_out)

        # Write to ffmpeg pipe
        try:
            proc.stdin.write(annotated_frame.tobytes())
        except Exception as e:
            print(f"Error writing to ffmpeg at frame {frame_idx}: {e}")
            break

        if frame_idx % 250 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / proc_fps if proc_fps > 0 else 0
            print(f"Processed [{frame_idx}/{total_frames}] frames ({(frame_idx/total_frames)*100:.1f}%) - Speed: {proc_fps:.1f} fps - ETA: {eta:.1f}s")

    cap.release()
    proc.stdin.close()
    proc.wait()

    total_time = time.time() - start_time
    print(f"\nProcessing complete in {total_time:.2f} seconds ({frame_idx / total_time:.1f} fps avg)!")

    # Save summary stats
    import json
    stats = {
        'total_frames': frame_idx,
        'frames_with_detections': frames_with_detections,
        'detection_counts': detection_counts,
        'quality_samples_count': len(in_brightness_list),
        'input': {
            'mean_brightness': float(np.mean(in_brightness_list)),
            'mean_contrast': float(np.mean(in_contrast_list)),
            'mean_sharpness': float(np.mean(in_sharpness_list)),
        },
        'output_annotated': {
            'mean_brightness': float(np.mean(out_brightness_list)),
            'mean_contrast': float(np.mean(out_contrast_list)),
            'mean_sharpness': float(np.mean(out_sharpness_list)),
        }
    }

    with open('/home/aiml/Desktop/deployment_model/person-detection-reid-pipeline/detection_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("Saved stats to detection_stats.json")

if __name__ == '__main__':
    main()
