<div align="center">

# 🎯 `person-detection-reid-pipeline`

### **Autonomous Aerial Target Detection, OSNet AIN Multi-Feature Re-Identification, Telemetry Fusion & VLM Threat Assessment Engine**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-Custom%20Trained-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![OSNet AIN](https://img.shields.io/badge/ReID-OSNet%20AIN%20Feature%20Bank-00E676?style=for-the-badge)](https://github.com/KaiyangZhou/deep-person-reid)
[![Ollama VLM](https://img.shields.io/badge/VLM-Qwen2.5--VL%20%2F%20LLaVA-FF6F00?style=for-the-badge)](https://ollama.ai/)
[![Azure](https://img.shields.io/badge/Azure-Blob%20Storage-0089D6?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/)

---

</div>

<br/>

## 📖 Master Documentation & Overview

> 📑 **Full Technical Specification Documents Included**:
> The complete, production-ready system documentation (including architecture diagrams, module matrices, REST API parameters, Azure SAS upload paths, VLM JSON schemas, and non-null severity rules) is included directly in the repository:
> 
> 🌐 [**View Detection Pipeline & VLM Architecture Specification (Markdown)**](Detection_Pipeline_Documentation.md)
> 
> 🌐 [**View REST API & Background Worker Integration Specification (Markdown)**](API_Integration_Documentation.md)

<br/>

## ⚡ Key Highlights & System Capabilities

- 🎯 **Custom-Trained YOLO Detector & Dynamic Classes**: Fine-tuned for small-object aerial/drone surveillance person, vehicle, and drone detection (`Person_001`, `Vehicle_001`, `Drone_001`, etc.).
- 🔄 **ByteTrack Short-Term Association**: Handles frame-to-frame spatial continuity and short-term occlusions across video streams.
- 🧬 **OSNet AIN Deep Feature Re-ID (Adaptive Instance Normalization)**: Uses `osnet_ain_x1_0` to filter out lighting, pose, shadow, and motion blur variations.
- 🏛️ **Multi-Embedding Feature Bank Gallery Engine**: Stores up to 10 diverse pose/angle keyframe embeddings per target, eliminating duplicate ID generation when targets turn around or re-appear.
- 🤖 **Vision-Language Model (VLM) Threat Engine**: Integrates local Ollama (`qwen2.5vl:3b` / `llava`) with model weights stored inside `ml_pipeline/app/vlm_models/`.
- 🛡️ **Guaranteed Non-Null Severity**: Dual-pass severity resolution (`resolve_severity()`) ensures target threat level (`1` to `4`) and severity labels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) are **100% non-null**.
- 🎖️ **Rich Tactical Directives**: Post-processor (`ensure_detailed_suggested_action()`) expands VLM responses into complete, detailed field directives.
- 🛰️ **GPS Telemetry Fusion**: Interpolates drone flight logs (`offline_teledata_` & `offline_flightdata_`) to tag real-world latitude, longitude, altitude, cardinal direction, and ISO timestamps.
- 🧹 **Automatic Disk Space Cleanup**: Downloaded videos, telemetry CSVs, and local output directories are **automatically deleted** after uploading evidence to Azure and sending completion callbacks.
- ⚙️ **Automatic Multi-Server Job Polling**: Single unified worker pool (`app.py`) continuously monitors and pulls pending jobs from **both** Test Server and Production Server queues.

<br/>

## 🏗️ End-to-End Pipeline Architecture

```gantt
+---------------------------------------------------------------------------------------------------+
|                                 END-TO-END PIPELINE ARCHITECTURE                                  |
+---------------------------------------------------------------------------------------------------+

 [Auto Multi-Server Polling] ──► [Download Media & CSV] ──► [Frame Decimation Engine (10 FPS)]
                                                                    │
                                                                    ▼
 [Disk & Temp Directory Cleanup] ◄── [OSNet AIN Feature Bank] ◄── [Custom YOLO + ByteTrack]
            │
            ▼
 [VLM Keyframe Analysis] ──► [Non-Null Severity & Action] ──► [Telemetry GPS Fusion] ──► [API Callbacks]
```

<br/>

## 📂 Repository Directory Tree

```text
person-detection-reid-pipeline/ml_pipeline/app
├── app.py                      # Production & Multi-Server Worker Service Entry Point
├── downloader.py               # Media & Telemetry File Downloader
├── models/                     # Deep Learning Model Checkpoints
│   ├── best.pt                 # Custom-Trained YOLO Weights
│   └── osnet_ain_x1_0_...pth   # OSNet AIN ReID Checkpoint Weights
├── vlm_models/                 # Local Ollama VLM Model Blobs (Qwen2.5-VL / LLaVA)
├── worker/                     # Autonomous Service Layer
│   ├── api_client.py           # Backend Server REST API Client
│   ├── result_builder.py       # API Schema Parser & JSON Payload Builder
│   └── worker.py               # Multi-Server Auto Poller, VLM Engine, Azure Uploader & Cleanup
└── pipeline/                   # Core Computer Vision Engine
    ├── video_processor.py      # Main Pipeline Orchestrator & Decimation Engine
    ├── detector.py             # Custom YOLO Detector & ByteTrack Integration
    ├── person_manager.py       # Lifetime Target Session Aggregator
    ├── identity_manager.py     # OSNet Re-ID Gallery Cosine Matching & Identity Brain
    ├── vlm_verifier.py         # Keyframe Visual Class & Threat Verification
    ├── telemetry.py            # Drone GPS Log Interpolation Engine
    ├── screenshot_manager.py   # Crop Quality Selection & Keyframe Exporter
    ├── json_writer.py          # Output Analytics & Metadata Writer
    ├── utils.py                # Severity Resolution & Action Formatting Helpers
    └── reid/                   # OSNet Deep Re-Identification Subsystem
        ├── config.py           # Re-ID Settings (similarity_threshold = 0.28)
        ├── identity_manager.py # Gallery Cosine Matching & Search
        ├── embedding_database.py# Thread-Safe Embedding Feature Bank Database
        └── reid_model.py       # Batched PyTorch OSNet AIN Feature Extractor
```

<br/>

## ⚙️ Key System Configuration

| Parameter | Setting / Value | Technical Description |
| :--- | :--- | :--- |
| **Detection Model** | Custom-Trained YOLO (`best.pt`) | Fine-tuned object detection weights for aerial imagery |
| **Detection Confidence** | `0.25` (with auto-fallback) | Threshold required for object detection validation |
| **Tracking Engine** | ByteTrack (`bytetrack.yaml`) | Frame-to-frame spatial multi-object tracker |
| **Re-ID Model** | OSNet AIN (`osnet_ain_x1_0`) | Adaptive Instance Normalization feature extractor |
| **Re-ID Weights Path** | `models/osnet_ain_x1_0_imagenet.pth` | Self-contained local checkpoint file |
| **Gallery Feature Bank** | Up to 10 pose embeddings / target | Multi-feature storage per target for angle-robust ReID |
| **Cosine Similarity Cutoff**| `0.28` | Multi-embedding gallery cutoff score |
| **VLM Model Engine** | Ollama (`qwen2.5vl:3b` / `llava`) | Local Vision-Language model for visual threat assessment |
| **VLM Storage Path** | `ml_pipeline/app/vlm_models/` | Self-contained local VLM model blobs directory |
| **Severity Mapping** | Levels 1 (LOW) to 4 (CRITICAL) | Non-null threat assessment resolved via VLM + ReID/duration fallback |
| **Disk Cleanup** | Automated `finally` block cleanup | Deletes downloaded `.mp4`, `.csv`, and output folders post-job |

<br/>

## 📦 Completion API Payload JSON Schema

```json
{
  "status": "COMPLETED",
  "results": [
    {
      "id_number": "Person_001",
      "class": "person",
      "entry_time": "00:00:02",
      "exit_time": "00:00:18",
      "total_duration": 16.0,
      "reidentification_count": 2,
      "severity_level": 3,
      "severity": "HIGH",
      "description": "Individual in dark jacket walking along path near water edge.",
      "suggest_action": "MONITOR — Keep close watch on target person movements and behavior.",
      "evidence": {
        "file_name": "Person_001_detection.jpg",
        "file_size": 245120,
        "file_url": "https://pdrlimage.blob.core.windows.net/user-1022c811-4f72-4254-81e2-8f9762d2b237/threatimages/flight_123/Person_001_detection.jpg"
      },
      "appearances": [
        {
          "appearance_id": 1,
          "entry_time": "00:00:02",
          "exit_time": "00:00:10",
          "duration": 8.0,
          "confidence": 0.89,
          "severity_level": 3,
          "severity": "HIGH",
          "entry_location": { "lat": 18.52043, "lon": 73.85674, "alt": 45.2, "direction": "NE" },
          "exit_location": { "lat": 18.52046, "lon": 73.85677, "alt": 45.2, "direction": "NE" }
        }
      ]
    }
  ]
}
```

<br/>

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Start Local Ollama VLM Service
Ensure Ollama is running locally pointing to your local `vlm_models` folder:
```bash
export OLLAMA_MODELS="$(pwd)/ml_pipeline/app/vlm_models"
ollama serve
```

### 3. Launch Automatic Multi-Server Pipeline Worker
```bash
python ml_pipeline/app/app.py
```
Starts the production background worker service, polling automatically across Test and Production servers.

---

<div align="center">

*Developed for High-Precision Drone Surveillance, Automated Person Re-Identification & VLM Threat Intelligence.*

</div>
