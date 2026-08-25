<div align="center">

# 🎯 `person-detection-reid-pipeline`

### **Autonomous Aerial Person Detection, Deep Re-Identification, Telemetry Fusion & VLM Threat Assessment Engine**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-Custom%20Trained-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![OSNet](https://img.shields.io/badge/ReID-OSNet%20Deep%20Embeddings-00E676?style=for-the-badge)](https://github.com/KaiyangZhou/deep-person-reid)
[![LLaVA VLM](https://img.shields.io/badge/VLM-Ollama%20LLaVA%20Threat%20Engine-FF6F00?style=for-the-badge)](https://ollama.ai/)
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

## ⚡ Key Highlights & Capabilities

- 🎯 **Custom-Trained YOLO Detector**: Fine-tuned for small-object aerial and drone surveillance person detection (`class_id = 0`).
- 🔄 **ByteTrack Short-Term Association**: Handles frame-to-frame spatial continuity and short-term occlusions.
- 🧬 **OSNet Deep Feature Re-ID**: Computes 512-dimensional L2-normalized embeddings to re-identify individuals across long absences and re-entries (`similarity_threshold = 0.68`).
- 🤖 **Vision-Language Model (VLM) Threat Engine**: Integrates local Ollama (`llava:latest`) for visual keyframe threat assessment, returning detailed target descriptions, visual class verification, and tactical actions.
- 🛡️ **Guaranteed Non-Null Severity**: Dual-pass severity resolution (`resolve_severity()`) ensures target threat level (`1` to `4`) and severity labels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) are **100% non-null**, falling back gracefully to ReID frequency and duration metrics if VLM visual data is missing.
- 🎖️ **Rich Tactical Suggested Actions**: Automated post-processor (`ensure_detailed_suggested_action()`) expands single-word responses into complete, detailed Army-Chief style field directives.
- 🛰️ **GPS Telemetry Fusion**: Interpolates drone flight logs (`offline_teledata_` & `offline_flightdata_`) to tag real-world latitude, longitude, relative altitude, cardinal direction, and ISO timestamps.
- ☁️ **Azure Blob Storage Uploads**: Automatically uploads crop evidence snapshots (`best_detection.jpg`) using BlockBlob SAS authorization.
- ⚙️ **Autonomous Background Worker**: Polling engine (`worker.py`) that queries backend API endpoints, processes detection jobs, and pushes structured completion callbacks.

<br/>

## 🏗️ System Architecture & Data Flow

```gantt
+---------------------------------------------------------------------------------------------------+
|                                 END-TO-END PIPELINE ARCHITECTURE                                  |
+---------------------------------------------------------------------------------------------------+

 [Pending Task Polling] ──► [Download Media & CSV] ──► [Frame Decimation Engine (10 FPS)]
                                                                  │
                                                                  ▼
 [Azure Blob Snapshot Upload] ◄── [OSNet Re-ID (0.68 Sim)] ◄── [Custom YOLO + ByteTrack]
            │
            ▼
 [VLM Keyframe Analysis] ──► [Non-Null Severity & Action] ──► [Telemetry GPS Fusion] ──► [API Callbacks]
```

<br/>

## 📂 Repository Directory Tree

```text
person-detection-reid-pipeline/ml_pipeline/app
├── app.py                      # Production Worker Service & Management Entry Point
├── downloader.py               # Media & Telemetry File Downloader
├── models/                     # Deep Learning Checkpoint Storage
│   ├── best.pt                 # Custom-Trained YOLO Weights
│   └── osnet_x1_0_market_...pth# OSNet Re-ID Model Weights
├── worker/                     # Autonomous Service Layer
│   ├── api_client.py           # Backend Server REST API Client
│   ├── result_builder.py       # API Schema Parser & JSON Payload Builder
│   └── worker.py               # Background Task Polling Loop, VLM Engine & Azure Uploader
└── pipeline/                   # Core Computer Vision Engine
    ├── video_processor.py      # Main Pipeline Orchestrator & Decimation Engine
    ├── detector.py             # Custom YOLO Detector & ByteTrack Integration
    ├── person_manager.py       # Lifetime Target Session Aggregator
    ├── identity_manager.py     # OSNet Re-ID Gallery Cosine Matching & Identity Brain
    ├── vlm_verifier.py         # Keyframe Visual Class & Threat Verification
    ├── telemetry.py            # Drone GPS Log Interpolation Engine
    ├── screenshot_manager.py   # Crop Quality Selection & Keyframe Exporter
    ├── json_writer.py          # Local Output Analytics & Metadata Writer
    ├── utils.py                # Severity Resolution & Action Formatting Helpers
    └── reid/                   # OSNet Deep Re-Identification Subsystem
        ├── config.py           # Re-ID Subsystem Settings (Threshold = 0.68)
        ├── identity_manager.py # Gallery Cosine Matching & Search
        ├── embedding_database.py# Thread-Safe Embedding Memory Database
        └── reid_model.py       # Batched PyTorch OSNet Feature Extractor
```

<br/>

## 📋 File-by-File Responsibility Matrix

| File Path | Module Category | Technical Functionality |
| :--- | :--- | :--- |
| 🛠️ `app/worker/worker.py` | **Background Worker & VLM** | Autonomous polling event loop, VLM LLaVA prompt execution, weapon escalation guardrails, Azure Blob snapshot uploads, and completion callbacks. |
| 📡 `app/worker/api_client.py` | **REST API Client** | Handles `GET /detection/pending` queries and status update callbacks to backend servers. |
| 🏗️ `app/worker/result_builder.py` | **API Result Builder** | Converts tracking and VLM analytics into production JSON payloads with non-null severities. |
| 🎬 `app/pipeline/video_processor.py` | **Pipeline Orchestrator** | Manages frame decimation (10 FPS), YOLO detection, ByteTrack tracking, Re-ID extraction, keyframe sampling, and VLM assessment. |
| 🔍 `app/pipeline/detector.py` | **Detector & Tracker** | Loads custom YOLO weights (`best.pt`) for person detection (`class_id=0`) and executes ByteTrack multi-object tracking. |
| 🧠 `app/pipeline/identity_manager.py` | **Re-ID Core Engine** | Fast-path lookup, gallery cosine similarity search, embedding updates, and stable Person ID assignment. |
| 👁️ `app/pipeline/vlm_verifier.py` | **VLM Verifier** | Performs multi-crop visual class verification using Ollama LLaVA VLM. |
| 🛠️ `app/pipeline/utils.py` | **Pipeline Utilities** | Implements `resolve_severity()` (non-null fallback engine) and `ensure_detailed_suggested_action()` (tactical action expansion). |
| 🌐 `app/pipeline/telemetry.py` | **GPS Fusion Engine** | Parses drone CSV logs (`offline_teledata_`, `offline_flightdata_`) and interpolates real-world GPS coordinates per frame. |

<br/>

## ⚙️ Hyperparameters & Configuration

| Parameter | Value / Setting | Technical Description |
| :--- | :--- | :--- |
| **Detection Model** | Custom-Trained YOLO (`best.pt`) | Fine-tuned object detection weights for aerial imagery |
| **Detection Confidence** | `0.50` | Threshold required for person detection validation |
| **Tracking Engine** | ByteTrack (`bytetrack.yaml`) | Frame-to-frame spatial multi-object tracker |
| **Re-ID Feature Model** | OSNet (`osnet_x1_0`) | Deep convolutional feature extractor |
| **Embedding Dimension** | `512` Float32 Values | L2-normalized feature representation vector |
| **Cosine Similarity Cutoff** | `0.68` | Minimum similarity score required for identity re-entry match |
| **VLM Engine** | Ollama LLaVA (`llava:latest`) | Vision-Language model for visual threat assessment & action recommendation |
| **VLM Temperature** | `0.2` | Low temperature setting for deterministic, non-creative threat evaluation |
| **Severity Mapping** | Levels 1 (LOW) to 4 (CRITICAL) | Non-null threat assessment resolved via VLM + ReID/duration fallback |
| **Identity TTL** | `None` (Disabled) | Preserves person gallery identities across full video length |

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
      "description": "Individual in dark jacket observed loitering near perimeter fence.",
      "suggest_action": "ALERT — Raise force readiness level and notify perimeter security units regarding Target Person_001.",
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
pip install -r requriment.txt
pip install azure-storage-blob requests opencv-python Pillow
```

### 2. Start Local Ollama VLM Service
Ensure Ollama is installed and running locally with the LLaVA model:
```bash
ollama run llava:latest
```

### 3. Launch Pipeline Server / Worker
```bash
python ml_pipeline/app/app.py
```
Runs the production background worker service polling for pending jobs across Test and Production servers.

---

<div align="center">

*Developed for High-Precision Drone Surveillance, Automated Person Re-Identification & VLM Threat Intelligence.*

</div>
