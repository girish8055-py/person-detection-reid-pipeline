<div align="center">

# 🎯 `person-detection-reid-pipeline`

### **Autonomous Aerial Person Detection, Deep Re-Identification & Telemetry Fusion Engine**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-Custom%20Trained-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![OSNet](https://img.shields.io/badge/ReID-OSNet%20Deep%20Embeddings-00E676?style=for-the-badge)](https://github.com/KaiyangZhou/deep-person-reid)
[![Azure](https://img.shields.io/badge/Azure-Blob%20Storage-0089D6?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/)
[![Documentation](https://img.shields.io/badge/Documentation-Master%20DOCX%20Included-7B1FA2?style=for-the-badge&logo=microsoftword&logoColor=white)](Person_Detection_ReID_Pipeline_Documentation.docx)

---

</div>

<br/>

## 📖 Master Documentation

> 📑 **Full Technical Specification Document Included**:
> The complete, production-ready system documentation (including ASCII directory trees, file-by-file responsibility matrix, REST API parameters, Azure SAS upload paths, and JSON payload schemas) is included directly in the repository:
> 
> 📥 [**Download Person_Detection_ReID_Pipeline_Documentation.docx**](Person_Detection_ReID_Pipeline_Documentation.docx)

<br/>

## ⚡ Key Highlights & Capabilities

- 🎯 **Custom-Trained YOLO Detector**: Fine-tuned for small-object aerial and drone surveillance person detection (`class_id = 0`).
- 🔄 **ByteTrack Short-Term Association**: Handles frame-to-frame spatial continuity and short-term occlusions.
- 🧬 **OSNet Deep Feature Re-ID**: Computes 512-dimensional L2-normalized embeddings to re-identify individuals across long absences and re-entries (`similarity_threshold = 0.68`).
- 🛰️ **GPS Telemetry Fusion**: Interpolates drone flight logs (`offline_teledata_` & `offline_flightdata_`) to tag real-world latitude, longitude, relative altitude, and ISO timestamps.
- ☁️ **Azure Blob Storage Uploads**: Automatically uploads crop evidence snapshots (`best_detection.jpg`) using BlockBlob SAS authorization.
- 🤖 **Autonomous Background Worker**: Polling engine (`worker.py`) that queries backend API endpoints (`http://114.143.215.196:5005/AeroMegh/idms`) and pushes completion callbacks.

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
 [Telemetry GPS Fusion] ──► [Structured JSON Payload] ──► [API Completion Update Callbacks]
```

<br/>

## 📂 Repository Directory Tree

```text
d:\deployment_model\ml_pipeline\app
├── app.py                      # Flask Web Application & Management Interface (http://localhost:5000)
├── downloader.py               # Media & Telemetry File Downloader
├── main.py                     # Standalone CLI Entry Point
├── models/                     # Deep Learning Checkpoint Storage
│   ├── best.pt                 # Custom-Trained YOLO Weights
│   └── osnet_x1_0_market_...pth# OSNet Re-ID Model Weights
├── worker/                     # Autonomous Service Layer
│   ├── api_client.py           # Backend Server REST API Client
│   ├── result_builder.py       # API Schema Parser & JSON Payload Builder
│   └── worker.py               # Background Task Polling Loop & Azure SAS Uploader
└── pipeline/                   # Core Computer Vision Engine
    ├── video_processor.py      # Main Pipeline Orchestrator & Decimation Engine
    ├── detector.py             # Custom YOLO Detector & ByteTrack Integration
    ├── person_manager.py       # Lifetime Target Session Aggregator
    ├── telemetry.py            # Drone GPS Log Interpolation Engine
    ├── screenshot_manager.py   # Crop Quality Selection & Snapshot Exporter
    ├── json_writer.py          # Local Output Analytics Writer
    └── reid/                   # OSNet Deep Re-Identification Subsystem
        ├── config.py           # Re-ID Subsystem Settings (Threshold = 0.68)
        ├── identity_manager.py # Gallery Cosine Matching & Identity Search Brain
        ├── embedding_database.py# Thread-Safe Embedding Memory Database
        └── reid_model.py       # Batched PyTorch OSNet Feature Extractor
```

<br/>

## 📋 File-by-File Responsibility Matrix

| File Path | Module Category | Technical Functionality |
| :--- | :--- | :--- |
| 🛠️ `app/worker/worker.py` | **Background Worker** | Autonomous polling event loop, downloads assets, runs ML pipeline, uploads snapshots to Azure Storage, and notifies backend API server. |
| 📡 `app/worker/api_client.py` | **REST API Client** | Handles `GET /detection/pending` queries and `PUT /detection/update/{task_id}` completion status callbacks. |
| 🎬 `app/pipeline/video_processor.py` | **Pipeline Orchestrator** | Manages frame decimation (10 FPS), YOLO detection, ByteTrack tracking, Re-ID extraction, and telemetry binding. |
| 🔍 `app/pipeline/detector.py` | **Detector & Tracker** | Loads custom YOLO weights (`best.pt`) for person detection (`class_id=0`) and executes ByteTrack multi-object tracking. |
| 🧠 `app/pipeline/reid/identity_manager.py` | **Re-ID Core Engine** | Fast-path lookup, gallery cosine similarity search, embedding updates, and stable Person ID assignment. |
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
| **Embedding EMA Alpha** | `0.2` | Exponential moving average update rate for gallery embeddings |
| **Identity TTL** | `None` (Disabled) | Preserves person gallery identities across full video length |
| **File Sizes** | Bytes (Raw Integer) | Measurement unit for evidence file and directory size tracking |

<br/>

## 📦 Completion API Payload JSON Schema

```json
{
  "status": "COMPLETED",
  "results": [
    {
      "person_id": "Person_001",
      "total_duration": 14.5,
      "reidentification_count": 1,
      "evidence": {
        "file_name": "Person_001_sec_10",
        "file_size": 245120,
        "file_url": "https://pdrlimage.blob.core.windows.net/user-1022c811-4f72-4254-81e2-8f9762d2b237/threatimages/flight_123/Person_001_sec_10.jpg"
      },
      "appearances": [
        {
          "appearance_id": 1,
          "entry_time": "00:00:00",
          "exit_time": "00:00:08",
          "duration": 8.0,
          "confidence": 0.88,
          "entry_location": { "lat": 18.52043, "lon": 73.85674, "alt": 45.2 },
          "exit_location": { "lat": 18.52046, "lon": 73.85677, "alt": 45.2 }
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
pip install azure-storage-blob requests opencv-python python-docx
```

### 2. Run Worker & Local Operator Interface
```bash
python ml_pipeline/app/app.py
```
Open `http://localhost:5000` in your web browser to submit jobs manually or view local progress.

---

<div align="center">

*Developed for High-Precision Drone Surveillance & Automated Person Re-Identification.*

</div>
