# AeroMegh IDMS — Person Detection & Re-Identification ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Custom%20Trained-00FFFF.svg)](https://ultralytics.com)
[![OSNet](https://img.shields.io/badge/ReID-OSNet-green.svg)](https://github.com/KaiyangZhou/deep-person-reid)
[![Azure](https://img.shields.io/badge/Azure-Blob%20Storage-0089D6.svg)](https://azure.microsoft.com/)
[![Documentation](https://img.shields.io/badge/Documentation-DOCX%20Included-blueviolet.svg)](Person_Detection_ReID_Pipeline_Documentation.docx)

An enterprise-grade, microservice-ready computer vision pipeline and background worker system. Designed for high-accuracy person detection, multi-object tracking, deep re-identification across frame absences/re-entries, and drone GPS telemetry fusion.

---

## 📄 Master System Documentation
The complete, end-to-end technical documentation (including directory trees, file-by-file responsibility matrix, API endpoint parameters, Azure SAS snapshot URLs, and JSON payload schemas) is available in Word document format:

👉 [**Download Person_Detection_ReID_Pipeline_Documentation.docx**](Person_Detection_ReID_Pipeline_Documentation.docx)

---

## 🏗️ System Architecture & Workflow

```
+------------------+     +------------------------+     +------------------------+
|  Pending Task    | --> | Frame Decimation Engine| --> |  Custom YOLO Detector  |
|  Polling (API)   |     |    (Target 10 FPS)     |     |   (Bbox Generation)    |
+------------------+     +------------------------+     +------------------------+
                                                                     |
                                                                     v
+------------------+     +------------------------+     +------------------------+
|  Azure Storage   | <-- | Deep Re-ID (OSNet)     | <-- |   ByteTrack Engine     |
| (Blob Uploads)   |     | (Cosine Matching 0.68) |     | (Short-term Track IDs) |
+------------------+     +------------------------+     +------------------------+
         |                                                           
         v                                                           
+------------------+     +------------------------+                  
| Telemetry Fusion | --> | API Completion Update  |                  
| (GPS/Time Align) |     |  (Structured Payloads) |                  
+------------------+     +------------------------+                  
```

---

## 📁 Repository Directory Structure

```
.
├── Person_Detection_ReID_Pipeline_Documentation.docx  # Master Complete Documentation File
├── README.md                                          # Front Page Overview
├── requriment.txt                                     # Python Dependencies
├── ml_pipeline/app/
│   ├── app.py                                         # Flask Management Web UI (http://localhost:5000)
│   ├── downloader.py                                  # Media & Telemetry File Downloader
│   ├── main.py                                        # Standalone CLI Entry Point
│   ├── models/                                        # Deep Learning Checkpoint Storage (YOLO best.pt & OSNet)
│   ├── worker/                                        # Background Worker Service Layer
│   │   ├── api_client.py                              # Backend Server REST API Client
│   │   ├── result_builder.py                          # Schema Parser & Output Builder
│   │   └── worker.py                                  # Autonomous Event Loop & Azure SAS Uploader
│   └── pipeline/                                      # Core Computer Vision Engine
│       ├── video_processor.py                         # Video Orchestrator & Frame Decimator
│       ├── detector.py                                # Custom YOLO & ByteTrack Tracker Module
│       ├── person_manager.py                          # Lifetime Person Session Aggregator
│       ├── telemetry.py                               # Drone GPS Log Interpolator
│       ├── screenshot_manager.py                      # Best Crop Snapshot Generator
│       ├── json_writer.py                             # Local Output Analytics Writer
│       └── reid/                                      # OSNet Deep Re-Identification Subsystem
│           ├── config.py                              # Re-ID Hyperparameters (Similarity = 0.68)
│           ├── identity_manager.py                    # Re-ID Matching & Gallery Search Engine
│           ├── embedding_database.py                  # Thread-Safe Embedding Memory Database
│           └── reid_model.py                          # Batched PyTorch OSNet Feature Extractor
```

---

## ⚙️ Core Modules & File Responsibilities

| File Path | Module Category | Technical Responsibility |
| :--- | :--- | :--- |
| `app/worker/worker.py` | **Background Worker** | Autonomous polling loop, downloads media, runs ML pipeline, uploads snapshots to Azure Storage, and notifies API server. |
| `app/worker/api_client.py` | **REST API Client** | Manages `GET /detection/pending` queries and `PUT /detection/update/{task_id}` completion status callbacks. |
| `app/pipeline/video_processor.py` | **Pipeline Orchestrator** | Handles video frame decimation (10 FPS), object detection, target tracking, Re-ID extraction, and telemetry alignment. |
| `app/pipeline/detector.py` | **Detector & Tracker** | Loads custom YOLO weights (`best.pt`) for person detection (`class_id=0`) and runs ByteTrack tracking. |
| `app/pipeline/reid/config.py` | **Re-ID Settings** | Holds Re-ID hyperparameters (`similarity_threshold=0.68`, `ema_alpha=0.2`, `identity_ttl_seconds=None`). |
| `app/pipeline/reid/identity_manager.py` | **Re-ID Core** | Fast-path lookup, gallery cosine similarity search, embedding updates, and stable Person ID assignment. |
| `app/pipeline/telemetry.py` | **GPS Fusion Engine** | Parses drone CSV logs (`offline_teledata_`, `offline_flightdata_`) and interpolates real-world GPS coordinates per frame. |

---

## 📊 Technical Hyperparameters

- **Detection Model**: Custom-Trained / Fine-Tuned YOLO (`best.pt`)
- **Tracking Algorithm**: ByteTrack (`bytetrack.yaml`)
- **Re-ID Feature Extractor**: OSNet (`osnet_x1_0_market_256x128.pth`)
- **Feature Vector Length**: 512 Float32 L2-Normalized Values
- **Re-ID Cosine Threshold**: `0.68` (Optimized for target re-entry matching)
- **Embedding Update Weight (EMA)**: `0.2` (`ema_alpha`)
- **Identity TTL**: `None` (Preserves identity memory across entire video duration)
- **Snapshot Storage**: Direct Azure Blob Container upload via BlockBlob SAS URLs

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requriment.txt
pip install azure-storage-blob requests opencv-python python-docx
```

### 2. Run the Worker & Local Management Server
```bash
python ml_pipeline/app/app.py
```
Access the operator interface in your browser at `http://localhost:5000`.

---

## 🌐 API Communications Specification

- **Base Server API URL**: `http://114.143.215.196:5005/AeroMegh/idms`
- **Pending Task Request**: `GET /detection/pending`
- **Status Callback Request**: `PUT /detection/update/{task_id}`
- **Azure Blob Upload Path**: `PUT {container_url}/threatimages/{flight_id}/{filename}.jpg` with header `'x-ms-blob-type': 'BlockBlob'`
