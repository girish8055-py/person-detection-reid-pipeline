# 📡 API Integration & Background Worker Specification
## Master REST API & Cloud Integration Document

---

## 📑 Executive Summary

This document specifies the REST API contracts, background worker polling mechanics, Azure Blob Storage upload workflow, and payload schemas for integrating the **Person Detection, Re-Identification & VLM Threat Assessment Pipeline** with central backend management servers.

---

## 🔄 1. Autonomous Worker Architecture (`worker.py`)

The background service operates as an autonomous event loop:

```
 +-------------------------------------------------------------------------------+
 |                           BACKGROUND WORKER EVENT LOOP                         |
 +-------------------------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 1. Query Pending Job: GET /detection/pending                  |
         +---------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 2. Download Video File & Telemetry CSV to local cache         |
         +---------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 3. Execute Detection, OSNet Re-ID & VLM Threat Assessment    |
         +---------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 4. Upload Target Snapshots to Azure Blob Storage (SAS Token)  |
         +---------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 5. Post Completion Callback: PUT /detection/update/{task_id}  |
         +---------------------------------------------------------------+
```

---

## 🌐 2. Endpoints & REST API Specifications

### 2.1 Fetch Pending Tasks (`GET /detection/pending`)
- **URL**: `http://<SERVER_IP>:<PORT>/AeroMegh/idms/detection/pending`
- **Method**: `GET`
- **Response HTTP 200 (Task Available)**:
```json
{
    "task_id": "task_987654",
    "flight_id": "flight_123",
    "video_url": "https://storage.provider.com/videos/flight_123.mp4",
    "telemetry_url": "https://storage.provider.com/telemetry/flight_123.csv",
    "sas_url": "https://pdrlimage.blob.core.windows.net/user-container?sv=2026-06-06&..."
}
```
- **Response HTTP 204**: No pending jobs available.

---

### 2.2 Update Job Status / Completion (`PUT /detection/update/{task_id}`)
- **URL**: `http://<SERVER_IP>:<PORT>/AeroMegh/idms/detection/update/{task_id}`
- **Method**: `PUT`
- **Headers**: `Content-Type: application/json`

#### Payload Schema (Success - `COMPLETED`):
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
            "description": "Individual in dark jacket observed loitering near perimeter boundary.",
            "suggest_action": "ALERT — Raise force readiness level and notify perimeter security units regarding Target Person_001.",
            "evidence": {
                "file_name": "Person_001_detection.jpg",
                "file_size": 245120,
                "file_url": "https://pdrlimage.blob.core.windows.net/user-container/threatimages/flight_123/Person_001_detection.jpg"
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
                    "entry_location": {
                        "lat": 18.52043,
                        "lon": 73.85674,
                        "alt": 45.2,
                        "distance": 12.4,
                        "heading": 45.0,
                        "direction": "NE"
                    },
                    "exit_location": {
                        "lat": 18.52046,
                        "lon": 73.85677,
                        "alt": 45.2,
                        "distance": 12.6,
                        "heading": 46.5,
                        "direction": "NE"
                    },
                    "bounding_box": {
                        "x": 320.5,
                        "y": 140.2,
                        "width": 85.0,
                        "height": 190.4
                    },
                    "frame_range": {
                        "start": 20,
                        "end": 100
                    }
                }
            ]
        }
    ]
}
```

---

## ☁️ 3. Azure Blob Storage Upload Integration

Target evidence crop images (`Person_001_detection.jpg`) are automatically uploaded to Azure Blob Storage using container SAS authorization tokens.

- **Storage Target Path**:
  `https://<blob-account>.blob.core.windows.net/<container>/threatimages/<flight_id>/<filename>.jpg`
- **HTTP Method**: `PUT`
- **Required Header**: `x-ms-blob-type: BlockBlob`
- **Fallback URL**: If Azure upload fails or SAS token expires, the system gracefully logs the failure and generates a formatted fallback URL to ensure pipeline continuity.

---

## 🛡️ 4. Payload Field Guarantee: Zero Null Severities & Detailed Actions

As guaranteed by the worker engine and result builder (`result_builder.py`):
1. **`severity_level`**: Always an integer `[1, 4]`. Never `null`.
2. **`severity`**: Always a string `["LOW", "MEDIUM", "HIGH", "CRITICAL"]`. Never `null`.
3. **`suggest_action`**: Always a complete tactical directive (e.g. `"MONITOR — Maintain passive surveillance..."`). Never a single-word string.
4. **`description`**: Always a visual summary sentence (VLM or spatial trajectory fallback).
