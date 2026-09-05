# 📡 API Integration & Background Worker Specification
## Master REST API & Cloud Integration Document

---

## 📑 Executive Summary

This document specifies the REST API contracts, automatic multi-server background worker polling mechanics, Azure Blob Storage upload workflow, disk space cleanup routines, and payload schemas for integrating the **Person Detection, OSNet AIN Re-Identification & VLM Threat Assessment Pipeline** with backend management servers.

---

## 🔄 1. Automatic Multi-Server Worker Architecture (`worker.py`)

The background service operates as an autonomous event loop polling across configured servers:

```
 +-------------------------------------------------------------------------------+
 |                     AUTOMATIC MULTI-SERVER WORKER EVENT LOOP                  |
 +-------------------------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 1. Auto-Poll Servers: GET /detection/pending                  |
         |    (Polls Test & Production servers automatically)             |
         +---------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 2. Download Video File & Telemetry CSV to temporary cache     |
         +---------------------------------------------------------------+
                                         |
                                         v
         +---------------------------------------------------------------+
         | 3. Execute Detection, OSNet AIN Re-ID & VLM Threat Engine    |
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
                                         |
                                         v
         +---------------------------------------------------------------+
         | 6. Automatic Cleanup: Delete Video, CSV, & Output Directory   |
         +---------------------------------------------------------------+
```

---

## 🌐 2. Endpoints & REST API Specifications

### 2.1 Fetch Pending Tasks (`GET /detection/pending`)
- **Server Endpoints**:
  - Test Server: `http://114.143.215.196:5005/AeroMegh/idms/detection/pending`
  - Production Server: `https://idmsapi.aerogcs.com/AeroMegh/idms/detection/pending`
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
- **URL**: `{active_base_url}/detection/update/{task_id}`
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
            "description": "Individual in dark jacket observed walking near perimeter path.",
            "suggest_action": "MONITOR — Keep close watch on target person movements and behavior.",
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
                    "entry_location": { "lat": 18.52043, "lon": 73.85674, "alt": 45.2, "direction": "NE" },
                    "exit_location": { "lat": 18.52046, "lon": 73.85677, "alt": 45.2, "direction": "NE" }
                }
            ]
        }
    ]
}
```

---

## 🧹 3. Automatic Disk Space Cleanup Protocol

When running live on servers, disk space is preserved by executing automated cleanup in a `finally` block:
1. **Downloaded Input Files**: Temporary `.mp4` video downloads and `.csv` telemetry files are deleted immediately after processing.
2. **Local Output Directories**: The local output folder (`outputs/{task_id}/`) containing intermediate keyframe crops, annotated videos, and temporary files is recursively deleted after evidence upload and callback completion.
