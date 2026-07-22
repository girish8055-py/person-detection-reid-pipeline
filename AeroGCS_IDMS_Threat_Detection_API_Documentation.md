# AeroGCS IDMS
## AI Threat Detection & Threat Management
### Backend API & Workflow Documentation

Covers: flight video capture → detection task creation → Python AI service polling → object detection → automatic threat creation → threat lifecycle management

---

## 1. System Overview

This module automates threat detection on recorded drone flight videos. When a flight ends and its video is uploaded to Azure Blob Storage, a detection task is created and queued. A separate Python AI service polls for pending tasks, downloads the video, runs object detection, uploads evidence images, and reports results back. The backend then converts each detection result into a Threat record with evidence and a full status history, which operators can view, comment on, assign, and manage through the Threats APIs.

### 1.1 End-to-End Flow

1. **Flight Ends & Video Upload**: The drone flight ends, and the recorded video finishes uploading to Azure Blob Storage under the user's container (`DroneStreams/{flightId}/{streamkey}.mp4`).
2. **Stream Key Lookup**: The client calls the Stream Key Lookup API to confirm the upload. This API records the final file size, generates a short-lived read SAS URL for the video and a short-lived write SAS URL (upload container) for evidence images, and creates a Detection Task with status `PENDING`.
3. **AI Task Polling**: The Python AI/object-detection service periodically calls `GET /AeroMegh/idms/detection/pending` to fetch all tasks currently in `PENDING` status, ordered by newest first.
4. **Processing & Upload**: The Python service downloads the video using the task's `video_url`, runs object/person detection, tracking, Re-ID, and telemetry interpolation, and uploads any evidence images (detected frames/snapshots) directly to Azure Blob Storage using the task's `upload_url` (SAS token).
5. **Reporting Results**: The Python service reports results back by calling `PUT /AeroMegh/idms/detection/update/:taskid` with the new status (`COMPLETED` or `FAILED`), an array of detection results (containing coordinates, **approximate distance**, **heading**, and **cardinal direction**), and any optional error messages.
6. **Threat Creation**: For every object in the results array, the backend creates a new Threat record (status `NEW`, source_type `AI_DETECTION`), links it to the flight/drone/detection task, stores its evidence image, and writes an initial entry into the threat status history—all inside a single database transaction.
7. **Task Status Update**: The detection task row is updated with its final status and error message (if any). Raw AI results are not persisted on the task; they only exist as the Threats created from them.
8. **Threat Management**: Operators then use the Threats APIs to list, filter, review, comment on, assign, update, or delete the generated threats.

### 1.2 Detection Task Status Values

| Status | Meaning |
| :--- | :--- |
| **PENDING** | Task created after video upload; waiting to be picked up by the AI service. |
| **COMPLETED** | AI service finished processing and reported results (threats created if any objects were detected). |
| **FAILED** | AI service could not process the video; `error_message` explains the failure. |

### 1.3 Threat Status Values

Threats move through a lifecycle tracked in `threat_status_history` every time status changes (via automatic creation or a manual update through the Update Threat API).

| Status | Typical Meaning |
| :--- | :--- |
| **NEW** | Default status when a threat is first created (manually or by AI detection). |
| **IN_PROGRESS / REVIEW** | Assigned and being investigated by a user or team (project-specific values). |
| **RESOLVED / CLOSED** | Investigation complete; no further action needed. |

---

## 2. Node.js Backend ↔ Python AI Service Interaction Diagram

Below is the step-by-step sequence detailing how the Node.js backend and the Python AI detection service interact through Azure Blob Storage:

```
 Drone / Client App            Node.js Backend (API)          Azure Blob Storage            Python AI Service
         │                             │                              │                             │
         ├─ A. Flight Ends & Video ───►│                              │                             │
         │  Upload (DroneStreams)      │                              │                             │
         │                             │                              │                             │
         ├─ 2. GET stream key lookup ─►│                              │                             │
         │                             ├─ 3. Verify blob, size, ─────►│                             │
         │                             │  and mint SAS tokens         │                             │
         │                             ├─ 5. INSERT PENDING task ────┐│                             │
         │                             │◄────────────────────────────┘│                             │
         │◄─ 6. 200 OK (details) ──────┤                              │                             │
         │                             │                              │                             │
         │                             │◄────── 7. GET pending tasks (polling) ─────────────────────┤
         │                             ├───────────────────────────────────────────────────────────►│
         │                             │        8. 200 OK - task list [ {task_id, video_url...} ]    │
         │                             │                              │                             │
         │                             │                              ├─ 9. Download video ─────────┤
         │                             │                              │◄────────────────────────────┤
         │                             │                              │                             ├─ 10. Run YOLO/Re-ID
         │                             │                              │◄─ 11. Upload crop evidence ─┤
         │                             │                              │   (SAS upload_url)          │
         │                             │                              │                             │
         │                             │◄────── 12. PUT update task (status, results array) ────────┤
         │                             ├─ 13. Create Threat, evidence │                             │
         │                             │  images & status history     │                             │
         │                             ├─ 14. UPDATE status: COMPLETED│                             │
         │                             ├─────────────────────────────── ───────────────────────────►│
         │                             │        15. 200 OK (updatedTask)                            │
```

### 2.1 Phase Summary

* **Phase A: Flight ends & video upload**: Video lands in Blob Storage; Stream Key Lookup confirms the upload, mints SAS URLs, and inserts a `PENDING` detection task.
* **Phase B: AI service polls for pending work**: The Python service repeatedly calls `GET /detection/pending` on its own schedule until it receives tasks to process.
* **Phase C: Detection & evidence capture**: The Python service downloads the video with the SAS `video_url`, runs detection/re-identification, interpolates drone telemetry, and uploads evidence images with the SAS `upload_url`.
* **Phase D: Report results & auto-create threats**: The Python service calls `PUT /detection/update/:taskid`; the backend creates a Threat (with evidence, GPS location, distance, direction, and status history) per result and marks the task `COMPLETED` or `FAILED`.

---

## 3. Referenced Tables

These tables are read from or written to by the APIs documented in this file:

| Table | Purpose |
| :--- | :--- |
| **`flights`** | Flight records; `streamkey`, `blob_container`, `size`, `status`, `end_time`, `drone_id`. |
| **`threat-detection`** | One row per detection task: `flight_id`, `user_id`, `video_url`, `upload_url`, `status`, `error_message`, timestamps. |
| **`threats`** | Core threat records: `threat_code`, `title`, `description`, `priority`, `severity`, `status`, `assignment`, `location`, links to flight/drone/detection task. |
| **`threat_evidence_images`** | Evidence image(s) attached to a threat (`image_url`, `file_name`, `file_size`, `uploaded_by`/`at`). |
| **`threat_status_history`** | Audit trail of every status change on a threat (`old_status` → `new_status`, `changed_by`, `reason`, `changed_at`). |
| **`threat_comments`** | Free-text comments/notes added to a threat by users. |
| **`threat_assignments`** | History of assignment changes (`assigned_user_id` / `assigned_team_id`, `assigned_by`, `note`, `assigned_at`). |

---

## 4. API Reference

Base path: `/AeroMegh/idms` — all endpoints require a valid Bearer token in the headers (`Authorization: Bearer <token>`) unless noted otherwise.

### 4.1 Flight Upload & Detection Task Creation

#### 4.1.1 Stream Key Lookup (creates the Detection Task)

* **Method**: `GET`
* **Endpoint**: `/AeroMegh/idms/flight/stream/:streamKey/:fileSize`
* **Purpose**: Called once the flight's recorded video has finished uploading to Blob Storage. Confirms the flight by streamkey, records the final video size, updates storage usage, generates SAS URLs, and triggers creation of a PENDING detection task for the AI service.

**Path Parameters:**
* `streamKey` (Path, Required): The flight's unique stream key.
* `fileSize` (Path, Required): Size (bytes) of the uploaded video, used to compute/validate storage usage.

**Response Codes:**
* `200`: Flight found and processed; detection task creation attempted (failure to create the task does not fail this request).
* `404`: No flight matches the given stream key.
* `500`: Internal server error.

**Notes:**
* Looks up the flight (and the owning user's `blob_container`) by `streamkey`.
* Confirms the video blob exists at `DroneStreams/{flightId}/{streamkey}.mp4` and reads its actual size from storage.
* Generates a write SAS token (5 min, container-level) — this becomes the detection task's `upload_url`, used later by the Python service to upload evidence images.
* Generates a read-only SAS token (5 min) for the video blob itself — this becomes the detection task's `video_url`.
* Inserts the `PENDING` row into the `threat-detection` table.

---

#### 4.1.2 Get Pending Detection Tasks

* **Method**: `GET`
* **Endpoint**: `/AeroMegh/idms/detection/pending`
* **Purpose**: Polled by the Python AI service to retrieve all detection tasks currently awaiting processing.

**Example Request:**
`GET /AeroMegh/idms/detection/pending`

**Example Response (200 OK):**
```json
{
  "message": "Pending detection tasks fetched successfully.",
  "tasks": [
    {
      "task_id": "b3f1c84d-295e-4b2a-89a1-7c5e5332f102",
      "video_url": "https://storage.blob.core.windows.net/DroneStreams/12/streamkey.mp4?sp=r&st=...",
      "flight_id": 12,
      "status": "PENDING",
      "upload_url": "https://storage.blob.core.windows.net/user-container?sp=w&st=...",
      "created_at": "2026-07-20T10:15:00.000Z",
      "updated_at": "2026-07-20T10:15:00.000Z"
    }
  ]
}
```

**Response Codes:**
* `200`: Returns the list of pending tasks (empty array allowed).
* `204`: No pending tasks found.
* `500`: Internal server error.

---

#### 4.1.3 Update Detection Task (reports AI results, auto-creates Threats)

* **Method**: `PUT`
* **Endpoint**: `/AeroMegh/idms/detection/update/:taskid`
* **Purpose**: Called by the Python AI service once processing finishes, to mark the task `COMPLETED` or `FAILED` and report detected objects. Every entry in results is converted into a new Threat with evidence, inside a single transaction.

**Path Parameters:**
* `taskid` (Path, Required): The ID of the `threat-detection` row to update.

**Request Body Schema:**
* `status` (string, Required): New task status (e.g. `COMPLETED` or `FAILED`).
* `results` (array, Optional): Detected objects/persons. Each item becomes one Threat when present.
* `error_message` (string, Optional): Failure reason; stored on the task if provided.

**Example Request:**
```http
PUT /AeroMegh/idms/detection/update/482
Content-Type: application/json

{
  "status": "COMPLETED",
  "results": [
    {
      "person_id": "Person_1",
      "total_duration": 14.50,
      "reidentification_count": 1,
      "appearances": [
        {
          "appearance_id": 1,
          "entry_time": "00:00:00",
          "exit_time": "00:00:08",
          "duration": 8.0,
          "confidence": 0.88,
          "entry_location": { 
            "lat": 19.9451248, 
            "lon": 73.6006139, 
            "alt": 45.2,
            "distance": 8.45,
            "heading": 82.77,
            "direction": "NE"
          },
          "exit_location": { 
            "lat": 19.9451251, 
            "lon": 73.6006142, 
            "alt": 45.2,
            "distance": 8.50,
            "heading": 82.79,
            "direction": "NE"
          }
        }
      ],
      "evidence": {
        "file_url": "https://pdrlimage.blob.core.windows.net/user-container/threatimages/flight_12/Person_1_sec_10.jpg",
        "file_name": "Person_1_sec_10",
        "file_size": 234011
      }
    }
  ]
}
```

**Response Codes:**
* `200`: Task updated; returns `updatedTask` and the list of `insertedThreats`.
* `404`: No detection task found for the given `taskid`.
* `500`: Internal server error (transaction rolled back).

---

### 4.2 Threats Management

#### 4.2.1 Add New Threat

* **Method**: `POST`
* **Endpoint**: `/AeroMegh/idms/threats/new`
* **Purpose**: Manually create a threat record (in addition to threats auto-created from AI detections).

**Request Body:**
* `title` (string, Required): Short title for the threat.
* `description` (string, Required): Description of the threat.
* `priority` (string, Required): e.g. `LOW` / `MEDIUM` / `HIGH`.
* `severity` (integer, Required): 1–5; validated server-side.
* `source_type` (string, Required): Origin of the threat, e.g. `MANUAL`, `AI_DETECTION`.
* `latitude / longitude` (number, Required): Location of the threat.

**Response Codes:**
* `200`: Threat created successfully.
* `400`: Missing required fields, or invalid severity.
* `500`: Internal server error.

---

#### 4.2.2 Get All Threats

* **Method**: `GET`
* **Endpoint**: `/AeroMegh/idms/threats`
* **Purpose**: List threats created by the authenticated user, with optional filters and pagination.

**Query Parameters:**
* `priority` (Query, Optional): Filter by priority.
* `severity` (Query, Optional): Filter by severity.
* `status` (Query, Optional): Filter by status.
* `pageNumber` (Query, Optional): 1-based page number (default 1).
* `pageSize` (Query, Optional): Page size (default 10).

---

#### 4.2.3 Get Threat Details

* **Method**: `GET`
* **Endpoint**: `/AeroMegh/idms/threats/:id`
* **Purpose**: Fetch a single threat with its full evidence images, status history, and comments.

---

#### 4.2.4 Delete Threat

* **Method**: `DELETE`
* **Endpoint**: `/AeroMegh/idms/threats/delete/:id`
* **Purpose**: Permanently delete a threat record.

---

#### 4.2.5 Update Threat

* **Method**: `PUT`
* **Endpoint**: `/AeroMegh/idms/threats/update/:id`
* **Purpose**: Update one or more fields of an existing threat. Logs a status history entry automatically when status changes.

---

#### 4.2.6 Add Comment to Threat

* **Method**: `POST`
* **Endpoint**: `/AeroMegh/idms/threats/comment/:id`
* **Purpose**: Add a text comment/note to a threat.

---

#### 4.2.7 Assign Threat

* **Method**: `POST`
* **Endpoint**: `/AeroMegh/idms/threats/assign/:id`
* **Purpose**: Assign (or reassign) a threat to a user and/or a team, keeping a full assignment history.

---

### 4.3 Detection Models Management

#### 4.3.1 Add New Detection Model

* **Method**: `POST`
* **Endpoint**: `/AeroMegh/idms/models/addNewModel`
* **Purpose**: Uploads a new AI detection model to Azure Blob storage and saves metadata record in database.

**Request Body (Multipart Form-Data):**
* `model_name` (string, Required): Name of the detection model.
* `file` (file, Required): Binary model file (e.g. `.pt`, `.onnx`, `.engine`).

**Response Codes:**
* `200`: Model uploaded and metadata saved successfully.
* `400`: Missing required fields or invalid file format.
* `500`: Internal server error.

---

#### 4.3.2 Get All Detection Models

* **Method**: `GET`
* **Endpoint**: `/AeroMegh/idms/models`
* **Purpose**: Fetches all uploaded detection models sorted by version number descending.

**Query Parameters (Optional):**
* `is_active` (boolean string, Optional): Filter models by active state (`true` or `false`).

**Response Codes:**
* `200`: Returns list of models successfully.
* `500`: Internal server error.

**Example Response (200 OK):**
```json
{
  "message": "Detection models fetched successfully.",
  "models": [
    {
      "id": 1,
      "model_name": "Threat Detection Model v1",
      "version": "v1",
      "version_number": 1,
      "file_name": "yolov8_model.pt",
      "blob_url": "https://<azure_account>.blob.core.windows.net/detection-models/yolov8_model.pt",
      "file_size": 45123890,
      "is_active": true,
      "created_at": "2026-07-22T17:42:00.000Z",
      "updated_at": "2026-07-22T17:42:00.000Z",
      "sas_token": "<sas_token>",
      "download_url": "https://<azure_account>.blob.core.windows.net/detection-models/yolov8_model.pt?<sas_token>"
    }
  ]
}
```

---

## 5. Route Summary

| Method | Route | Short Description |
| :--- | :--- | :--- |
| **GET** | `/AeroMegh/idms/flight/stream/:streamKey/:fileSize` | Confirms uploaded flight video and triggers the `PENDING` detection task. |
| **GET** | `/AeroMegh/idms/detection/pending` | Returns all detection tasks awaiting picking up by AI. |
| **PUT** | `/AeroMegh/idms/detection/update/:taskid` | Reports results and auto-creates Threats on completion. |
| **POST** | `/AeroMegh/idms/threats/new` | Manually creates a new threat. |
| **GET** | `/AeroMegh/idms/threats` | List/filter/paginate threats. |
| **GET** | `/AeroMegh/idms/threats/:id` | Get full details (evidence, comments, history) of a threat. |
| **DELETE**| `/AeroMegh/idms/threats/delete/:id` | Permanently deletes a threat. |
| **PUT** | `/AeroMegh/idms/threats/update/:id` | Updates threat fields and audits status changes. |
| **POST** | `/AeroMegh/idms/threats/comment/:id` | Adds a comment to a threat. |
| **POST** | `/AeroMegh/idms/threats/assign/:id` | Assigns/reassigns a threat to a user or team. |
| **POST** | `/AeroMegh/idms/models/addNewModel` | Uploads a new AI detection model to Azure Blob and registers in DB. |
| **GET** | `/AeroMegh/idms/models` | Fetches all detection models sorted by version (descending). |
