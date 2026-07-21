# Drone Video Detection & ReID Pipeline

A complete object detection and Person Re-Identification (ReID) pipeline designed to process drone footage, track individuals, and correlate spatial-temporal details with telemetry logs.

---

## 💻 System Components

### 1. Web Application (`app.py`)
- Provides a Web UI for uploading drone video footage (along with optional telemetry CSV files) or submitting public video/telemetry URLs.
- Initiates background processing jobs and allows operators to poll for job progress and view final detection results.

### 2. Background Worker (`worker/worker.py`)
- Automatically polls the server for pending detection tasks.
- Downloads assets locally (video and telemetry CSV files).
- Feeds raw footage through the pipeline to perform object detection (YOLO) and ReID matching (OSNet).
- Renames cropped best-detections of individuals using the format `{person_id}_{time_of_detection}` (e.g., `Person_001_2026-07-20T11_22_25`).
- Uploads screenshots directly to Azure Storage (using the server-provided task SAS URL or default SAS URL) nested under `/threatimages/{flight_id}/{filename}.jpg`.
- Communicates task results back to the server.
- Performs automatic cleanup to save local disk space.

### 3. API Client (`worker/api_client.py`)
- Fetches pending tasks from the API.
- Updates task states (`IN_PROGRESS`, `FAILED`, `COMPLETED`).
- Delivers detection details and file metadata (such as file sizes) to the database/endpoint.

---

## 🚀 Getting Started

### 1. Requirements & Dependencies
Make sure Python dependencies are installed in your active virtual environment (`venv`):
```bash
pip install -r requirements.txt
pip install azure-storage-blob requests opencv-python
```

### 2. Running the Application
To run the Web UI and start the background task polling loop simultaneously:
```bash
python ml_pipeline/app/app.py
```
Then open `http://localhost:5000` in your web browser.

---

## ⚙️ Configuration & Customization

### Local File Deletion (Cleanup Toggle)
By default, the worker deletes raw downloaded videos and CSV files after detection finishes to optimize disk space. You can toggle this behavior in `ml_pipeline/app/worker/worker.py` under the `Cleanup` section:

```python
# To KEEP downloaded files, comment out this loop:
for path in (video_path, csv_path, flightdata_path):
    if path and os.path.exists(path):
        os.remove(path)
```

### Azure Storage Uploads
The worker uploads screenshots to Azure Blob Storage using a dynamic SAS URL supplied by the server. If the server does not supply a URL, it defaults to the configured environment variable `AZURE_CONTAINER_SAS_URL` or a default fallback container URL:
- Blobs are uploaded using native REST requests `PUT` operations to:
  `{container_url}/threatimages/{flight_id}/{filename}.jpg`
- The generated URL is returned inside the final payload under the `evidence` object:
  ```json
  "evidence": {
      "file_name": "Person_001_2026-07-20T11_22_25",
      "file_size": 0.2339,
      "file_url": "https://pdrlimage.blob.core.windows.net/.../threatimages/{flight_id}/Person_001_2026-07-20T11_22_25.jpg"
  }
  ```
