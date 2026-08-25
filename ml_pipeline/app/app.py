"""
app.py

Web UI:
  1. Paste a public video URL.
  2. Click "Start".
  3. Server downloads the video locally, then runs the full detection +
     ReID + person-tracking pipeline on it.
  4. Page polls for progress and shows results (unique persons + durations)
     when done.

Run:
    python app.py
Then open http://localhost:5000 in a browser (or http://<your-ip>:5000
from another machine on the same network).

Only one video processes at a time (simple global lock) -- this matches
"paste a URL, hit Start, whole process happens" from a single operator.
If your manager needs multiple concurrent jobs later, swap the global lock
for a proper job queue (e.g. Celery / RQ) -- flag it to me and I'll build
that version.
"""

import logging
import os
import threading
import uuid

import requests
from flask import Flask, jsonify, render_template_string, request

from pipeline.video_processor import VideoProcessor
from pipeline.utils import get_severity_from_label, resolve_severity, ensure_detailed_suggested_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
REID_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "upload")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB safety cap

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB, matches MAX_DOWNLOAD_BYTES below

# Models load ONCE at startup, reused across every job.
processor = VideoProcessor(
    model_path=MODEL_PATH,
    reid_weights_path=REID_WEIGHTS_PATH,
    confidence=0.5,
    target_fps=10,
)

# ---------------------------------------------------------------------- #
# In-memory job store (fine for single-operator use; swap for Redis/DB if
# this needs to survive server restarts or scale to multiple workers)
# ---------------------------------------------------------------------- #
jobs = {}
jobs_lock = threading.Lock()
processing_lock = threading.Lock()  # ensures only one video processes at a time


def _set_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def _download_file(url: str, dest_path: str, job_id: str, label: str = "file") -> None:
    _set_job(job_id, status="downloading", progress=0.0, message=f"Downloading {label}...")

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)

            if downloaded > MAX_DOWNLOAD_BYTES:
                raise RuntimeError(f"{label} exceeds max allowed size (2GB).")

            if total > 0:
                pct = min(99.0, 100.0 * downloaded / total)
                _set_job(job_id, progress=pct, message=f"Downloading {label}... {downloaded // (1024*1024)}MB")


def _run_job(job_id: str, url: str, csv_url: str = None):
    """URL path: download the video (and optional telemetry CSV) first, then hand off to the shared processor."""
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp4")
    csv_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.csv") if csv_url else None

    try:
        _download_file(url, video_path, job_id, label="video")
        if csv_url:
            _download_file(csv_url, csv_path, job_id, label="telemetry CSV")
    except Exception as e:
        logger.exception("Job %s failed during download", job_id)
        _set_job(job_id, status="error", message=f"Download failed: {e}")
        for p in (video_path, csv_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return

    _run_job_local_file(job_id, video_path, csv_path=csv_path)


def _run_job_local_file(job_id: str, video_path: str, csv_path: str = None):
    """
    Shared by both entry points (URL-download and direct-upload) once the
    video (and optional telemetry CSV) is already sitting on local disk.
    """
    output_folder = os.path.join(OUTPUT_ROOT, job_id)

    # Determine model to use: check downloaded_model/ for the latest downloaded model, fallback to models/best.pt
    downloaded_dir = os.path.join(BASE_DIR, "downloaded_model")
    chosen_model = None
    if os.path.exists(downloaded_dir):
        pt_files = [
            os.path.join(downloaded_dir, f)
            for f in os.listdir(downloaded_dir)
            if f.endswith(".pt") and os.path.isfile(os.path.join(downloaded_dir, f))
        ]
        if pt_files:
            chosen_model = max(pt_files, key=os.path.getmtime)

    if not chosen_model:
        chosen_model = MODEL_PATH

    try:
        with processing_lock:
            _set_job(job_id, status="processing", progress=0.0, message=f"Running detection pipeline with model {os.path.basename(chosen_model)}...")

            def progress_cb(pct, msg):
                _set_job(job_id, progress=pct, message=msg)

            persons = processor.process_video(
                video_path,
                output_folder,
                progress_callback=progress_cb,
                teledata_csv_path=csv_path,
                model_path=chosen_model,
                model_name=os.path.basename(chosen_model),
            )
            from worker.result_builder import build_result_json
            from worker.worker import get_vlm_behavior_description_and_action

            payload = build_result_json(persons)

            # Enrich results with VLM description and action recommendation
            first_person_data = list(persons.values())[0] if persons else {}
            for person_result in payload["results"]:
                p_id_str = person_result.get("id_number")
                p_id = None
                if p_id_str:
                    try:
                        p_id = int(p_id_str)
                    except ValueError:
                        p_id = p_id_str
                
                p_data = persons.get(p_id) if p_id is not None else None
                if not p_data:
                    p_data = first_person_data
                
                kf_paths = p_data.get("keyframes", []) if p_data else []

                # Severity/description/action are decided by VLM or fallback
                if p_data and (p_data.get("description") or p_data.get("severity")):
                    person_result["description"] = p_data.get("description", "")
                    vlm_class = p_data.get("class")
                    if vlm_class and vlm_class != person_result.get("class"):
                        person_result["class"] = vlm_class
                    sev_level, sev_label = resolve_severity(p_data.get("severity"), person_result.get("reidentification_count", 0))
                    person_result["severity_level"] = sev_level
                    person_result["severity"] = sev_label
                    person_result["suggest_action"] = ensure_detailed_suggested_action(
                        p_data.get("suggest_action", ""),
                        sev_label,
                        person_result.get("id_number", "target"),
                        person_result.get("total_duration", 0),
                        person_result.get("reidentification_count", 0)
                    )
                else:
                    description, suggest_action, vlm_class, vlm_severity = get_vlm_behavior_description_and_action(person_result, kf_paths)
                    person_result["description"] = description
                    if vlm_class and vlm_class != person_result.get("class"):
                        person_result["class"] = vlm_class

                    sev_level, sev_label = resolve_severity(vlm_severity, person_result.get("reidentification_count", 0))
                    person_result["severity_level"], person_result["severity"] = sev_level, sev_label
                    person_result["suggest_action"] = ensure_detailed_suggested_action(
                        suggest_action,
                        sev_label,
                        person_result.get("id_number", "target"),
                        person_result.get("total_duration", 0),
                        person_result.get("reidentification_count", 0)
                    )

                # Every appearance inherits the person's non-null severity.
                for app_entry in person_result.get("appearances", []):
                    app_entry["severity_level"] = person_result.get("severity_level")
                    app_entry["severity"] = person_result.get("severity")

            print(payload)
        # Summary shown in the UI: use the non-null severity from payload
        vlm_severity_by_id = {
            r.get("id_number"): (r.get("severity_level"), r.get("severity"))
            for r in payload["results"]
        }
        summary = []
        for p in persons.values():
            sev_level, sev_label = vlm_severity_by_id.get(
                str(p["person_id"]),
                resolve_severity(p.get("severity"), p.get("reidentification_count", 0))
            )
            summary.append({
                "person_id": p["person_id"],
                "total_duration": p["total_duration"],
                "reidentification_count": p["reidentification_count"],
                "severity_level": sev_level,
                "severity": sev_label,
                "appearances": len(p["appearances"]),
            })

        _set_job(
            job_id,
            status="done",
            progress=100.0,
            message=f"Done. {len(summary)} unique person(s) found.",
            results=summary,
            payload=payload,
            output_folder=output_folder,
        )

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _set_job(job_id, status="error", message=str(e))

    finally:
        # Clean up video + csv to save disk space. Comment this out if you
        # want to keep raw source files for later re-processing.
        for p in (video_path, csv_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/start", methods=["POST"])
def start():
    body = request.json or {}
    url = body.get("url", "").strip()
    csv_url = body.get("csv_url", "").strip() or None

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if processing_lock.locked():
        return jsonify({"error": "Another video is currently processing. Please wait."}), 409

    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0.0, "message": "Queued", "url": url, "csv_url": csv_url}

    thread = threading.Thread(target=_run_job, args=(job_id, url, csv_url), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/start_url_with_csv", methods=["POST"])
def start_url_with_csv():
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if processing_lock.locked():
        return jsonify({"error": "Another video is currently processing. Please wait."}), 409

    job_id = uuid.uuid4().hex[:12]
    csv_path = None
    csv_file = request.files.get("csv")
    if csv_file is not None and csv_file.filename != "":
        csv_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.csv")
        csv_file.save(csv_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "url": url,
            "csv_filename": csv_file.filename if csv_file else None,
        }

    def _run_url_job_with_local_csv():
        video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp4")
        try:
            _download_file(url, video_path, job_id, label="video")
        except Exception as e:
            logger.exception("Job %s failed during video download", job_id)
            _set_job(job_id, status="error", message=f"Download failed: {e}")
            for p in (video_path, csv_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            return
        _run_job_local_file(job_id, video_path, csv_path=csv_path)

    thread = threading.Thread(target=_run_url_job_with_local_csv, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/start_upload", methods=["POST"])
def start_upload():
    """
    Direct file upload -- no external hosting site involved. The browser
    sends the video (and optional telemetry CSV) straight to this Flask
    server as multipart/form-data.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video file selected"}), 400

    if processing_lock.locked():
        return jsonify({"error": "Another video is currently processing. Please wait."}), 409

    job_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    file.save(video_path)  # Flask streams this to disk, no extra buffering needed

    csv_path = None
    csv_file = request.files.get("csv")
    if csv_file is not None and csv_file.filename != "":
        csv_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.csv")
        csv_file.save(csv_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "filename": file.filename,
            "csv_filename": csv_file.filename if csv_file else None,
        }

    thread = threading.Thread(target=_run_job_local_file, args=(job_id, video_path), kwargs={"csv_path": csv_path}, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)


@app.route("/video/<job_id>")
def get_video(job_id):
    from flask import send_from_directory
    folder = os.path.join(OUTPUT_ROOT, job_id)
    return send_from_directory(folder, "detection_annotated.mp4", mimetype="video/mp4")


# ---------------------------------------------------------------------- #
# Minimal single-page UI (kept inline so this is a single ready-to-run file)
# ---------------------------------------------------------------------- #
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Drone Video Person Detection</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; background:#0f1115; color:#e6e6e6; }
    h1 { font-size: 1.3rem; }
    input[type=text] { width: 100%; padding: 10px; font-size: 1rem; box-sizing: border-box; border-radius: 6px; border: 1px solid #333; background:#1a1d24; color:#e6e6e6; }
    button { margin-top: 12px; padding: 10px 20px; font-size: 1rem; cursor: pointer; border-radius: 6px; border: none; background:#4f8cff; color:white; }
    button:disabled { background:#555; cursor: not-allowed; }
    #status { margin-top: 20px; padding: 14px; border-radius: 8px; background:#1a1d24; display:none; }
    .bar-bg { background:#333; border-radius: 6px; overflow:hidden; height: 10px; margin-top: 8px; }
    .bar-fill { background:#4f8cff; height: 100%; width:0%; transition: width 0.3s; }
    table { width:100%; border-collapse: collapse; margin-top: 16px; }
    td, th { text-align:left; padding: 6px 8px; border-bottom: 1px solid #333; font-size: 0.9rem; }
    .err { color:#ff6b6b; }
    .tabs { display:flex; gap:8px; margin-bottom: 12px; }
    .tab { padding: 8px 14px; border-radius: 6px; cursor:pointer; background:#1a1d24; border:1px solid #333; }
    .tab.active { background:#4f8cff; color:white; border-color:#4f8cff; }
    .panel { display:none; }
    .panel.active { display:block; }
    input[type=file] { width: 100%; padding: 10px; background:#1a1d24; border:1px solid #333; border-radius: 6px; color:#e6e6e6; }
  </style>
</head>
<body>
  <h1>Drone Video &mdash; Person Detection &amp; ReID</h1>

  <div class="tabs">
    <div class="tab active" id="tabUpload" onclick="switchTab('upload')">Upload from computer</div>
    <div class="tab" id="tabUrl" onclick="switchTab('url')">Paste a URL</div>
  </div>

  <div class="panel active" id="panelUpload">
    <label style="font-size:0.85rem; opacity:0.8;">Video file</label>
    <input type="file" id="videoFile" accept="video/*">
    <label style="font-size:0.85rem; opacity:0.8; margin-top:10px; display:block;">Telemetry CSV (optional -- offline_teledata_*.csv, for GPS tagging)</label>
    <input type="file" id="csvFile" accept=".csv">
    <br>
    <button id="startUploadBtn" onclick="startUpload()">Start</button>
  </div>

  <div class="panel" id="panelUrl">
    <label style="font-size:0.85rem; opacity:0.8;">Video URL</label>
    <input type="text" id="url" placeholder="https://example.com/video.mp4">
    <label style="font-size:0.85rem; opacity:0.8; margin-top:10px; display:block;">Telemetry CSV File from Computer (optional -- offline_teledata_*.csv)</label>
    <input type="file" id="urlCsvFile" accept=".csv">
    <label style="font-size:0.85rem; opacity:0.8; margin-top:10px; display:block;">OR Telemetry CSV URL (optional)</label>
    <input type="text" id="csvUrl" placeholder="https://example.com/offline_teledata_....csv">
    <br>
    <button id="startBtn" onclick="startJob()">Start</button>
  </div>

  <div id="status">
    <div id="msg">Queued</div>
    <div class="bar-bg"><div class="bar-fill" id="bar"></div></div>
    <div id="results"></div>
  </div>

<script>
let pollTimer = null;

function switchTab(which) {
    document.getElementById('tabUpload').classList.toggle('active', which === 'upload');
    document.getElementById('tabUrl').classList.toggle('active', which === 'url');
    document.getElementById('panelUpload').classList.toggle('active', which === 'upload');
    document.getElementById('panelUrl').classList.toggle('active', which === 'url');
}

async function startUpload() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    if (!file) { alert('Choose a video file first.'); return; }

    const csvInput = document.getElementById('csvFile');
    const csvFile = csvInput.files[0];

    document.getElementById('startUploadBtn').disabled = true;
    document.getElementById('status').style.display = 'block';
    document.getElementById('results').innerHTML = '';
    document.getElementById('msg').textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('video', file);
    if (csvFile) formData.append('csv', csvFile);

    const resp = await fetch('/start_upload', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok) {
        document.getElementById('msg').innerHTML = '<span class="err">' + data.error + '</span>';
        document.getElementById('startUploadBtn').disabled = false;
        return;
    }

    pollTimer = setInterval(() => poll(data.job_id, 'startUploadBtn'), 1500);
}

async function startJob() {
    const url = document.getElementById('url').value.trim();
    if (!url) { alert('Paste a video URL first.'); return; }

    const urlCsvFile = document.getElementById('urlCsvFile').files[0];
    const csvUrl = document.getElementById('csvUrl').value.trim();

    document.getElementById('startBtn').disabled = true;
    document.getElementById('status').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    let resp, data;
    if (urlCsvFile) {
        document.getElementById('msg').textContent = 'Uploading telemetry CSV...';
        const formData = new FormData();
        formData.append('url', url);
        formData.append('csv', urlCsvFile);
        resp = await fetch('/start_url_with_csv', { method: 'POST', body: formData });
        data = await resp.json();
    } else {
        resp = await fetch('/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url, csv_url: csvUrl})
        });
        data = await resp.json();
    }

    if (!resp.ok) {
        document.getElementById('msg').innerHTML = '<span class="err">' + data.error + '</span>';
        document.getElementById('startBtn').disabled = false;
        return;
    }

    pollTimer = setInterval(() => poll(data.job_id, 'startBtn'), 1500);
}

async function poll(jobId, btnId) {
    const resp = await fetch('/status/' + jobId);
    const job = await resp.json();

    document.getElementById('msg').textContent = job.message || job.status;
    document.getElementById('bar').style.width = (job.progress || 0) + '%';

    if (job.status === 'done') {
        clearInterval(pollTimer);
        document.getElementById(btnId).disabled = false;
        renderResults(job.results, job.payload, jobId);
    } else if (job.status === 'error') {
        clearInterval(pollTimer);
        document.getElementById(btnId).disabled = false;
        document.getElementById('msg').innerHTML = '<span class="err">Error: ' + job.message + '</span>';
    }
}

function renderResults(results, payload, jobId) {
    let html = '';
    if (jobId) {
        html += '<h3 style="margin-top:20px; font-size:1rem;">Annotated Detection Video (Verification):</h3>';
        html += `<video controls width="100%" style="border-radius:8px; background:#000; margin-bottom:15px;" src="/video/${jobId}"></video>`;
    }

    if (!results || results.length === 0) {
        html += '<p>No persons detected.</p>';
    } else {
        const SEV_COLORS = {1: '#4caf50', 2: '#ff9800', 3: '#ff5722', 4: '#f44336'};
        html += '<table><tr><th>Detected ID / Target</th><th>Total Duration (s)</th><th>Re-ID Count</th><th>Severity</th><th>Appearances</th></tr>';
        for (const r of results) {
            const sevColor = SEV_COLORS[r.severity_level] || '#777';
            html += `<tr><td>${r.person_id}</td><td>${r.total_duration}</td><td>${r.reidentification_count}</td>`
                + `<td><span style="background:${sevColor};color:#fff;padding:2px 10px;border-radius:10px;font-size:0.8rem;">Level ${r.severity_level} &mdash; ${r.severity}</span></td>`
                + `<td>${r.appearances}</td></tr>`;
        }
        html += '</table>';
    }

    if (payload) {
        html += '<h3 style="margin-top:20px; font-size:1rem;">Admin Payload (JSON Output):</h3>';
        html += '<pre style="background:#111; color:#0f0; padding:12px; border-radius:6px; max-height:300px; overflow:auto; font-size:0.85rem;">' + JSON.stringify(payload, null, 2) + '</pre>';
    }

    document.getElementById('results').innerHTML = html;
}
</script>
</body>
</html>
"""
from worker.worker import start_worker

if __name__ == "__main__":

    start_worker(app_module=__import__(__name__))

    app.run(
        host="0.0.0.0",
        port=5050,
        debug=False
    )