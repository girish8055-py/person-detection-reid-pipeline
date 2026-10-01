import base64
import json
import logging
import os
import requests
import requests.exceptions
import shutil
import sys
import threading
import time
from pathlib import Path

# Add parent directory (app/) to sys.path to enable module resolution
_app_dir = str(Path(__file__).resolve().parent.parent)
# Additional imports for VLM image processing
import io
from PIL import Image
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Set OLLAMA_MODELS to point directly to local vlm_models folder inside the project
_vlm_models_path = os.path.join(_app_dir, "vlm_models")
if os.path.exists(_vlm_models_path):
    os.environ["OLLAMA_MODELS"] = _vlm_models_path

from downloader import download_video, download_file, download_model
from pipeline.telemetry import parse_video_start_epoch
from pipeline.utils import get_severity_from_label, resolve_severity, ensure_detailed_suggested_action, limit_text_words

try:
    from worker.api_client import (
        fetch_pending_job,
        update_status,
        update_completed,
        update_failed,
        fetch_active_model_url,
        TEST_BASE_URL,
        PROD_BASE_URL,
    )
    from worker.result_builder import build_result_json
except ModuleNotFoundError:
    from api_client import (
        fetch_pending_job,
        update_status,
        update_completed,
        update_failed,
        fetch_active_model_url,
        TEST_BASE_URL,
        PROD_BASE_URL,
    )
    from result_builder import build_result_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weapon/threat guardrail: the VLM sometimes describes a gun in the image but
# still labels the scene "LOW". These keywords are scanned in the VLM's own
# description/action text and force-escalate the severity so an armed subject
# is NEVER reported as low threat.
# ---------------------------------------------------------------------------
FIRING_KEYWORDS = (
    "firing", "shoot", "shot", "muzzle flash", "discharging",
    "opening fire", "opens fire", "gunfire",
)
WEAPON_KEYWORDS = (
    "gun", "rifle", "pistol", "revolver", "shotgun", "firearm",
    "weapon", "armed with", "holding a", "carrying a", "ak-47", "ak47",
    "ak-74", "m4 ", "m16", "machine gun", "assault", "ammunition",
    "ammo", "magazine", "holster", "bullet",
)
# Words that contain weapon keywords but mean the opposite/nothing dangerous.
_WEAPON_FALSE_FRIENDS = ("unarmed", "alarmed", "harmed", "swarmed", "forearmed")


def enforce_weapon_escalation(description, action, severity, target_id="N/A"):
    """
    Safety net applied to every VLM verdict. If the VLM's own words mention
    weapons or active firing while the assigned severity is lower than the
    threat demands, escalate it:

        active firing / muzzle flash  -> CRITICAL (level 4)
        any visible/mentioned weapon  -> at least HIGH   (level 3)

    Returns the (possibly escalated) severity label string.
    """
    blob = f"{description} {action} {severity}".lower()
    for friend in _WEAPON_FALSE_FRIENDS:
        blob = blob.replace(friend, "")

    current = get_severity_from_label(severity)
    current_level = current[0] if current else 1

    if any(kw in blob for kw in FIRING_KEYWORDS):
        needed_level, needed_label, why = 4, "CRITICAL", "active firing detected"
    elif any(kw in blob for kw in WEAPON_KEYWORDS):
        needed_level, needed_label, why = 3, "HIGH", "weapon detected"
    else:
        return severity

    if current_level < needed_level:
        print(
            f"[VLM ENGINE] THREAT ESCALATION Target ID {target_id}: "
            f"severity '{severity}' -> '{needed_label}' ({why} in VLM analysis)."
        )
        logger.warning(
            "Threat escalation for %s: '%s' -> '%s' (%s)",
            target_id, severity, needed_label, why,
        )
        return needed_label

    return severity


def get_vlm_behavior_description_and_action(person_result, keyframe_paths):
    """
    Sends keyframe images and telemetry metadata of a target detection to
    a local VLM (llava) via Ollama, requesting a visual summary, recommended
    action and a severity assessment based on the visual analysis.
    """
    ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    
    images_b64 = []
    max_frames = 3
    for kf_path in keyframe_paths[:max_frames]:
        if os.path.exists(kf_path):
            try:
                # Resize image to max 512x512 for efficient transmission
                with Image.open(kf_path) as img:
                    img.thumbnail((512, 512))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            except Exception as e:
                logger.error("Failed to process keyframe %s: %s", kf_path, e)
        else:
            logger.warning("Keyframe path does not exist: %s", kf_path)

    target_id = person_result.get('id_number', 'N/A')
    target_class = person_result.get('class', 'target')

    logger.info("Calling local VLM (LLaVA) for Target ID %s (%s) with %d keyframe image(s)...", 
                target_id, target_class, len(images_b64))
    print(f"\n[VLM ENGINE] Requesting VLM analysis for Target ID {target_id} ({target_class}) with {len(images_b64)} keyframe(s)...")

    # Telemetry summary for the prompt
    duration = person_result.get('total_duration', 0)
    entry_t = person_result.get('entry_time', 'N/A')
    exit_t = person_result.get('exit_time', 'N/A')
    recon_count = person_result.get('reidentification_count', 0)
    appearances = person_result.get('appearances', [])

    prompt = f"""You are a Military Intelligence Analyst examining aerial surveillance frames of a detected {target_class}.

CRITICAL TARGET FOCUS INSTRUCTIONS:
- You are provided with sequential frames (sampled 1 frame every 2 seconds) of a detected target ({target_class}) highlighted inside a RED BOUNDING BOX.
- Focus your description EXCLUSIVELY on the single target object highlighted inside the RED BOUNDING BOX.
- Use the surrounding scene context (fences, buildings, vehicles, terrain) to describe the real-world activity, appearance, and movement of the target.
- NEVER write the words "red box", "bounding box", "crop", "red outline", "Target ID", or "Class" anywhere in your text description.
- Describe the subject directly as if observing them in real life (e.g. "A person wearing a dark jacket walking near the perimeter fence").
- Ignore all other people, vehicles, or objects not inside the target's bounding box.

YOUR TASK:
Provide a RICH, NATURAL SITUATIONAL DESCRIPTION (MAX 30 WORDS) of ONLY the physical visual appearance, movement, and behavior of the target {target_class}.

NON-NEGOTIABLE DESCRIPTION RULES:
- DO NOT mention "red box", "bounding box", "red outline", "crop", "Target ID", "Class", or timestamps in the description text itself.
- Describe ONLY the target object's real-world appearance and behavior directly.

STEP 1 — MANDATORY WEAPON & THREAT SCAN:
- Scan the target for WEAPONS: firearms, rifles, pistols, RPGs, explosives, knives.
- Scan for HOSTILE ACTS or MILITARY INDICATORS: tactical gear, uniforms, body armour, aiming/firing pose.

STEP 2 — DECISION & ACTION:
- Description: 1 concise sentence (MAX 30 WORDS) detailing the exact physical appearance, color, type, environment, and visual activity of the target object.
- Suggested Action: <ACTION_VERB> - <tactical order based on target activity, MAX 30 WORDS> (Verb must be MONITOR, TRACK, ALERT, or DETAIN).
- Severity: LOW, MEDIUM, HIGH, or CRITICAL based on threat/weapon presence.

RESPOND IN VALID JSON ONLY:
{{"description":"<rich situational description of ONLY the target object, max 30 words>","suggested_action":"<ACTION_VERB> - <tactical command, max 30 words>","severity":"<LOW/MEDIUM/HIGH/CRITICAL>"}}
"""

    payload = {
        "model": "qwen2.5vl:3b",
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "format": "json",
        # Low temperature -> consistent, deterministic threat assessments
        # instead of creative variation between runs.
        "options": {"temperature": 0.2},
    }

    # Retry VLM up to 3 times on failure
    last_error = None
    for attempt in range(3):
        try:
            if attempt > 0:
                print(f"[VLM ENGINE] Retrying VLM call for Target ID {target_id} (attempt {attempt + 1})...")
            response = requests.post(ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            raw_resp = response_data.get("response", "{}")
            try:
                model_output = json.loads(raw_resp)
            except Exception:
                model_output = {}
            if not isinstance(model_output, dict):
                model_output = {}
            verified_cls = target_class
            desc = model_output.get("description", "").strip()
            act = (
                model_output.get("suggested_action")
                or model_output.get("action_taken")
                or ""
            )
            act = str(act).strip()
            sev_raw = model_output.get("severity", "")
            if desc:
                # Guardrail: never let an armed/firing subject be reported
                # as a low threat, even if the VLM misjudges it.
                final_severity = enforce_weapon_escalation(str(desc), act, str(sev_raw).strip(), target_id)
                sev_level, sev_label = resolve_severity(final_severity, recon_count)
                act = ensure_detailed_suggested_action(act, sev_label, target_id, duration, recon_count, obj_class=verified_cls)
                desc = limit_text_words(desc, 30)
                act = limit_text_words(act, 30)
                print(f"[VLM ENGINE] Target ID {target_id} VLM analysis complete.\n  -> Description: {desc}\n  -> Suggested Action: {act}\n  -> Severity: {sev_label}\n")
                return desc, act, str(verified_cls), sev_label
            else:
                last_error = "VLM returned empty description"
                logger.warning("VLM returned empty description for %s, attempt %d", target_id, attempt + 1)
        except Exception as e:
            last_error = str(e)
            logger.error("VLM call failed for %s (attempt %d): %s", target_id, attempt + 1, e)
            print(f"[VLM ENGINE] VLM call failed for Target ID {target_id} (attempt {attempt + 1}): {e}")

    # ---- Smart fallback when VLM is unavailable ----
    print(f"[VLM ENGINE] VLM unavailable after retries. Using smart fallback for Target ID {target_id}.")

    # Analyze appearances for smarter fallback
    total_appearances = len(appearances)
    # Determine if the target moved by checking if any appearance has more than one coordinate point
    moving = any(len(app.get("coordinates", [])) > 1 for app in appearances)

    # Compute spatial spread of coordinates to differentiate between roaming and loitering
    all_coords = []
    for app in appearances:
        for coord in app.get("coordinates", []):
            all_coords.append(coord)
    if all_coords:
        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        area_spread = span_x * span_y
    else:
        area_spread = 0

    # Thresholds for deciding narrative style (tuned empirically)
    LARGE_AREA = 5000  # pixels squared, adjust as needed

    if total_appearances > 2:
        if moving:
            if area_spread > LARGE_AREA:
                movement_desc = "traversed a wide area"
            else:
                movement_desc = "moved within a confined zone"
        else:
            movement_desc = "appeared intermittently without clear motion"
        desc = f"{target_class.title()} {target_id} {movement_desc} over {duration}s across {total_appearances} sightings."
    elif moving:
        desc = f"{target_class.title()} {target_id} was observed moving from {entry_t} to {exit_t} ({duration}s)."
    else:
        desc = f"{target_class.title()} {target_id} remained in a localized area between {entry_t} and {exit_t} ({duration}s)."

    # Generate a context-aware actionable recommendation based on class, duration, and re‑identifications
    is_vehicle = "vehicle" in target_class.lower() or "vehicle" in str(target_id).lower()
    if is_vehicle:
        if duration > 120 or recon_count > 3:
            act = f"ALERT — Vehicle {target_id} loitering in sector for {int(duration)}s across {recon_count} sightings. Dispatch mobile patrol to inspect."
        elif duration > 60:
            act = f"TRACK — Vehicle {target_id} active for {int(duration)}s in area. Maintain continuous optical sensor tracking."
        elif recon_count > 2:
            act = f"TRACK — Vehicle {target_id} re-identified {recon_count} times. Monitor movement path across sector."
        else:
            act = f"MONITOR — Vehicle {target_id} observed briefly for {int(duration)}s. Maintain routine passive surveillance."
    else:
        if duration > 120 or recon_count > 3:
            act = f"ALERT — Target {target_id} persisted for {int(duration)}s with {recon_count} re-appearances. Alert perimeter security and dispatch patrol."
        elif duration > 60:
            act = f"TRACK — Target {target_id} active for {int(duration)}s. Continue tracking movement across sector."
        elif recon_count > 2:
            act = f"TRACK — Target {target_id} re-identified {recon_count} times. Log keyframes and track position."
        else:
            act = f"MONITOR — Target {target_id} observed briefly ({int(duration)}s). Maintain standard surveillance."

    # Fallback severity based on ReID count and duration when VLM is unavailable (never null)
    sev_level, sev_label = resolve_severity(None, recon_count)
    act = ensure_detailed_suggested_action(act, sev_label, target_id, duration, recon_count, obj_class=target_class)
    desc = limit_text_words(desc, 30)
    act = limit_text_words(act, 30)
    return desc, act, target_class, sev_label


def upload_to_azure(file_path, filename, flight_id="default", sas_url=None, base_url=None):
    azure_filename = filename if filename.lower().endswith((".jpg", ".jpeg")) else f"{filename}.jpg"
    if not base_url:
        base_url = PROD_BASE_URL
    fallback_url = f"{base_url}/uploads/threatimages/{flight_id}/{azure_filename}"
    
    # Priority: Environment variable override > server task sas_url > default fallback
    env_sas = os.getenv("AZURE_CONTAINER_SAS_URL")
    if env_sas:
        sas_url = env_sas
    elif not sas_url:
        sas_url = os.getenv(
            "AZURE_CONTAINER_SAS_URL",
            "https://pdrlimage.blob.core.windows.net/user-1022c811-4f72-4254-81e2-8f9762d2b237?sv=2026-06-06&st=2026-07-18T21%3A59%3A09Z&se=2026-07-20T12%3A19%3A09Z&sr=c&sp=cw&sig=cd0wCdPPDTiHTFArrxQ2Szak2b64TEtshg4DbpXgIe0%3D"
        )
    
    try:
        parts = sas_url.split("?")
        container_url = parts[0].rstrip("/")
        sas_token = "?" + parts[1] if len(parts) > 1 else ""
        
        # Append /threatimages/{flightid}/{filename}.jpg inside the URL
        upload_url = f"{container_url}/threatimages/{flight_id}/{azure_filename}{sas_token}"
        
        with open(file_path, "rb") as data:
            response = requests.put(
                upload_url,
                data=data,
                headers={"x-ms-blob-type": "BlockBlob"},
                timeout=60
            )
            response.raise_for_status()
            
        return f"{container_url}/threatimages/{flight_id}/{azure_filename}"
    except Exception as e:
        if "403" in str(e):
            logger.warning("[AZURE SAS EXPIRED] Azure SAS Token authentication failed (403 Forbidden). The SAS token sent by the backend server is expired. Using fallback URL.")
        else:
            logger.warning("Failed to upload to Azure Container SAS (%s). Using fallback URL.", e)
        return fallback_url


SERVERS_CONFIG = [
    ("Test Server", TEST_BASE_URL),
    ("Production Server", PROD_BASE_URL),
]


def worker_loop(app_module, server_name="Auto Server Queue", base_url=None):
    logger.info("[%s] Automatic Multi-Server Pipeline Worker Started", server_name)

    while True:
        task_id = None
        video_path = None
        csv_path = None
        flightdata_path = None
        active_server_name = None
        active_base_url = None

        try:
            if app_module.processing_lock.locked():
                time.sleep(5)
                continue

            # Auto-poll: check all configured servers for an available pending job
            job = None
            for s_name, s_url in SERVERS_CONFIG:
                job = fetch_pending_job(base_url=s_url)
                if job is not None:
                    active_server_name = s_name
                    active_base_url = s_url
                    break

            if job is None:
                time.sleep(5)
                continue

            server_name = active_server_name
            base_url = active_base_url
            task_id = job["task_id"]
            flight_id = job["flight_id"]

            logger.info(
                "[%s] Auto-pulled pending job from server: Task ID = %s",
                server_name,
                task_id
            )

            # -----------------------------
            # Update Status
            # -----------------------------

            update_status(
                task_id,
                "IN_PROGRESS",
                base_url=base_url
            )

            # -----------------------------
            # Download Video & Telemetry
            # -----------------------------

            try:
                video_path = download_video(
                    job["video_url"],
                    task_id
                )
            except RuntimeError as video_err:
                # Check if this is a network connectivity error (server down, refused, timeout).
                # In that case, reset to PENDING so the task can be retried later.
                err_str = str(video_err).lower()
                is_network_error = any(kw in err_str for kw in [
                    "connection refused", "connectionrefused",
                    "failed to establish", "max retries exceeded",
                    "newconnectionerror", "timeout", "timed out",
                    "name or service not known", "no route to host",
                    "connection reset",
                ])
                if is_network_error:
                    logger.warning(
                        "[%s] Task %s: Video download failed due to network error. "
                        "Resetting task back to PENDING for retry. Error: %s",
                        server_name, task_id, video_err
                    )
                    update_status(task_id, "PENDING", base_url=base_url)
                    task_id = None  # Prevent except block from calling update_failed()
                    time.sleep(30)
                    continue
                else:
                    # Non-network error (e.g., 404, corrupt file) — let it fail permanently
                    raise

            csv_path = None
            flightdata_path = None
            video_start_epoch = None
            telemetry_data = job.get("telemetry")

            if telemetry_data is not None:
                logger.info("[%s] Using in-memory telemetry data from job payload.", server_name)
                video_start_epoch = parse_video_start_epoch(video_path=video_path, flightdata_list=job.get("flightdata"))
            elif job.get("csv_url"):
                try:
                    csv_path = download_file(
                        job["csv_url"],
                        f"{task_id}.csv"
                    )
                    # Derive and download flightdata CSV URL by replacing offline_teledata_ with offline_flightdata_
                    flightdata_url = job["csv_url"].replace("offline_teledata_", "offline_flightdata_")
                    flightdata_path = download_file(
                        flightdata_url,
                        f"{task_id}_flightdata.csv"
                    )
                    # Parse video start epoch from video metadata or flightdata
                    video_start_epoch = parse_video_start_epoch(video_path=video_path, flightdata_path=flightdata_path)
                except Exception as telemetry_err:
                    logger.warning("[%s] Failed to download or parse telemetry/flightdata: %s", server_name, telemetry_err)

            # -----------------------------
            # Download Detection Model (if URL provided or active model fetched)
            # -----------------------------
            model_url = job.get("model_url")
            model_name = job.get("model_name")
            if not model_url:
                logger.info("[%s] No model URL in task payload. Fetching active model from server...", server_name)
                model_info = fetch_active_model_url(base_url=base_url)
                if model_info:
                    model_url = model_info.get("download_url")
                    model_name = model_info.get("model_name")

            model_path = None
            if model_url:
                try:
                    url_clean = model_url.split("?")[0]
                    # Determine target local filename: use descriptive model_name if available, otherwise URL basename
                    if model_name:
                        safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in model_name])
                        model_filename = f"{safe_name}.pt"
                    else:
                        model_filename = os.path.basename(url_clean)
                        if not model_filename.lower().endswith(".pt"):
                            model_filename = "custom_model.pt"
                    
                    logger.info("[%s] Downloading detection model from: %s", server_name, model_url)
                    dl_path = download_model(model_url, model_filename)
                    if dl_path and os.path.exists(dl_path) and os.path.getsize(dl_path) > 0:
                        model_path = dl_path
                    else:
                        raise RuntimeError("Downloaded model file is empty or invalid")
                except Exception as model_err:
                    err_msg = f"Failed to download requested detection model from '{model_url}': {model_err}"
                    logger.error("[%s] %s (No fallback allowed).", server_name, err_msg)
                    raise RuntimeError(err_msg)
            else:
                default_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                downloaded_model_path = os.path.join(default_dir, "downloaded_model", "best.pt")
                if os.path.exists(downloaded_model_path):
                    logger.info(
                        "[%s] No custom model URL in task. Using downloaded model: %s",
                        server_name, downloaded_model_path
                    )
                    model_path = downloaded_model_path
                else:
                    msg = f"[{server_name}] Model URL is not present and downloaded model ({downloaded_model_path}) is missing."
                    print(msg)
                    logger.warning(
                        "[%s] Task %s has no model URL and downloaded model 'downloaded_model/best.pt' is missing. "
                        "Resetting task back to PENDING for retry.",
                        server_name, task_id
                    )
                    update_status(task_id, "PENDING", base_url=base_url)
                    task_id = None
                    time.sleep(30)
                    continue

            output_folder = os.path.join(
                app_module.OUTPUT_ROOT,
                task_id
            )

            # -----------------------------
            # Run Detection
            # -----------------------------

            with app_module.processing_lock:

                persons = app_module.processor.process_video(

                    video_path,

                    output_folder,

                    teledata_csv_path=csv_path,

                    telemetry_data=telemetry_data,

                    video_start_epoch=video_start_epoch,

                    model_path=model_path,

                    model_name=model_name

                )

            # -----------------------------
            # -----------------------------
            # Rename Output Files, Gather Size & Upload to Azure
            # -----------------------------
            files_to_upload = {}
            file_sizes = {}
            
            for pk, pval in persons.items():
                # Keep the class-matched prefix (Person_001 / Vehicle_001 / Drone_001)
                raw_id = str(pk)
                det_cls = str(pval.get("class", "person")).strip().capitalize()
                raw_person_id = raw_id if "_" in raw_id else f"{det_cls}_{raw_id}"
                
                # Format time of detection string
                time_str = "unknown"
                det_time = pval.get("entry_detection_time")
                if det_time:
                    time_str = str(det_time).replace(":", "_").replace("+", "_").replace(" ", "_")
                else:
                    entry_time = pval.get("current_entry", 0)
                    time_str = f"sec_{int(entry_time)}"
                
                # Check for the screenshot and rename it (ensure .jpg extension)
                person_dir = os.path.join(output_folder, f"person_{raw_person_id}")
                if not os.path.exists(person_dir):
                    alt_dir = os.path.join(output_folder, raw_person_id.lower())
                    if os.path.exists(alt_dir):
                        person_dir = alt_dir
                    else:
                        alt_dir2 = os.path.join(output_folder, f"person_{pk}")
                        if os.path.exists(alt_dir2):
                            person_dir = alt_dir2
                
                evidence_items = [
                    ("first_detection.jpg", f"{raw_person_id}_first_{time_str}.jpg"),
                    ("middle_detection.jpg", f"{raw_person_id}_middle_{time_str}.jpg"),
                    ("last_detection.jpg", f"{raw_person_id}_last_{time_str}.jpg"),
                    ("best_detection.jpg", f"{raw_person_id}_{time_str}.jpg"),
                ]
                
                uploaded_evidence_map = {}
                primary_file_name = None
                primary_file_size = 0
                primary_file_url = None

                for src_name, target_dest_name in evidence_items:
                    src_file = os.path.join(person_dir, src_name)
                    if not os.path.exists(src_file):
                        continue

                    dest_file = os.path.join(person_dir, target_dest_name)
                    if src_file != dest_file:
                        try:
                            if os.path.exists(dest_file):
                                os.remove(dest_file)
                            os.rename(src_file, dest_file)
                        except Exception as rename_err:
                            dest_file = src_file

                    if os.path.exists(dest_file):
                        f_sz = os.path.getsize(dest_file)
                        try:
                            files_to_upload[target_dest_name] = open(dest_file, "rb")
                        except Exception as open_err:
                            logger.warning("[%s] Could not open file %s for upload: %s", server_name, dest_file, open_err)

                        f_url = upload_to_azure(
                            dest_file,
                            target_dest_name,
                            flight_id=job.get("flight_id", "default"),
                            sas_url=job.get("upload_url"),
                            base_url=base_url
                        )
                        key_name = src_name.replace(".jpg", "")
                        uploaded_evidence_map[key_name] = {
                            "file_name": target_dest_name,
                            "file_size": f_sz,
                            "file_url": f_url
                        }
                        if not primary_file_name or "middle" in key_name or "best" in key_name:
                            primary_file_name = target_dest_name
                            primary_file_size = f_sz
                            primary_file_url = f_url

                if os.path.exists(person_dir):
                    # compute folder size
                    for root_dir, _, filenames in os.walk(person_dir):
                        for f in filenames:
                            fp = os.path.join(root_dir, f)
                            if os.path.exists(fp):
                                folder_size += os.path.getsize(fp)

                # Collect unique evidence blob URLs into an array (list of strings)
                evidence_urls = []
                for k in ["first_detection", "middle_detection", "last_detection", "best_detection"]:
                    if k in uploaded_evidence_map and uploaded_evidence_map[k].get("file_url"):
                        url_val = uploaded_evidence_map[k]["file_url"]
                        if url_val not in evidence_urls:
                            evidence_urls.append(url_val)

                if not evidence_urls and primary_file_url:
                    evidence_urls.append(primary_file_url)

                pval["file_name"] = primary_file_name
                pval["file_size"] = primary_file_size
                pval["file_url"] = primary_file_url
                pval["evidence"] = evidence_urls

                file_sizes[primary_file_name or raw_person_id] = {
                    "file_size": primary_file_size,
                    "folder_size": folder_size
                }

            # -----------------------------
            # Build JSON
            # -----------------------------

            payload = build_result_json(
                persons
            )

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

                # description = VLM's visual observation of the detected object's behaviour.
                # It is always written by VLM from keyframe images in video_processor.py
                # then carried forward here unchanged. It is NEVER overwritten with a template.
                p_obj_class = (p_data.get("class") or person_result.get("class") or "person") if p_data else person_result.get("class", "person")

                if p_data and (p_data.get("description") or p_data.get("severity")):
                    # VLM already ran during video processing — use its description and action directly.
                    person_result["description"] = limit_text_words(p_data.get("description", ""), 30)
                    sev_level, sev_label = resolve_severity(p_data.get("severity"), person_result.get("reidentification_count", 0))
                    person_result["severity_level"] = sev_level
                    person_result["severity"] = sev_label
                    person_result["suggest_action"] = ensure_detailed_suggested_action(
                        p_data.get("suggest_action", ""),
                        sev_label,
                        person_result.get("id_number", "target"),
                        person_result.get("total_duration", 0),
                        person_result.get("reidentification_count", 0),
                        obj_class=p_obj_class
                    )
                else:
                    # VLM not yet called — call it now with the keyframes.
                    description, suggest_action, vlm_class, vlm_severity = get_vlm_behavior_description_and_action(person_result, kf_paths)
                    person_result["description"] = limit_text_words(description, 30)
                    sev_level, sev_label = resolve_severity(vlm_severity, person_result.get("reidentification_count", 0))
                    person_result["severity_level"], person_result["severity"] = sev_level, sev_label
                    person_result["suggest_action"] = ensure_detailed_suggested_action(
                        suggest_action,
                        sev_label,
                        person_result.get("id_number", "target"),
                        person_result.get("total_duration", 0),
                        person_result.get("reidentification_count", 0),
                        obj_class=p_obj_class
                    )

                # Every appearance inherits the person's non-null severity.
                for app_entry in person_result.get("appearances", []):
                    app_entry["severity_level"] = person_result.get("severity_level")
                    app_entry["severity"] = person_result.get("severity")

            # -----------------------------
            # Save Temporary Local Debug JSON
            # -----------------------------
            try:
                debug_payload = {
                    "status": "COMPLETED",
                    "task_id": task_id,
                    "results": payload["results"]
                }
                app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                app_debug_file = os.path.join(app_dir, "admin_output_debug.json")
                with open(app_debug_file, "w", encoding="utf-8") as f:
                    json.dump(debug_payload, f, indent=4)
                logger.info("Saved temporary admin payload debug file to: %s", app_debug_file)
                
                if output_folder and os.path.exists(output_folder):
                    job_debug_file = os.path.join(output_folder, "admin_output_debug.json")
                    with open(job_debug_file, "w", encoding="utf-8") as f:
                        json.dump(debug_payload, f, indent=4)
            except Exception as debug_err:
                logger.warning("Temporary debug JSON write skipped/failed: %s", debug_err)

            # -----------------------------
            # Send Completed
            # -----------------------------

            try:
                update_completed(
                    task_id,
                    payload["results"],
                    files=files_to_upload,
                    file_sizes=file_sizes,
                    base_url=base_url
                )
                logger.info("[%s] Task %s Completed and sent to admin.", server_name, task_id)
            # -----------------------------
            # Cleanup: Delete downloaded video, CSV files, and local output folder after processing.
            # -----------------------------
            finally:
                for f in files_to_upload.values():
                    try:
                        f.close()
                    except Exception:
                        pass

                for path in (video_path, csv_path, flightdata_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            logger.info("[%s] Deleted temporary downloaded file: %s", server_name, path)
                        except Exception as clean_err:
                            logger.warning("[%s] Could not delete temporary file %s: %s", server_name, path, clean_err)

                # Delete local job output directory (keyframes, annotated video, local crops)
                if output_folder and os.path.exists(output_folder):
                    try:
                        shutil.rmtree(output_folder)
                        logger.info("[%s] Cleaned up local job output directory: %s", server_name, output_folder)
                    except Exception as clean_out_err:
                        logger.warning("[%s] Could not delete local output directory %s: %s", server_name, output_folder, clean_out_err)

                # Remove temporary local debug JSON if created
                app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                app_debug_file = os.path.join(app_dir, "admin_output_debug.json")
                if os.path.exists(app_debug_file):
                    try:
                        os.remove(app_debug_file)
                    except Exception:
                        pass

        except Exception as e:

            logger.exception("[%s] Error in worker loop: %s", server_name, e)

            if task_id is not None:
                try:
                    update_failed(
                        task_id,
                        str(e),
                        base_url=base_url
                    )
                except Exception:
                    pass

            time.sleep(5)


def start_worker(app_module):
    worker_thread = threading.Thread(
        target=worker_loop,
        args=(app_module, "Auto Server Queue"),
        daemon=True
    )
    worker_thread.start()
    logger.info("Automatic Multi-Server Pipeline Worker Started (Polling Test & Production Servers)")