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
    ollama_url = "http://localhost:11434/api/generate"
    
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

    prompt = f"""You are the ARMY CHIEF (Chief of Army Staff) personally reviewing drone surveillance evidence. You have decades of battlefield command experience and you issue clear, decisive tactical orders. Analyze the provided keyframe images of a detected {target_class} as if briefing your field commanders.

SITREP (Situation Report):
TARGET: {target_id} | Class: {target_class}
Duration in area of operations: {duration}s | Observed: {entry_t} to {exit_t} | Re‑appeared: {recon_count} time(s)

STEP 1 — MANDATORY WEAPON & THREAT SCAN (do this FIRST):
- Zoom into every image mentally and scan for WEAPONS: guns, rifles, pistols, revolvers, shotguns, AK/M4/M16 style weapons, machine guns, knives, explosives, rocket launchers.
- Scan for HOSTILE ACTS: aiming a weapon, firing (muzzle flash, smoke, recoil pose), taking cover, dragging a casualty.
- Scan for HOSTILE INDICATORS: military/tactical gear, body armour, helmets, ammunition magazines, holsters, uniformed combatants, attack formation.

STEP 2 — COMMANDER'S DECISION:
- Describe in ONE concise sentence (MAXIMUM 30 WORDS) the target's appearance (clothing, build, vehicle type) and activity — if a weapon is present or being fired, you MUST mention it explicitly in the description.
- AS THE ARMY CHIEF, issue a UNIQUE, SITUATION-SPECIFIC TACTICAL ACTION COMMAND (MAXIMUM 30 WORDS) for your troops based on the visual evidence you observe. DO NOT COPY OR REPEAT GENERIC BOILERPLATE SENTENCES. Provide an action verb (MONITOR/TRACK/ALERT/DETAIN) followed by a hyphen and a specific command tailored to what you see.
- Assess ONE overall THREAT LEVEL using these NON-NEGOTIABLE rules:
    * ANY firearm visible, held, aimed, or fired  -> severity MUST be "CRITICAL". An armed subject is NEVER "LOW" or "MEDIUM". No exceptions.
    * Military/tactical gear or hostile posture without visible weapon -> at least "HIGH".
    * Suspicious conduct only (loitering near assets, concealment, evasive behaviour) -> "HIGH".
    * Unusual but plausibly innocent activity -> "MEDIUM".
    * Routine civilian movement, clearly benign -> "LOW".
- A commander who calls an armed gunman a "LOW" threat gets his soldiers killed. When in doubt, ESCALATE.
- If the target lingers >60s without clear purpose, treat it as potential reconnaissance of our position and raise severity accordingly.

RESPOND IN VALID JSON ONLY (no markdown, no extra text):
{{"verified_class":"{target_class}","description":"<one concise specific sentence description, max 30 words>","suggested_action":"<ACTION_VERB> - <unique tactical command based on visual evidence, max 30 words>","severity":"<LOW/MEDIUM/HIGH/CRITICAL>"}}
"""

    payload = {
        "model": "llava:latest",
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
            verified_cls = model_output.get("verified_class", target_class)
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


def worker_loop(app_module, server_name="Production Server", base_url=PROD_BASE_URL):

    logger.info("[%s] Worker Started", server_name)

    while True:
        task_id = None
        try:

            # Only one job at a time
            if app_module.processing_lock.locked():
                time.sleep(5)
                continue

            job = fetch_pending_job(base_url=base_url)

            if job is None:
                logger.info("[%s] No Pending Jobs", server_name)
                time.sleep(5)
                continue

            task_id = job["task_id"]

            flight_id = job["flight_id"]

            logger.info(
                "[%s] Processing Task : %s",
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
                # No specific model URL in task: use default base model (models/best.pt)
                default_local_model = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models",
                    "best.pt"
                )
                if os.path.exists(default_local_model):
                    logger.info(
                        "[%s] No custom model URL in task. Using base model: %s",
                        server_name, default_local_model
                    )
                    model_path = default_local_model
                else:
                    logger.warning(
                        "[%s] Task %s has no model URL and base model 'models/best.pt' is missing. "
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
                # Keep the class-matched prefix (Person_001 / Vehicle_001)
                raw_id = str(pk)
                det_cls = str(pval.get("class", "person")).strip().capitalize()
                raw_person_id = raw_id if raw_id.lower().startswith(("person_", "vehicle_")) else f"{det_cls}_{raw_id}"
                
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
                
                src_file = os.path.join(person_dir, "best_detection.jpg")
                dest_filename = f"{raw_person_id}_{time_str}.jpg"
                dest_file = os.path.join(person_dir, dest_filename)
                
                file_size = 0
                folder_size = 0
                
                if os.path.exists(src_file):
                    try:
                        if os.path.exists(dest_file):
                            os.remove(dest_file)
                        os.rename(src_file, dest_file)
                        logger.info("[%s] Saved detection screenshot: %s", server_name, dest_file)
                    except Exception as rename_err:
                        logger.warning("[%s] Could not rename %s to %s: %s", server_name, src_file, dest_file, rename_err)
                
                if not os.path.exists(dest_file) and os.path.exists(src_file):
                    dest_file = src_file
                    dest_filename = f"{raw_person_id}_{time_str}.jpg"

                if os.path.exists(dest_file):
                    file_size = os.path.getsize(dest_file)
                    try:
                        files_to_upload[dest_filename] = open(dest_file, "rb")
                    except Exception as open_err:
                        logger.warning("[%s] Could not open file %s for upload: %s", server_name, dest_file, open_err)
                
                if os.path.exists(person_dir):
                    # compute folder size
                    for root_dir, _, filenames in os.walk(person_dir):
                        for f in filenames:
                            fp = os.path.join(root_dir, f)
                            if os.path.exists(fp):
                                folder_size += os.path.getsize(fp)
                
                # Upload to Azure / SAS Upload URL
                file_url = None
                if os.path.exists(dest_file):
                    file_url = upload_to_azure(
                        dest_file,
                        dest_filename,
                        flight_id=job.get("flight_id", "default"),
                        sas_url=job.get("upload_url"),
                        base_url=base_url
                    )
                    logger.info("[%s] Detection screenshot uploaded. URL: %s", server_name, file_url)
                
                pval["file_name"] = dest_filename
                pval["file_size"] = file_size
                pval["file_url"] = file_url
                
                file_sizes[dest_filename] = {
                    "file_size": file_size,
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

                # Severity/description/action are decided by VLM or resolved fallback
                if p_data and (p_data.get("description") or p_data.get("severity")):
                    person_result["description"] = limit_text_words(p_data.get("description", ""), 30)
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
                    vlm_class = p_data.get("class")
                    if vlm_class and vlm_class != person_result.get("class"):
                        person_result["class"] = vlm_class
                else:
                    description, suggest_action, vlm_class, vlm_severity = get_vlm_behavior_description_and_action(person_result, kf_paths)
                    person_result["description"] = limit_text_words(description, 30)
                    sev_level, sev_label = resolve_severity(vlm_severity, person_result.get("reidentification_count", 0))
                    person_result["severity_level"], person_result["severity"] = sev_level, sev_label
                    person_result["suggest_action"] = ensure_detailed_suggested_action(
                        suggest_action,
                        sev_label,
                        person_result.get("id_number", "target"),
                        person_result.get("total_duration", 0),
                        person_result.get("reidentification_count", 0)
                    )
                    if vlm_class and vlm_class != person_result.get("class"):
                        person_result["class"] = vlm_class

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
            finally:
                for f in files_to_upload.values():
                    try:
                        f.close()
                    except Exception:
                        pass

            # -----------------------------
            # Cleanup: Delete video and CSV files after detection.
            # To KEEP downloaded files, comment out this loop.
            # -----------------------------
            '''for path in (video_path, csv_path, flightdata_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info("Deleted temporary download file: %s", path)
                    except Exception as clean_err:
                        logger.warning("Could not delete %s: %s", path, clean_err)'''

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

    # Start Test Server background worker
    test_thread = threading.Thread(
        target=worker_loop,
        args=(app_module, "Test Server", TEST_BASE_URL),
        daemon=True
    )
    test_thread.start()
    logger.info("Background Worker for Test Server Started")

    # Start Production Server background worker
    prod_thread = threading.Thread(
        target=worker_loop,
        args=(app_module, "Production Server", PROD_BASE_URL),
        daemon=True
    )
    prod_thread.start()
    logger.info("Background Worker for Production Server Started")