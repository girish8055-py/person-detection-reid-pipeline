import logging
import os
import requests
import shutil
import threading
import time

from downloader import download_video, download_file, download_model
from pipeline.telemetry import parse_video_start_epoch
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

logger = logging.getLogger(__name__)


def upload_to_azure(file_path, filename, flight_id="default", sas_url=None, base_url=None):
    azure_filename = filename if filename.lower().endswith((".jpg", ".jpeg")) else f"{filename}.jpg"
    if not base_url:
        base_url = PROD_BASE_URL
    fallback_url = f"{base_url}/uploads/threatimages/{flight_id}/{azure_filename}"
    
    if not sas_url:
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
        logger.exception("Failed to upload to Azure Container SAS via requests: %s. Using fallback URL.", e)
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

            video_path = download_video(
                job["video_url"],
                task_id
            )

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
                    
                    target_dir = r"D:\deployment_model\ml_pipeline\app\downloaded_model"
                    logger.info("[%s] Downloading detection model from: %s", server_name, model_url)
                    model_path = download_model(model_url, model_filename)
                except Exception as model_err:
                    err_msg = f"Failed to download detection model from '{model_url}': {model_err}"
                    logger.error("[%s] %s", server_name, err_msg)
                    raise RuntimeError(f"Model download failed: {err_msg}")

            if not model_path:
                raise RuntimeError("Detection model could not be loaded because model URL/file is missing or download failed. Default base model fallback is disabled.")

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
                raw_person_id = f"Person_{pk}" if not str(pk).startswith("Person_") else str(pk)
                
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