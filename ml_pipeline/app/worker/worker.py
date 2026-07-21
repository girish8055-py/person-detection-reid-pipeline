import logging
import os
import requests
import shutil
import threading
import time

from downloader import download_video, download_file
from pipeline.telemetry import parse_video_start_epoch
from worker.api_client import (
    fetch_pending_job,
    update_status,
    update_completed,
    update_failed,
)

from worker.result_builder import build_result_json

logger = logging.getLogger(__name__)


def upload_to_azure(file_path, filename, flight_id="default", sas_url=None):
    azure_filename = filename if filename.lower().endswith((".jpg", ".jpeg")) else f"{filename}.jpg"
    fallback_url = f"http://114.143.215.196:5005/AeroMegh/idms/uploads/threatimages/{flight_id}/{azure_filename}"
    
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


def worker_loop(app_module):

    logger.info("Worker Started")

    while True:

        try:

            # Only one job at a time
            if app_module.processing_lock.locked():
                time.sleep(5)
                continue

            job = fetch_pending_job()

            if job is None:
                logger.info("No Pending Jobs")
                time.sleep(5)
                continue

            task_id = job["task_id"]

            flight_id = job["flight_id"]

            logger.info(
                "Processing Task : %s",
                task_id
            )

            # -----------------------------
            # Update Status
            # -----------------------------

            update_status(
                task_id,
                "IN_PROGRESS"
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
            if job.get("csv_url"):
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
                    # Parse video start epoch from flightdata
                    video_start_epoch = parse_video_start_epoch(flightdata_path)
                except Exception as telemetry_err:
                    logger.warning("Failed to download or parse telemetry/flightdata: %s", telemetry_err)

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

                    video_start_epoch=video_start_epoch

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
                
                # Check for the screenshot and rename it
                person_dir = os.path.join(output_folder, f"person_{raw_person_id}")
                src_file = os.path.join(person_dir, "best_detection.jpg")
                dest_filename = f"{raw_person_id}_{time_str}"
                dest_file = os.path.join(person_dir, dest_filename)
                
                file_size = 0
                folder_size = 0
                
                if os.path.exists(src_file):
                    try:
                        if os.path.exists(dest_file):
                            os.remove(dest_file)
                        os.rename(src_file, dest_file)
                    except Exception as rename_err:
                        logger.warning("Could not rename %s to %s: %s", src_file, dest_file, rename_err)
                
                if os.path.exists(dest_file):
                    file_size = os.path.getsize(dest_file)
                    try:
                        files_to_upload[dest_filename] = open(dest_file, "rb")
                    except Exception as open_err:
                        logger.warning("Could not open file %s for upload: %s", dest_file, open_err)
                
                if os.path.exists(person_dir):
                    # compute folder size
                    for root_dir, _, filenames in os.walk(person_dir):
                        for f in filenames:
                            fp = os.path.join(root_dir, f)
                            if os.path.exists(fp):
                                folder_size += os.path.getsize(fp)
                
                # Upload to Azure
                file_url = None
                if os.path.exists(dest_file):
                    file_url = upload_to_azure(
                        dest_file,
                        dest_filename,
                        flight_id=job.get("flight_id", "default"),
                        sas_url=job.get("upload_url")
                    )
                
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
                    file_sizes=file_sizes
                )
            finally:
                for f in files_to_upload.values():
                    try:
                        f.close()
                    except Exception:
                        pass

            logger.info(
                "Task %s Completed",
                task_id
            )

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

            logger.exception(e)

            try:

                update_failed(
                    task_id,
                    str(e)
                )

            except Exception:
                pass

            time.sleep(5)


def start_worker(app_module):

    thread = threading.Thread(

        target=worker_loop,

        args=(app_module,),

        daemon=True

    )

    thread.start()

    logger.info("Background Worker Started")