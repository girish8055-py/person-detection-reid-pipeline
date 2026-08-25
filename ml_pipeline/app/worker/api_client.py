import json
import logging
import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Server Configuration
# --------------------------------------------------------

TEST_BASE_URL = "http://114.143.215.196:5005/AeroMegh/idms"
PROD_BASE_URL = "https://idmsapi.aerogcs.com/AeroMegh/idms"

# For backwards compatibility
BASE_URL = TEST_BASE_URL
PENDING_API = f"{BASE_URL}/detection/pending"

# --------------------------------------------------------
# Common Headers
# --------------------------------------------------------

HEADERS = {
    "Content-Type": "application/json"
}

# --------------------------------------------------------
# Fetch Pending Job
# --------------------------------------------------------

def fetch_active_model_url(base_url=TEST_BASE_URL):
    try:
        url = f"{base_url}/models?is_active=true"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 204 or not response.text.strip():
            logger.info("No active models returned from server.")
            return None

        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        
        if models:
            # Return the download URL and model name of the first active model
            model = models[0]
            download_url = model.get("download_url")
            if not download_url and model.get("blob_url") and model.get("sas_token"):
                # Clean up any duplicate query string markers
                sas_token = model.get("sas_token")
                if sas_token.startswith("?"):
                    download_url = f"{model.get('blob_url')}{sas_token}"
                else:
                    download_url = f"{model.get('blob_url')}?{sas_token}"
            return {
                "download_url": download_url,
                "model_name": model.get("model_name") or model.get("file_name")
            }
            
    except requests.exceptions.RequestException as req_err:
        logger.warning("Could not connect to model API server at %s: %s", base_url, req_err)
    except Exception as e:
        logger.error("Failed to fetch active model URL from %s: %s", base_url, e)
    return None


# --------------------------------------------------------
# Fetch Pending Job
# --------------------------------------------------------

def fetch_pending_job(base_url=TEST_BASE_URL):
    try:
        pending_api = f"{base_url}/detection/pending"
        response = requests.get(pending_api, timeout=30)

        print("=" * 60)
        print("Status Code:", response.status_code)
        print("Response Text:")
        print(response.text)
        print("=" * 60)

        response.raise_for_status()

        if response.status_code == 204 or not response.text.strip():
            logger.info("No pending jobs available.")
            return None

        data = response.json()

        tasks = data.get("tasks", [])

        if not tasks:
            return None

        task = tasks[0]

        flight_logs = task.get("flight_logs") or []
        telemetry = task.get("telemetry") or task.get("teledata")
        if not telemetry and flight_logs:
            # Extract logs of type AS_TELE_DATA
            telemetry = [log for log in flight_logs if log.get("key") == "AS_TELE_DATA"]

        flightdata = task.get("flightdata") or task.get("flight_data")
        if not flightdata and flight_logs:
            # Fallback to all logs so parse_video_start_epoch can scan log messages
            flightdata = flight_logs

        return {
            "task_id": task["task_id"],
            "flight_id": task["flight_id"],
            "video_url": task["video_url"],
            "csv_url": task.get("csv_url"),
            "telemetry": telemetry,
            "flightdata": flightdata,
            "status": task["status"],
            "upload_url": task.get("upload_url") or task.get("sas_url") or task.get("azure_sas_url") or task.get("upload_sas_url"),
            "model_url": task.get("model_url") or task.get("detection_model_url") or task.get("weights_url") or task.get("model_path"),
            "model_name": task.get("model_name") or task.get("detection_model_name"),
        }

    except requests.exceptions.RequestException as req_err:
        logger.warning("Could not connect to server at %s (Connection Refused/Offline): %s", base_url, req_err)
        return None
    except Exception as e:
        logger.error("Error fetching pending job from %s: %s", base_url, e)
        return None


# --------------------------------------------------------
# Update Task Status
# --------------------------------------------------------

def update_status(task_id, status, base_url=TEST_BASE_URL):
    try:
        url = f"{base_url}/detection/update/{task_id}"

        response = requests.put(
            url,
            json={"status": status},
            headers=HEADERS,
            timeout=60,
        )

        response.raise_for_status()
        logger.info("Task %s -> %s", task_id, status)
        return True

    except requests.exceptions.RequestException as req_err:
        logger.warning("Could not update status to %s on %s: %s", status, base_url, req_err)
        return False
    except Exception as e:
        logger.error("Failed to update status on %s: %s", base_url, e)
        return False


# --------------------------------------------------------
# Update Completed
# --------------------------------------------------------

def update_completed(task_id, results, files=None, file_sizes=None, base_url=TEST_BASE_URL):
    try:
        url = f"{base_url}/detection/update/{task_id}"

        if files:
            try:
                headers = HEADERS.copy()
                if "Content-Type" in headers:
                    del headers["Content-Type"]
                
                # Gather form data, including file sizes
                data = {
                    "status": "COMPLETED",
                    "results": json.dumps(results),
                }
                if file_sizes:
                    for filename, sizes in file_sizes.items():
                        data[f"{filename}_size"] = str(sizes.get("file_size", 0))
                        data[f"{filename}_folder_size"] = str(sizes.get("folder_size", 0))
                        data[f"size_{filename}"] = str(sizes.get("file_size", 0))

                response = requests.put(
                    url,
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=120,
                )
                response.raise_for_status()
                logger.info("Task %s Completed (multipart file upload successful)", task_id)
                return True
            except Exception as multipart_err:
                logger.warning("Failed to upload files via multipart: %s. Falling back to JSON-only update.", multipart_err)

        response = requests.put(
            url,
            json={
                "status": "COMPLETED",
                "results": results,
            },
            headers=HEADERS,
            timeout=120,
        )

        response.raise_for_status()
        logger.info("Task %s Completed", task_id)
        return True

    except requests.exceptions.RequestException as req_err:
        logger.warning("Could not update completion status for task %s on %s: %s", task_id, base_url, req_err)
        return False
    except Exception as e:
        logger.error("Failed to update completion status for task %s: %s", task_id, e)
        return False


# --------------------------------------------------------
# Update Failed
# --------------------------------------------------------

def update_failed(task_id, message="Processing Failed", base_url=TEST_BASE_URL):
    try:
        url = f"{base_url}/detection/update/{task_id}"

        response = requests.put(
            url,
            json={
                "status": "FAILED",
                "error": message,
            },
            headers=HEADERS,
            timeout=60,
        )

        response.raise_for_status()
        logger.info("Task %s Failed", task_id)
        return True

    except requests.exceptions.RequestException as req_err:
        logger.warning("Could not send failure status for task %s to %s: %s", task_id, base_url, req_err)
        return False
    except Exception as e:
        logger.error("Failed to send failure status for task %s: %s", task_id, e)
        return False