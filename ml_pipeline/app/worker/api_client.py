import json
import logging
import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Server Configuration
# --------------------------------------------------------

BASE_URL = "http://114.143.215.196:5005/AeroMegh/idms"
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

def fetch_active_model_url():
    try:
        url = f"{BASE_URL}/models?is_active=true"
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
            
    except Exception as e:
        logger.exception("Failed to fetch active model URL: %s", e)
    return None


# --------------------------------------------------------
# Fetch Pending Job
# --------------------------------------------------------

def fetch_pending_job():
    try:
        response = requests.get(PENDING_API, timeout=30)

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

        return {
            "task_id": task["task_id"],
            "flight_id": task["flight_id"],
            "video_url": task["video_url"],
            "csv_url": task.get("csv_url"),
            "status": task["status"],
            "upload_url": task.get("upload_url") or task.get("sas_url") or task.get("azure_sas_url") or task.get("upload_sas_url"),
            "model_url": task.get("model_url") or task.get("detection_model_url") or task.get("weights_url") or task.get("model_path"),
            "model_name": task.get("model_name") or task.get("detection_model_name"),
        }

    except Exception as e:
        logger.exception(e)
        return None


# --------------------------------------------------------
# Update Task Status
# --------------------------------------------------------

def update_status(task_id, status):
    try:
        url = f"{BASE_URL}/detection/update/{task_id}"

        response = requests.put(
            url,
            json={"status": status},
            headers=HEADERS,
            timeout=60,
        )

        response.raise_for_status()
        logger.info("Task %s -> %s", task_id, status)
        return True

    except Exception as e:
        logger.exception(e)
        return False


# --------------------------------------------------------
# Update Completed
# --------------------------------------------------------

def update_completed(task_id, results, files=None, file_sizes=None):
    try:
        url = f"{BASE_URL}/detection/update/{task_id}"

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

    except Exception as e:
        logger.exception(e)
        return False


# --------------------------------------------------------
# Update Failed
# --------------------------------------------------------

def update_failed(task_id, message="Processing Failed"):
    try:
        url = f"{BASE_URL}/detection/update/{task_id}"

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

    except Exception as e:
        logger.exception(e)
        return False