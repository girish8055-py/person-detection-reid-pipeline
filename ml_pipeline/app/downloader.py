import os
import requests


def download_file(url, filename):
    """
    Downloads any file from the given URL.

    Returns:
        Local file path
    """
    os.makedirs("upload", exist_ok=True)
    dest_path = os.path.join("upload", filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {dest_path}")
    return dest_path


def download_video(video_url, task_id):
    """
    Downloads the video from the given URL.

    Returns:
        Local video path
    """
    return download_file(video_url, f"{task_id}.mp4")


def download_model(url, filename):
    """
    Downloads the model to the 'downloaded_model' directory.
    """
    dest_dir = r"D:\deployment_model\ml_pipeline\app\downloaded_model"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded model: {dest_path}")
    return dest_path