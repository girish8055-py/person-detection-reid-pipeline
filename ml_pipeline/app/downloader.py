import os
import requests


def download_file(url, filename):
    """
    Downloads any file from the given URL to the 'upload' folder.

    Returns:
        Local file path
    """
    dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded: {dest_path}")
        return dest_path
    except Exception as e:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        raise RuntimeError(f"Could not download file from '{url}': {e}")


def download_video(video_url, task_id):
    """
    Downloads the video from the given URL into the 'upload' folder.

    Returns:
        Local video path
    """
    return download_file(video_url, f"{task_id}.mp4")


def download_model(url, filename):
    """
    Downloads the model to the 'downloaded_model' directory.
    Strictly downloads the model from the provided URL.
    """
    dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_model")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    temp_path = dest_path + ".tmp"

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)
            print(f"Downloaded model successfully: {dest_path}")
            return dest_path
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError("Downloaded model file is empty")
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise RuntimeError(f"Could not download model file from '{url}': {e}")