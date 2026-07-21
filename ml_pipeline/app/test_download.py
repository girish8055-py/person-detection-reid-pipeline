from fetch_job import fetch_pending_job
from downloader import download_video

job = fetch_pending_job()

if job:

    path = download_video(
        job["video_url"],
        job["task_id"]
    )

    print(path)