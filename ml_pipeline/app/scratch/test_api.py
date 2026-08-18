import sys
import os

# Add app path to sys.path so we can import from worker.api_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from worker.api_client import fetch_pending_job
job = fetch_pending_job()
print("Pending Job:", job)
