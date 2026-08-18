import sys
import os

# Add app path to sys.path so we can import from worker.api_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from worker.api_client import fetch_pending_job, TEST_BASE_URL, PROD_BASE_URL

print("Testing fetch from Test Server:")
print("Base URL:", TEST_BASE_URL)
try:
    job = fetch_pending_job(TEST_BASE_URL)
    print("Test Server job result:", job)
except Exception as e:
    print("Error fetching from Test Server:", e)

print("\nTesting fetch from Production Server:")
print("Base URL:", PROD_BASE_URL)
try:
    job = fetch_pending_job(PROD_BASE_URL)
    print("Production Server job result:", job)
except Exception as e:
    print("Error fetching from Production Server:", e)
