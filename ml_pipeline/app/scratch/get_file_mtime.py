import os
import datetime

files = [
    r"D:\deployment_model\ml_pipeline\pending_approval_170ce267-3d4a-4347-a81a-95925e14fcf7.json",
    r"D:\deployment_model\ml_pipeline\app\pipeline\approve_170ce267-3d4a-4347-a81a-95925e14fcf7.txt containing",
]

for f in files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"{os.path.basename(f)}: Modified at {dt}")
    else:
        print(f"{os.path.basename(f)}: Does not exist")
