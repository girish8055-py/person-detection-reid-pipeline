import os
import datetime

output_dir = r"D:\deployment_model\ml_pipeline\app\outputs"
if os.path.exists(output_dir):
    items = os.listdir(output_dir)
    print(f"Total items in outputs: {len(items)}")
    for item in items:
        item_path = os.path.join(output_dir, item)
        mtime = os.path.getmtime(item_path)
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f" - {item}: Modified at {dt} ({'Dir' if os.path.isdir(item_path) else 'File'})")
else:
    print("Outputs directory does not exist")
