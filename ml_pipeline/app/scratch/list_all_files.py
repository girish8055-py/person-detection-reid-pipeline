import os

exclude_dirs = {".git", "venv", "__pycache__"}

for root, dirs, files in os.walk(r"D:\deployment_model"):
    # Filter out excluded directories in-place
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, r"D:\deployment_model")
        size = os.path.getsize(full_path)
        print(f"{rel_path} ({size} bytes)")
