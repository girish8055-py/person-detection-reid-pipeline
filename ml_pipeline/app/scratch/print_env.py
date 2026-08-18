import os
for k, v in os.environ.items():
    if any(x in k.lower() for x in ["auth", "api", "key", "pass", "user", "secret", "token", "azure"]):
        print(f"{k}: {v}")
