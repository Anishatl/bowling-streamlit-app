import os
import joblib
import requests

def download_and_cache_model(url, filename, cache_dir="models"):
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, filename)
    
    if not os.path.exists(file_path):
        print(f"Downloading {filename} from Google Drive...")
        r = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(r.content)
    
    return joblib.load(file_path)
