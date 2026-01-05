import os
import joblib
import gdown  # pip install gdown

# Use this to actually get ML models, currently stored on google drive, so those IDs should work
# Paths for local repo files
CLIP_SCALER_PATH = "clip_scaler.pkl"
CLIP_LABEL_ENCODER_PATH = "clip_label_encoder.pkl"
JOINT_LABEL_ENCODER_PATH = "label_encoder.pkl"

# Google Drive IDs for large models
CLIP_LEVEL_RF_MODEL_GDRIVE_ID = "1j0Gk9rsp4p1F0V7JPQdaxBD7DqGnd5hZ"
JOINT_RISK_RF_MODELS_GDRIVE_ID = "1cSvUNOhczi8JxbP2MAmRmCoZg6Ook0Pj"

# Local cache folder for downloaded models***
CACHE_DIR = "downloaded_models"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_from_gdrive(file_id: str, dest_path: str):
    """
    Download a file from Google Drive by its file ID if it doesn't exist locally.
    """
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {dest_path} from Google Drive...")
        gdown.download(url, dest_path, quiet=False)
    else:
        print(f"Using cached file {dest_path}")

def load_clip_level_rf_model():
    local_path = os.path.join(CACHE_DIR, "clip_level_rf_model.pkl")
    download_from_gdrive(CLIP_LEVEL_RF_MODEL_GDRIVE_ID, local_path)
    return joblib.load(local_path)

def load_joint_risk_rf_models():
    local_path = os.path.join(CACHE_DIR, "joint_risk_rf_models.pkl")
    download_from_gdrive(JOINT_RISK_RF_MODELS_GDRIVE_ID, local_path)
    return joblib.load(local_path)

def load_clip_scaler():
    return joblib.load(CLIP_SCALER_PATH)

def load_clip_label_encoder():
    return joblib.load(CLIP_LABEL_ENCODER_PATH)

def load_joint_label_encoder():
    return joblib.load(JOINT_LABEL_ENCODER_PATH)

def load_all_models():
    clip_model = load_clip_level_rf_model()
    joint_models = load_joint_risk_rf_models()
    clip_scaler = load_clip_scaler()
    clip_label_encoder = load_clip_label_encoder()
    joint_label_encoder = load_joint_label_encoder()

    return {
        "clip_model": clip_model,
        "joint_models": joint_models,
        "clip_scaler": clip_scaler,
        "clip_label_encoder": clip_label_encoder,
        "joint_label_encoder": joint_label_encoder,
    }
