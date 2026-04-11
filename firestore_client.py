import os
import json
from pathlib import Path
from google.oauth2 import service_account
from google.cloud import firestore

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
LOCAL_SERVICE_ACCOUNT_PATH = REPO_ROOT / "serviceAccountKey.json"


def load_firestore_credentials():
    raw = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if raw:
        info = json.loads(raw)
        credentials = service_account.Credentials.from_service_account_info(info)
        return info["project_id"], credentials

    if LOCAL_SERVICE_ACCOUNT_PATH.exists():
        with LOCAL_SERVICE_ACCOUNT_PATH.open() as f:
            info = json.load(f)
        credentials = service_account.Credentials.from_service_account_file(
            str(LOCAL_SERVICE_ACCOUNT_PATH)
        )
        return info["project_id"], credentials

    return None, None


project_id, credentials = load_firestore_credentials()

db = (
    firestore.Client(project=project_id, credentials=credentials)
    if project_id and credentials
    else None
)
