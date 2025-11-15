# firestore_client.py
import os
import json
from google.oauth2 import service_account
from google.cloud import firestore

# ---------------------------------------
# Load service account from Render env var
# ---------------------------------------
raw = os.environ.get("FIREBASE_CREDENTIALS_JSON")
if not raw:
    raise RuntimeError(
        "🔥 FIREBASE_CREDENTIALS_JSON environment variable not set on Render!"
    )

info = json.loads(raw)

credentials = service_account.Credentials.from_service_account_info(info)

# ---------------------------------------
# Firestore Client
# ---------------------------------------
db = firestore.Client(
    project=info["project_id"],
    credentials=credentials
)