import joblib
import os

def load_viewership_model():
    path = "viewership_model_log.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Missing viewership_model_log.joblib")

    data = joblib.load(path)

    if isinstance(data, dict):
        model = data["model"]
        model.team_counts = data.get("team_counts", {})
        model.smearing_factor = data.get("smearing_factor", 1.0)
        return model

    # fallback if saved as plain model
    data.smearing_factor = 1.0
    return data


def load_brand_model():
    path = "brand_model.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Missing brand_model.joblib")

    data = joblib.load(path)
    if isinstance(data, dict):
        model = data["model"]
        model.team_counts = data.get("team_counts", {})
        return model

    return data