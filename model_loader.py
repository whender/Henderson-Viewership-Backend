import joblib
import os
import hashlib
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _attach_pregame_artifact_metadata(model, artifact=None):
    """Attach optional pregame artifact metadata without changing loader callers."""
    artifact = artifact if isinstance(artifact, dict) else {}
    model.team_counts = artifact.get("team_counts", getattr(model, "team_counts", {}))
    model.smearing_factor = artifact.get(
        "smearing_factor",
        getattr(model, "smearing_factor", 1.0),
    )

    # Canonical artifact contract used by slate competition inference. The
    # longer aliases keep artifacts created during development readable.
    model.intrinsic_model = artifact.get(
        "intrinsic_model",
        artifact.get("intrinsic_competition_model"),
    )
    model.intrinsic_smearing_factor = artifact.get(
        "intrinsic_smearing_factor",
        artifact.get("intrinsic_competition_smearing_factor"),
    )
    model.intrinsic_team_counts = artifact.get(
        "intrinsic_team_counts",
        artifact.get("intrinsic_competition_team_counts", {}),
    )
    model.competition_score_by_tier = artifact.get(
        "competition_score_by_tier",
        {},
    )
    model.pregame_ensemble = artifact.get("pregame_ensemble")
    return model

def load_viewership_model():
    path = os.path.join(BASE_DIR, "viewership_model_log.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing viewership_model_log.joblib")

    data = joblib.load(path)

    if isinstance(data, dict):
        model = _attach_pregame_artifact_metadata(data["model"], data)
        calibration_path = os.path.join(BASE_DIR, "opening_week_calibration.json")
        if os.path.exists(calibration_path):
            with open(calibration_path) as source:
                calibration = json.load(source)
            with open(path, "rb") as artifact:
                digest = hashlib.file_digest(artifact, "sha256").hexdigest()
            if calibration.get("model_sha256") != digest:
                raise ValueError("Opening-week calibration must be revalidated for this model artifact")
            model.opening_week_calibration = calibration
        return model

    # fallback if saved as plain model
    return _attach_pregame_artifact_metadata(data)


def load_brand_model():
    path = os.path.join(BASE_DIR, "brand_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing brand_model.joblib")

    data = joblib.load(path)
    if isinstance(data, dict):
        model = data["model"]
        model.team_counts = data.get("team_counts", {})
        return model

    return data

def load_postgame_model():
    path = os.path.join(BASE_DIR, "viewership_postgame_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing viewership_postgame_model.joblib")

    data = joblib.load(path)

    if isinstance(data, dict):
        model = data["model"]
        model.team_counts = data.get("team_counts", {})
        model.smearing_factor = data.get("smearing_factor", 1.0)
        return model

    # fallback if saved as plain model
    data.smearing_factor = 1.0
    return data
