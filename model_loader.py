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
    week1_path = os.path.join(BASE_DIR, 'week1_major_days.joblib')
    if os.path.exists(week1_path):
        week1 = joblib.load(week1_path)
        with open(os.path.join(BASE_DIR, 'viewership_model_log.joblib'), 'rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if (week1.get('version') != 1 or week1.get('primary_sha256') != digest
                or week1.get('prediction_year') != week1.get('training_max_year', 0) + 1):
            raise ValueError('Week 1 model must be revalidated for this primary artifact')
        model.week1_major_days = week1
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
        monday_path = os.path.join(BASE_DIR, "monday_calibration.json")
        if os.path.exists(monday_path):
            with open(monday_path) as source:
                monday = json.load(source)
            with open(path, "rb") as artifact:
                digest = hashlib.file_digest(artifact, "sha256").hexdigest()
            if monday.get("model_sha256") != digest:
                raise ValueError("Monday calibration must be revalidated for this model artifact")
            model.monday_calibration = monday
        nonlinear_path = os.path.join(BASE_DIR, "nonlinear_pregame.joblib")
        if os.path.exists(nonlinear_path):
            nonlinear = joblib.load(nonlinear_path)
            with open(path, "rb") as artifact:
                digest = hashlib.file_digest(artifact, "sha256").hexdigest()
            if (nonlinear.get('version') != 1 or nonlinear.get('primary_sha256') != digest
                    or nonlinear.get('promotion_passed') is not True):
                raise ValueError('Nonlinear model must be revalidated for this primary artifact')
            model.nonlinear_pregame = nonlinear
        from audience_interest import load_audience_interest
        model.audience_interest = load_audience_interest(BASE_DIR)
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
