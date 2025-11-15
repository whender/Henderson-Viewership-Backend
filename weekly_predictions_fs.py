# weekly_predictions_fs.py

import numpy as np
import pandas as pd
from datetime import datetime
from firestore_client import db

from predict import (
    model, normalize_team, rank_to_coefs, team_conferences,
    rivalries, FEUD_START, FEUD_END, MODEL_TEAM_NAMES, format_viewers
)

# -------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------

def parse_viewership(val):
    """Convert '1.22M' or '850K' or raw numbers to millions."""
    try:
        if val is None:
            return None
        val = str(val).strip().upper()
        if "\n" in val:
            val = val.split("\n")[0]
        if "(" in val:
            val = val.split("(")[0]
        if val.endswith("M"):
            return float(val[:-1])
        if val.endswith("K"):
            return float(val[:-1]) / 1000
        return float(val)
    except:
        return None


def calc_error(p, a):
    p = parse_viewership(p)
    a = parse_viewership(a)
    if p is None or a is None or a <= 0:
        return None
    return abs((p - a) / a) * 100


# -------------------------------------------------------------
# Prediction creation (identical to your Streamlit predictor)
# -------------------------------------------------------------
def generate_prediction(row):
    try:
        team1 = normalize_team(row["team1"])
        team2 = normalize_team(row["team2"])

        rank1 = int(row["rank1"])
        rank2 = int(row["rank2"])
        spread = float(row["spread"])

        network = row["network"]
        time_slot = row["time_slot"]
        comp_tier1 = int(row.get("comp_tier1", 0))

        day = row["day"]
        date_str = row["date"]

        conf1 = team_conferences.get(team1, "Group of 6")
        conf2 = team_conferences.get(team2, "Group of 6")

        both_ranked = rank1 > 0 and rank2 > 0
        same_conf = conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"]

        t1_top10, t1_mid = rank_to_coefs(rank1)
        t2_top10, t2_mid = rank_to_coefs(rank2)

        top10 = t1_top10 + t2_top10
        rank_25_11 = t1_mid + t2_mid

        # Friday override (matches your Streamlit logic)
        is_friday = str(day).strip().lower() == "fri" or "fri" in str(day).lower()

        # rivalry flag
        auto_rivalry = next(
            (r for r, (a, b) in rivalries.items() if {team1, team2} == {a, b}),
            None
        )

        # Build feature vector exactly like your Streamlit logic
        features = {
            "Spread": spread,
            "Competing Tier 1": comp_tier1,

            "ABC": int(network == "ABC"),
            "CBS": int(network == "CBS"),
            "NBC": int(network == "NBC"),
            "FOX": int(network == "FOX"),
            "ESPN": int(network == "ESPN"),
            "ESPN2": int(network == "ESPN2"),
            "ESPNU": int(network == "ESPNU"),
            "FS1": int(network == "FS1"),
            "FS2": int(network == "FS2"),
            "BTN": int(network == "BTN"),
            "CW": int(network == "CW"),
            "NFLN": int(network == "NFLN"),
            "ESPNNEWS": int(network == "ESPNNEWS"),

            "Conf Champ": 0,

            "Sun": int("Sunday" in time_slot),
            "Monday": int("Monday" in time_slot),
            "Weekday": int("Weekday" in time_slot),
            "Friday": int(is_friday or "Friday" in time_slot),

            "Sat Early": int(("Early" in time_slot) and not is_friday),
            "Sat Mid": int(("Mid" in time_slot) and not is_friday),
            "Sat Late": int(("Late" in time_slot) and not is_friday),

            "Top 10 Rankings": top10,
            "25-11 Rankings": rank_25_11,

            "SEC": (conf1 == "SEC") + (conf2 == "SEC"),
            "Big 10": (conf1 == "Big 10") + (conf2 == "Big 10"),
            "ACC": (conf1 == "ACC") + (conf2 == "ACC"),
            "Big 12": (conf1 == "Big 12") + (conf2 == "Big 12"),
        }

        # Postseason implications
        for conf_tag, flag_name in {
            "SEC": "SEC_PostseasonImplications",
            "Big 10": "Big10_PostseasonImplications",
            "Big 12": "Big12_PostseasonImplications",
            "ACC": "ACC_PostseasonImplications",
        }.items():
            features[flag_name] = int(both_ranked and same_conf and conf1 == conf_tag)

        # rivalry dummies
        for r in rivalries:
            features[r] = int(r == auto_rivalry)

        # YTTV flags
        features["YTTV_ABC"] = 0
        features["YTTV_ESPN"] = 0
        now = datetime.now()
        if FEUD_START <= now <= FEUD_END:
            if network in ["ABC", "ESPN"]:
                features[f"YTTV_{network}"] = 1

        # team dummy flags
        for col in model.params.index:
            if col in MODEL_TEAM_NAMES:
                features[col] = int(col in [team1, team2])

        features["const"] = 1.0

        # Ensure all model columns exist
        for c in model.params.index:
            if c not in features:
                features[c] = 0.0

        # Ohio State × BTN interaction
        if "OhioSt_BTN" in model.params.index:
            features["OhioSt_BTN"] = int(
                ("Ohio St." in [team1, team2]) and network == "BTN"
            )

        X = pd.DataFrame([[features[c] for c in model.params.index]], columns=model.params.index)

        # Log prediction + smearing
        pred_ln = float(model.predict(X)[0])
        smearing = getattr(model, "smearing_factor", 1.0)
        pred = (np.exp(pred_ln) - 1) * smearing  # thousands → real viewers
        pred_real = pred * 1000

        return format_viewers(pred_real)

    except Exception as e:
        return f"Error: {e}"