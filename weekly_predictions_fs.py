import numpy as np
import pandas as pd
from datetime import datetime

from predict import (
    model, normalize_team, rank_to_coefs, team_conferences,
    rivalries, FEUD_START, FEUD_END, MODEL_TEAM_NAMES, format_viewers
)

# ---------------------------------------------------------------------
# ✔ VIEWERSHIP PARSER / ERROR FUNCTION
# ---------------------------------------------------------------------

def parse_viewership(val):
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


def calc_error(pred, actual):
    p = parse_viewership(pred)
    a = parse_viewership(actual)

    if p is None or a is None or a <= 0:
        return None

    return abs((p - a) / a) * 100


# ---------------------------------------------------------------------
# ⭐ MAIN PREDICTION FUNCTION — IDENTICAL TO STREAMLIT
# ---------------------------------------------------------------------
def generate_prediction(row):

    try:
        # ------------------------
        # TEAM / BASIC INPUTS
        # ------------------------
        team1 = normalize_team(row["team1"])
        team2 = normalize_team(row["team2"])

        rank1 = int(row["rank1"])
        rank2 = int(row["rank2"])
        spread = float(row["spread"])
        network = row["network"]
        time_slot = str(row["time_slot"])
        comp_tier1 = int(row.get("comp_tier1", 0))

        day = row["day"]

        conf1 = team_conferences.get(team1, "Group of 6")
        conf2 = team_conferences.get(team2, "Group of 6")

        both_ranked = rank1 > 0 and rank2 > 0
        same_conf = conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"]

        t1_top10, t1_mid = rank_to_coefs(rank1)
        t2_top10, t2_mid = rank_to_coefs(rank2)

        top10 = t1_top10 + t2_top10
        rank_25_11 = t1_mid + t2_mid

        # ------------------------
        # FRIDAY OVERRIDE
        # ------------------------
        is_friday = (
            str(day).strip().lower() == "fri"
            or "fri" in str(day).lower()
        )

        # ------------------------
        # RIVALRY DETECTION
        # ------------------------
        auto_rivalry = next(
            (r for r, (a, b) in rivalries.items() if {team1, team2} == {a, b}),
            None
        )

        # ------------------------
        # TIME BUCKETS (MATCHING STREAMLIT EXACTLY!)
        # ------------------------

        # FULL timestamp-based bucketing
        early_keywords = ["11:00a", "11:30a", "12:00p", "12:30p", "1:00p", "1:30p", "2:00p"]
        mid_keywords = ["2:30p", "3:00p", "3:30p", "4:00p", "4:30p", "5:00p", "5:30p", "6:00p", "6:30p"]
        late_keywords = ["9:30p", "10:00p", "10:30", "11:00p", "11:30p"]

        sat_early = (
            not is_friday and (
                "Early" in time_slot or any(t in time_slot for t in early_keywords)
            )
        )
        sat_mid = (
            not is_friday and (
                "Mid" in time_slot or any(t in time_slot for t in mid_keywords)
            )
        )
        sat_late = (
            not is_friday and (
                "Late" in time_slot or any(t in time_slot for t in late_keywords)
            )
        )

        # ------------------------
        # FEATURE VECTOR
        # ------------------------
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

            "Sat Early": int(sat_early),
            "Sat Mid": int(sat_mid),
            "Sat Late": int(sat_late),

            "Top 10 Rankings": top10,
            "25-11 Rankings": rank_25_11,

            "SEC": (conf1 == "SEC") + (conf2 == "SEC"),
            "Big 10": (conf1 == "Big 10") + (conf2 == "Big 10"),
            "ACC": (conf1 == "ACC") + (conf2 == "ACC"),
            "Big 12": (conf1 == "Big 12") + (conf2 == "Big 12"),
        }

        # postseason flags
        for conf_tag, flag_name in {
            "SEC": "SEC_PostseasonImplications",
            "Big 10": "Big10_PostseasonImplications",
            "Big 12": "Big12_PostseasonImplications",
            "ACC": "ACC_PostseasonImplications",
        }.items():
            features[flag_name] = int(both_ranked and same_conf and conf1 == conf_tag)

        # rivalry flags
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

        # const
        features["const"] = 1.0

        # fill missing model params
        for c in model.params.index:
            if c not in features:
                features[c] = 0.0

        # OhioSt × BTN
        if "OhioSt_BTN" in model.params.index:
            features["OhioSt_BTN"] = int(
                ("Ohio St." in [team1, team2]) and network == "BTN"
            )

        # ------------------------
        # PREDICTION + CONFIDENCE INTERVAL (MATCHES STREAMLIT)
        # ------------------------

        X = pd.DataFrame([[features[c] for c in model.params.index]], columns=model.params.index)

        try:
            pred_res = model.get_prediction(X)
            ci = pred_res.summary_frame(alpha=0.32)

            smearing = getattr(model, "smearing_factor", 1.0)

            pred_ln = ci["mean"].iloc[0]
            low_ln = ci["obs_ci_lower"].iloc[0]
            high_ln = ci["obs_ci_upper"].iloc[0]

            pred = (np.exp(pred_ln) - 1) * smearing
            low = max(0, (np.exp(low_ln) - 1) * smearing)
            high = max(low, (np.exp(high_ln) - 1) * smearing)

        except:
            pred = float(model.predict(X)[0])
            low = max(pred - 500, 0)
            high = pred + 500

        pred_fmt = f"{pred/1_000:.2f}M\n({low/1_000:.2f}–{high/1_000:.2f}M)"
        return pred_fmt

    except Exception as e:
        return f"Error: {e}"