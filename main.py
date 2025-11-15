from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
import os
import math
from datetime import datetime

# ======================================================
# 🚀 FASTAPI SETUP
# ======================================================

app = FastAPI(
    title="Henderson Viewership Model API",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 🧹 JSON SANITIZER (fixes your NaN crash)
# ======================================================

def clean_nan(obj):
    """Recursively convert NaN/inf to None so FastAPI can JSON encode."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    return obj

# ======================================================
# 📦 IMPORT PREDICTOR LOGIC
# ======================================================
from predict import predict_viewership, teams_list

class GameInput(BaseModel):
    team1: str
    team2: str
    rank1: int
    rank2: int
    spread: float
    network: str
    time_slot: str
    comp_tier1: int = 0

@app.get("/")
def root():
    return {"status": "running", "message": "Henderson Viewership Model API"}

@app.get("/teams")
def get_teams():
    return {
        "teams": [
            {"label": frontend_name, "value": backend_name}
            for frontend_name, backend_name in teams_list.items()
        ]
    }

@app.post("/predict")
def predict_game(game: GameInput):
    result = predict_viewership(game.dict())
    return {
        "prediction_raw": result["raw"],
        "prediction_formatted": result["formatted"]
    }

# ======================================================
# BRAND RANKINGS (UNCHANGED)
# ======================================================

numeric_features = [
    "Spread","Competing Tier 1","FOX","ESPN","ESPN2","ESPNU","FS1","FS2","NBC","CBS","ABC","BTN",
    "CW","NFLN","ESPNNEWS","Conf Champ","Sun","Monday","Weekday","Friday","Sat Early",
    "Sat Mid","Sat Late","Top 10 Rankings","25-11 Rankings","SEC_PostseasonImplications",
    "Big10_PostseasonImplications","Big12_PostseasonImplications","ACC_PostseasonImplications",
    "YTTV_ABC","YTTV_ESPN"
]

rivalry_features = [
    "Michigan_OhioSt","Texas_Oklahoma","Alabama_Auburn","Georgia_Florida","NotreDame_USC",
    "Florida_Tennessee","Oregon_Washington","BYU_Utah","Iowa_IowaSt","OleMiss_MississippiSt",
    "Clemson_SouthCarolina","Arizona_ArizonaSt","Miami_FloridaSt","Texas_TexasA&M",
    "Oregon_OregonSt","USC_UCLA","Louisville_Kentucky","OhioSt_PennSt","Alabama_LSU"
]

team_conferences = {
    "Alabama":"SEC","Auburn":"SEC","Georgia":"SEC","Florida":"SEC","LSU":"SEC","Tennessee":"SEC","Texas A&M":"SEC",
    "Kentucky":"SEC","South Carolina":"SEC","Mississippi":"SEC","Mississippi St.":"SEC","Arkansas":"SEC",
    "Missouri":"SEC","Vanderbilt":"SEC","Texas":"SEC","Oklahoma":"SEC",

    "Michigan":"Big 10","Ohio St.":"Big 10","Penn St.":"Big 10","Wisconsin":"Big 10","Iowa":"Big 10",
    "Michigan St.":"Big 10","Nebraska":"Big 10","Minnesota":"Big 10","Illinois":"Big 10","Indiana":"Big 10",
    "Purdue":"Big 10","Northwestern":"Big 10","Maryland":"Big 10","Rutgers":"Big 10","UCLA":"Big 10",
    "USC":"Big 10","Oregon":"Big 10","Washington":"Big 10",

    "Clemson":"ACC","Florida St.":"ACC","Miami":"ACC","North Carolina":"ACC","Duke":"ACC",
    "North Carolina St.":"ACC","Virginia":"ACC","Virginia Tech":"ACC","Louisville":"ACC","Syracuse":"ACC",
    "Boston College":"ACC","Wake Forest":"ACC","Pittsburgh":"ACC","Georgia Tech":"ACC","California":"ACC",
    "Stanford":"ACC","SMU":"ACC",

    "BYU":"Big 12","UCF":"Big 12","Houston":"Big 12","Cincinnati":"Big 12","Baylor":"Big 12",
    "Texas Tech":"Big 12","TCU":"Big 12","Kansas":"Big 12","Kansas St.":"Big 12","Iowa St.":"Big 12",
    "Oklahoma St.":"Big 12","West Virginia":"Big 12","Utah":"Big 12","Arizona":"Big 12",
    "Arizona St.":"Big 12","Colorado":"Big 12"
}

cleaned_path = "viewership_cleaned.csv"
df_all = pd.read_csv(cleaned_path, low_memory=False)

def compute_brand_rankings(df):
    power4_set = set(team_conferences.keys()) | {"Notre Dame"}

    d1 = pd.get_dummies(df["Team 1"])
    d2 = pd.get_dummies(df["Team 2"])
    team_dummies = d1.add(d2, fill_value=0)

    valid_counts = team_dummies.sum()
    team_dummies = team_dummies[valid_counts[valid_counts >= 3].index]
    if team_dummies.empty:
        return []

    feature_cols = [c for c in df.columns if c in numeric_features + rivalry_features]
    X = pd.concat([df[feature_cols], team_dummies], axis=1).fillna(0)
    X = sm.add_constant(X)

    y = np.log(df["Persons 2+"].astype(float) + 1)
    ridge = RidgeCV(alphas=[1.0])
    ridge.fit(X, y)

    params = pd.Series(ridge.coef_, index=X.columns)
    team_coefs = params[team_dummies.columns]
    team_coefs = team_coefs[team_coefs.index.isin(power4_set)]

    valid_counts = valid_counts.reindex(team_coefs.index).fillna(0)
    adjusted = team_coefs.copy()

    for t in team_coefs.index:
        n = valid_counts[t]
        if n <= 4:
            adjusted[t] = team_coefs[t] * (n / (n + 5))

    boost_pct = (np.exp(adjusted) - 1) * 100

    results = []
    for rank, (team, lift) in enumerate(
        boost_pct.sort_values(ascending=False).items(), start=1
    ):
        results.append({
            "rank": rank,
            "team": team,
            "viewership_lift_pct": float(round(lift, 1)),
            "games_used": int(valid_counts[team])
        })
    return results

brand_rankings_cache = {}
available_years = sorted(df_all["Year"].dropna().unique().tolist())

brand_rankings_cache["all"] = compute_brand_rankings(df_all)
for y in available_years:
    brand_rankings_cache[str(y)] = compute_brand_rankings(df_all[df_all["Year"] == y])

@app.get("/brand-years")
def brand_years():
    return {"years": available_years}

@app.get("/brand-rankings")
def brand_rankings(year: str = "all"):
    if year not in brand_rankings_cache:
        return {"rows": []}
    return {"rows": brand_rankings_cache[year]}

# ======================================================
# 📅 WEEKLY PREDICTIONS
# ======================================================

from predict import (
    model, normalize_team, rank_to_coefs, team_conferences,
    rivalries, FEUD_START, FEUD_END, MODEL_TEAM_NAMES, format_viewers
)

@app.get("/weekly-predictions")
def weekly_predictions():

    df = pd.read_csv("weekly_predictions.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Ensure prediction column exists
    if "Predicted Viewers" not in df.columns:
        df["Predicted Viewers"] = ""

    df["Predicted Viewers"] = df["Predicted Viewers"].replace(["nan", "NaN"], "")

    # --- helper to parse existing M/K values ---
    def parse_viewership(val):
        try:
            if val is None or pd.isna(val): return None
            val = str(val).strip().upper()

            if "\n" in val:  # remove CI if present
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

    # Rank buckets (same as Streamlit)
    def rank_to_coefs(r):
        if 1 <= r <= 10:
            return (1, 0)
        elif 11 <= r <= 25:
            return (0, 1)
        else:
            return (0, 0)

    preds = []
    for _, row in df.iterrows():

        # Skip rows already predicted
        existing = str(row["Predicted Viewers"]).strip().lower()
        if existing not in ["", "nan"]:
            preds.append(row["Predicted Viewers"])
            continue

        try:
            # ---------------------------
            # NORMALIZE TEAM NAMES
            # ---------------------------
            team1 = normalize_team(row["Team 1"])
            team2 = normalize_team(row["Team 2"])

            rank1 = int(row["Rank 1"])
            rank2 = int(row["Rank 2"])
            spread = float(row["Spread"])
            network = row["Network"]
            time_slot = str(row["Time Slot"])
            comp_tier1 = int(row.get("Competing Tier 1", 0))

            # ---------------------------
            # CONF / POSTSEASON FLAGS
            # ---------------------------
            conf1 = team_conferences.get(team1, "Group of 6")
            conf2 = team_conferences.get(team2, "Group of 6")

            both_ranked = rank1 > 0 and rank2 > 0
            same_conf = conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"]

            # Ranking buckets
            t1_top10, t1_25 = rank_to_coefs(rank1)
            t2_top10, t2_25 = rank_to_coefs(rank2)
            top10 = t1_top10 + t2_top10
            rank_25_11 = t1_25 + t2_25

            # ---------------------------
            # FRIDAY LOGIC (exact match)
            # ---------------------------
            raw_day = str(row["Day"]).strip().lower()
            is_friday = raw_day == "fri" or "fri" in raw_day

            # ---------------------------
            # RIVALRY (exact match)
            # ---------------------------
            auto_rivalry = next(
                (r for r, (a, b) in rivalries.items() if {team1, team2} == {a, b}),
                None
            )

            # ---------------------------
            # TIME SLOT LOGIC (exact match)
            # ---------------------------
            features = {
                "Spread": spread,
                "Competing Tier 1": comp_tier1,

                # Networks
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

                # Time
                "Sun": int("Sunday" in time_slot),
                "Monday": int("Monday" in time_slot),
                "Weekday": int("Weekday" in time_slot),
                "Friday": int(is_friday or "Friday" in time_slot),

                # EXACT STREAMLIT SLOT RULES
                "Sat Early": int(
                    not is_friday and (
                        "Early" in time_slot or
                        any(t in time_slot for t in [
                            "11:00a","11:30a","12:00p","12:30p","1:00p","1:30p","2:00p"
                        ])
                    )
                ),
                "Sat Mid": int(
                    not is_friday and (
                        "Mid" in time_slot or
                        any(t in time_slot for t in [
                            "2:30p","3:00p","3:30p","4:00p","4:30p",
                            "5:00p","5:30p","6:00p","6:30p"
                        ])
                    )
                ),
                "Sat Late": int(
                    not is_friday and (
                        "Late" in time_slot or
                        any(t in time_slot for t in [
                            "9:30p","10:00p","11:00p","11:30p"
                        ])
                    )
                ),

                # Ranking buckets
                "Top 10 Rankings": top10,
                "25-11 Rankings": rank_25_11,

                # Conf dummies
                "SEC": (conf1 == "SEC") + (conf2 == "SEC"),
                "Big 10": (conf1 == "Big 10") + (conf2 == "Big 10"),
                "ACC": (conf1 == "ACC") + (conf2 == "ACC"),
                "Big 12": (conf1 == "Big 12") + (conf2 == "Big 12"),
            }

            # Postseason flags
            for conf_tag, flag_name in {
                "SEC": "SEC_PostseasonImplications",
                "Big 10": "Big10_PostseasonImplications",
                "Big 12": "Big12_PostseasonImplications",
                "ACC": "ACC_PostseasonImplications",
            }.items():
                features[flag_name] = int(both_ranked and same_conf and conf1 == conf_tag)

            # Rivalries
            for r in rivalries:
                features[r] = int(r == auto_rivalry)

            # YTTV feud flags
            now = datetime.now()
            feud_active = FEUD_START <= now <= FEUD_END
            features["YTTV_ABC"] = int(feud_active and network == "ABC")
            features["YTTV_ESPN"] = int(feud_active and network == "ESPN")

            # Team dummy columns (CRITICAL)
            for col in model.params.index:
                if col in MODEL_TEAM_NAMES:
                    features[col] = int(col in [team1, team2])

            # Add constant + any missing features
            features["const"] = 1.0
            for c in model.params.index:
                if c not in features:
                    features[c] = 0.0

            # BTN × Ohio State interaction
            if "OhioSt_BTN" in model.params.index:
                features["OhioSt_BTN"] = int(
                    ("Ohio St." in [team1, team2]) and network == "BTN"
                )

            # ---------------------------
            # PREDICT
            # ---------------------------
            X = pd.DataFrame([[features[c] for c in model.params.index]],
                             columns=model.params.index)

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
                # Fallback (rare)
                pred = float(model.predict(X)[0])
                low = pred - 500
                high = pred + 500

            # ---------------------------
            # FORMAT EXACTLY LIKE STREAMLIT
            # ---------------------------
            pred_fmt = (
                f"{pred/1_000:.2f}M "
                f"({low/1_000:.2f}–{high/1_000:.2f}M)"
            )

            preds.append(pred_fmt)

        except Exception as e:
            preds.append(f"Error: {e}")

    # Write predictions back only to empty rows
    for i, val in enumerate(preds):
        if str(df.loc[i, "Predicted Viewers"]).strip().lower() in ["", "nan"]:
            df.loc[i, "Predicted Viewers"] = val

    df.to_csv("weekly_predictions.csv", index=False)

    # ---------------------------
    # ERROR CALCULATIONS
    # ---------------------------
    df["% Error"] = [
        abs((parse_viewership(p) - parse_viewership(a)) / parse_viewership(a)) * 100
        if parse_viewership(p) and parse_viewership(a) and parse_viewership(a) > 0
        else None
        for p, a in zip(df["Predicted Viewers"], df["Actual Viewers"])
    ]

    # Accuracy emojis
    def indicator(e):
        if e is None: return ""
        if e < 5: return "🟢🎯"
        if e < 25: return "🟢"
        if e < 35: return "🟡"
        return "🔴"

    df["Accuracy"] = [indicator(e) for e in df["% Error"]]

    # Matchup formatting
    def format_matchup(t1, r1, t2, r2):
        t1s = f"#{int(r1)} {t1}" if r1 > 0 else t1
        t2s = f"#{int(r2)} {t2}" if r2 > 0 else t2
        return f"{t1s} @ {t2s}"

    df["Matchup"] = [
        format_matchup(t1, r1, t2, r2)
        for t1, r1, t2, r2 in zip(df["Team 1"], df["Rank 1"], df["Team 2"], df["Rank 2"])
    ]

    # Safe date formatting
    def fmt_date(d, day):
        try:
            dt = pd.to_datetime(d)
            return f"{day}, {dt.strftime('%b %-d')}"
        except:
            return str(day)

    df["DateFmt"] = [
        fmt_date(d, day) for d, day in zip(df["Date"], df["Day"])
    ]

    # Group weeks
    df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year

    weeks = []
    for week, group in df.groupby("Week"):
        games = []
        for _, r in group.iterrows():
            games.append({
                "date": r["DateFmt"],
                "time_slot": r["Time Slot"],
                "matchup": r["Matchup"],
                "spread": r["Spread"],
                "network": r["Network"],
                "predicted": r["Predicted Viewers"],
                "actual": r["Actual Viewers"],
                "percent_error": r["% Error"],
                "accuracy": r["Accuracy"],
            })

        year = group["Year"].iloc[0]
        weeks.append({
            "week": int(week),
            "year": int(year) if not pd.isna(year) else None,
            "games": games
        })

    weeks = sorted(weeks, key=lambda w: w["week"], reverse=True)

    # Summary metrics
    errors = df["% Error"].dropna()
    metrics = {
        "median_error": float(errors.median()) if len(errors) else None,
        "mean_error": float(errors.mean()) if len(errors) else None,
        "pct_within_10": int((errors < 10).mean() * 100) if len(errors) else None,
        "pct_within_25": int((errors < 25).mean() * 100) if len(errors) else None,
    }

    return clean_nan({
        "weeks": weeks,
        "metrics": metrics
    })