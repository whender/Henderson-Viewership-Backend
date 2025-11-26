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

# Import weekly prediction helpers
from weekly_predictions_fs import (
    generate_prediction,
    calc_error,
    compute_competition_score
)
from firestore_client import db

# ======================================================
# 🚀 FASTAPI SETUP
# ======================================================

app = FastAPI(
    title="Henderson Viewership Model API",
    version="1.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 🧹 JSON SANITIZER
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
    comp_tier1: int = 0   # ignored for main game predictor


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
# 🏆 BRAND RANKINGS (POST-GAME MODEL VERSION)
# ======================================================

numeric_features_post = [
    "Spread","Competing Tier 1","FOX","ESPN","ESPN2","ESPNU","FS1","FS2","NBC","CBS",
    "ABC","BTN","CW","NFLN","ESPNNEWS","Conf Champ","Sun","Monday","Weekday","Friday",
    "Sat Early","Sat Mid","Sat Late","Top 10 Rankings","25-11 Rankings",
    "SEC_PostseasonImplications","Big10_PostseasonImplications",
    "Big12_PostseasonImplications","ACC_PostseasonImplications",
    "YTTV_ABC","YTTV_ESPN",
    "Score Diff"
]

rivalry_features = [
    "Michigan_OhioSt", "Texas_Oklahoma",
    "Alabama_Auburn", "Georgia_Florida",
    "NotreDame_USC", "Florida_Tennessee",
    "Oregon_Washington", "BYU_Utah",
    "Iowa_IowaSt", "OleMiss_MississippiSt",
    "Clemson_SouthCarolina", "Arizona_ArizonaSt",
    "Miami_FloridaSt", "Texas_TexasA&M",
    "Oregon_OregonSt", "USC_UCLA",
    "Louisville_Kentucky", "Washington_WashingtonSt",
    "Kansas_KansasSt", "Minnesota_Wisconsin",
    "Army_Navy", "OhioSt_PennSt",
    "Alabama_LSU"
]

team_conferences = {
    "Alabama":"SEC","Auburn":"SEC","Georgia":"SEC","Florida":"SEC","LSU":"SEC",
    "Tennessee":"SEC","Texas A&M":"SEC","Kentucky":"SEC","South Carolina":"SEC",
    "Mississippi":"SEC","Mississippi St.":"SEC","Arkansas":"SEC","Missouri":"SEC",
    "Vanderbilt":"SEC","Texas":"SEC","Oklahoma":"SEC",

    "Michigan":"Big 10","Ohio St.":"Big 10","Penn St.":"Big 10","Wisconsin":"Big 10",
    "Iowa":"Big 10","Michigan St.":"Big 10","Nebraska":"Big 10","Minnesota":"Big 10",
    "Illinois":"Big 10","Indiana":"Big 10","Purdue":"Big 10","Northwestern":"Big 10",
    "Maryland":"Big 10","Rutgers":"Big 10","UCLA":"Big 10","USC":"Big 10","Oregon":"Big 10",
    "Washington":"Big 10",

    "Clemson":"ACC","Florida St.":"ACC","Miami":"ACC","North Carolina":"ACC","Duke":"ACC",
    "North Carolina St.":"ACC","Virginia":"ACC","Virginia Tech":"ACC","Louisville":"ACC",
    "Syracuse":"ACC","Boston College":"ACC","Wake Forest":"ACC","Pittsburgh":"ACC",
    "Georgia Tech":"ACC","California":"ACC","Stanford":"ACC","SMU":"ACC",

    "BYU":"Big 12","UCF":"Big 12","Houston":"Big 12","Cincinnati":"Big 12","Baylor":"Big 12",
    "Texas Tech":"Big 12","TCU":"Big 12","Kansas":"Big 12","Kansas St.":"Big 12",
    "Iowa St.":"Big 12","Oklahoma St.":"Big 12","West Virginia":"Big 12","Utah":"Big 12",
    "Arizona":"Big 12","Arizona St.":"Big 12","Colorado":"Big 12"
}

power4_set = set(team_conferences.keys()) | {"Notre Dame"}

df_all = pd.read_csv("viewership_cleaned.csv", low_memory=False)

def compute_brand_rankings(df):
    d1 = pd.get_dummies(df["Team 1"])
    d2 = pd.get_dummies(df["Team 2"])
    team_dummies = d1.add(d2, fill_value=0)

    counts = team_dummies.sum()
    valid_teams = counts[counts >= 3].index
    team_dummies = team_dummies[valid_teams]

    if team_dummies.empty:
        return []

    cols = [c for c in df.columns if c in numeric_features_post + rivalry_features]

    X = pd.concat([df[cols], team_dummies], axis=1).fillna(0)
    X = sm.add_constant(X)

    y = np.log(df["Persons 2+"].astype(float) + 1)

    ridge = RidgeCV(alphas=[1.0])
    ridge.fit(X, y)

    params = pd.Series(ridge.coef_, index=X.columns)

    team_coefs = params[team_dummies.columns]
    team_coefs = team_coefs[team_coefs.index.isin(power4_set)]

    counts = counts.reindex(team_coefs.index).fillna(0)
    adjusted = team_coefs.copy()
    for t in team_coefs.index:
        n = counts[t]
        if n <= 4:
            adjusted[t] = team_coefs[t] * (n / (n + 5))

    lift_pct = (np.exp(adjusted) - 1) * 100

    rows = []
    for i, (team, lift) in enumerate(lift_pct.sort_values(ascending=False).items(), start=1):
        rows.append({
            "rank": i,
            "team": team,
            "viewership_lift_pct": float(round(lift, 1)),
            "games_used": int(counts[team])
        })
    return rows


brand_rankings_cache = {
    "all": compute_brand_rankings(df_all)
}

available_years = sorted(df_all["Year"].dropna().unique().tolist())
for y in available_years:
    df_y = df_all[df_all["Year"] == y]
    brand_rankings_cache[str(y)] = compute_brand_rankings(df_y)


@app.get("/brand-years")
def brand_years():
    return {"years": available_years}


@app.get("/brand-rankings")
def brand_rankings(year: str = "all"):
    return {"rows": brand_rankings_cache.get(year, [])}

# ======================================================
# 📅 WEEKLY PREDICTIONS — NOW WITH COMPETITION VARIABLE
# ======================================================

@app.get("/weekly-predictions")
def weekly_predictions():

    docs = db.collection("weekly-predictions").stream()
    weeks_output = []

    for doc in docs:
        data = doc.to_dict()
        week = data["week"]
        year = data["year"]
        games = data["games"]

        updated = False

        for g in games:

            # 1️⃣ AUTOMATIC COMPETITION SCORE (new)
            g["comp_tier1"] = compute_competition_score(games, g)

            # 2️⃣ Prediction (only if missing)
            if not g.get("predicted") or g["predicted"] in ["", None, "nan", "NaN"]:
                g["predicted"] = generate_prediction(g)
                updated = True

            # 3️⃣ Error + accuracy
            g["percent_error"] = calc_error(g["predicted"], g.get("actual"))
            e = g["percent_error"]

            if e is None:
                g["accuracy"] = ""
            elif e < 5:
                g["accuracy"] = "🟢🎯"
            elif e < 25:
                g["accuracy"] = "🟢"
            elif e < 35:
                g["accuracy"] = "🟡"
            else:
                g["accuracy"] = "🔴"

        if updated:
            db.collection("weekly-predictions").document(doc.id).set(data)

        weeks_output.append({
            "week": week,
            "year": year,
            "games": games
        })

    # Summary stats
    all_errors = []
    for w in weeks_output:
        for g in w["games"]:
            e = g.get("percent_error")
            if isinstance(e, (int, float)) and not math.isnan(e):
                all_errors.append(e)

    if len(all_errors) > 0:
        median_error = float(np.median(all_errors))
        mean_error = float(np.mean(all_errors))
        pct10 = int(np.mean([e < 10 for e in all_errors]) * 100)
        pct25 = int(np.mean([e < 25 for e in all_errors]) * 100)
    else:
        median_error = None
        mean_error = None
        pct10 = None
        pct25 = None

    metrics = {
        "median_error": median_error,
        "mean_error": mean_error,
        "pct_within_10": pct10,
        "pct_within_25": pct25
    }

    weeks_output = sorted(weeks_output, key=lambda w: w["week"], reverse=True)

    return clean_nan({
        "weeks": weeks_output,
        "metrics": metrics
    })