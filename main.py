from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
import os

# ======================================================
# 🚀 FASTAPI SETUP
# ======================================================

app = FastAPI(
    title="Henderson Viewership Model API",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 📦 IMPORT VIEWERSHIP PREDICTION LOGIC
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
# 🧠 BRAND RANKINGS — PRECOMPUTED AT SERVER START
# ======================================================

# ---------- these MUST match your Streamlit dashboard ----------
numeric_features = [
    "Spread","Competing Tier 1",
    "FOX","ESPN","ESPN2","ESPNU","FS1","FS2","NBC","CBS","ABC","BTN","CW","NFLN","ESPNNEWS",
    "Conf Champ","Sun","Monday","Weekday","Friday","Sat Early","Sat Mid","Sat Late",
    "Top 10 Rankings","25-11 Rankings",
    "SEC_PostseasonImplications","Big10_PostseasonImplications",
    "Big12_PostseasonImplications","ACC_PostseasonImplications",
    "YTTV_ABC","YTTV_ESPN"
]

rivalry_features = [
    "Michigan_OhioSt","Texas_Oklahoma","Alabama_Auburn","Georgia_Florida",
    "NotreDame_USC","Florida_Tennessee","Oregon_Washington","BYU_Utah",
    "Iowa_IowaSt","OleMiss_MississippiSt","Clemson_SouthCarolina",
    "Arizona_ArizonaSt","Miami_FloridaSt","Texas_TexasA&M","Oregon_OregonSt",
    "USC_UCLA","Louisville_Kentucky","OhioSt_PennSt","Alabama_LSU"
]

# ---------- Conference mapping pulled from Streamlit ----------
team_conferences = {
    # SEC
    "Alabama":"SEC","Auburn":"SEC","Georgia":"SEC","Florida":"SEC","LSU":"SEC",
    "Tennessee":"SEC","Texas A&M":"SEC","Kentucky":"SEC","South Carolina":"SEC",
    "Mississippi":"SEC","Mississippi St.":"SEC","Arkansas":"SEC","Missouri":"SEC",
    "Vanderbilt":"SEC","Texas":"SEC","Oklahoma":"SEC",
    # Big 10
    "Michigan":"Big 10","Ohio St.":"Big 10","Penn St.":"Big 10","Wisconsin":"Big 10",
    "Iowa":"Big 10","Michigan St.":"Big 10","Nebraska":"Big 10","Minnesota":"Big 10",
    "Illinois":"Big 10","Indiana":"Big 10","Purdue":"Big 10","Northwestern":"Big 10",
    "Maryland":"Big 10","Rutgers":"Big 10","UCLA":"Big 10","USC":"Big 10",
    "Oregon":"Big 10","Washington":"Big 10",
    # ACC
    "Clemson":"ACC","Florida St.":"ACC","Miami":"ACC","North Carolina":"ACC",
    "Duke":"ACC","North Carolina St.":"ACC","Virginia":"ACC","Virginia Tech":"ACC",
    "Louisville":"ACC","Syracuse":"ACC","Boston College":"ACC","Wake Forest":"ACC",
    "Pittsburgh":"ACC","Georgia Tech":"ACC","California":"ACC","Stanford":"ACC","SMU":"ACC",
    # Big 12
    "BYU":"Big 12","UCF":"Big 12","Houston":"Big 12","Cincinnati":"Big 12",
    "Baylor":"Big 12","Texas Tech":"Big 12","TCU":"Big 12","Kansas":"Big 12",
    "Kansas St.":"Big 12","Iowa St.":"Big 12","Oklahoma St.":"Big 12",
    "West Virginia":"Big 12","Utah":"Big 12","Arizona":"Big 12","Arizona St.":"Big 12",
    "Colorado":"Big 12"
}

# ======================================================
# 📄 LOAD CLEANED DATASET
# ======================================================
cleaned_path = "viewership_cleaned.csv"

if not os.path.exists(cleaned_path):
    raise RuntimeError("Missing viewership_cleaned.csv — required for brand rankings")

df_all = pd.read_csv(cleaned_path, low_memory=False)


# ======================================================
# 🧮 BRAND RANKINGS ENGINE
# ======================================================

def compute_brand_rankings(df):
    """Compute RidgeCV brand lift for any subset of the dataset."""

    power4_set = set(team_conferences.keys()) | {"Notre Dame"}

    # team dummies (Team1 + Team2)
    d1 = pd.get_dummies(df["Team 1"])
    d2 = pd.get_dummies(df["Team 2"])
    team_dummies = d1.add(d2, fill_value=0)

    valid_counts = team_dummies.sum()
    team_dummies = team_dummies[valid_counts[valid_counts >= 3].index]

    if team_dummies.empty:
        return []

    # feature columns
    feature_cols = [c for c in df.columns if c in numeric_features + rivalry_features]

    X = pd.concat([df[feature_cols], team_dummies], axis=1).fillna(0)
    X = sm.add_constant(X)

    # log viewers
    y = np.log(df["Persons 2+"].astype(float) + 1)

    # one alpha instead of [0.1, 1, 10] → MUCH faster
    ridge = RidgeCV(alphas=[1.0])
    ridge.fit(X, y)

    params = pd.Series(ridge.coef_, index=X.columns)
    team_coefs = params[team_dummies.columns]
    team_coefs = team_coefs[team_coefs.index.isin(power4_set)]

    valid_counts = valid_counts.reindex(team_coefs.index).fillna(0)

    # Shrink small sample teams
    adjusted = team_coefs.copy()
    for t in team_coefs.index:
        n = valid_counts[t]
        if n <= 4:
            adjusted[t] = team_coefs[t] * (n / (n + 5))

    # convert to % lift
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


# ======================================================
# 🚀 PRECOMPUTE BRAND RANKINGS ON SERVER START
# ======================================================

brand_rankings_cache = {}

available_years = sorted(df_all["Year"].dropna().unique().tolist())

# “All Years”
brand_rankings_cache["all"] = compute_brand_rankings(df_all)

# Individual years
for y in available_years:
    brand_rankings_cache[str(y)] = compute_brand_rankings(df_all[df_all["Year"] == y])


# ======================================================
# 📡 API ENDPOINTS — LIGHTNING FAST NOW
# ======================================================

@app.get("/brand-years")
def brand_years():
    return {"years": available_years}


@app.get("/brand-rankings")
def brand_rankings(year: str = "all"):
    if year not in brand_rankings_cache:
        return {"year": year, "rows": []}
    return {"year": year, "rows": brand_rankings_cache[year]}