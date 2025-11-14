from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV

from predict import predict_viewership
from predict import teams_list  # dict: frontend_name -> backend_name

app = FastAPI(
    title="Henderson Viewership Model API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔢 GAME PREDICTOR SCHEMA
# =========================
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
    """
    Returns list of teams for the dropdowns:
    [{ "label": "BYU", "value": "BYU" }, ...]
    """
    return {
        "teams": [
            {"label": frontend_name, "value": backend_name}
            for frontend_name, backend_name in teams_list.items()
        ]
    }


@app.post("/predict")
def predict_game(game: GameInput):
    """
    Game predictor endpoint (same behavior as Streamlit tab 1).
    """
    result = predict_viewership(game.dict())
    return {
        "prediction_raw": result["raw"],
        "prediction_formatted": result["formatted"]
    }

# ==========================================
# 🏆 BRAND RANKINGS (Power 4 + Notre Dame)
# ==========================================

# These match your Streamlit Brand Rankings tab
numeric_features = [
    "Spread", "Competing Tier 1",
    "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "FS2", "NBC", "CBS", "ABC", "BTN", "CW", "NFLN", "ESPNNEWS",
    "Conf Champ", "Sun", "Monday", "Weekday", "Friday", "Sat Early", "Sat Mid", "Sat Late",
    "Top 10 Rankings", "25-11 Rankings",
    "SEC_PostseasonImplications", "Big10_PostseasonImplications", "Big12_PostseasonImplications", "ACC_PostseasonImplications",
    "YTTV_ABC", "YTTV_ESPN"
]

rivalry_features = [
    "Michigan_OhioSt", "Texas_Oklahoma", "Alabama_Auburn", "Georgia_Florida",
    "NotreDame_USC", "Florida_Tennessee", "Oregon_Washington", "BYU_Utah",
    "Iowa_IowaSt", "OleMiss_MississippiSt", "Clemson_SouthCarolina",
    "Arizona_ArizonaSt", "Miami_FloridaSt", "Texas_TexasA&M", "Oregon_OregonSt",
    "USC_UCLA", "Louisville_Kentucky", "OhioSt_PennSt", "Alabama_LSU"
]

# Same Power 4 conference map you used in Streamlit
team_conferences = {
    # SEC
    "Alabama": "SEC", "Auburn": "SEC", "Georgia": "SEC", "Florida": "SEC", "LSU": "SEC",
    "Tennessee": "SEC", "Texas A&M": "SEC", "Kentucky": "SEC", "South Carolina": "SEC",
    "Mississippi": "SEC", "Mississippi St.": "SEC", "Arkansas": "SEC", "Missouri": "SEC",
    "Vanderbilt": "SEC", "Texas": "SEC", "Oklahoma": "SEC",
    # Big 10
    "Michigan": "Big 10", "Ohio St.": "Big 10", "Penn St.": "Big 10", "Wisconsin": "Big 10",
    "Iowa": "Big 10", "Michigan St.": "Big 10", "Nebraska": "Big 10", "Minnesota": "Big 10",
    "Illinois": "Big 10", "Indiana": "Big 10", "Purdue": "Big 10", "Northwestern": "Big 10",
    "Maryland": "Big 10", "Rutgers": "Big 10", "UCLA": "Big 10", "USC": "Big 10",
    "Oregon": "Big 10", "Washington": "Big 10",
    # ACC
    "Clemson": "ACC", "Florida St.": "ACC", "Miami": "ACC", "North Carolina": "ACC",
    "Duke": "ACC", "North Carolina St.": "ACC", "Virginia": "ACC", "Virginia Tech": "ACC",
    "Louisville": "ACC", "Syracuse": "ACC", "Boston College": "ACC", "Wake Forest": "ACC",
    "Pittsburgh": "ACC", "Georgia Tech": "ACC", "California": "ACC", "Stanford": "ACC", "SMU": "ACC",
    # Big 12
    "BYU": "Big 12", "UCF": "Big 12", "Houston": "Big 12", "Cincinnati": "Big 12",
    "Baylor": "Big 12", "Texas Tech": "Big 12", "TCU": "Big 12", "Kansas": "Big 12",
    "Kansas St.": "Big 12", "Iowa St.": "Big 12", "Oklahoma St.": "Big 12",
    "West Virginia": "Big 12", "Utah": "Big 12", "Arizona": "Big 12", "Arizona St.": "Big 12",
    "Colorado": "Big 12",
}

# Load the cleaned data once on startup (same CSV as Streamlit tab 2)
try:
    df_all = pd.read_csv("viewership_cleaned.csv", low_memory=False)
except FileNotFoundError:
    df_all = None


@app.get("/brand-years")
def brand_years():
    """
    Return the list of distinct years available in viewership_cleaned.csv.
    Used to populate the year dropdown.
    """
    if df_all is None or "Year" not in df_all.columns:
        return {"years": []}

    years = sorted(df_all["Year"].dropna().unique().tolist())
    return {"years": years}


@app.get("/brand-rankings")
def brand_rankings(year: Optional[str] = "all"):
    """
    Recreates the Brand Rankings tab logic from Streamlit.

    Query param:
      - year = "all" (default) or a specific year like "2024"
    """
    if df_all is None:
        return {"year": year, "rows": []}

    # ---- Filter by year (same as Streamlit) ----
    if year is None or year == "" or year.lower() == "all":
        df = df_all.copy()
        year_label = "All Years"
    else:
        try:
            yr = int(year)
        except ValueError:
            yr = None

        if yr is None:
            df = df_all.copy()
            year_label = "All Years"
        else:
            df = df_all[df_all["Year"] == yr].copy()
            year_label = str(yr)

    if df.empty:
        return {"year": year_label, "rows": []}

    # ---- Team universe: Power 4 + Notre Dame ----
    power4_set = set(team_conferences.keys()) | {"Notre Dame"}

    # Team dummies (Team 1 + Team 2)
    team_dummies_1 = pd.get_dummies(df["Team 1"])
    team_dummies_2 = pd.get_dummies(df["Team 2"])
    team_dummies = team_dummies_1.add(team_dummies_2, fill_value=0)

    # Require >= 3 games for stability
    valid_counts = team_dummies.sum()
    team_dummies = team_dummies[valid_counts[valid_counts >= 3].index]

    if team_dummies.empty:
        return {"year": year_label, "rows": []}

    # Feature matrix: numeric features + rivalry features + team dummies
    feature_cols = [c for c in df.columns if c in (numeric_features + rivalry_features)]
    X = pd.concat([df[feature_cols], team_dummies], axis=1).fillna(0.0)
    X = sm.add_constant(X)

    # Target: log viewers
    y = np.log(df["Persons 2+"].astype(float) + 1.0)

    # Ridge regression (same as Streamlit)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X, y)

    # Coefficients aligned to column names
    params = pd.Series(ridge.coef_, index=X.columns)

    team_coefs = params[team_dummies.columns]
    team_coefs = team_coefs[team_coefs.index.isin(power4_set)]

    if team_coefs.empty:
        return {"year": year_label, "rows": []}

    valid_counts = valid_counts.reindex(team_coefs.index).fillna(0)

    # Shrink small sample sizes (n <= 4) like your Streamlit version
    adjusted = team_coefs.copy()
    for t in team_coefs.index:
        n = valid_counts[t]
        if n <= 4:
            adjusted[t] = team_coefs[t] * (n / (n + 5))

    # Convert to % lift: (exp(beta) - 1) * 100
    boost_pct = (np.exp(adjusted) - 1.0) * 100.0

    brand_df = (
        pd.DataFrame({
            "Team": team_coefs.index,
            "Viewership Lift (%)": boost_pct.round(1),
            "Games Used": valid_counts[team_coefs.index].values,
        })
        .sort_values(by="Viewership Lift (%)", ascending=False)
        .reset_index(drop=True)
    )
    brand_df.insert(0, "Rank", range(1, len(brand_df) + 1))

    # Convert to nice JSON structure for React
    rows = []
    for _, r in brand_df.iterrows():
        rows.append({
            "rank": int(r["Rank"]),
            "team": str(r["Team"]),
            "viewership_lift_pct": float(r["Viewership Lift (%)"]),
            "games_used": int(r["Games Used"]),
        })

    return {
        "year": year_label,
        "rows": rows,
    }