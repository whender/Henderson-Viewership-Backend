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

# ======================================================
# 📅 WEEKLY PREDICTIONS ENDPOINT
# ======================================================
import pandas as pd
import numpy as np
from fastapi import APIRouter
from datetime import datetime

from predict import (
    model,
    normalize_team,
    rank_to_coefs,
    team_conferences,
    rivalries,
    FEUD_START,
    FEUD_END,
    MODEL_TEAM_NAMES,
    teams_list,
    format_viewers
)

router = APIRouter()

@router.get("/weekly-predictions")
def weekly_predictions():
    # Load CSV
    df = pd.read_csv("weekly_predictions.csv")

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Ensure column exists
    if "Predicted Viewers" not in df.columns:
        df["Predicted Viewers"] = ""

    # Normalize blanks
    df["Predicted Viewers"] = df["Predicted Viewers"].replace(["nan", "NaN"], "")

    # Helper for computing % error
    def parse_viewership(val):
        try:
            if pd.isna(val) or val is None:
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

    # Main prediction loop
    new_preds = []
    for _, row in df.iterrows():
        val = row.get("Predicted Viewers", "")
        if str(val).strip() not in ["", "nan", "NaN"]:
            new_preds.append(val)
            continue

        # Extract row data
        try:
            team1 = normalize_team(row["Team 1"])
            team2 = normalize_team(row["Team 2"])
            rank1 = int(row["Rank 1"])
            rank2 = int(row["Rank 2"])

            spread = float(row["Spread"])
            network = row["Network"]
            time_slot = str(row["Time Slot"])
            comp_tier1 = int(row.get("Competing Tier 1", 0))

            conf1 = team_conferences.get(team1, "Group of 6")
            conf2 = team_conferences.get(team2, "Group of 6")

            both_ranked = (rank1 > 0 and rank2 > 0)
            same_conf = (conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"])

            t1_top10, t1_25_11 = rank_to_coefs(rank1)
            t2_top10, t2_25_11 = rank_to_coefs(rank2)
            top10 = t1_top10 + t2_top10
            rank_25_11 = t1_25_11 + t2_25_11

            is_friday = "fri" in str(row["Day"]).lower()

            # Identify rivalry
            auto_rivalry = next(
                (r for r, (a, b) in rivalries.items() if {team1, team2} == {a, b}),
                None
            )

            # Build feature dict exactly like Streamlit
            features = {
                "Spread": spread,
                "Competing Tier 1": comp_tier1,

                "ABC": int(network == "ABC"), "CBS": int(network == "CBS"),
                "NBC": int(network == "NBC"), "FOX": int(network == "FOX"),
                "ESPN": int(network == "ESPN"), "ESPN2": int(network == "ESPN2"),
                "ESPNU": int(network == "ESPNU"), "FS1": int(network == "FS1"),
                "FS2": int(network == "FS2"), "BTN": int(network == "BTN"),
                "NFLN": int(network == "NFLN"), "CW": int(network == "CW"),
                "ESPNNEWS": int(network == "ESPNNEWS"),

                "Sun": int("Sunday" in time_slot),
                "Monday": int("Monday" in time_slot),
                "Weekday": int("Weekday" in time_slot),
                "Friday": int("Friday" in time_slot or is_friday),
                "Sat Early": int("Early" in time_slot),
                "Sat Mid": int("Mid" in time_slot),
                "Sat Late": int("Late" in time_slot),

                "Top 10 Rankings": top10,
                "25-11 Rankings": rank_25_11,

                "SEC": (conf1 == "SEC") + (conf2 == "SEC"),
                "Big 10": (conf1 == "Big 10") + (conf2 == "Big 10"),
                "ACC": (conf1 == "ACC") + (conf2 == "ACC"),
                "Big 12": (conf1 == "Big 12") + (conf2 == "Big 12"),
            }

            # Add postseason flags
            for conf_tag, flag_name in {
                "SEC": "SEC_PostseasonImplications",
                "Big 10": "Big10_PostseasonImplications",
                "Big 12": "Big12_PostseasonImplications",
                "ACC": "ACC_PostseasonImplications",
            }.items():
                features[flag_name] = int(both_ranked and same_conf and conf1 == conf_tag)

            # Rivalry features
            for r in rivalries:
                features[r] = int(r == auto_rivalry)

            # YouTube TV blackout
            features["YTTV_ABC"] = 0
            features["YTTV_ESPN"] = 0
            now = datetime.now()
            if FEUD_START <= now <= FEUD_END:
                if network in ["ABC", "ESPN"]:
                    features[f"YTTV_{network}"] = 1

            # Team dummy columns
            for col in model.params.index:
                if col in MODEL_TEAM_NAMES:
                    features[col] = int(col in [team1, team2])

            features["const"] = 1.0
            for c in model.params.index:
                if c not in features:
                    features[c] = 0.0

            # BTN × OSU interaction
            if "OhioSt_BTN" in model.params.index:
                features["OhioSt_BTN"] = int("Ohio St." in [team1, team2] and network == "BTN")

            # Create input DF
            X = pd.DataFrame([[features[c] for c in model.params.index]], columns=model.params.index)

            # Predict w/ CI
            try:
                pred_res = model.get_prediction(X)
                ci = pred_res.summary_frame(alpha=0.32)

                smearing = getattr(model, "smearing_factor", 1.0)

                pred_ln = ci["mean"].iloc[0]
                ci_low_ln = ci["obs_ci_lower"].iloc[0]
                ci_high_ln = ci["obs_ci_upper"].iloc[0]

                pred = (np.exp(pred_ln) - 1) * smearing
                low = max(0, (np.exp(ci_low_ln) - 1) * smearing)
                high = max(low, (np.exp(ci_high_ln) - 1) * smearing)

            except:
                pred = float(model.predict(X)[0])
                low = pred - 500
                high = pred + 500

            pred_fmt = f"{pred/1_000:.2f}M ({low/1_000:.2f}–{high/1_000:.2f}M)"
            new_preds.append(pred_fmt)

        except Exception as e:
            new_preds.append(f"Error: {e}")

    # Assign new predictions back to blank rows
    for i, val in enumerate(new_preds):
        if str(df.loc[i, "Predicted Viewers"]).strip() in ["", "nan", "NaN"]:
            df.loc[i, "Predicted Viewers"] = val

    # Save CSV (same behavior as Streamlit)
    df.to_csv("weekly_predictions.csv", index=False)

    # Compute % error & accuracy
    def calc_error(pred_str, actual_str):
        p = parse_viewership(pred_str)
        a = parse_viewership(actual_str)
        if p is None or a is None or a <= 0:
            return None
        return abs((p - a) / a) * 100

    df["% Error"] = [
        calc_error(p, a) for p, a in zip(df["Predicted Viewers"], df["Actual Viewers"])
    ]

    def indicator(e):
        if e is None: return ""
        if e < 5: return "🟢🎯"
        if e < 25: return "🟢"
        if e < 35: return "🟡"
        return "🔴"

    df["Accuracy"] = [indicator(e) for e in df["% Error"]]

    # Format matchup
    def format_matchup(t1, r1, t2, r2):
        t1s = f"#{int(r1)} {t1}" if r1 > 0 else t1
        t2s = f"#{int(r2)} {t2}" if r2 > 0 else t2
        return f"{t1s} @ {t2s}"

    df["Matchup"] = [
        format_matchup(t1, r1, t2, r2)
        for t1, r1, t2, r2 in zip(df["Team 1"], df["Rank 1"], df["Team 2"], df["Rank 2"])
    ]

    # Format date
    df["DateFmt"] = [
        f"{day}, {pd.to_datetime(date).strftime('%b %-d')}"
        for date, day in zip(df["Date"], df["Day"])
    ]

    # Group by week
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

    # Sort weeks descending
    weeks = sorted(weeks, key=lambda w: w["week"], reverse=True)

    # Summary stats
    errors = df["% Error"].dropna()
    metrics = {
        "median_error": float(errors.median()) if len(errors) else 0,
        "mean_error": float(errors.mean()) if len(errors) else 0,
        "pct_within_10": int((errors < 10).mean() * 100) if len(errors) else 0,
        "pct_within_25": int((errors < 25).mean() * 100) if len(errors) else 0,
    }

    return {
        "weeks": weeks,
        "metrics": metrics
    }


app.include_router(router)