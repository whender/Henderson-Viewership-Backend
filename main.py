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

from weekly_predictions_fs import (
    generate_pregame_prediction,
    generate_postgame_prediction,
    calc_error,
    build_features,
    parse_viewership
)

from firestore_client import db

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
# 🧹 JSON SANITIZER
# ======================================================

def clean_nan(obj):
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
# 📦 IMPORT PREGAME PREDICTOR LOGIC
# ======================================================
from predict import predict_viewership, teams_list, normalize_team, model as pregame_model, MODEL_TEAM_NAMES

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
# 🏆 BRAND RANKINGS (POSTGAME MODEL)
# ======================================================

numeric_features_post = [
    "Spread","Competing Tier 1","FOX","ESPN","ESPN2","ESPNU","FS1","FS2","NBC","CBS",
    "ABC","BTN","CW","NFLN","ESPNNEWS",
    "SEC_ConfChamp","Big10_ConfChamp","Big12_ConfChamp","ACC_ConfChamp","Other_ConfChamp",
    "Sun","Monday","Weekday","Friday",
    "Sat Early","Sat Mid","Sat Late","Top 10 Rankings","25-11 Rankings",
    "SEC_PostseasonImplications","Big10_PostseasonImplications",
    "Big12_PostseasonImplications","ACC_PostseasonImplications",
    "YTTV_ABC","YTTV_ESPN","Score Diff"
]

rivalry_features = [
    "Michigan_OhioSt","Texas_Oklahoma","Alabama_Auburn","Georgia_Florida",
    "NotreDame_USC","Florida_Tennessee","Oregon_Washington","BYU_Utah",
    "Iowa_IowaSt","OleMiss_MississippiSt","Clemson_SouthCarolina","Arizona_ArizonaSt",
    "Miami_FloridaSt","Texas_TexasA&M","Oregon_OregonSt","USC_UCLA",
    "Louisville_Kentucky","Washington_WashingtonSt","Kansas_KansasSt",
    "Minnesota_Wisconsin","Army_Navy","Army_AirForce","Navy_AirForce",
    "OhioSt_PennSt","Alabama_LSU"
]

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
    "Pittsburgh":"ACC","Georgia Tech":"ACC","California":"ACC",
    "Stanford":"ACC","SMU":"ACC",

    # Big 12
    "BYU":"Big 12","UCF":"Big 12","Houston":"Big 12","Cincinnati":"Big 12",
    "Baylor":"Big 12","Texas Tech":"Big 12","TCU":"Big 12","Kansas":"Big 12",
    "Kansas St.":"Big 12","Iowa St.":"Big 12","Oklahoma St.":"Big 12",
    "West Virginia":"Big 12","Utah":"Big 12","Arizona":"Big 12",
    "Arizona St.":"Big 12","Colorado":"Big 12",

    # AAC
    "Charlotte":"AAC","East Carolina":"AAC","FAU":"AAC","Memphis":"AAC",
    "North Texas":"AAC","Rice":"AAC","South Florida":"AAC","Temple":"AAC",
    "Tulane":"AAC","Tulsa":"AAC","UAB":"AAC","UTSA":"AAC","Army":"AAC",
    "Navy":"AAC",

    # Sun Belt
    "Appalachian St.":"Sun Belt","Arkansas St.":"Sun Belt","Coastal Carolina":"Sun Belt",
    "Georgia Southern":"Sun Belt","Georgia St.":"Sun Belt","James Madison":"Sun Belt",
    "Louisiana":"Sun Belt","Louisiana Monroe":"Sun Belt","Marshall":"Sun Belt",
    "Old Dominion":"Sun Belt","South Alabama":"Sun Belt","Southern Miss":"Sun Belt",
    "Texas St.":"Sun Belt","Troy":"Sun Belt",

    # Mountain West
    "Air Force":"Mountain West","Boise St.":"Mountain West","Colorado St.":"Mountain West",
    "Fresno St.":"Mountain West","Hawaii":"Mountain West","Nevada":"Mountain West",
    "New Mexico":"Mountain West","San Diego St.":"Mountain West","San Jose St.":"Mountain West",
    "UNLV":"Mountain West","Utah St.":"Mountain West","Wyoming":"Mountain West",

    # MAC
    "Akron":"MAC","Ball St.":"MAC","Bowling Green":"MAC","Buffalo":"MAC",
    "Central Michigan":"MAC","Eastern Michigan":"MAC","Kent St.":"MAC",
    "Miami Ohio":"MAC","Northern Illinois":"MAC","Ohio":"MAC","Toledo":"MAC",
    "Western Michigan":"MAC",

    # Conference USA
    "Delaware":"Conference USA","FIU":"Conference USA","Jacksonville St.":"Conference USA",
    "Kennesaw State":"Conference USA","Liberty":"Conference USA","Louisiana Tech":"Conference USA",
    "Middle Tennessee St.":"Conference USA","New Mexico St.":"Conference USA",
    "Sam Houston":"Conference USA","UTEP":"Conference USA","Missouri State":"Conference USA",
    "Kennesaw St.":"Conference USA","Western Kentucky":"Conference USA",

    # Independents / other FBS
    "Massachusetts":"Independent","Notre Dame":"Independent","Connecticut":"Independent",
    "Washington St.":"Pac-12","Oregon St.":"Pac-12"
}

power4_set = set(team_conferences.keys()) | {"Notre Dame"}
excluded_brand_teams = {"North Dakota"}
excluded_viewership_ranking_teams = {
    "Western Carolina",
    "North Alabama",
    "Youngstown State",
    "Austin Peay",
    "South Dakota State",
    "South Dakota",
    "Illinois State",
    "Idaho",
    "North Dakota State",
    "Lafayette",
    "UT Martin",
    "Howard",
    "Portland State",
    "Indiana State",
    "VMI",
    "Chattanooga",
    "Cal Poly",
    "McNeese",
    "Lamar",
    "Eastern Illinois",
    "Alcorn State",
}

df_all = pd.read_csv("viewership_cleaned.csv", low_memory=False)

NETWORK_LABELS = [
    "ABC", "CBS", "NBC", "FOX", "ESPN2",
    "ESPNU", "FS1", "FS2", "BTN", "CW", "NFLN", "ESPNNEWS"
]


def detect_network(row):
    return next(
        (network for network in NETWORK_LABELS if row.get(network) == 1),
        "ESPN",
    )


def detect_time_bucket(row):
    if row.get("Sun") == 1:
        return "Sunday"
    if row.get("Monday") == 1:
        return "Monday"
    if row.get("Weekday") == 1:
        return "Weekday"
    if row.get("Friday") == 1:
        return "Friday"
    if row.get("Sat Early") == 1:
        return "Sat Early"
    if row.get("Sat Mid") == 1:
        return "Sat Mid"
    if row.get("Sat Late") == 1:
        return "Sat Late"
    return "Other"


def is_primetime_baseline(row):
    return not any(
        row.get(flag) == 1
        for flag in ["Sun", "Monday", "Weekday", "Friday", "Sat Early", "Sat Mid", "Sat Late"]
    )


def display_time_bucket(row):
    if is_primetime_baseline(row):
        return "Primetime"
    return row.get("scenario_time_bucket") or "Other"


def detect_rank_bucket(row):
    top_10 = pd.to_numeric(row.get("Top 10 Rankings"), errors="coerce")
    rank_11_25 = pd.to_numeric(row.get("25-11 Rankings"), errors="coerce")
    total_ranked = (0 if pd.isna(top_10) else top_10) + (0 if pd.isna(rank_11_25) else rank_11_25)

    if total_ranked >= 2:
        return "Both Ranked"
    if total_ranked == 1:
        return "One Ranked"
    return "Unranked"


def detect_rank_detail(row):
    top_10 = int(pd.to_numeric(row.get("Top 10 Rankings"), errors="coerce") or 0)
    rank_11_25 = int(pd.to_numeric(row.get("25-11 Rankings"), errors="coerce") or 0)
    return f"{top_10} Top 10 / {rank_11_25} ranked 11-25"


def detect_competing_tier(row):
    comp = pd.to_numeric(row.get("Competing Tier 1"), errors="coerce")
    if pd.isna(comp) or comp <= 0:
        return "None"
    if comp == 1:
        return "1 Major Game"
    return "2+ Major Games"


df_all["scenario_network"] = df_all.apply(detect_network, axis=1)
df_all["scenario_time_bucket"] = df_all.apply(detect_time_bucket, axis=1)
df_all["scenario_rank_bucket"] = df_all.apply(detect_rank_bucket, axis=1)
df_all["scenario_rank_detail"] = df_all.apply(detect_rank_detail, axis=1)
df_all["scenario_competing_bucket"] = df_all.apply(detect_competing_tier, axis=1)
df_all["scenario_conf_champ"] = df_all["Conf Champ"].fillna(0).astype(int)


def _compute_average_team_effect(df):
    team_columns = [column for column in pregame_model.params.index if column in MODEL_TEAM_NAMES]
    team_effects = pd.to_numeric(pregame_model.params.reindex(team_columns), errors="coerce").dropna()

    team_dummies_1 = pd.get_dummies(df["Team 1"])
    team_dummies_2 = pd.get_dummies(df["Team 2"])
    team_dummies = team_dummies_1.add(team_dummies_2, fill_value=0)
    team_counts = team_dummies.sum()
    valid_teams = set(team_counts[team_counts >= 5].index)
    modeled_teams = set(team_columns)
    omitted_baseline_teams = valid_teams - modeled_teams

    total_effect = float(team_effects.sum())
    total_count = len(team_effects) + len(omitted_baseline_teams)

    if total_count == 0:
        return 0.0

    return total_effect / total_count


AVERAGE_TEAM_EFFECT = _compute_average_team_effect(df_all)
MODELED_TEAM_COLUMNS = {column for column in pregame_model.params.index if column in MODEL_TEAM_NAMES}


def _build_pregame_design_matrix(df):
    model_columns = list(pregame_model.params.index)
    X = pd.DataFrame(index=df.index)

    for column in model_columns:
        if column == "const":
            X[column] = 1.0
        elif column in df.columns:
            X[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(float)
        else:
            X[column] = 0.0

    team_dummies = pd.get_dummies(df["Team 1"]).add(pd.get_dummies(df["Team 2"]), fill_value=0)
    team_dummies = team_dummies.reindex(columns=[c for c in model_columns if c in MODELED_TEAM_COLUMNS], fill_value=0)

    for column in team_dummies.columns:
        X[column] = team_dummies[column].astype(float)

    return X[model_columns]


def _predict_viewers_from_design_matrix(X, team_effect_adjustment):
    ln_pred = pregame_model.predict(X) + team_effect_adjustment
    smearing = getattr(pregame_model, "smearing_factor", 1.0)
    expected = (np.exp(ln_pred) - 1) * smearing
    return pd.to_numeric(expected, errors="coerce")


def _compute_expected_viewers(df):
    X = _build_pregame_design_matrix(df)

    # Neutral matchup baseline: replace both teams with an average-FBS-team effect.
    for column in MODELED_TEAM_COLUMNS:
        if column in X.columns:
            X[column] = 0.0

    return _predict_viewers_from_design_matrix(X, 2 * AVERAGE_TEAM_EFFECT)


def _compute_team_specific_expected_viewers(df, focal_team_column):
    X = _build_pregame_design_matrix(df)
    focal_teams = df[focal_team_column].fillna("")

    for team_name in focal_teams[focal_teams.isin(MODELED_TEAM_COLUMNS)].unique():
        X.loc[focal_teams == team_name, team_name] = 0.0

    # Replace only the focal team's brand effect with an average-FBS-team baseline.
    return _predict_viewers_from_design_matrix(X, AVERAGE_TEAM_EFFECT)


df_all["expected_viewers"] = _compute_expected_viewers(df_all)
df_all["expected_viewers_team1"] = _compute_team_specific_expected_viewers(df_all, "Team 1")
df_all["expected_viewers_team2"] = _compute_team_specific_expected_viewers(df_all, "Team 2")
df_all["actual_minus_expected"] = (
    pd.to_numeric(df_all["Persons 2+"], errors="coerce") - df_all["expected_viewers"]
)
df_all["actual_vs_expected_pct"] = np.where(
    df_all["expected_viewers"] > 0,
    ((pd.to_numeric(df_all["Persons 2+"], errors="coerce") / df_all["expected_viewers"]) - 1) * 100,
    np.nan,
)

rank_detail_options = ["all"] + sorted(
    df_all["scenario_rank_detail"].dropna().unique().tolist(),
    key=lambda value: (
        int(str(value).split(" Top 10 / ")[0]),
        int(str(value).split(" / ")[1].split(" ranked")[0]),
    ),
)

def compute_brand_rankings(df):
    d1 = pd.get_dummies(df["Team 1"])
    d2 = pd.get_dummies(df["Team 2"])
    team_dummies = d1.add(d2, fill_value=0)

    counts = team_dummies.sum()
    valid = counts[counts >= 3].index
    team_dummies = team_dummies[valid]

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
    team_coefs = team_coefs[~team_coefs.index.isin(excluded_brand_teams)]

    counts = counts.reindex(team_coefs.index).fillna(0)
    adjusted = team_coefs.copy()

    for t in team_coefs.index:
        n = counts[t]
        if n <= 25:
            adjusted[t] = team_coefs[t] * (n / (n + 10))

    lift_pct = (np.exp(adjusted) - 1) * 100

    rows = []
    for i, (team, lift) in enumerate(lift_pct.sort_values(ascending=False).items(), start=1):
        rows.append({
            "rank": i,
            "team": team,
            "conference": team_conferences.get(team, "Independent"),
            "viewership_lift_pct": float(round(lift, 1)),
            "games_used": int(counts[team])
        })
    return rows

def _load_csv_brand_rankings():
    csv_path = os.path.join(os.path.dirname(__file__), "brand_rankings.csv")

    if os.path.exists(csv_path):
        try:
            brand_df = pd.read_csv(csv_path)
            if {"Team", "Adjusted (Shrinkage)"}.issubset(brand_df.columns):
                brand_df = brand_df[~brand_df["Team"].isin(excluded_brand_teams)]
                brand_df["Adjusted (Shrinkage)"] = pd.to_numeric(
                    brand_df["Adjusted (Shrinkage)"], errors="coerce"
                )
                brand_df = brand_df.dropna(subset=["Team", "Adjusted (Shrinkage)"])
                brand_df = brand_df.sort_values(
                    "Adjusted (Shrinkage)", ascending=False
                ).reset_index(drop=True)

                rows = []
                for idx, row in brand_df.iterrows():
                    team_name = str(row["Team"])
                    lift_pct = (math.exp(float(row["Adjusted (Shrinkage)"])) - 1) * 100
                    rows.append({
                        "rank": idx + 1,
                        "team": team_name,
                        "conference": team_conferences.get(team_name, "Independent"),
                        "viewership_lift_pct": float(round(lift_pct, 1)),
                        "games_used": int(pd.to_numeric(row.get("Games Used"), errors="coerce") or 0),
                    })

                if rows:
                    return rows
        except Exception:
            pass

    return []


csv_brand_rankings = _load_csv_brand_rankings()

brand_rankings_cache = {}
brand_rankings_cache["all"] = csv_brand_rankings or compute_brand_rankings(df_all)

available_years = sorted(df_all["Year"].dropna().unique().tolist())
for y in available_years:
    df_y = df_all[df_all["Year"] == y]
    brand_rankings_cache[str(y)] = compute_brand_rankings(df_y)

def _load_profile_brand_rank_lookup():
    if csv_brand_rankings:
        return {
            row["team"]: {
                "brand_rank": row["rank"],
                "viewership_lift_pct": row["viewership_lift_pct"],
                "conference": row["conference"],
            }
            for row in csv_brand_rankings
        }

    csv_path = os.path.join(os.path.dirname(__file__), "brand_rankings.csv")

    if os.path.exists(csv_path):
        try:
            brand_df = pd.read_csv(csv_path)
            if {"Team", "Adjusted (Shrinkage)"}.issubset(brand_df.columns):
                brand_df = brand_df[~brand_df["Team"].isin(excluded_brand_teams)]
                brand_df["Adjusted (Shrinkage)"] = pd.to_numeric(
                    brand_df["Adjusted (Shrinkage)"], errors="coerce"
                )
                brand_df = brand_df.dropna(subset=["Team", "Adjusted (Shrinkage)"])
                brand_df = brand_df.sort_values(
                    "Adjusted (Shrinkage)", ascending=False
                ).reset_index(drop=True)

                lookup = {}
                for idx, row in brand_df.iterrows():
                    team_name = str(row["Team"])
                    lift_pct = (math.exp(float(row["Adjusted (Shrinkage)"])) - 1) * 100
                    lookup[team_name] = {
                        "brand_rank": idx + 1,
                        "viewership_lift_pct": float(round(lift_pct, 1)),
                        "conference": team_conferences.get(team_name, "Independent"),
                    }

                if lookup:
                    return lookup
        except Exception:
            pass

    return {
        row["team"]: {
            "brand_rank": row["rank"],
            "viewership_lift_pct": row["viewership_lift_pct"],
            "conference": row["conference"],
        }
        for row in brand_rankings_cache["all"]
    }


brand_rank_lookup = _load_profile_brand_rank_lookup()

@app.get("/brand-years")
def brand_years():
    return {"years": available_years}

@app.get("/brand-rankings")
def brand_rankings(year: str = "all"):
    return {"rows": brand_rankings_cache.get(year, [])}


def _build_team_profile_cache(df):
    records = []

    for _, row in df.iterrows():
        viewers = pd.to_numeric(row.get("Persons 2+"), errors="coerce")
        year = row.get("Year")

        if pd.isna(viewers) or pd.isna(year):
            continue

        detected_network = detect_network(row)

        game_meta = {
            "date": row.get("Date"),
            "network": detected_network,
            "time_bucket": display_time_bucket(row),
            "rank_detail": row.get("scenario_rank_detail"),
            "competing_bucket": row.get("scenario_competing_bucket"),
            "viewers": float(viewers),
            "expected_viewers": float(pd.to_numeric(row.get("expected_viewers"), errors="coerce"))
            if pd.notna(pd.to_numeric(row.get("expected_viewers"), errors="coerce")) else None,
            "actual_minus_expected": float(pd.to_numeric(row.get("actual_minus_expected"), errors="coerce"))
            if pd.notna(pd.to_numeric(row.get("actual_minus_expected"), errors="coerce")) else None,
            "actual_vs_expected_pct": float(pd.to_numeric(row.get("actual_vs_expected_pct"), errors="coerce"))
            if pd.notna(pd.to_numeric(row.get("actual_vs_expected_pct"), errors="coerce")) else None,
            "conf_champ": bool(row.get("Conf Champ") == 1),
        }

        team_1 = row.get("Team 1")
        team_2 = row.get("Team 2")

        if pd.notna(team_1):
            expected_team_1 = pd.to_numeric(row.get("expected_viewers_team1"), errors="coerce")
            records.append({
                "team": str(team_1),
                "opponent": str(team_2) if pd.notna(team_2) else None,
                "matchup": f"{team_1} vs {team_2}" if pd.notna(team_2) else str(team_1),
                "year": int(year),
                **{
                    **game_meta,
                    "expected_viewers": float(expected_team_1) if pd.notna(expected_team_1) else None,
                    "actual_minus_expected": float(viewers - expected_team_1) if pd.notna(expected_team_1) else None,
                    "actual_vs_expected_pct": (
                        float(((viewers / expected_team_1) - 1) * 100)
                        if pd.notna(expected_team_1) and expected_team_1 > 0
                        else None
                    ),
                },
            })

        if pd.notna(team_2):
            expected_team_2 = pd.to_numeric(row.get("expected_viewers_team2"), errors="coerce")
            records.append({
                "team": str(team_2),
                "opponent": str(team_1) if pd.notna(team_1) else None,
                "matchup": f"{team_1} vs {team_2}" if pd.notna(team_1) else str(team_2),
                "year": int(year),
                **{
                    **game_meta,
                    "expected_viewers": float(expected_team_2) if pd.notna(expected_team_2) else None,
                    "actual_minus_expected": float(viewers - expected_team_2) if pd.notna(expected_team_2) else None,
                    "actual_vs_expected_pct": (
                        float(((viewers / expected_team_2) - 1) * 100)
                        if pd.notna(expected_team_2) and expected_team_2 > 0
                        else None
                    ),
                },
            })

    long_df = pd.DataFrame(records)
    if long_df.empty:
        return {}

    profile_cache = {}

    for team, team_df in long_df.groupby("team"):
        yearly_rows = []

        for year, year_df in team_df.groupby("year"):
            year_df = year_df.sort_values("viewers", ascending=False)
            peak_game = year_df.iloc[0]
            low_game = year_df.iloc[-1]
            year_expected = pd.to_numeric(year_df["expected_viewers"], errors="coerce").dropna()
            year_actual = pd.to_numeric(year_df["viewers"], errors="coerce").dropna()
            year_delta_series = pd.to_numeric(year_df["actual_minus_expected"], errors="coerce")

            yearly_rows.append({
                "year": int(year),
                "games": int(len(year_df)),
                "average_viewers": float(round(year_df["viewers"].mean(), 1)),
                "median_viewers": float(round(year_df["viewers"].median(), 1)),
                "expected_average_viewers": (
                    float(round(year_expected.mean(), 1))
                    if not year_expected.empty
                    else None
                ),
                "average_minus_expected": (
                    float(round(year_actual.mean() - year_expected.mean(), 1))
                    if not year_actual.empty and not year_expected.empty
                    else None
                ),
                "overperformance_pct": (
                    float(round(((year_actual.sum() / year_expected.sum()) - 1) * 100, 1))
                    if not year_actual.empty and not year_expected.empty and year_expected.sum() > 0
                    else None
                ),
                "games_above_expected": int((year_delta_series > 0).sum()) if not year_delta_series.empty else 0,
                "peak_viewers": float(round(peak_game["viewers"], 1)),
                "peak_matchup": peak_game["matchup"],
                "peak_network": peak_game["network"],
                "low_viewers": float(round(low_game["viewers"], 1)),
                "low_matchup": low_game["matchup"],
            })

        team_df = team_df.sort_values(["year", "viewers"], ascending=[False, False])
        top_games_df = team_df.sort_values(["viewers", "year"], ascending=[False, False]).head(10)
        latest_year = max(row["year"] for row in yearly_rows)
        latest_year_row = next(row for row in yearly_rows if row["year"] == latest_year)
        expected_series = pd.to_numeric(team_df["expected_viewers"], errors="coerce").dropna()
        actual_series = pd.to_numeric(team_df["viewers"], errors="coerce").dropna()
        expected_avg = float(round(expected_series.mean(), 1)) if not expected_series.empty else None
        avg_delta = (
            float(round(actual_series.mean() - expected_series.mean(), 1))
            if not actual_series.empty and not expected_series.empty
            else None
        )
        overperformance_pct = (
            float(round(((actual_series.sum() / expected_series.sum()) - 1) * 100, 1))
            if not actual_series.empty and not expected_series.empty and expected_series.sum() > 0
            else None
        )
        games_above_expected = (
            int((pd.to_numeric(team_df["actual_minus_expected"], errors="coerce") > 0).sum())
            if "actual_minus_expected" in team_df
            else 0
        )

        profile_cache[team] = {
            "team": team,
            "years": sorted(yearly_rows, key=lambda row: row["year"], reverse=True),
            "top_games": [
                {
                    "rank": idx + 1,
                    "year": int(game["year"]),
                    "date": game["date"],
                    "matchup": game["matchup"],
                    "opponent": game["opponent"],
                    "network": game["network"],
                    "time_bucket": game["time_bucket"],
                    "rank_detail": game["rank_detail"],
                    "competing_bucket": game["competing_bucket"],
                    "viewers": float(round(game["viewers"], 1)),
                    "expected_viewers": float(round(game["expected_viewers"], 1)) if pd.notna(game["expected_viewers"]) else None,
                    "actual_minus_expected": float(round(game["actual_minus_expected"], 1)) if pd.notna(game["actual_minus_expected"]) else None,
                    "actual_vs_expected_pct": float(round(game["actual_vs_expected_pct"], 1)) if pd.notna(game["actual_vs_expected_pct"]) else None,
                    "conf_champ": bool(game["conf_champ"]),
                }
                for idx, (_, game) in enumerate(top_games_df.iterrows())
            ],
            "games": [
                {
                    "year": int(game["year"]),
                    "date": game["date"],
                    "matchup": game["matchup"],
                    "opponent": game["opponent"],
                    "network": game["network"],
                    "time_bucket": game["time_bucket"],
                    "rank_detail": game["rank_detail"],
                    "competing_bucket": game["competing_bucket"],
                    "viewers": float(round(game["viewers"], 1)),
                    "expected_viewers": float(round(game["expected_viewers"], 1)) if pd.notna(game["expected_viewers"]) else None,
                    "actual_minus_expected": float(round(game["actual_minus_expected"], 1)) if pd.notna(game["actual_minus_expected"]) else None,
                    "actual_vs_expected_pct": float(round(game["actual_vs_expected_pct"], 1)) if pd.notna(game["actual_vs_expected_pct"]) else None,
                    "conf_champ": bool(game["conf_champ"]),
                }
                for _, game in team_df.iterrows()
            ],
            "summary": {
                "games": int(len(team_df)),
                "years_available": [row["year"] for row in sorted(yearly_rows, key=lambda row: row["year"])],
                "average_viewers": float(round(team_df["viewers"].mean(), 1)),
                "median_viewers": float(round(team_df["viewers"].median(), 1)),
                "expected_average_viewers": expected_avg,
                "average_minus_expected": avg_delta,
                "overperformance_pct": overperformance_pct,
                "games_above_expected": games_above_expected,
                "peak_viewers": float(round(team_df.iloc[0]["viewers"], 1)),
                "peak_matchup": team_df.iloc[0]["matchup"],
                "latest_year": int(latest_year),
                "latest_year_average": latest_year_row["average_viewers"],
                "brand_rank": brand_rank_lookup.get(team, {}).get("brand_rank"),
                "viewership_lift_pct": brand_rank_lookup.get(team, {}).get("viewership_lift_pct"),
                "conference": brand_rank_lookup.get(team, {}).get("conference", team_conferences.get(team, "Independent")),
            },
        }

    return profile_cache


team_profile_cache = _build_team_profile_cache(df_all)
team_profile_teams = sorted(team_profile_cache.keys())


@app.get("/team-profile-teams")
def team_profile_teams_endpoint():
    return {"teams": team_profile_teams}


@app.get("/team-profile")
def team_profile(team: str):
    normalized_team = normalize_team(team)
    profile = team_profile_cache.get(normalized_team)

    if profile is None:
        return {"error": f"No profile data found for {team}."}

    return clean_nan(profile)


@app.get("/team-scenario-compare")
def team_scenario_compare(
    team: str,
    compare_team: str,
    network: str = "all",
    time_bucket: str = "all",
    rank_bucket: str = "all",
    competing_bucket: str = "all",
    year_start: str = "all",
    year_end: str = "all",
    include_conf_champ: bool = True,
):
    normalized_team = normalize_team(team)
    normalized_compare_team = normalize_team(compare_team)

    def build_team_games(team_name):
        team_mask = (df_all["Team 1"] == team_name) | (df_all["Team 2"] == team_name)
        games = df_all.loc[team_mask].copy()
        games["team"] = team_name
        games["opponent"] = np.where(
            games["Team 1"] == team_name,
            games["Team 2"],
            games["Team 1"],
        )
        games["matchup"] = games["Team 1"].astype(str) + " vs " + games["Team 2"].astype(str)
        games["expected_viewers"] = np.where(
            games["Team 1"] == team_name,
            pd.to_numeric(games["expected_viewers_team1"], errors="coerce"),
            pd.to_numeric(games["expected_viewers_team2"], errors="coerce"),
        )
        games["actual_minus_expected"] = (
            pd.to_numeric(games["Persons 2+"], errors="coerce")
            - pd.to_numeric(games["expected_viewers"], errors="coerce")
        )
        games["actual_vs_expected_pct"] = np.where(
            pd.to_numeric(games["expected_viewers"], errors="coerce") > 0,
            (
                (pd.to_numeric(games["Persons 2+"], errors="coerce") / pd.to_numeric(games["expected_viewers"], errors="coerce"))
                - 1
            ) * 100,
            np.nan,
        )
        return games

    def apply_filters(df):
        filtered = df
        if network != "all":
            filtered = filtered[filtered["scenario_network"] == network]
        if time_bucket != "all":
            filtered = filtered[filtered["scenario_time_bucket"] == time_bucket]
        if rank_bucket != "all":
            filtered = filtered[filtered["scenario_rank_detail"] == rank_bucket]
        if competing_bucket != "all":
            filtered = filtered[filtered["scenario_competing_bucket"] == competing_bucket]
        if year_start != "all":
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") >= int(year_start)]
        if year_end != "all":
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") <= int(year_end)]
        if not include_conf_champ:
            filtered = filtered[filtered["scenario_conf_champ"] != 1]
        return filtered

    team_filtered = apply_filters(build_team_games(normalized_team))
    compare_filtered = apply_filters(build_team_games(normalized_compare_team))

    team_viewers = pd.to_numeric(team_filtered["Persons 2+"], errors="coerce").dropna()
    compare_viewers = pd.to_numeric(compare_filtered["Persons 2+"], errors="coerce").dropna()
    team_expected = pd.to_numeric(team_filtered["expected_viewers"], errors="coerce").dropna()
    compare_expected = pd.to_numeric(compare_filtered["expected_viewers"], errors="coerce").dropna()

    team_avg = float(team_viewers.mean()) if not team_viewers.empty else None
    compare_avg = float(compare_viewers.mean()) if not compare_viewers.empty else None
    team_expected_avg = float(team_expected.mean()) if not team_expected.empty else None
    compare_expected_avg = float(compare_expected.mean()) if not compare_expected.empty else None
    team_vs_expected = (
        team_avg - team_expected_avg
        if team_avg is not None and team_expected_avg is not None
        else None
    )
    compare_vs_expected = (
        compare_avg - compare_expected_avg
        if compare_avg is not None and compare_expected_avg is not None
        else None
    )
    team_overperformance_pct = (
        ((team_viewers.sum() / team_expected.sum()) - 1) * 100
        if not team_viewers.empty and not team_expected.empty and team_expected.sum() > 0
        else None
    )
    compare_overperformance_pct = (
        ((compare_viewers.sum() / compare_expected.sum()) - 1) * 100
        if not compare_viewers.empty and not compare_expected.empty and compare_expected.sum() > 0
        else None
    )
    delta = (team_avg - compare_avg) if team_avg is not None and compare_avg is not None else None
    delta_pct = (
        ((team_avg / compare_avg) - 1) * 100
        if team_avg is not None and compare_avg not in [None, 0]
        else None
    )

    shared_opponents = sorted(
        set(team_filtered["opponent"].dropna().tolist())
        & set(compare_filtered["opponent"].dropna().tolist())
    )

    opponent_rows = []
    for opponent in shared_opponents:
        team_opp_games = team_filtered[team_filtered["opponent"] == opponent]
        compare_opp_games = compare_filtered[compare_filtered["opponent"] == opponent]
        team_opp_viewers = pd.to_numeric(team_opp_games["Persons 2+"], errors="coerce").dropna()
        compare_opp_viewers = pd.to_numeric(compare_opp_games["Persons 2+"], errors="coerce").dropna()

        team_opp_avg = float(team_opp_viewers.mean()) if not team_opp_viewers.empty else None
        compare_opp_avg = float(compare_opp_viewers.mean()) if not compare_opp_viewers.empty else None

        opponent_rows.append({
            "opponent": opponent,
            "team_average_viewers": team_opp_avg,
            "compare_average_viewers": compare_opp_avg,
            "difference_viewers": (
                team_opp_avg - compare_opp_avg
                if team_opp_avg is not None and compare_opp_avg is not None
                else None
            ),
            "team_games": int(len(team_opp_games)),
            "compare_games": int(len(compare_opp_games)),
        })

    head_to_head_mask = (
        ((df_all["Team 1"] == normalized_team) & (df_all["Team 2"] == normalized_compare_team))
        | ((df_all["Team 1"] == normalized_compare_team) & (df_all["Team 2"] == normalized_team))
    )
    head_to_head_df = apply_filters(df_all.loc[head_to_head_mask].copy())
    head_to_head_rows = []
    for _, game in head_to_head_df.sort_values(["Year", "Date"], ascending=[False, False]).iterrows():
        viewers = pd.to_numeric(game.get("Persons 2+"), errors="coerce")
        head_to_head_rows.append({
            "year": int(game["Year"]) if pd.notna(game.get("Year")) else None,
            "date": game.get("Date"),
            "matchup": f"{game['Team 1']} vs {game['Team 2']}",
            "network": game.get("scenario_network"),
            "time_bucket": display_time_bucket(game),
            "viewers": float(viewers) if pd.notna(viewers) else None,
            "conf_champ": bool(game.get("scenario_conf_champ") == 1),
        })

    return clean_nan({
        "team": normalized_team,
        "compare_team": normalized_compare_team,
        "filters": {
            "network": network,
            "time_bucket": time_bucket,
            "rank_bucket": rank_bucket,
            "competing_bucket": competing_bucket,
            "year_start": year_start,
            "year_end": year_end,
            "include_conf_champ": include_conf_champ,
        },
        "team_average_viewers": team_avg,
        "compare_team_average_viewers": compare_avg,
        "team_expected_average_viewers": team_expected_avg,
        "compare_team_expected_average_viewers": compare_expected_avg,
        "team_average_minus_expected": team_vs_expected,
        "compare_team_average_minus_expected": compare_vs_expected,
        "team_overperformance_pct": team_overperformance_pct,
        "compare_team_overperformance_pct": compare_overperformance_pct,
        "difference_viewers": delta,
        "difference_pct": delta_pct,
        "team_sample_size": int(len(team_viewers)),
        "compare_team_sample_size": int(len(compare_viewers)),
        "shared_opponents": opponent_rows,
        "shared_opponent_count": int(len(opponent_rows)),
        "head_to_head_games": head_to_head_rows,
        "head_to_head_count": int(len(head_to_head_rows)),
        "available_filters": {
            "networks": ["all", "ABC", "CBS", "NBC", "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "FS2", "BTN", "CW", "NFLN", "ESPNNEWS"],
            "time_buckets": ["all", "Sat Early", "Sat Mid", "Sat Late", "Friday", "Monday", "Sunday", "Weekday", "Other"],
            "rank_buckets": rank_detail_options,
            "competing_buckets": ["all", "None", "1 Major Game", "2+ Major Games"],
            "years": ["all"] + [str(year) for year in sorted(pd.to_numeric(df_all["Year"], errors="coerce").dropna().astype(int).unique().tolist())],
        },
    })


@app.get("/team-viewership-rankings")
def team_viewership_rankings(
    network: str = "all",
    time_bucket: str = "all",
    rank_bucket: str = "all",
    opponent: str = "all",
    min_games: int = 1,
    include_conf_champ: bool = True,
):
    team_aliases = {
        "Kennesaw St.": "Kennesaw State",
    }

    records = []
    opponent_options = set()

    for _, row in df_all.iterrows():
        viewers = pd.to_numeric(row.get("Persons 2+"), errors="coerce")
        year = row.get("Year")

        if pd.isna(viewers) or pd.isna(year):
            continue

        if network != "all" and row.get("scenario_network") != network:
            continue
        if time_bucket == "Primetime":
            if not is_primetime_baseline(row):
                continue
        elif time_bucket == "Other":
            if row.get("scenario_time_bucket") != "Other" or is_primetime_baseline(row):
                continue
        elif time_bucket != "all" and row.get("scenario_time_bucket") != time_bucket:
            continue
        if rank_bucket != "all" and row.get("scenario_rank_detail") != rank_bucket:
            continue
        if not include_conf_champ and row.get("scenario_conf_champ") == 1:
            continue

        team_1 = row.get("Team 1")
        team_2 = row.get("Team 2")

        if pd.notna(team_1):
            opponent_options.add(team_aliases.get(str(team_2), str(team_2)) if pd.notna(team_2) else None)
            normalized_opponent = team_aliases.get(str(team_2), str(team_2)) if pd.notna(team_2) else None
            if opponent != "all" and normalized_opponent != opponent:
                pass
            else:
                records.append({
                    "team": team_aliases.get(str(team_1), str(team_1)),
                    "viewers": float(viewers),
                })
        if pd.notna(team_2):
            opponent_options.add(team_aliases.get(str(team_1), str(team_1)) if pd.notna(team_1) else None)
            normalized_opponent = team_aliases.get(str(team_1), str(team_1)) if pd.notna(team_1) else None
            if opponent != "all" and normalized_opponent != opponent:
                pass
            else:
                records.append({
                    "team": team_aliases.get(str(team_2), str(team_2)),
                    "viewers": float(viewers),
                })

    rankings_df = pd.DataFrame(records)
    if rankings_df.empty:
        return clean_nan({
            "rows": [],
            "filters": {
                "network": network,
                "time_bucket": time_bucket,
                "rank_bucket": rank_bucket,
                "opponent": opponent,
                "min_games": min_games,
                "include_conf_champ": include_conf_champ,
            },
            "available_filters": {
                "networks": ["all", "ABC", "CBS", "NBC", "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "FS2", "BTN", "CW", "NFLN", "ESPNNEWS"],
                "time_buckets": ["all", "Primetime", "Sat Early", "Sat Mid", "Sat Late", "Friday", "Monday", "Sunday", "Weekday", "Other"],
                "rank_buckets": rank_detail_options,
                "opponents": ["all"] + sorted(option for option in opponent_options if option),
            },
        })

    rows = []
    grouped = rankings_df.groupby("team")["viewers"]
    for team, viewers in grouped:
        if team in excluded_brand_teams or team in excluded_viewership_ranking_teams:
            continue

        game_count = int(len(viewers))
        if game_count < max(1, int(min_games)):
            continue

        median_viewers = float(viewers.median())
        average_viewers = float(viewers.mean())
        rows.append({
            "team": team,
            "conference": team_conferences.get(team, "Independent"),
            "games": game_count,
            "average_viewers": round(average_viewers, 1),
            "median_viewers": round(median_viewers, 1),
        })

    rows = sorted(
        rows,
        key=lambda row: (
            -(row["average_viewers"] or 0),
            -(row["median_viewers"] or 0),
            row["team"],
        ),
    )

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return clean_nan({
        "rows": rows,
        "filters": {
            "network": network,
            "time_bucket": time_bucket,
            "rank_bucket": rank_bucket,
            "opponent": opponent,
            "min_games": min_games,
            "include_conf_champ": include_conf_champ,
        },
        "available_filters": {
            "networks": ["all", "ABC", "CBS", "NBC", "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "FS2", "BTN", "CW", "NFLN", "ESPNNEWS"],
            "time_buckets": ["all", "Primetime", "Sat Early", "Sat Mid", "Sat Late", "Friday", "Monday", "Sunday", "Weekday", "Other"],
            "rank_buckets": rank_detail_options,
            "opponents": ["all"] + sorted(option for option in opponent_options if option),
        },
    })

# WEEKLY PREDICTIONS

@app.get("/weekly-predictions")
def weekly_predictions():
    if db is None:
        return clean_nan({
            "weeks": [],
            "metrics": None,
            "error": "Firestore is not configured."
        })

    docs = db.collection("weekly-predictions").stream()
    weeks_output = []

    pre_errors = []
    post_errors = []

    for doc in docs:
        data = doc.to_dict()
        week = data["week"]
        year = data["year"]
        games = data["games"]

        updated = False  # If anything changes, we write back

        for g in games:

            # =====================================================
            # 🔵 SAVE OLD VALUES FOR CHANGE DETECTION
            # =====================================================
            pre_old      = g.get("predicted", "")
            post_old     = g.get("post_predicted", "")
            actual_old   = g.get("actual")
            pre_err_old  = g.get("percent_error")
            acc_old      = g.get("accuracy")
            post_err_old = g.get("post_percent_error")
            post_acc_old = g.get("post_accuracy")

            # =====================================================
            # 🔵 PRE-GAME PREDICTION (NEVER OVERWRITE)
            # =====================================================
            missing_pred = (
                not g.get("predicted")
                or g["predicted"] in ["", None, "nan", "NaN"]
            )

            if missing_pred:
                g["predicted"] = generate_pregame_prediction(g)

            if g.get("predicted") != pre_old:
                updated = True

            # =====================================================
            # 🔵 PREGAME ERROR — ONLY UPDATE WHEN NEEDED
            # =====================================================
            has_actual = g.get("actual") not in [None, "", "nan", "NaN"]
            has_pred = bool(g.get("predicted"))

            # sanitize old error
            old_err = pre_err_old if isinstance(pre_err_old, (int, float)) else None

            # compute possible new error
            new_err = calc_error(g.get("predicted"), g.get("actual")) if (has_actual and has_pred) else None

            # CASE 1: error is missing, but now we CAN compute → compute it
            if old_err is None and new_err is not None:
                g["percent_error"] = new_err
                updated = True

            # CASE 2: prediction or actual changed → recompute
            elif (g.get("predicted") != pre_old or g.get("actual") != actual_old) and new_err is not None:
                g["percent_error"] = new_err
                updated = True

            # CASE 3: keep the old error
            else:
                g["percent_error"] = old_err
                new_err = old_err

            # Accuracy emoji
            if new_err is None:
                g["accuracy"] = ""
            elif new_err < 5:
                g["accuracy"] = "🟢🎯"
            elif new_err < 25:
                g["accuracy"] = "🟢"
            elif new_err < 35:
                g["accuracy"] = "🟡"
            else:
                g["accuracy"] = "🔴"

            if g["accuracy"] != acc_old:
                updated = True

            # Metrics collector
            if isinstance(new_err, (int, float)) and not math.isnan(new_err):
                pre_errors.append(new_err)

            # =====================================================
            # 🔴 POST-GAME PREDICTION (NEVER OVERWRITE)
            # =====================================================
            scores_exist = (
                g.get("score1") is not None and
                g.get("score2") is not None
            )
            has_post = bool(g.get("post_predicted"))

            if scores_exist and not has_post:
                try:
                    post_pred_str = generate_postgame_prediction(g)
                    if post_pred_str:
                        g["post_predicted"] = post_pred_str
                        updated = True
                except:
                    pass

            post_pred_str = g.get("post_predicted") or ""

            # =====================================================
            # 🔴 POSTGAME ERROR
            # =====================================================
            post_pred_m = parse_viewership(post_pred_str)
            actual_m = parse_viewership(g.get("actual"))

            if post_pred_m and actual_m:
                e_post = abs((post_pred_m - actual_m) / actual_m) * 100
            else:
                e_post = None

            g["post_percent_error"] = e_post

            # badge
            if e_post is None:
                g["post_accuracy"] = ""
            elif e_post < 5:
                g["post_accuracy"] = "🟢🎯"
            elif e_post < 25:
                g["post_accuracy"] = "🟢"
            elif e_post < 35:
                g["post_accuracy"] = "🟡"
            else:
                g["post_accuracy"] = "🔴"

            # metrics
            if isinstance(e_post, (int, float)) and not math.isnan(e_post):
                post_errors.append(e_post)

            # detect change
            if g.get("post_predicted") != post_old:
                updated = True
            if g.get("post_percent_error") != post_err_old:
                updated = True
            if g.get("post_accuracy") != post_acc_old:
                updated = True

        # =====================================================
        # 🔥 WRITE BACK TO FIRESTORE IF ANY FIELD CHANGED
        # =====================================================
        if updated:
            db.collection("weekly-predictions").document(doc.id).set(data)

        weeks_output.append({
            "week": week,
            "year": year,
            "games": games
        })

    # =====================================================
    # 📊 METRICS
    # =====================================================
    def compute_stats(arr):
        if len(arr) == 0:
            return (None, None, None, None)
        return (
            float(np.median(arr)),
            float(np.mean(arr)),
            int(np.mean([e < 10 for e in arr]) * 100),
            int(np.mean([e < 25 for e in arr]) * 100),
        )

    pre_median, pre_mean, pre_pct10, pre_pct25 = compute_stats(pre_errors)
    post_median, post_mean, post_pct10, post_pct25 = compute_stats(post_errors)

    metrics = {
        "pregame": {
            "median_error": pre_median,
            "mean_error": pre_mean,
            "pct_within_10": pre_pct10,
            "pct_within_25": pre_pct25,
        },
        "postgame": {
            "median_error": post_median,
            "mean_error": post_mean,
            "pct_within_10": post_pct10,
            "pct_within_25": post_pct25,
        },
    }

    weeks_output = sorted(weeks_output, key=lambda w: w["week"], reverse=True)

    return clean_nan({
        "weeks": weeks_output,
        "metrics": metrics
    })
