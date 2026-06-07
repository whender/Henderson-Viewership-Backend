from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
import os
import math
from datetime import datetime
from collections import Counter, defaultdict

from weekly_predictions_fs import (
    generate_pregame_prediction,
    generate_postgame_prediction,
    calc_error,
    build_features,
    parse_viewership
)

from firestore_client import db
from cbb_viewership import (
    CBBGameInput,
    cbb_filter_options,
    cbb_metadata,
    cbb_team_effects,
    cbb_team_options,
    cbb_team_profile,
    cbb_viewership_rankings,
    predict_cbb_viewership,
)

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
from predict import (
    predict_viewership,
    teams_list,
    normalize_team,
    model as pregame_model,
    MODEL_TEAM_NAMES,
    rivalries,
    black_friday_date,
)

class GameInput(BaseModel):
    team1: str
    team2: str
    rank1: int
    rank2: int
    network: str
    time_slot: str
    comp_tier1: int = 0


class RealignmentSimulationInput(BaseModel):
    conference: str = "Big 10"
    expansion_teams: list[str] = Field(default_factory=lambda: ["Utah"])
    protected_opponents: list[str] = Field(default_factory=list)
    protected_opponents_by_team: dict[str, list[str]] = Field(default_factory=dict)
    games_per_team: int = 9
    network_policy: str = "big_ten_tv_mix"
    ranking_policy: str = "brand_tiers"


class SuperleagueSimulationInput(BaseModel):
    teams: list[str] = Field(default_factory=list)
    games_per_team: int = 9
    ranking_policy: str = "brand_tiers"


class LeagueRealignmentSimulationInput(BaseModel):
    memberships: dict[str, str] = Field(default_factory=dict)
    protected_matchups_by_team: dict[str, list[str]] = Field(default_factory=dict)
    games_per_team: int = 9
    ranking_policy: str = "brand_tiers"

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
        "prediction_formatted": result["formatted"],
        "warnings": result.get("warnings", []),
    }


# ======================================================
# 🏀 COLLEGE BASKETBALL MODEL
# ======================================================

@app.get("/cbb/metadata")
def cbb_model_metadata():
    return cbb_metadata()


@app.get("/cbb/teams")
def cbb_teams():
    return cbb_team_options()


@app.get("/cbb/filters")
def cbb_filters():
    return cbb_filter_options()


@app.post("/cbb/predict")
def predict_cbb_game(game: CBBGameInput):
    return predict_cbb_viewership(game)


@app.get("/cbb/team-effects")
def cbb_team_effect_rankings(conference: str = "all"):
    return cbb_team_effects(conference=conference)


@app.get("/cbb/viewership-rankings")
def cbb_team_viewership_rankings(
    network: str = "all",
    time_slot: str = "all",
    stage: str = "all",
    season: str = "all",
    opponent: str = "all",
    min_games: int = 1,
    include_tournament: bool = True,
):
    return cbb_viewership_rankings(
        network=network,
        time_slot=time_slot,
        stage=stage,
        season=season,
        opponent=opponent,
        min_games=min_games,
        include_tournament=include_tournament,
    )


@app.get("/cbb/team-profile")
def cbb_profile(team: str):
    return cbb_team_profile(team)

# ======================================================
# 🏆 BRAND RANKINGS (POSTGAME MODEL)
# ======================================================

numeric_features_post = [
    "Competing Tier 1","FOX","ESPN","ESPN2","ESPNU","FS1","FS2","NBC","CBS",
    "ABC","BTN","CW","NFLN","ESPNNEWS",
    "SEC_ConfChamp","Big10_ConfChamp","Big12_ConfChamp","ACC_ConfChamp","Other_ConfChamp",
    "Sun","Monday","Weekday","Friday Power","Friday Non-Power","Black Friday","Week 0 Power","Week 1 Power",
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

parsed_game_dates = pd.to_datetime(df_all.get("ParsedDate"), errors="coerce")
df_all["Black Friday"] = parsed_game_dates.map(
    lambda value: int(pd.notna(value) and value.date() == black_friday_date(value.year))
)
df_all["Friday"] = np.where(
    df_all["Black Friday"].eq(1),
    0,
    pd.to_numeric(df_all.get("Friday"), errors="coerce").fillna(0).astype(int),
)
POWER_FOOTBALL_CONFERENCES = {"SEC", "Big 10", "ACC", "Big 12"}
df_all["Friday Power"] = (
    df_all["Friday"].eq(1)
    & df_all["Team 1"].map(lambda team: team_conferences.get(team) in POWER_FOOTBALL_CONFERENCES or team == "Notre Dame")
    & df_all["Team 2"].map(lambda team: team_conferences.get(team) in POWER_FOOTBALL_CONFERENCES or team == "Notre Dame")
).astype(int)
df_all["Friday Non-Power"] = (df_all["Friday"].eq(1) & df_all["Friday Power"].eq(0)).astype(int)
df_all["Week 0 Power"] = (
    parsed_game_dates.notna()
    & parsed_game_dates.dt.month.eq(8)
    & parsed_game_dates.dt.day.le(28)
    & df_all["Team 1"].map(lambda team: team_conferences.get(team) in POWER_FOOTBALL_CONFERENCES or team == "Notre Dame")
    & df_all["Team 2"].map(lambda team: team_conferences.get(team) in POWER_FOOTBALL_CONFERENCES or team == "Notre Dame")
).astype(int)
labor_day_dates = parsed_game_dates.map(
    lambda value: (
        pd.Timestamp(value.year, 9, 1)
        + pd.Timedelta(days=(0 - pd.Timestamp(value.year, 9, 1).weekday()) % 7)
    ).date()
    if pd.notna(value)
    else None
)
df_all["Week 1 Power"] = (
    parsed_game_dates.notna()
    & (parsed_game_dates.dt.date >= parsed_game_dates.map(lambda value: pd.Timestamp(value.year, 8, 29).date() if pd.notna(value) else None))
    & (parsed_game_dates.dt.date <= labor_day_dates)
    & df_all["Team 1"].map(lambda team: team_conferences.get(team) in POWER_FOOTBALL_CONFERENCES or team == "Notre Dame")
    & df_all["Team 2"].map(lambda team: team_conferences.get(team) in POWER_FOOTBALL_CONFERENCES or team == "Notre Dame")
).astype(int)

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
    if row.get("Black Friday") == 1:
        return "Black Friday"
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
        for flag in ["Sun", "Monday", "Weekday", "Friday", "Black Friday", "Sat Early", "Sat Mid", "Sat Late"]
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
    for column in ["DeionEra", "DeionEra25"]:
        if column in X.columns:
            X[column] = 0.0

    return _predict_viewers_from_design_matrix(X, 2 * AVERAGE_TEAM_EFFECT)


def _compute_team_specific_expected_viewers(df, focal_team_column):
    X = _build_pregame_design_matrix(df)
    focal_teams = df[focal_team_column].fillna("")

    for team_name in focal_teams[focal_teams.isin(MODELED_TEAM_COLUMNS)].unique():
        X.loc[focal_teams == team_name, team_name] = 0.0
    colorado_mask = focal_teams == "Colorado"
    for column in ["DeionEra", "DeionEra25"]:
        if column in X.columns:
            X.loc[colorado_mask, column] = 0.0

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

FOOTBALL_MEDIA_RIGHTS_WEIGHT = 0.85
BASKETBALL_MEDIA_RIGHTS_WEIGHT = 0.15

FOOTBALL_BRAND_NAME_ALIASES = {
    "Appalachian St.": "Appalachian State",
    "Arizona St.": "Arizona State",
    "Arkansas St.": "Arkansas State",
    "Ball St.": "Ball State",
    "Boise St.": "Boise State",
    "Connecticut": "UConn",
    "Colorado St.": "Colorado State",
    "FAU": "Florida Atlantic",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Georgia St.": "Georgia State",
    "Iowa St.": "Iowa State",
    "Jacksonville St.": "Jacksonville State",
    "Kansas St.": "Kansas State",
    "Kent St.": "Kent State",
    "Miami": "Miami (FL)",
    "Miami Ohio": "Miami (OH)",
    "Middle Tennessee St.": "Middle Tennessee",
    "Michigan St.": "Michigan State",
    "Mississippi": "Ole Miss",
    "Mississippi St.": "Mississippi State",
    "Massachusetts": "UMass",
    "New Mexico St.": "New Mexico State",
    "North Carolina St.": "NC State",
    "Ohio St.": "Ohio State",
    "Oklahoma St.": "Oklahoma State",
    "Oregon St.": "Oregon State",
    "Penn St.": "Penn State",
    "Pittsburgh": "Pitt",
    "San Diego St.": "San Diego State",
    "San Jose St.": "San Jose State",
    "Texas St.": "Texas State",
    "Utah St.": "Utah State",
    "Washington St.": "Washington State",
}


def display_brand_team_name(team):
    return FOOTBALL_BRAND_NAME_ALIASES.get(team, team)


def _mean_std(values):
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 2:
        return float(series.mean() or 0), 1.0
    std = float(series.std(ddof=0))
    if not std or math.isnan(std):
        std = 1.0
    return float(series.mean()), std


def _load_football_brand_scores():
    csv_path = os.path.join(os.path.dirname(__file__), "brand_rankings.csv")
    scores = {}
    if not os.path.exists(csv_path):
        for row in brand_rankings_cache.get("all", []):
            coefficient = math.log((float(row.get("viewership_lift_pct", 0)) / 100) + 1)
            team = display_brand_team_name(row["team"])
            scores[team] = {
                "team": team,
                "football_coefficient": coefficient,
                "football_lift_pct": float(row.get("viewership_lift_pct", 0)),
                "football_games": int(row.get("games_used", 0)),
                "football_conference": row.get("conference", "Independent"),
            }
        return scores

    try:
        brand_df = pd.read_csv(csv_path)
        if not {"Team", "Adjusted (Shrinkage)"}.issubset(brand_df.columns):
            return scores
        brand_df = brand_df[~brand_df["Team"].isin(excluded_brand_teams)].copy()
        brand_df["Adjusted (Shrinkage)"] = pd.to_numeric(brand_df["Adjusted (Shrinkage)"], errors="coerce")
        brand_df["Games Used"] = pd.to_numeric(brand_df.get("Games Used"), errors="coerce").fillna(0)
        brand_df = brand_df.dropna(subset=["Team", "Adjusted (Shrinkage)"])
        for _, row in brand_df.iterrows():
            raw_team = str(row["Team"])
            team = display_brand_team_name(raw_team)
            coefficient = float(row["Adjusted (Shrinkage)"])
            scores[team] = {
                "team": team,
                "football_coefficient": coefficient,
                "football_lift_pct": float(round((math.exp(coefficient) - 1) * 100, 1)),
                "football_games": int(row["Games Used"]),
                "football_conference": team_conferences.get(raw_team, team_conferences.get(team, "Independent")),
            }
    except Exception:
        return {}
    return scores


def colorado_deion_adjustment():
    model_path = os.path.join(os.path.dirname(__file__), "viewership_postgame_model.joblib")
    data_path = os.path.join(os.path.dirname(__file__), "viewership_cleaned.csv")
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return 0.0
    try:
        model_artifact = joblib.load(model_path)
        params = model_artifact["model"].params
        deion_era_coef = float(params.get("DeionEra", 0.0))
        deion_era_25_coef = float(params.get("DeionEra25", 0.0))

        games = pd.read_csv(data_path)
        colorado_games = games[
            games["Team 1"].eq("Colorado") | games["Team 2"].eq("Colorado")
        ].copy()
        if colorado_games.empty:
            return 0.0
        deion_era_games = pd.to_numeric(colorado_games.get("DeionEra"), errors="coerce").fillna(0).sum()
        deion_era_25_games = pd.to_numeric(colorado_games.get("DeionEra25"), errors="coerce").fillna(0).sum()
        total_games = len(colorado_games)
        return float(((deion_era_games * deion_era_coef) + (deion_era_25_games * deion_era_25_coef)) / total_games)
    except Exception:
        return 0.0


def apply_colorado_deion_control(football_scores):
    adjustment = colorado_deion_adjustment()
    colorado = football_scores.get("Colorado")
    if not colorado or not adjustment:
        return football_scores
    adjusted_scores = {
        team: values.copy()
        for team, values in football_scores.items()
    }
    adjusted_coefficient = adjusted_scores["Colorado"]["football_coefficient"] - adjustment
    adjusted_scores["Colorado"]["football_coefficient"] = adjusted_coefficient
    adjusted_scores["Colorado"]["football_lift_pct"] = float(round((math.exp(adjusted_coefficient) - 1) * 100, 1))
    adjusted_scores["Colorado"]["deion_adjustment"] = float(round(adjustment, 6))
    adjusted_scores["Colorado"]["deion_controlled"] = True
    return adjusted_scores


def _load_basketball_brand_scores():
    try:
        basketball_rows = cbb_team_effects().get("rows", [])
    except Exception:
        basketball_rows = []

    scores = {}
    for row in basketball_rows:
        team = display_brand_team_name(row["team"])
        coefficient = float(row.get("coefficient", 0))
        scores[team] = {
            "team": team,
            "basketball_coefficient": coefficient,
            "basketball_lift_pct": float(round((math.exp(coefficient) - 1) * 100, 1)),
            "basketball_games": int(row.get("appearances", 0)),
            "basketball_conference": row.get("conference", "Unknown"),
        }
    return scores


def football_brand_rankings_for_main_tab(control_deion=False):
    football_scores = _load_football_brand_scores()
    if control_deion:
        football_scores = apply_colorado_deion_control(football_scores)
    rows = []
    for row in football_scores.values():
        rows.append({
            "team": row["team"],
            "conference": row["football_conference"],
            "viewership_lift_pct": row["football_lift_pct"],
            "games_used": row["football_games"],
            "deion_controlled": bool(row.get("deion_controlled", False)),
            "deion_adjustment": row.get("deion_adjustment"),
        })
    rows = sorted(rows, key=lambda row: (-row["viewership_lift_pct"], -row["games_used"], row["team"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def combined_media_brand_rankings(control_deion=False):
    football_scores = _load_football_brand_scores()
    if control_deion:
        football_scores = apply_colorado_deion_control(football_scores)
    basketball_scores = _load_basketball_brand_scores()
    football_mean, football_std = _mean_std(
        row["football_coefficient"] for row in football_scores.values()
    )
    basketball_mean, basketball_std = _mean_std(
        row["basketball_coefficient"] for row in basketball_scores.values()
    )
    last_place_football_z = min(
        ((row["football_coefficient"] - football_mean) / football_std for row in football_scores.values()),
        default=0.0,
    )
    last_place_basketball_z = min(
        ((row["basketball_coefficient"] - basketball_mean) / basketball_std for row in basketball_scores.values()),
        default=0.0,
    )

    rows = []
    teams = sorted(set(football_scores) | set(basketball_scores))
    for team in teams:
        football = football_scores.get(team, {})
        basketball = basketball_scores.get(team, {})
        football_z = (
            (football["football_coefficient"] - football_mean) / football_std
            if football else last_place_football_z
        )
        basketball_z = (
            (basketball["basketball_coefficient"] - basketball_mean) / basketball_std
            if basketball else last_place_basketball_z
        )
        weighted_score = (
            FOOTBALL_MEDIA_RIGHTS_WEIGHT * football_z
            + BASKETBALL_MEDIA_RIGHTS_WEIGHT * basketball_z
        )
        media_index = 100 + 15 * weighted_score
        conference = football.get("football_conference") or basketball.get("basketball_conference") or "Independent"
        if basketball and conference == "Independent" and team != "Notre Dame":
            conference = basketball.get("basketball_conference") or conference
        primary_driver = "Football"
        if basketball and not football:
            primary_driver = "Basketball"
        elif football and basketball:
            primary_driver = "Football" if abs(FOOTBALL_MEDIA_RIGHTS_WEIGHT * football_z) >= abs(BASKETBALL_MEDIA_RIGHTS_WEIGHT * basketball_z) else "Basketball"

        rows.append({
            "team": team,
            "conference": conference,
            "media_brand_index": float(round(media_index, 1)),
            "combined_score": float(round(weighted_score, 4)),
            "_sort_score": float(weighted_score),
            "football_score": float(round(football_z, 3)),
            "basketball_score": float(round(basketball_z, 3)),
            "football_contribution": float(round(FOOTBALL_MEDIA_RIGHTS_WEIGHT * football_z, 3)),
            "basketball_contribution": float(round(BASKETBALL_MEDIA_RIGHTS_WEIGHT * basketball_z, 3)),
            "football_lift_pct": football.get("football_lift_pct"),
            "basketball_lift_pct": basketball.get("basketball_lift_pct"),
            "football_games": int(football.get("football_games", 0)),
            "basketball_games": int(basketball.get("basketball_games", 0)),
            "primary_driver": primary_driver,
            "deion_controlled": bool(football.get("deion_controlled", False)),
            "deion_adjustment": football.get("deion_adjustment"),
        })

    rows = sorted(
        rows,
        key=lambda row: (-row["_sort_score"], -row["football_games"], -row["basketball_games"], row["team"]),
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row.pop("_sort_score", None)
    return rows


def basketball_brand_rankings_for_main_tab():
    rows = []
    for row in cbb_team_effects().get("rows", []):
        lift_pct = (float(row.get("viewer_multiplier", 1)) - 1) * 100
        rows.append({
            "rank": row["rank"],
            "team": row["team"],
            "conference": row["conference"],
            "viewership_lift_pct": float(round(lift_pct, 1)),
            "games_used": int(row["appearances"]),
            "basketball_lift_pct": float(round(lift_pct, 1)),
            "basketball_games": int(row["appearances"]),
        })
    return rows

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


REALIGNMENT_NETWORK_PLANS = {
    "Big 10": {
        "label": "Big Ten TV Mix",
        "premium": [
            {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 0},
            {"network": "CBS", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 0},
            {"network": "NBC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "FOX", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
        ],
        "secondary": [
            {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
            {"network": "CBS", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "NBC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
            {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
        ],
        "cable": [
            {"network": "BTN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
            {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "BTN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2},
            {"network": "FS1", "time_slot": "Friday", "comp_tier1": 1},
        ],
    },
    "SEC": {
        "label": "SEC ESPN/ABC Mix",
        "premium": [
            {"network": "ABC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 0},
            {"network": "ABC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
        ],
        "secondary": [
            {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
            {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
        ],
        "cable": [
            {"network": "ESPN2", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
            {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
            {"network": "ESPN2", "time_slot": "Friday", "comp_tier1": 1},
        ],
    },
    "Big 12": {
        "label": "Big 12 ESPN/FOX Mix",
        "premium": [
            {"network": "ABC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 0},
            {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 0},
            {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "FOX", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
        ],
        "secondary": [
            {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
            {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
        ],
        "cable": [
            {"network": "FS1", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
            {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2},
            {"network": "FS1", "time_slot": "Friday", "comp_tier1": 1},
        ],
    },
    "ACC": {
        "label": "ACC ESPN/ABC/CW Mix",
        "premium": [
            {"network": "ABC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 0},
            {"network": "ABC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
        ],
        "secondary": [
            {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
            {"network": "CW", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "ESPN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
        ],
        "cable": [
            {"network": "ESPN2", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
            {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "CW", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2},
            {"network": "ESPN2", "time_slot": "Friday", "comp_tier1": 1},
        ],
    },
    "Superleague": {
        "label": "Superleague National TV Mix",
        "premium": [
            {"network": "ABC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 0},
            {"network": "CBS", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 0},
            {"network": "NBC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0},
            {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
        ],
        "secondary": [
            {"network": "ABC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "FOX", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
            {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "NBC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1},
        ],
        "cable": [
            {"network": "ESPN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1},
            {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1},
            {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2},
            {"network": "ESPN2", "time_slot": "Friday", "comp_tier1": 1},
        ],
    },
}

DEFAULT_REALIGNMENT_NETWORK_PLAN = {
    "label": "ESPN/ABC Mix",
    "premium": REALIGNMENT_NETWORK_PLANS["SEC"]["premium"],
    "secondary": REALIGNMENT_NETWORK_PLANS["SEC"]["secondary"],
    "cable": REALIGNMENT_NETWORK_PLANS["SEC"]["cable"],
}

REALIGNMENT_WEEKLY_TV_INVENTORY = {
    "Big 10": {"premium": 3, "secondary": 2, "cable": 2},
    "SEC": {"premium": 3, "secondary": 2, "cable": 1},
    "Big 12": {"premium": 2, "secondary": 2, "cable": 2},
    "ACC": {"premium": 2, "secondary": 2, "cable": 1},
    "Superleague": {"premium": 4, "secondary": 3, "cable": 2},
}

REALIGNMENT_REGULAR_SEASON_WEEKS = 13
REALIGNMENT_PRIMARY_CONFERENCE_WEEKS = list(range(4, 14))
REALIGNMENT_EARLY_CONFERENCE_WEEKS = [1, 2, 3]
REALIGNMENT_EARLY_CONFERENCE_GAME_SHARE = 0.015

REALIGNMENT_NATIONAL_TV_SCORE_FLOOR = {
    "Big 10": 1.05,
    "SEC": 1.00,
    "Big 12": 0.75,
    "ACC": 0.55,
    "Superleague": 0.0,
}

REALIGNMENT_SEASONAL_RATED_SHARE = {
    "Big 10": 0.85,
    "SEC": 0.70,
    "Big 12": 0.67,
    "ACC": 0.56,
    "Superleague": 1.00,
}

REALIGNMENT_SEASONAL_SLOT_MIX = {
    "Big 10": [
        {
            "share": 0.29,
            "slot": {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        },
        {
            "share": 0.19,
            "slot": {"network": "NBC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        },
        {
            "share": 0.16,
            "slot": {"network": "CBS", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "premium"},
        },
        {
            "share": 0.16,
            "slot": {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "share": 0.20,
            "slot": {"network": "BTN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 2, "tv_tier": "cable"},
        },
    ],
    "SEC": [
        {
            "share": 0.38,
            "slot": {"network": "ABC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 0, "tv_tier": "premium"},
        },
        {
            "share": 0.26,
            "slot": {"network": "ABC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0, "tv_tier": "premium"},
        },
        {
            "share": 0.07,
            "slot": {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        },
        {
            "share": 0.23,
            "slot": {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        },
        {
            "share": 0.06,
            "slot": {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "secondary"},
        },
    ],
    "Big 12": [
        {
            "share": 0.03,
            "slot": {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        },
        {
            "share": 0.25,
            "slot": {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        },
        {
            "share": 0.36,
            "slot": {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        },
        {
            "share": 0.22,
            "slot": {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "share": 0.08,
            "slot": {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "share": 0.06,
            "slot": {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable"},
        },
    ],
    "ACC": [
        {
            "share": 0.06,
            "slot": {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        },
        {
            "share": 0.59,
            "slot": {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        },
        {
            "share": 0.23,
            "slot": {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "share": 0.12,
            "slot": {"network": "CW", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable"},
        },
    ],
}

REALIGNMENT_WEEKLY_SLOT_POOLS = {
    "Big 10": [
        {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "FOX", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium", "cadence": 3, "offset": 0},
        {"network": "NBC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "CBS", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "FS1", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 0},
        {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "BTN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 2, "tv_tier": "cable"},
        {"network": "BTN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable"},
        {"network": "BTN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 2, "offset": 1},
        {"network": "FS1", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 1},
    ],
    "SEC": [
        {"network": "ABC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 0, "tv_tier": "premium"},
        {"network": "ABC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 0, "tv_tier": "premium"},
        {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium", "cadence": 2, "offset": 0},
        {"network": "ESPN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "secondary", "cadence": 2, "offset": 1},
        {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 0},
        {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 4, "offset": 1},
        {"network": "ESPN", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "secondary", "cadence": 4, "offset": 1},
        {"network": "ESPN2", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "cable", "cadence": 4, "offset": 3},
    ],
    "Big 12": [
        {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium", "cadence": 6, "offset": 1},
        {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "FOX", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium", "cadence": 2, "offset": 1},
        {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "secondary", "cadence": 2, "offset": 0},
        {"network": "ESPN", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "secondary", "cadence": 2, "offset": 1},
        {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "ESPN2", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 0},
        {"network": "FS1", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "FS1", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 0},
        {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 4, "offset": 0},
        {"network": "ESPN2", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 1},
        {"network": "FS1", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 0},
        {"network": "FS1", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 1},
        {"network": "CW", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 3, "offset": 1},
        {"network": "CW", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 4, "offset": 0},
    ],
    "ACC": [
        {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium", "cadence": 4, "offset": 0},
        {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "secondary", "cadence": 2, "offset": 1},
        {"network": "ESPN", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN2", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "cable", "cadence": 2, "offset": 1},
        {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 4, "offset": 2},
        {"network": "CW", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 2, "offset": 1},
        {"network": "CW", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 2, "offset": 0},
        {"network": "CW", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 2, "tv_tier": "cable", "cadence": 2, "offset": 1},
        {"network": "CW", "time_slot": "Friday", "comp_tier1": 2, "tv_tier": "cable", "cadence": 3, "offset": 2},
        {"network": "ESPN2", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "cable", "cadence": 3, "offset": 1},
        {"network": "ESPN2", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "cable", "cadence": 3, "offset": 0},
    ],
}

REALIGNMENT_EXPANSION_WEEKLY_SLOTS = {
    "Big 10": [
        {
            "min_teams": 20,
            "slot": {"network": "FS1", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "min_teams": 22,
            "slot": {"network": "BTN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 2, "tv_tier": "cable"},
        },
    ],
    "SEC": [
        {
            "min_teams": 18,
            "slot": {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "min_teams": 20,
            "slot": {"network": "ESPNU", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 2, "tv_tier": "cable"},
        },
    ],
    "Big 12": [
        {
            "min_teams": 18,
            "slot": {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "min_teams": 20,
            "slot": {"network": "FS1", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "cable"},
        },
        {
            "min_teams": 22,
            "slot": {"network": "ESPN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "secondary"},
        },
        {
            "min_teams": 24,
            "slot": {"network": "FOX", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium", "cadence": 2, "offset": 0},
        },
    ],
    "ACC": [
        {
            "min_teams": 18,
            "slot": {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable"},
        },
    ],
}

REALIGNMENT_BLACK_FRIDAY_WEEK = max(REALIGNMENT_PRIMARY_CONFERENCE_WEEKS)

REALIGNMENT_BLACK_FRIDAY_EXTRA_SLOTS = {
    "Big 10": [
        {"network": "FOX", "time_slot": "Black Friday Early", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "CBS", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "NBC", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "FS1", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "cable"},
    ],
    "SEC": [
        {"network": "ABC", "time_slot": "Black Friday Early", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "ABC", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "ESPN", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN2", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "cable"},
    ],
    "Big 12": [
        {"network": "FOX", "time_slot": "Black Friday Early", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "FOX", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "ABC", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "ESPN", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "FS1", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "ESPN2", "time_slot": "Black Friday Late", "comp_tier1": 1, "tv_tier": "cable"},
    ],
    "ACC": [
        {"network": "ABC", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "premium"},
        {"network": "ESPN", "time_slot": "Black Friday Early", "comp_tier1": 1, "tv_tier": "secondary"},
        {"network": "ESPN2", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "cable"},
        {"network": "CW", "time_slot": "Black Friday Primetime", "comp_tier1": 2, "tv_tier": "cable"},
    ],
}


def _realignment_network_plan(conference):
    return REALIGNMENT_NETWORK_PLANS.get(conference, DEFAULT_REALIGNMENT_NETWORK_PLAN)


def _realignment_weekly_tv_inventory(conference):
    return REALIGNMENT_WEEKLY_TV_INVENTORY.get(conference, REALIGNMENT_WEEKLY_TV_INVENTORY["SEC"])


def _realignment_seasonal_rated_cap(conference, scheduled_games):
    share = REALIGNMENT_SEASONAL_RATED_SHARE.get(conference, REALIGNMENT_SEASONAL_RATED_SHARE["SEC"])
    return max(0, min(int(scheduled_games), int(round(int(scheduled_games) * share))))


def _select_realignment_tv_slot(game_index, total_games, conference):
    plan = _realignment_network_plan(conference)
    premium_slots = plan["premium"]
    secondary_slots = plan["secondary"]
    cable_slots = plan["cable"]

    premium_cutoff = max(8, math.ceil(total_games * 0.25))
    secondary_cutoff = max(premium_cutoff + 1, math.ceil(total_games * 0.65))

    if game_index < premium_cutoff:
        return premium_slots[game_index % len(premium_slots)]
    if game_index < secondary_cutoff:
        return secondary_slots[(game_index - premium_cutoff) % len(secondary_slots)]
    return cable_slots[(game_index - secondary_cutoff) % len(cable_slots)]


def _slot_available_in_week(slot, week):
    cadence = int(slot.get("cadence") or 1)
    offset = int(slot.get("offset") or 0)
    return ((int(week or 1) - 1 - offset) % cadence) == 0


def _slot_identity(slot):
    return (
        slot.get("network"),
        slot.get("time_slot"),
        int(slot.get("comp_tier1") or 0),
        slot.get("tv_tier"),
        int(slot.get("cadence") or 1),
        int(slot.get("offset") or 0),
    )


def _clean_slot(slot):
    return {
        key: value
        for key, value in dict(slot).items()
        if key not in {"cadence", "offset"}
    }


def _slot_for_week(slot, week):
    cleaned = _clean_slot(slot)
    if int(week or 0) == REALIGNMENT_BLACK_FRIDAY_WEEK and cleaned.get("time_slot") == "Friday":
        cleaned["time_slot"] = "Black Friday Early"
    return cleaned


def _black_friday_extra_slots(conference, week):
    if int(week or 0) != REALIGNMENT_BLACK_FRIDAY_WEEK:
        return []
    return [dict(slot) for slot in REALIGNMENT_BLACK_FRIDAY_EXTRA_SLOTS.get(conference, [])]


def _is_top_saturday_slot(slot):
    time_slot = str(slot.get("time_slot") or "")
    return slot.get("tv_tier") == "premium" and "Friday" not in time_slot


def _weekly_tv_slots(conference, week=None, team_count=None, transferred_slots=None, removed_slots=None):
    if conference in REALIGNMENT_WEEKLY_SLOT_POOLS:
        slots = list(REALIGNMENT_WEEKLY_SLOT_POOLS[conference])
        if team_count is not None:
            slots.extend(
                expansion_slot["slot"]
                for expansion_slot in REALIGNMENT_EXPANSION_WEEKLY_SLOTS.get(conference, [])
                if int(team_count) >= int(expansion_slot["min_teams"])
            )
        removal_counts = Counter(_slot_identity(slot) for slot in (removed_slots or []))
        kept_slots = []
        for slot in slots:
            key = _slot_identity(slot)
            if removal_counts[key] > 0:
                removal_counts[key] -= 1
                continue
            kept_slots.append(slot)
        base_slots = kept_slots + list(transferred_slots or [])
        top_saturday_slots = [slot for slot in base_slots if _is_top_saturday_slot(slot)]
        remaining_slots = [slot for slot in base_slots if not _is_top_saturday_slot(slot)]
        slots = top_saturday_slots + _black_friday_extra_slots(conference, week) + remaining_slots
        return [
            _slot_for_week(slot, week)
            for slot in slots
            if week is None or _slot_available_in_week(slot, week)
        ]
    plan = _realignment_network_plan(conference)
    inventory = _realignment_weekly_tv_inventory(conference)
    slots = []
    top_saturday_slots = []
    remaining_slots = []
    for tier in ["premium", "secondary", "cable"]:
        tier_slots = plan[tier]
        for idx in range(int(inventory.get(tier, 0))):
            slot = dict(tier_slots[idx % len(tier_slots)])
            slot["tv_tier"] = tier
            if _is_top_saturday_slot(slot):
                top_saturday_slots.append(slot)
            else:
                remaining_slots.append(slot)
    slots = top_saturday_slots + _black_friday_extra_slots(conference, week) + remaining_slots
    slots = [_slot_for_week(slot, week) for slot in slots]
    return slots


def _slot_key(slot):
    return (
        slot.get("network"),
        slot.get("time_slot"),
        int(slot.get("comp_tier1") or 0),
        slot.get("tv_tier"),
    )


def _seasonal_slot_sequence(conference, slot_count):
    slot_mix = REALIGNMENT_SEASONAL_SLOT_MIX.get(conference)
    if not slot_mix or slot_count <= 0:
        return []

    counts = []
    allocated = 0
    for idx, item in enumerate(slot_mix):
        if idx == len(slot_mix) - 1:
            count = slot_count - allocated
        else:
            count = int(round(slot_count * float(item["share"])))
        count = max(0, count)
        counts.append(count)
        allocated += count

    while allocated > slot_count:
        for idx in range(len(counts) - 1, -1, -1):
            if counts[idx] > 0 and allocated > slot_count:
                counts[idx] -= 1
                allocated -= 1
    while allocated < slot_count:
        largest_idx = max(range(len(slot_mix)), key=lambda idx: slot_mix[idx]["share"])
        counts[largest_idx] += 1
        allocated += 1

    sequence = []
    for item, count in zip(slot_mix, counts):
        sequence.extend([dict(item["slot"]) for _ in range(count)])
    return sequence


def _seasonal_slot_counts(conference, slot_count):
    return Counter(slot.get("network") for slot in _seasonal_slot_sequence(conference, slot_count))


def _allocate_counts(total, weights_by_key):
    if total <= 0 or not weights_by_key:
        return {key: 0 for key in weights_by_key}
    weight_total = sum(weights_by_key.values())
    raw_allocations = {
        key: (total * weight / weight_total)
        for key, weight in weights_by_key.items()
    }
    allocations = {
        key: int(math.floor(value))
        for key, value in raw_allocations.items()
    }
    remaining = total - sum(allocations.values())
    for key, _ in sorted(
        raw_allocations.items(),
        key=lambda item: (-(item[1] - math.floor(item[1])), item[0]),
    )[:remaining]:
        allocations[key] += 1
    return allocations


def _conference_tv_slot_transfers(memberships):
    original_memberships = {
        team: conference
        for team, conference in team_conferences.items()
        if conference in REALIGNMENT_EDITABLE_CONFERENCES and team in MODEL_TEAM_NAMES
    }
    moved_by_source_destination = Counter()
    current_counts = Counter()
    original_counts = Counter(original_memberships.values())

    for team, destination in memberships.items():
        if destination in REALIGNMENT_EDITABLE_CONFERENCES:
            current_counts[destination] += 1
        source = original_memberships.get(team)
        if (
            source in REALIGNMENT_EDITABLE_CONFERENCES
            and destination in REALIGNMENT_EDITABLE_CONFERENCES
            and source != destination
        ):
            moved_by_source_destination[(source, destination)] += 1

    transferred_slots = {conference: [] for conference in REALIGNMENT_EDITABLE_CONFERENCES}
    removed_slots = {conference: [] for conference in REALIGNMENT_EDITABLE_CONFERENCES}
    transfer_rows = []

    for source in REALIGNMENT_EDITABLE_CONFERENCES:
        source_moves = {
            destination: count
            for (move_source, destination), count in moved_by_source_destination.items()
            if move_source == source
        }
        moved_out = sum(source_moves.values())
        if moved_out <= 0:
            continue
        source_pool = REALIGNMENT_WEEKLY_SLOT_POOLS.get(source, [])
        if not source_pool:
            continue
        original_count = max(1, original_counts.get(source, len(source_pool)))
        remaining_count = current_counts.get(source, 0)
        reclaimable_limit = len(source_pool) if remaining_count < 4 else max(0, len(source_pool) - 1)
        reclaimed_count = min(
            reclaimable_limit,
            max(0, int(round(len(source_pool) * (moved_out / original_count) * 1.5))),
        )
        if reclaimed_count <= 0:
            continue
        reclaimed_slots = list(source_pool[:reclaimed_count])
        removed_slots[source].extend(reclaimed_slots)

        allocations = _allocate_counts(reclaimed_count, source_moves)
        slot_index = 0
        for destination, slot_count in sorted(allocations.items(), key=lambda item: (-item[1], item[0])):
            for _ in range(slot_count):
                if slot_index >= len(reclaimed_slots):
                    break
                slot = dict(reclaimed_slots[slot_index])
                slot["transferred_from"] = source
                transferred_slots[destination].append(slot)
                transfer_rows.append({
                    "source": source,
                    "destination": destination,
                    "teams_moved": int(source_moves[destination]),
                    "network": slot.get("network"),
                    "time_slot": slot.get("time_slot"),
                    "tv_tier": slot.get("tv_tier"),
                })
                slot_index += 1

    return transferred_slots, removed_slots, transfer_rows


def _viewer_total(values):
    series = pd.Series(values, dtype="float64").dropna()
    return float(series.sum()) if not series.empty else 0.0


def _viewer_average(values):
    series = pd.Series(values, dtype="float64").dropna()
    return float(series.mean()) if not series.empty else 0.0


def _football_team_coefficient(team):
    normalized_team = normalize_team(team)
    try:
        return float(pregame_model.params.get(normalized_team, 0.0))
    except Exception:
        return 0.0


def _conference_members(conference):
    members = [
        team
        for team, team_conference in team_conferences.items()
        if team_conference == conference and team in MODEL_TEAM_NAMES
    ]
    return sorted(
        set(members),
        key=lambda team: (-_football_team_coefficient(team), team),
    )


@app.get("/realignment/conference-members")
def realignment_conference_members(conference: str = "Big 10"):
    return {"conference": conference, "teams": _conference_members(conference)}


REALIGNMENT_EDITABLE_CONFERENCES = ["Big 10", "SEC", "Big 12", "ACC"]


@app.get("/realignment/memberships")
def realignment_memberships():
    memberships = {
        team: conference
        for team, conference in team_conferences.items()
        if conference in REALIGNMENT_EDITABLE_CONFERENCES and team in MODEL_TEAM_NAMES
    }
    return {
        "conferences": REALIGNMENT_EDITABLE_CONFERENCES,
        "memberships": memberships,
        "conference_teams": {
            conference: sorted(
                [team for team, team_conference in memberships.items() if team_conference == conference],
                key=lambda team: (-_football_team_coefficient(team), team),
            )
            for conference in REALIGNMENT_EDITABLE_CONFERENCES
        },
    }


def _rivalry_bonus(team_1, team_2):
    rivalry_pairs = _rivalry_pair_keys()
    return 0.75 if tuple(sorted([team_1, team_2])) in rivalry_pairs else 0.0


def _rivalry_pair_keys():
    return {
        tuple(sorted([normalize_team(team_1), normalize_team(team_2)]))
        for team_1, team_2 in rivalries.values()
        if normalize_team(team_1) and normalize_team(team_2)
    }


RIVALRY_WEEK_PRIORITY = {
    tuple(sorted(pair)): priority
    for priority, pair in {
        100: ("Alabama", "Auburn"),
        98: ("Michigan", "Ohio St."),
        96: ("Florida", "Florida St."),
        94: ("Georgia", "Georgia Tech"),
        92: ("Clemson", "South Carolina"),
        90: ("USC", "UCLA"),
        88: ("Oregon", "Washington"),
        86: ("Washington", "Washington St."),
        84: ("Arizona", "Arizona St."),
        82: ("BYU", "Utah"),
        80: ("Kansas", "Kansas St."),
        78: ("Minnesota", "Wisconsin"),
        76: ("Texas", "Texas A&M"),
        74: ("Mississippi", "Mississippi St."),
        72: ("Louisville", "Kentucky"),
        70: ("Vanderbilt", "Tennessee"),
        68: ("Navy", "Army"),
    }.items()
}


def _rivalry_week_priority(team_1, team_2):
    return RIVALRY_WEEK_PRIORITY.get(tuple(sorted([team_1, team_2])), 0)


def _protected_rivalry_pairs_for_teams(teams):
    team_set = set(teams)
    return sorted(
        [
            pair
            for pair in _rivalry_pair_keys()
            if pair[0] in team_set and pair[1] in team_set
        ]
    )


ESPN_2026_PRESEASON_RANKINGS = {
    "Ohio St.": 1,
    "Oregon": 2,
    "Georgia": 3,
    "Notre Dame": 4,
    "Texas": 5,
    "Indiana": 6,
    "Miami": 7,
    "Texas Tech": 8,
    "Mississippi": 9,
    "Texas A&M": 10,
    "LSU": 11,
    "BYU": 12,
    "Oklahoma": 13,
    "Michigan": 14,
    "Penn St.": 15,
    "Alabama": 16,
    "Washington": 17,
    "Utah": 18,
    "Iowa": 19,
    "USC": 20,
    "Louisville": 21,
    "SMU": 22,
    "TCU": 23,
    "Houston": 24,
    "Tennessee": 25,
}

FINAL_AP_2021_2025_AGGREGATE_RANKINGS = {
    "Georgia": 1,
    "Ohio St.": 2,
    "Alabama": 3,
    "Oregon": 4,
    "Notre Dame": 5,
    "Michigan": 6,
    "Mississippi": 7,
    "Texas": 8,
    "Penn St.": 9,
    "Tennessee": 10,
    "Clemson": 11,
    "Utah": 12,
    "Washington": 13,
    "Indiana": 14,
    "Oklahoma": 15,
    "BYU": 16,
    "Florida St.": 17,
    "Miami": 18,
    "Oklahoma St.": 19,
    "TCU": 20,
    "Tulane": 21,
    "LSU": 22,
    "Missouri": 23,
    "Cincinnati": 24,
    "Baylor": 25,
}


def _build_rank_map(teams, ranking_policy):
    if ranking_policy == "unranked":
        return {team: 0 for team in teams}
    if ranking_policy == "espn_2026_preseason":
        return {
            team: ESPN_2026_PRESEASON_RANKINGS.get(team, 0)
            for team in teams
        }
    if ranking_policy == "final_ap_2021_2025":
        return {
            team: FINAL_AP_2021_2025_AGGREGATE_RANKINGS.get(team, 0)
            for team in teams
        }

    ranked_teams = sorted(
        teams,
        key=lambda team: (-_football_team_coefficient(team), team),
    )
    rank_slots = [3, 6, 9, 12, 15, 18, 21, 24]
    return {
        team: rank_slots[idx] if idx < len(rank_slots) else 0
        for idx, team in enumerate(ranked_teams)
    }


def _matchup_score(team_1, team_2, rank_map):
    rank_1 = rank_map.get(team_1, 0)
    rank_2 = rank_map.get(team_2, 0)
    ranking_score = 0.0
    for rank in [rank_1, rank_2]:
        if 1 <= rank <= 10:
            ranking_score += 0.45
        elif 11 <= rank <= 25:
            ranking_score += 0.25

    return (
        _football_team_coefficient(team_1)
        + _football_team_coefficient(team_2)
        + ranking_score
        + _rivalry_bonus(team_1, team_2)
    )


def _interleaved_schedule_order(teams):
    seeded = sorted(
        teams,
        key=lambda team: (-_football_team_coefficient(team), team),
    )
    tiers = [seeded[0::3], seeded[1::3], seeded[2::3]]
    ordered = []
    max_len = max((len(tier) for tier in tiers), default=0)
    for idx in range(max_len):
        for tier in tiers:
            if idx < len(tier):
                ordered.append(tier[idx])
    return ordered


def _round_robin_pairs(teams):
    rotation = _interleaved_schedule_order(teams)
    if len(rotation) % 2 == 1:
        rotation.append(None)

    rounds = []
    team_count = len(rotation)
    for _ in range(team_count - 1):
        round_pairs = []
        for idx in range(team_count // 2):
            team_1 = rotation[idx]
            team_2 = rotation[team_count - 1 - idx]
            if team_1 is not None and team_2 is not None:
                round_pairs.append((team_1, team_2))
        rounds.append(round_pairs)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return rounds


def _generate_conference_schedule(teams, games_per_team, rank_map, protected_pairs=None):
    teams = list(dict.fromkeys(teams))
    requested_games = max(1, min(int(games_per_team or 9), max(1, len(teams) - 1)))
    protected_pair_keys = {
        tuple(sorted([team_1, team_2]))
        for team_1, team_2 in (protected_pairs or [])
        if team_1 in teams and team_2 in teams and team_1 != team_2
    }
    counts = {team: 0 for team in teams}
    selected = []
    selected_keys = set()

    def add_pair(team_1, team_2, protected=False):
        pair_key = tuple(sorted([team_1, team_2]))
        if pair_key in selected_keys:
            return False
        if not protected and (counts[team_1] >= max_games or counts[team_2] >= max_games):
            return False
        selected.append({
            "team1": team_1,
            "team2": team_2,
            "score": _matchup_score(team_1, team_2, rank_map),
            "protected": protected,
            "rivalry_week_priority": _rivalry_week_priority(team_1, team_2),
        })
        selected_keys.add(pair_key)
        counts[team_1] += 1
        counts[team_2] += 1
        return True

    max_games = requested_games

    for team_1, team_2 in sorted(protected_pair_keys):
        add_pair(team_1, team_2, protected=True)

    max_games = max(requested_games, max(counts.values(), default=requested_games))

    for round_pairs in _round_robin_pairs(teams):
        for team_1, team_2 in round_pairs:
            add_pair(team_1, team_2, protected=False)
        if all(count >= max_games for count in counts.values()):
            break

    all_pairs = []
    for i, team_1 in enumerate(teams):
        for team_2 in teams[i + 1:]:
            pair_key = tuple(sorted([team_1, team_2]))
            if pair_key in selected_keys:
                continue
            all_pairs.append((team_1, team_2))

    while any(count < max_games for count in counts.values()):
        candidates = [
            (team_1, team_2)
            for team_1, team_2 in all_pairs
            if counts[team_1] < max_games and counts[team_2] < max_games
        ]
        if not candidates:
            break

        team_1, team_2 = sorted(
            candidates,
            key=lambda pair: (
                counts[pair[0]] + counts[pair[1]],
                abs(_football_team_coefficient(pair[0]) - _football_team_coefficient(pair[1])),
                pair[0],
                pair[1],
            ),
        )[0]
        add_pair(team_1, team_2, protected=False)
        pair_key = tuple(sorted([team_1, team_2]))
        all_pairs = [
            pair for pair in all_pairs
            if tuple(sorted(pair)) != pair_key
        ]

    return selected, counts


def _assign_schedule_weeks(schedule, teams, games_per_team):
    early_game_limit = min(
        len(REALIGNMENT_EARLY_CONFERENCE_WEEKS),
        int(round(len(schedule) * REALIGNMENT_EARLY_CONFERENCE_GAME_SHARE)),
    )
    early_game_limit = max(0, early_game_limit)
    weeks = [
        {"number": week_number, "games": [], "teams": set(), "score": 0.0}
        for week_number in [
            *REALIGNMENT_EARLY_CONFERENCE_WEEKS,
            *REALIGNMENT_PRIMARY_CONFERENCE_WEEKS,
        ]
    ]
    early_week_numbers = set(REALIGNMENT_EARLY_CONFERENCE_WEEKS)
    primary_week_numbers = set(REALIGNMENT_PRIMARY_CONFERENCE_WEEKS)
    early_pair_keys = {
        tuple(sorted([game["team1"], game["team2"]]))
        for game in sorted(
            [game for game in schedule if not game.get("protected", False)],
            key=lambda row: (row["score"], row["team1"], row["team2"]),
        )[:early_game_limit]
    }

    def add_game_to_week(game, target_week):
        game_with_week = {**game, "week": target_week["number"]}
        target_week["games"].append(game_with_week)
        target_week["teams"].update([game["team1"], game["team2"]])
        target_week["score"] += float(game["score"])

    remaining_games = []
    for game in schedule:
        pair_key = tuple(sorted([game["team1"], game["team2"]]))
        if pair_key in early_pair_keys:
            early_candidates = [
                week for week in weeks
                if week["number"] in early_week_numbers
                and game["team1"] not in week["teams"]
                and game["team2"] not in week["teams"]
            ]
            if early_candidates:
                add_game_to_week(
                    game,
                    sorted(early_candidates, key=lambda week: (len(week["games"]), week["number"]))[0],
                )
                continue
        remaining_games.append(game)

    primary_weeks = [week for week in weeks if week["number"] in primary_week_numbers]

    def schedule_with_primary_week_coloring(games_to_schedule):
        game_items = sorted(
            list(enumerate(games_to_schedule)),
            key=lambda item: (
                not item[1].get("protected", False),
                -int(item[1].get("rivalry_week_priority") or 0),
                -item[1]["score"],
                item[1]["team1"],
                item[1]["team2"],
            ),
        )
        used_team_weeks = {
            (team, week["number"])
            for week in weeks
            for scheduled_game in week["games"]
            for team in [scheduled_game["team1"], scheduled_game["team2"]]
        }
        week_loads = Counter({week["number"]: len(week["games"]) for week in primary_weeks})
        week_scores = Counter({week["number"]: week["score"] for week in primary_weeks})
        assignments = {}

        def available_weeks(game):
            return [
                week["number"]
                for week in primary_weeks
                if (game["team1"], week["number"]) not in used_team_weeks
                and (game["team2"], week["number"]) not in used_team_weeks
            ]

        def search(unassigned_items):
            if not unassigned_items:
                return True
            candidate_options = []
            for item_index, (game_index, game) in enumerate(unassigned_items):
                options = available_weeks(game)
                if not options:
                    return False
                candidate_options.append((
                    len(options),
                    not game.get("protected", False),
                    -int(game.get("rivalry_week_priority") or 0),
                    -game["score"],
                    game["team1"],
                    game["team2"],
                    item_index,
                    game_index,
                    game,
                    options,
                ))
            *_, item_index, game_index, game, options = sorted(candidate_options)[0]
            remaining_items = unassigned_items[:item_index] + unassigned_items[item_index + 1:]
            if game.get("protected", False):
                ordered_options = sorted(
                    options,
                    key=lambda number: (
                        -number,
                        week_loads[number],
                        week_scores[number],
                    ),
                )
            else:
                ordered_options = sorted(
                    options,
                    key=lambda number: (
                        week_loads[number],
                        week_scores[number],
                        number,
                    ),
                )
            for week_number in ordered_options:
                assignments[game_index] = week_number
                used_team_weeks.add((game["team1"], week_number))
                used_team_weeks.add((game["team2"], week_number))
                week_loads[week_number] += 1
                week_scores[week_number] += float(game["score"])
                if search(remaining_items):
                    return True
                week_scores[week_number] -= float(game["score"])
                week_loads[week_number] -= 1
                used_team_weeks.remove((game["team1"], week_number))
                used_team_weeks.remove((game["team2"], week_number))
                assignments.pop(game_index, None)
            return False

        if not search(game_items):
            return False
        weeks_by_number = {week["number"]: week for week in primary_weeks}
        for game_index, game in enumerate(games_to_schedule):
            add_game_to_week(game, weeks_by_number[assignments[game_index]])
        return True

    if schedule_with_primary_week_coloring(remaining_games):
        return [
            game
            for week in sorted(weeks, key=lambda row: row["number"])
            for game in week["games"]
        ]

    remaining_games = sorted(
        remaining_games,
        key=lambda row: (
            not row.get("protected", False),
            -int(row.get("rivalry_week_priority") or 0),
            -row["score"],
            row["team1"],
            row["team2"],
        ),
    )
    while remaining_games:
        made_progress = False
        for week in sorted(primary_weeks, key=lambda row: (len(row["games"]), row["score"], row["number"])):
            candidate = next(
                (
                    game for game in remaining_games
                    if game["team1"] not in week["teams"] and game["team2"] not in week["teams"]
                ),
                None,
            )
            if not candidate:
                continue
            add_game_to_week(candidate, week)
            remaining_games.remove(candidate)
            made_progress = True
        if not made_progress:
            break

    for game in remaining_games:
        target_week = sorted(
            primary_weeks,
            key=lambda week: (
                int(game["team1"] in week["teams"]) + int(game["team2"] in week["teams"]),
                len(week["games"]),
                week["score"],
                week["number"],
            ),
        )[0]
        add_game_to_week(game, target_week)

    def rebuild_week_state(week):
        week["teams"] = set()
        week["score"] = 0.0
        for scheduled_game in week["games"]:
            week["teams"].update([scheduled_game["team1"], scheduled_game["team2"]])
            week["score"] += float(scheduled_game["score"])

    for _ in range(200):
        moved_game = False
        for week in primary_weeks:
            games_by_team = defaultdict(list)
            for scheduled_game in week["games"]:
                games_by_team[scheduled_game["team1"]].append(scheduled_game)
                games_by_team[scheduled_game["team2"]].append(scheduled_game)
            conflicted_games = []
            for games in games_by_team.values():
                conflicted_games.extend(games[1:])
            for scheduled_game in conflicted_games:
                destination_week = next(
                    (
                        candidate_week for candidate_week in sorted(
                            primary_weeks,
                            key=lambda row: (len(row["games"]), row["score"], row["number"]),
                        )
                        if candidate_week is not week
                        and scheduled_game["team1"] not in candidate_week["teams"]
                        and scheduled_game["team2"] not in candidate_week["teams"]
                    ),
                    None,
                )
                if not destination_week:
                    continue
                week["games"].remove(scheduled_game)
                scheduled_game["week"] = destination_week["number"]
                destination_week["games"].append(scheduled_game)
                rebuild_week_state(week)
                rebuild_week_state(destination_week)
                moved_game = True
                break
            if moved_game:
                break
        if not moved_game:
            break

    return [
        game
        for week in sorted(weeks, key=lambda row: row["number"])
        for game in week["games"]
    ]


def _realistic_tv_slots_for_schedule(
    schedule,
    conference,
    team_count=None,
    transferred_slots=None,
    removed_slots=None,
):
    weeks = sorted({game.get("week", 1) for game in schedule})
    if not weeks:
        return {}
    score_floor = REALIGNMENT_NATIONAL_TV_SCORE_FLOOR.get(
        conference,
        REALIGNMENT_NATIONAL_TV_SCORE_FLOOR["SEC"],
    )
    selected_slots = {}
    used_weekly_slot_indexes = defaultdict(set)
    ordered_games = sorted(
        schedule,
        key=lambda row: (-row["score"], row.get("week", 1), row["team1"], row["team2"]),
    )
    for game in ordered_games:
        if float(game.get("score") or 0.0) < score_floor:
            continue
        week = game.get("week", 1)
        weekly_slots = _weekly_tv_slots(
            conference,
            week,
            team_count=team_count,
            transferred_slots=transferred_slots,
            removed_slots=removed_slots,
        )
        pair_key = tuple(sorted([game["team1"], game["team2"]]))
        if pair_key in selected_slots:
            continue
        for slot_idx, slot in enumerate(weekly_slots):
            if slot_idx in used_weekly_slot_indexes[week]:
                continue
            selected_slots[pair_key] = dict(slot)
            used_weekly_slot_indexes[week].add(slot_idx)
            break

    return selected_slots


def _global_tv_slot_key(slot):
    return (
        slot.get("network"),
        slot.get("time_slot"),
    )


def _mark_realignment_row_unrated(row):
    row["network"] = "Not nationally rated"
    row["time_slot"] = "No national TV window"
    row["comp_tier1"] = None
    row["tv_tier"] = "unrated"
    row["nationally_rated"] = False
    row["predicted_viewers"] = 0.0
    row["predicted_viewers_formatted"] = "Not rated"


def _apply_tv_slot_to_realignment_row(row, slot, conference, conference_teams):
    prediction = predict_viewership({
        "team1": row["team1"],
        "team2": row["team2"],
        "rank1": row.get("rank1", 0),
        "rank2": row.get("rank2", 0),
        "network": slot["network"],
        "time_slot": slot["time_slot"],
        "comp_tier1": slot["comp_tier1"],
        "conference_overrides": {team: conference for team in conference_teams},
    })
    row["network"] = slot["network"]
    row["time_slot"] = slot["time_slot"]
    row["comp_tier1"] = slot["comp_tier1"]
    row["tv_tier"] = slot.get("tv_tier")
    row["nationally_rated"] = True
    row["predicted_viewers"] = float(prediction["raw"])
    row["predicted_viewers_formatted"] = prediction["formatted"]


def _refresh_realignment_slate_summary(slate):
    rows = slate.get("rows", [])
    rated_rows = [row for row in rows if row.get("nationally_rated", True)]
    slate["summary"] = {
        "teams": len(slate.get("teams", [])),
        "games": len(rated_rows),
        "scheduled_games": len(rows),
        "nationally_rated_games": len(rated_rows),
        "unrated_games": len(rows) - len(rated_rows),
        "total_viewers": _viewer_total(row["predicted_viewers"] for row in rated_rows),
        "average_viewers": _viewer_average(row["predicted_viewers"] for row in rated_rows),
    }


def _apply_global_tv_slots_to_league_slates(conference_slates, transferred_slots_by_conference, removed_slots_by_conference):
    candidates = []
    for conference, slate in conference_slates.items():
        for row in slate.get("rows", []):
            _mark_realignment_row_unrated(row)
            candidates.append({
                "conference": conference,
                "row": row,
                "team_count": len(slate.get("teams", [])),
                "conference_teams": slate.get("teams", []),
            })

    used_slots_by_week = defaultdict(set)
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item["row"].get("matchup_score") or 0.0),
            int(item["row"].get("week") or 1),
            item["conference"],
            item["row"].get("team1", ""),
            item["row"].get("team2", ""),
        ),
    )

    for candidate in ordered_candidates:
        conference = candidate["conference"]
        row = candidate["row"]
        if float(row.get("matchup_score") or 0.0) < REALIGNMENT_NATIONAL_TV_SCORE_FLOOR.get(
            conference,
            REALIGNMENT_NATIONAL_TV_SCORE_FLOOR["SEC"],
        ):
            continue
        week = int(row.get("week") or 1)
        weekly_slots = _weekly_tv_slots(
            conference,
            week,
            team_count=candidate["team_count"],
            transferred_slots=transferred_slots_by_conference.get(conference, []),
            removed_slots=removed_slots_by_conference.get(conference, []),
        )
        for slot in weekly_slots:
            slot_key = _global_tv_slot_key(slot)
            if slot_key in used_slots_by_week[week]:
                continue
            used_slots_by_week[week].add(slot_key)
            _apply_tv_slot_to_realignment_row(
                row,
                slot,
                conference,
                candidate["conference_teams"],
            )
            break

    for slate in conference_slates.values():
        _refresh_realignment_slate_summary(slate)


def _predict_realignment_slate(
    teams,
    conference,
    sim,
    protected_pairs=None,
    realistic_tv_inventory=False,
    transferred_slots=None,
    removed_slots=None,
):
    normalized_teams = [normalize_team(team) for team in teams]
    rank_map = _build_rank_map(normalized_teams, sim.ranking_policy)
    schedule, counts = _generate_conference_schedule(
        normalized_teams,
        sim.games_per_team,
        rank_map,
        protected_pairs=protected_pairs,
    )
    if realistic_tv_inventory:
        schedule = _assign_schedule_weeks(schedule, normalized_teams, sim.games_per_team)
        scheduled_tv_slots = _realistic_tv_slots_for_schedule(
            schedule,
            conference,
            team_count=len(normalized_teams),
            transferred_slots=transferred_slots,
            removed_slots=removed_slots,
        )
    else:
        schedule = sorted(schedule, key=lambda row: (-row["score"], row["team1"], row["team2"]))
        scheduled_tv_slots = {}
    conference_overrides = {team: conference for team in normalized_teams}

    rows = []
    total_games = len(schedule)
    for idx, game in enumerate(schedule):
        pair_key = tuple(sorted([game["team1"], game["team2"]]))
        slot = (
            scheduled_tv_slots.get(pair_key)
            if realistic_tv_inventory
            else _select_realignment_tv_slot(idx, total_games, conference)
        )
        if slot:
            prediction = predict_viewership({
                "team1": game["team1"],
                "team2": game["team2"],
                "rank1": rank_map.get(game["team1"], 0),
                "rank2": rank_map.get(game["team2"], 0),
                "network": slot["network"],
                "time_slot": slot["time_slot"],
                "comp_tier1": slot["comp_tier1"],
                "conference_overrides": conference_overrides,
            })
            viewers = float(prediction["raw"])
            predicted_viewers_formatted = prediction["formatted"]
            network = slot["network"]
            time_slot = slot["time_slot"]
            comp_tier1 = slot["comp_tier1"]
            tv_tier = slot.get("tv_tier")
            nationally_rated = True
        else:
            viewers = 0.0
            predicted_viewers_formatted = "Not rated"
            network = "Not nationally rated"
            time_slot = "No national TV window"
            comp_tier1 = None
            tv_tier = "unrated"
            nationally_rated = False
        rows.append({
            "game_number": idx + 1,
            "week": game.get("week"),
            "team1": game["team1"],
            "team2": game["team2"],
            "matchup": f"{game['team1']} vs {game['team2']}",
            "rank1": rank_map.get(game["team1"], 0),
            "rank2": rank_map.get(game["team2"], 0),
            "network": network,
            "time_slot": time_slot,
            "comp_tier1": comp_tier1,
            "tv_tier": tv_tier,
            "nationally_rated": nationally_rated,
            "matchup_score": float(round(game["score"], 4)),
            "protected": bool(game.get("protected", False)),
            "predicted_viewers": viewers,
            "predicted_viewers_formatted": predicted_viewers_formatted,
        })

    rated_rows = [row for row in rows if row.get("nationally_rated", True)]
    return {
        "teams": normalized_teams,
        "team_game_counts": counts,
        "rank_map": rank_map,
        "rows": rows,
        "summary": {
            "teams": len(normalized_teams),
            "games": len(rated_rows) if realistic_tv_inventory else len(rows),
            "scheduled_games": len(rows),
            "nationally_rated_games": len(rated_rows),
            "unrated_games": len(rows) - len(rated_rows),
            "total_viewers": _viewer_total(row["predicted_viewers"] for row in rated_rows),
            "average_viewers": _viewer_average(row["predicted_viewers"] for row in rated_rows),
        },
    }


@app.post("/realignment/simulate")
def realignment_simulation(sim: RealignmentSimulationInput):
    conference = sim.conference or "Big 10"
    expansion_teams = [
        normalize_team(team)
        for team in sim.expansion_teams
        if normalize_team(team)
    ]
    protected_opponents = [
        normalize_team(team)
        for team in sim.protected_opponents
        if normalize_team(team)
    ]
    protected_opponents_by_team = {}
    for raw_team, raw_opponents in (sim.protected_opponents_by_team or {}).items():
        expansion_team = normalize_team(raw_team)
        if not expansion_team or expansion_team not in expansion_teams:
            continue
        protected_opponents_by_team[expansion_team] = [
            normalize_team(opponent)
            for opponent in (raw_opponents or [])
            if normalize_team(opponent)
        ]
    if expansion_teams and protected_opponents:
        first_expansion_team = expansion_teams[0]
        protected_opponents_by_team[first_expansion_team] = list(dict.fromkeys([
            *protected_opponents_by_team.get(first_expansion_team, []),
            *protected_opponents,
        ]))
    baseline_teams = [
        team
        for team in _conference_members(conference)
        if team not in expansion_teams
    ]
    expanded_teams = sorted(
        set(baseline_teams) | set(expansion_teams),
        key=lambda team: (-_football_team_coefficient(team), team),
    )

    user_protected_pairs = [
        (team, opponent)
        for team, opponents in protected_opponents_by_team.items()
        for opponent in opponents
        if team in expanded_teams and opponent in expanded_teams and opponent != team
    ]
    user_protected_pairs = sorted(set(
        tuple(sorted([team, opponent]))
        for team, opponent in user_protected_pairs
    ))
    normalized_protected_opponents_by_team = {
        team: [
            opponent
            for opponent in opponents
            if opponent in expanded_teams and opponent != team
        ]
        for team, opponents in protected_opponents_by_team.items()
        if team in expanded_teams
    }
    baseline_rivalry_pairs = _protected_rivalry_pairs_for_teams(baseline_teams)
    expanded_rivalry_pairs = _protected_rivalry_pairs_for_teams(expanded_teams)
    expanded_protected_pairs = sorted(
        set(expanded_rivalry_pairs) | {
            tuple(sorted([team_1, team_2]))
            for team_1, team_2 in user_protected_pairs
        }
    )

    baseline = _predict_realignment_slate(
        baseline_teams,
        conference,
        sim,
        protected_pairs=baseline_rivalry_pairs,
    )
    expanded = _predict_realignment_slate(
        expanded_teams,
        conference,
        sim,
        protected_pairs=expanded_protected_pairs,
    )

    baseline_summary = baseline["summary"]
    expanded_summary = expanded["summary"]
    expansion_game_rows = [
        row
        for row in expanded["rows"]
        if row["team1"] in expansion_teams or row["team2"] in expansion_teams
    ]
    scheduled_pair_keys = {
        tuple(sorted([row["team1"], row["team2"]]))
        for row in expanded["rows"]
    }
    protected_matchups = []
    for team_1, team_2 in user_protected_pairs:
        key = tuple(sorted([team_1, team_2]))
        game = next(
            (
                row for row in expanded["rows"]
                if tuple(sorted([row["team1"], row["team2"]])) == key
            ),
            None,
        )
        protected_matchups.append({
            "expansion_team": team_1 if team_1 in expansion_teams else team_2,
            "opponent": team_2 if team_1 in expansion_teams else team_1,
            "team1": team_1,
            "team2": team_2,
            "matchup": f"{team_1} vs {team_2}",
            "scheduled": key in scheduled_pair_keys,
            "game_number": game.get("game_number") if game else None,
            "predicted_viewers": game.get("predicted_viewers") if game else None,
        })

    total_delta = expanded_summary["total_viewers"] - baseline_summary["total_viewers"]
    average_delta = expanded_summary["average_viewers"] - baseline_summary["average_viewers"]
    expansion_average = _viewer_average(row["predicted_viewers"] for row in expansion_game_rows)
    expansion_vs_baseline_average = expansion_average - baseline_summary["average_viewers"]

    return clean_nan({
        "realignment_simulator_version": 3,
        "conference": conference,
        "expansion_teams": expansion_teams,
        "settings": {
            "games_per_team": sim.games_per_team,
            "network_policy": _realignment_network_plan(conference)["label"],
            "tv_network_plan": _realignment_network_plan(conference)["label"],
            "ranking_policy": sim.ranking_policy,
            "protected_opponents": protected_opponents,
            "protected_opponents_by_team": normalized_protected_opponents_by_team,
        },
        "protected_matchups": protected_matchups,
        "automatic_protected_rivalries": [
            {"team1": team_1, "team2": team_2, "matchup": f"{team_1} vs {team_2}"}
            for team_1, team_2 in expanded_rivalry_pairs
        ],
        "protected_missing_count": int(sum(1 for row in protected_matchups if not row["scheduled"])),
        "baseline": baseline,
        "expanded": expanded,
        "impact": {
            "total_viewer_delta": float(total_delta),
            "average_viewer_delta": float(average_delta),
            "game_inventory_delta": int(expanded_summary["games"] - baseline_summary["games"]),
            "expansion_team_games": int(len(expansion_game_rows)),
            "expansion_team_average_viewers": float(expansion_average),
            "expansion_vs_baseline_average": float(expansion_vs_baseline_average),
        },
        "top_expansion_games": sorted(
            expansion_game_rows,
            key=lambda row: row["predicted_viewers"],
            reverse=True,
        )[:12],
        "available_options": {
            "conferences": sorted(set(team_conferences.values())),
            "network_policies": [],
            "ranking_policies": ["brand_tiers", "espn_2026_preseason", "final_ap_2021_2025", "unranked"],
        },
        "methodology": (
            "Generates a deterministic conference slate by locking known rivalry games, "
            "then selecting available matchups until teams reach the requested conference-game cap. "
            "Expansion teams are temporarily treated as members of the target conference "
            "inside the football viewership model."
        ),
    })


@app.post("/realignment/superleague")
def superleague_simulation(sim: SuperleagueSimulationInput):
    selected_teams = sorted(
        {
            normalize_team(team)
            for team in sim.teams
            if normalize_team(team) in MODEL_TEAM_NAMES
        },
        key=lambda team: (-_football_team_coefficient(team), team),
    )
    if len(selected_teams) < 2:
        return clean_nan({
            "realignment_simulator_version": 3,
            "mode": "superleague",
            "settings": {
                "teams": selected_teams,
                "games_per_team": sim.games_per_team,
                "ranking_policy": sim.ranking_policy,
                "tv_network_plan": _realignment_network_plan("Superleague")["label"],
            },
            "slate": {
                "teams": selected_teams,
                "team_game_counts": {},
                "rank_map": {},
                "rows": [],
                "summary": {
                    "teams": len(selected_teams),
                    "games": 0,
                    "total_viewers": 0.0,
                    "average_viewers": 0.0,
                },
            },
            "top_games": [],
            "automatic_protected_rivalries": [],
            "methodology": "Draft at least two teams to generate a superleague slate.",
        })

    protected_rivalry_pairs = _protected_rivalry_pairs_for_teams(selected_teams)
    slate = _predict_realignment_slate(
        selected_teams,
        "Superleague",
        sim,
        protected_pairs=protected_rivalry_pairs,
    )
    top_games = sorted(
        slate["rows"],
        key=lambda row: row["predicted_viewers"],
        reverse=True,
    )[:40]

    return clean_nan({
        "realignment_simulator_version": 3,
        "mode": "superleague",
        "settings": {
            "teams": selected_teams,
            "games_per_team": sim.games_per_team,
            "ranking_policy": sim.ranking_policy,
            "tv_network_plan": _realignment_network_plan("Superleague")["label"],
        },
        "slate": slate,
        "impact": {
            "total_viewers": slate["summary"]["total_viewers"],
            "average_viewers": slate["summary"]["average_viewers"],
            "games": slate["summary"]["games"],
            "teams": slate["summary"]["teams"],
        },
        "top_games": top_games,
        "automatic_protected_rivalries": [
            {"team1": team_1, "team2": team_2, "matchup": f"{team_1} vs {team_2}"}
            for team_1, team_2 in protected_rivalry_pairs
        ],
        "available_options": {
            "ranking_policies": ["brand_tiers", "espn_2026_preseason", "final_ap_2021_2025", "unranked"],
        },
        "methodology": (
            "Builds a custom superleague from the drafted teams, locks known rivalry games, "
            "then fills a deterministic conference-style slate using the selected games-per-team cap. "
            "Games use a national Superleague TV mix and the football viewership model."
        ),
    })


@app.post("/realignment/league-simulate")
def league_realignment_simulation(sim: LeagueRealignmentSimulationInput):
    normalized_memberships = {}
    for raw_team, raw_conference in (sim.memberships or {}).items():
        team = normalize_team(raw_team)
        conference = raw_conference if raw_conference in REALIGNMENT_EDITABLE_CONFERENCES else None
        if team in MODEL_TEAM_NAMES and conference:
            normalized_memberships[team] = conference

    if not normalized_memberships:
        normalized_memberships = {
            team: conference
            for team, conference in team_conferences.items()
            if conference in REALIGNMENT_EDITABLE_CONFERENCES and team in MODEL_TEAM_NAMES
        }

    user_protected_pairs = set()
    for raw_team, raw_opponents in (sim.protected_matchups_by_team or {}).items():
        team = normalize_team(raw_team)
        if team not in normalized_memberships:
            continue
        for raw_opponent in raw_opponents or []:
            opponent = normalize_team(raw_opponent)
            if (
                opponent in normalized_memberships
                and opponent != team
                and normalized_memberships.get(opponent) == normalized_memberships.get(team)
            ):
                user_protected_pairs.add(tuple(sorted([team, opponent])))

    transferred_tv_slots, removed_tv_slots, tv_slot_transfer_rows = _conference_tv_slot_transfers(normalized_memberships)
    conference_slates = {}
    for conference in REALIGNMENT_EDITABLE_CONFERENCES:
        members = sorted(
            [
                team
                for team, team_conference in normalized_memberships.items()
                if team_conference == conference
            ],
            key=lambda team: (-_football_team_coefficient(team), team),
        )
        if len(members) < 2:
            conference_slates[conference] = {
                "teams": members,
                "team_game_counts": {},
                "rank_map": {},
                "rows": [],
                "summary": {
                    "teams": len(members),
                    "games": 0,
                    "total_viewers": 0.0,
                    "average_viewers": 0.0,
                },
            }
            continue

        protected_rivalry_pairs = _protected_rivalry_pairs_for_teams(members)
        protected_pairs = sorted(
            set(protected_rivalry_pairs) | {
                pair for pair in user_protected_pairs
                if pair[0] in members and pair[1] in members
            }
        )
        slate = _predict_realignment_slate(
            members,
            conference,
            sim,
            protected_pairs=protected_pairs,
            realistic_tv_inventory=True,
            transferred_slots=transferred_tv_slots.get(conference, []),
            removed_slots=removed_tv_slots.get(conference, []),
        )
        conference_slates[conference] = slate

    _apply_global_tv_slots_to_league_slates(
        conference_slates,
        transferred_tv_slots,
        removed_tv_slots,
    )

    all_rows = []
    for conference, slate in conference_slates.items():
        for row in slate["rows"]:
            all_rows.append({
                **row,
                "conference": conference,
                "tv_network_plan": _realignment_network_plan(conference)["label"],
            })

    all_rows = sorted(
        all_rows,
        key=lambda row: (-row["predicted_viewers"], row["conference"], row["game_number"]),
    )
    for idx, row in enumerate(all_rows):
        row["global_game_number"] = idx + 1

    conference_summaries = {
        conference: slate["summary"]
        for conference, slate in conference_slates.items()
    }
    rated_rows = [row for row in all_rows if row.get("nationally_rated", True)]
    total_viewers = _viewer_total(row["predicted_viewers"] for row in rated_rows)
    average_viewers = _viewer_average(row["predicted_viewers"] for row in rated_rows)

    return clean_nan({
        "realignment_simulator_version": 3,
        "mode": "league_realignment",
        "settings": {
            "games_per_team": sim.games_per_team,
            "ranking_policy": sim.ranking_policy,
            "conferences": REALIGNMENT_EDITABLE_CONFERENCES,
            "tv_slot_transfers": tv_slot_transfer_rows,
        },
        "memberships": normalized_memberships,
        "conference_slates": conference_slates,
        "conference_summaries": conference_summaries,
        "rows": all_rows,
        "summary": {
            "conferences": len(REALIGNMENT_EDITABLE_CONFERENCES),
            "teams": len(normalized_memberships),
            "games": len(rated_rows),
            "scheduled_games": len(all_rows),
            "nationally_rated_games": len(rated_rows),
            "unrated_games": len(all_rows) - len(rated_rows),
            "total_viewers": total_viewers,
            "average_viewers": average_viewers,
        },
        "top_games": rated_rows[:40],
        "methodology": (
            "Schedules every edited conference using the selected games-per-team cap, "
            "locks known rivalry games within each conference, assigns a limited weekly "
            "national TV inventory from each conference's partner mix, marks the rest as "
            "not nationally rated, and aggregates rated TV games by projected viewership."
        ),
    })


@app.get("/brand-years")
def brand_years():
    return {"years": available_years}

@app.get("/brand-rankings")
def brand_rankings(year: str = "all", scope: str = "combined", control_deion: bool = False):
    normalized_scope = str(scope or "combined").lower()
    if normalized_scope == "football":
        return {
            "rows": football_brand_rankings_for_main_tab(control_deion=control_deion) if year == "all" else brand_rankings_cache.get(year, []),
            "metadata": {
                "scope": "football",
                "description": "Football-only brand pull ranking.",
                "control_deion": control_deion,
                "colorado_deion_adjustment": float(round(colorado_deion_adjustment(), 6)) if control_deion else 0.0,
            },
        }
    if normalized_scope == "basketball":
        return {
            "rows": basketball_brand_rankings_for_main_tab(),
            "metadata": {
                "scope": "basketball",
                "description": "Men's basketball-only brand pull ranking.",
                "control_deion": False,
            },
        }
    return {
        "rows": combined_media_brand_rankings(control_deion=control_deion),
        "metadata": {
            "scope": "combined",
            "football_weight": FOOTBALL_MEDIA_RIGHTS_WEIGHT,
            "basketball_weight": BASKETBALL_MEDIA_RIGHTS_WEIGHT,
            "description": "Media rights brand index using 85% football and 15% men's basketball weights.",
            "control_deion": control_deion,
            "colorado_deion_adjustment": float(round(colorado_deion_adjustment(), 6)) if control_deion else 0.0,
        },
    }


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
            "time_buckets": ["all", "Sat Early", "Sat Mid", "Sat Late", "Friday", "Black Friday", "Monday", "Sunday", "Weekday", "Other"],
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
                "time_buckets": ["all", "Primetime", "Sat Early", "Sat Mid", "Sat Late", "Friday", "Black Friday", "Monday", "Sunday", "Weekday", "Other"],
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
            "time_buckets": ["all", "Primetime", "Sat Early", "Sat Mid", "Sat Late", "Friday", "Black Friday", "Monday", "Sunday", "Weekday", "Other"],
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
