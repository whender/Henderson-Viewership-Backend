import math
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "cbb_viewership_model.joblib")
GAMES_PATH = os.path.join(BASE_DIR, "cbb_games.csv")

NETWORKS = [
    "ABC",
    "ABC + ESPN + ESPNU",
    "BTN",
    "CBS",
    "CNBC",
    "CW",
    "ESPN",
    "ESPN + ESPN2",
    "ESPN + ESPNU",
    "ESPN2",
    "ESPNU",
    "FOX",
    "FS1",
    "NBC",
    "TBS",
    "TBS + TNT + truTV",
    "TBS + truTV",
    "TNT",
    "TNT + truTV",
    "truTV",
    "USA",
]

TIME_SLOTS = ["Early", "Mid", "Prime", "Late", "Unknown"]
DAY_OPTIONS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "March",
    10: "October",
    11: "November",
    12: "December",
}
STAGES = [
    "Regular Season",
    "Conference Championship",
    "First Four",
    "First Round",
    "Second Round",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "National Championship",
    "Postseason",
]
TEAM_EFFECT_SHRINKAGE_DENOMINATOR = 25

CBB_TEAM_CONFERENCES = {
    "Akron": "MAC",
    "Alabama": "SEC",
    "Arizona": "Big 12",
    "Arizona State": "Big 12",
    "Arkansas": "SEC",
    "Arkansas State": "Sun Belt",
    "Auburn": "SEC",
    "BYU": "Big 12",
    "Baylor": "Big 12",
    "Boise State": "Mountain West",
    "Boston College": "ACC",
    "Bradley": "Missouri Valley",
    "Butler": "Big East",
    "California": "ACC",
    "Charlotte": "American",
    "Chattanooga": "SoCon",
    "Cincinnati": "Big 12",
    "Cleveland State": "Horizon",
    "Clemson": "ACC",
    "Colorado": "Big 12",
    "Colorado State": "Mountain West",
    "Creighton": "Big East",
    "Davidson": "Atlantic 10",
    "Dayton": "Atlantic 10",
    "DePaul": "Big East",
    "Drake": "Missouri Valley",
    "Duke": "ACC",
    "Duquesne": "Atlantic 10",
    "ETSU": "SoCon",
    "East Carolina": "American",
    "Florida": "SEC",
    "Florida Atlantic": "American",
    "Florida State": "ACC",
    "Fordham": "Atlantic 10",
    "Furman": "SoCon",
    "George Mason": "Atlantic 10",
    "George Washington": "Atlantic 10",
    "Georgetown": "Big East",
    "Georgia": "SEC",
    "Georgia Tech": "ACC",
    "Gonzaga": "WCC",
    "Grand Canyon": "Mountain West",
    "Hawaii": "Big West",
    "High Point": "Big South",
    "Houston": "Big 12",
    "Howard": "MEAC",
    "Illinois": "Big Ten",
    "Illinois State": "Missouri Valley",
    "Indiana": "Big Ten",
    "Indiana State": "Missouri Valley",
    "Iowa": "Big Ten",
    "Iowa State": "Big 12",
    "Jackson State": "SWAC",
    "Kansas": "Big 12",
    "Kansas State": "Big 12",
    "Kent State": "MAC",
    "Kentucky": "SEC",
    "LSU": "SEC",
    "La Salle": "Atlantic 10",
    "Liberty": "Conference USA",
    "Little Rock": "OVC",
    "Louisville": "ACC",
    "Loyola Chicago": "Atlantic 10",
    "Loyola Marymount": "WCC",
    "Longwood": "Big South",
    "Marquette": "Big East",
    "Marist": "MAAC",
    "Maryland": "Big Ten",
    "McNeese": "Southland",
    "Memphis": "American",
    "Miami (FL)": "ACC",
    "Miami (OH)": "MAC",
    "Michigan": "Big Ten",
    "Michigan State": "Big Ten",
    "Milwaukee": "Horizon",
    "Minnesota": "Big Ten",
    "Mississippi State": "SEC",
    "Missouri": "SEC",
    "Morehead State": "OVC",
    "Murray State": "Missouri Valley",
    "NC State": "ACC",
    "Nebraska": "Big Ten",
    "Nevada": "Mountain West",
    "New Mexico": "Mountain West",
    "Norfolk State": "MEAC",
    "North Carolina": "ACC",
    "North Texas": "American",
    "Northwestern": "Big Ten",
    "Notre Dame": "ACC",
    "Ohio": "MAC",
    "Ohio State": "Big Ten",
    "Oakland": "Horizon",
    "Oklahoma": "SEC",
    "Oklahoma State": "Big 12",
    "Ole Miss": "SEC",
    "Oregon": "Big Ten",
    "Oregon State": "WCC",
    "Penn State": "Big Ten",
    "Pitt": "ACC",
    "Princeton": "Ivy League",
    "Providence": "Big East",
    "Purdue": "Big Ten",
    "Purdue Fort Wayne": "Horizon",
    "Quinnipiac": "MAAC",
    "Rhode Island": "Atlantic 10",
    "Rice": "American",
    "Richmond": "Atlantic 10",
    "Robert Morris": "Horizon",
    "Rutgers": "Big Ten",
    "SIUE": "OVC",
    "SMU": "ACC",
    "Saint Joseph's": "Atlantic 10",
    "Saint Louis": "Atlantic 10",
    "Saint Mary's": "WCC",
    "Samford": "SoCon",
    "San Diego State": "Mountain West",
    "San Francisco": "WCC",
    "San Jose State": "Mountain West",
    "Santa Clara": "WCC",
    "Seton Hall": "Big East",
    "Siena": "MAAC",
    "South Carolina": "SEC",
    "South Florida": "American",
    "Southern": "SWAC",
    "Southeast Missouri": "OVC",
    "St. Bonaventure": "Atlantic 10",
    "St. John's": "Big East",
    "Stanford": "ACC",
    "Stephen F. Austin": "Southland",
    "Syracuse": "ACC",
    "TCU": "Big 12",
    "Temple": "American",
    "Tennessee": "SEC",
    "Tennessee State": "OVC",
    "Texas": "SEC",
    "Texas A&M": "SEC",
    "Texas Tech": "Big 12",
    "Texas Southern": "SWAC",
    "Troy": "Sun Belt",
    "Tulane": "American",
    "Tulsa": "American",
    "UAB": "American",
    "UC Irvine": "Big West",
    "UC San Diego": "Big West",
    "UCF": "Big 12",
    "UCLA": "Big Ten",
    "UConn": "Big East",
    "UMass": "MAC",
    "UC Santa Barbara": "Big West",
    "UNC Asheville": "Big South",
    "UNC Greensboro": "SoCon",
    "UNI": "Missouri Valley",
    "UNLV": "Mountain West",
    "UT Martin": "OVC",
    "USC": "Big Ten",
    "UTSA": "American",
    "Utah": "Big 12",
    "Utah State": "Mountain West",
    "VCU": "Atlantic 10",
    "Vanderbilt": "SEC",
    "Villanova": "Big East",
    "Virginia": "ACC",
    "Virginia Tech": "ACC",
    "Wake Forest": "ACC",
    "Washington": "Big Ten",
    "Washington State": "WCC",
    "West Virginia": "Big 12",
    "Wichita State": "American",
    "Western Illinois": "OVC",
    "Wisconsin": "Big Ten",
    "Wright State": "Horizon",
    "Xavier": "Big East",
    "Yale": "Ivy League",
}

TEAM_NAME_ALIASES = {
    "A&M CC": "Texas A&M-Corpus Christi",
    "A&M-Corpus Christi": "Texas A&M-Corpus Christi",
    "App State": "Appalachian State",
    "Cal": "California",
    "Cal Baptist": "California Baptist",
    "Call State Fullerton": "Cal State Fullerton",
    "Call State Northridge": "Cal State Northridge",
    "FAU": "Florida Atlantic",
    "FL)": "Miami (FL)",
    "Hawai'i": "Hawaii",
    "Jaksonville State": "Jacksonville State",
    "Miami FL": "Miami (FL)",
    "Mississppi State": "Mississippi State",
    "NC Central": "North Carolina Central",
    "NM State": "New Mexico State",
    "Qunnipiac": "Quinnipiac",
    "SFA": "Stephen F. Austin",
    "Saint Louis": "Saint Louis",
    "Saint Mary's": "Saint Mary's",
    "South Florida": "South Florida",
    "Stephen F Austin": "Stephen F. Austin",
    "St Louis": "Saint Louis",
    "St. John's": "St. John's",
    "St. John’s": "St. John's",
    "St.John's": "St. John's",
    "St. Louis": "Saint Louis",
    "St. Mary's": "Saint Mary's",
    "Syraucse": "Syracuse",
    "UAlbany": "Albany",
    "Ualbany": "Albany",
    "Uconn": "UConn",
    "UNCG": "UNC Greensboro",
    "USF": "South Florida",
    "Virgnia": "Virginia",
    "WKU": "Western Kentucky",
}


class CBBGameInput(BaseModel):
    team1: str
    team2: str
    network: str
    time_slot: str = "Prime"
    day_of_week: str = "Saturday"
    month: int = 3
    is_tournament: int = 0
    stage: str = "Regular Season"
    team1_rank: int = 0
    team2_rank: int = 0
    team1_seed: int = 0
    team2_seed: int = 0


def clean_nan(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {key: clean_nan(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(value) for value in obj]
    return obj


def format_viewers(value):
    if value is None:
        return "N/A"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def canonical_team_name(team):
    if team is None or pd.isna(team):
        return team
    name = str(team).strip()
    name = name.replace("’", "'")
    name = " ".join(name.split())
    return TEAM_NAME_ALIASES.get(name, name)


def canonical_team_key(team):
    return canonical_team_name(team)


@lru_cache(maxsize=1)
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Missing cbb_viewership_model.joblib")
    return joblib.load(MODEL_PATH)


def canonical_team_feature_map():
    artifact = load_artifact()
    rows_by_team = {
        canonical_team_key(row["team"]): row
        for row in artifact.get("team_effects", [])
    }
    feature_map = {}
    feature_appearances = {}
    for column in artifact.get("team_features", []):
        raw_team = column.replace("team_fe__", "", 1)
        canonical = canonical_team_key(raw_team)
        next_appearances = rows_by_team.get(canonical, {}).get("appearances", -1)
        if canonical not in feature_map or next_appearances > feature_appearances[canonical]:
            feature_map[canonical] = column
            feature_appearances[canonical] = next_appearances
    return feature_map


@lru_cache(maxsize=1)
def load_games():
    if not os.path.exists(GAMES_PATH):
        raise FileNotFoundError("Missing cbb_games.csv")
    df = pd.read_csv(GAMES_PATH)
    df["viewers"] = pd.to_numeric(df["viewers"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team1_raw"] = df["team1"]
    df["team2_raw"] = df["team2"]
    df["team1"] = df["team1"].apply(canonical_team_name)
    df["team2"] = df["team2"].apply(canonical_team_name)
    if "month_name" not in df.columns:
        df["month_name"] = pd.to_numeric(df["month"], errors="coerce").map(MONTH_NAMES).fillna("Other")
    df["season_label"] = df["season"].astype(str)
    return df


def all_teams():
    artifact = load_artifact()
    df = load_games()
    teams = sorted(
        set(canonical_team_name(team) for team in artifact.get("frequent_teams", []))
        | set(pd.concat([df["team1"], df["team2"]]).dropna().astype(str).str.strip())
    )
    return teams


def team_effect_column(team):
    return f"team_fe__{team}"


def normalize_rank(value):
    try:
        value = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return value if 1 <= value <= 25 else 0


def normalize_seed(value):
    try:
        value = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return value if 1 <= value <= 16 else 0


def add_rank_seed_features(row):
    team1_rank = normalize_rank(row.get("team1_rank"))
    team2_rank = normalize_rank(row.get("team2_rank"))
    team1_seed = normalize_seed(row.get("team1_seed"))
    team2_seed = normalize_seed(row.get("team2_seed"))

    seeds = [seed for seed in [team1_seed, team2_seed] if seed > 0]
    best_seed = min(seeds) if seeds else 0

    row.update({
        "team1_rank": team1_rank,
        "team2_rank": team2_rank,
        "team1_top_10": int(0 < team1_rank <= 10),
        "team2_top_10": int(0 < team2_rank <= 10),
        "team1_rank_11_25": int(10 < team1_rank <= 25),
        "team2_rank_11_25": int(10 < team2_rank <= 25),
        "top_10_teams": int(0 < team1_rank <= 10) + int(0 < team2_rank <= 10),
        "rank_11_25_teams": int(10 < team1_rank <= 25) + int(10 < team2_rank <= 25),
        "team1_seed": team1_seed,
        "team2_seed": team2_seed,
        "team1_seeded": int(team1_seed > 0),
        "team2_seeded": int(team2_seed > 0),
        "seeded_teams": len(seeds),
        "best_seed": best_seed,
        "seed_score": 17 - best_seed if best_seed else 0,
        "seed_diff": abs(team1_seed - team2_seed) if team1_seed and team2_seed else 0,
    })


def build_prediction_frame(payload):
    artifact = load_artifact()
    row = dict(payload)
    row["team1"] = canonical_team_name(row.get("team1"))
    row["team2"] = canonical_team_name(row.get("team2"))
    row["month_name"] = MONTH_NAMES.get(int(row.get("month") or 0), "Other")
    row["is_weekend"] = int(row.get("day_of_week") in ["Saturday", "Sunday"])
    add_rank_seed_features(row)
    feature_map = canonical_team_feature_map()

    for column in artifact.get("team_features", []):
        row[column] = 0

    for team in [row.get("team1"), row.get("team2")]:
        team = canonical_team_key(team)
        column = feature_map.get(team)
        if column in row:
            row[column] = 1

    for feature in artifact.get("features", []):
        row.setdefault(feature, 0)

    return pd.DataFrame([row])


def predict_cbb_viewership(game: CBBGameInput):
    artifact = load_artifact()
    frame = build_prediction_frame(game.dict())
    prediction = float(np.exp(artifact["model"].predict(frame[artifact["features"]])[0]))
    stage = game.stage or "Regular Season"
    warning = None
    if artifact.get("training_scope") == "regular_season" and stage != "Regular Season":
        warning = "Basketball model is trained on regular-season games only; tournament predictions should be treated as directional."
    return {
        "prediction_raw": round(prediction),
        "prediction_formatted": format_viewers(prediction),
        "model_mae_viewers": round(artifact["mae_viewers"]),
        "model_mae_pct": round(float(artifact.get("mae_pct", 0)), 1),
        "baseline_mae_viewers": round(artifact.get("baseline_mae_viewers", 0)),
        "baseline_mae_pct": round(float(artifact.get("baseline_mae_pct", 0)), 1),
        "training_scope": artifact.get("training_scope", "all_games"),
        "warning": warning,
    }


def predict_expected_without_team_effects(df):
    artifact = load_artifact()
    frame = df.copy()
    zero_team_effects = {
        column: pd.Series(0, index=frame.index)
        for column in artifact.get("team_features", [])
    }
    if zero_team_effects:
        frame = pd.concat([frame, pd.DataFrame(zero_team_effects, index=frame.index)], axis=1)
    predictions = np.exp(artifact["model"].predict(frame[artifact["features"]]))
    return pd.Series(predictions, index=df.index)


def predict_expected_with_focal_team_removed(df, focal_team):
    artifact = load_artifact()
    feature_map = canonical_team_feature_map()
    frame = df.copy()
    team_effects = {
        column: pd.Series(0, index=frame.index)
        for column in artifact.get("team_features", [])
    }
    if team_effects:
        frame = pd.concat([frame, pd.DataFrame(team_effects, index=frame.index)], axis=1)

    focal_team = canonical_team_key(focal_team)
    for idx, row in frame.iterrows():
        for team in [row.get("team1"), row.get("team2")]:
            team_key = canonical_team_key(team)
            if team_key == focal_team:
                continue
            column = feature_map.get(team_key)
            if column in frame.columns:
                frame.at[idx, column] = 1

    predictions = np.exp(artifact["model"].predict(frame[artifact["features"]]))
    return pd.Series(predictions, index=df.index)


@lru_cache(maxsize=1)
def enriched_games():
    df = load_games().copy()
    df["matchup_label"] = df["team1"].astype(str) + " vs " + df["team2"].astype(str)
    regular_mask = df["stage"].astype(str).eq("Regular Season")
    df["expected_neutral_viewers"] = np.nan
    if regular_mask.any():
        df.loc[regular_mask, "expected_neutral_viewers"] = predict_expected_without_team_effects(df.loc[regular_mask])
    df["actual_minus_expected"] = df["viewers"] - df["expected_neutral_viewers"]
    df["actual_vs_expected_pct"] = np.where(
        df["expected_neutral_viewers"] > 0,
        ((df["viewers"] / df["expected_neutral_viewers"]) - 1) * 100,
        np.nan,
    )
    return df


def add_focal_expected_fields(team_df, team):
    out = team_df.copy()
    regular_mask = out["stage"].astype(str).eq("Regular Season")
    out["expected_team_viewers"] = np.nan
    if regular_mask.any():
        out.loc[regular_mask, "expected_team_viewers"] = predict_expected_with_focal_team_removed(
            out.loc[regular_mask],
            team,
        )
    out["team_actual_minus_expected"] = out["viewers"] - out["expected_team_viewers"]
    out["team_actual_vs_expected_pct"] = np.where(
        out["expected_team_viewers"] > 0,
        ((out["viewers"] / out["expected_team_viewers"]) - 1) * 100,
        np.nan,
    )
    return out


def cbb_metadata():
    artifact = load_artifact()
    df = load_games()
    raw_team_effects = len(artifact.get("team_features", []))
    display_team_effects = len({
        canonical_team_name(row["team"])
        for row in artifact.get("team_effects", [])
    })
    return clean_nan({
        "trained_rows": artifact.get("trained_rows"),
        "model_mae_viewers": round(artifact.get("mae_viewers", 0)),
        "model_mae_pct": round(float(artifact.get("mae_pct", 0)), 1),
        "baseline_mae_viewers": round(artifact.get("baseline_mae_viewers", 0)),
        "baseline_mae_pct": round(float(artifact.get("baseline_mae_pct", 0)), 1),
        "training_scope": artifact.get("training_scope", "all_games"),
        "team_fixed_effects": raw_team_effects,
        "raw_team_fixed_effects": raw_team_effects,
        "display_team_fixed_effects": display_team_effects,
        "seasons": sorted(df["season_label"].dropna().unique().tolist()),
        "games": int(len(df)),
    })


def cbb_team_options():
    return {"teams": [{"label": team, "value": team} for team in all_teams()]}


def cbb_filter_options():
    df = load_games()
    return {
        "networks": ["all"] + sorted(df["network"].dropna().unique().tolist()),
        "time_slots": ["all"] + sorted(df["time_slot"].dropna().unique().tolist()),
        "stages": ["all"] + sorted(df["stage"].dropna().unique().tolist()),
        "seasons": ["all"] + sorted(df["season_label"].dropna().unique().tolist()),
        "opponents": ["all"] + all_teams(),
        "conferences": ["all"] + sorted(set(CBB_TEAM_CONFERENCES.values()) | {"Unknown"}),
    }


def cbb_team_effects(conference="all"):
    artifact = load_artifact()
    grouped = {}
    for row in artifact.get("team_effects", []):
        team = canonical_team_name(row["team"])
        appearances = int(row["appearances"])
        bucket = grouped.setdefault(team, {"team": team, "weighted_coef": 0.0, "appearances": 0})
        bucket["weighted_coef"] += float(row["coef_log_viewers"]) * appearances
        bucket["appearances"] += appearances

    rows = []
    for row in grouped.values():
        appearances = int(row["appearances"])
        coefficient = row["weighted_coef"] / appearances if appearances else 0
        shrunk_coefficient = coefficient
        if appearances <= 25:
            shrunk_coefficient = coefficient * (appearances / (appearances + TEAM_EFFECT_SHRINKAGE_DENOMINATOR))
        team_conference = CBB_TEAM_CONFERENCES.get(row["team"], "Unknown")
        rows.append({
            "team": row["team"],
            "conference": team_conference,
            "coefficient": round(shrunk_coefficient, 6),
            "raw_coefficient": round(coefficient, 6),
            "viewer_multiplier": round(float(np.exp(shrunk_coefficient)), 3),
            "raw_viewer_multiplier": round(float(np.exp(coefficient)), 3),
            "appearances": appearances,
            "shrinkage_factor": round(float(appearances / (appearances + TEAM_EFFECT_SHRINKAGE_DENOMINATOR)), 3) if appearances <= 25 else 1.0,
        })
    if conference != "all":
        rows = [row for row in rows if row["conference"] == conference]
    rows = sorted(rows, key=lambda row: (-row["viewer_multiplier"], -row["appearances"], row["team"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return {"rows": rows, "metadata": cbb_metadata(), "available_filters": cbb_filter_options()}


def filtered_games(network="all", time_slot="all", stage="all", season="all", include_tournament=True):
    df = enriched_games()
    if network != "all":
        df = df[df["network"] == network]
    if time_slot != "all":
        df = df[df["time_slot"] == time_slot]
    if stage != "all":
        df = df[df["stage"] == stage]
    if season != "all":
        df = df[df["season_label"] == season]
    if not include_tournament:
        df = df[df["is_tournament"] != 1]
    return df


def cbb_viewership_rankings(
    network="all",
    time_slot="all",
    stage="all",
    season="all",
    opponent="all",
    min_games=1,
    include_tournament=True,
):
    df = filtered_games(network, time_slot, stage, season, include_tournament)
    records = []

    for _, row in df.iterrows():
        for team_col, opponent_col in [("team1", "team2"), ("team2", "team1")]:
            team = row.get(team_col)
            row_opponent = row.get(opponent_col)
            if pd.isna(team):
                continue
            if opponent != "all" and row_opponent != opponent:
                continue
            records.append({"team": str(team), "viewers": float(row["viewers"])})

    rankings_df = pd.DataFrame(records)
    if rankings_df.empty:
        return clean_nan({"rows": [], "available_filters": cbb_filter_options()})

    rows = []
    for team, viewers in rankings_df.groupby("team")["viewers"]:
        if len(viewers) < max(1, int(min_games)):
            continue
        rows.append({
            "team": team,
            "games": int(len(viewers)),
            "average_viewers": float(round(viewers.mean(), 1)),
            "median_viewers": float(round(viewers.median(), 1)),
            "peak_viewers": float(round(viewers.max(), 1)),
        })

    rows = sorted(rows, key=lambda row: (-row["average_viewers"], -row["median_viewers"], row["team"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return clean_nan({"rows": rows, "available_filters": cbb_filter_options()})


def short_date_label(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return value
    return f"{parsed.month}/{parsed.day}/{str(parsed.year)[-2:]}"


def cbb_game_viewership_rankings(
    network="all",
    time_slot="all",
    stage="all",
    season="all",
    conference="all",
    team="all",
    rank_bucket="all",
    include_tournament=True,
):
    df = filtered_games(network, time_slot, stage, season, include_tournament)
    if conference != "all":
        df = df[
            df["team1"].map(lambda value: CBB_TEAM_CONFERENCES.get(value, "Unknown") == conference)
            | df["team2"].map(lambda value: CBB_TEAM_CONFERENCES.get(value, "Unknown") == conference)
        ]
    if team != "all":
        df = df[(df["team1"] == team) | (df["team2"] == team)]

    if rank_bucket == "Any Ranked":
        df = df[(df["team1_rank"].fillna(0).astype(int) > 0) | (df["team2_rank"].fillna(0).astype(int) > 0)]
    elif rank_bucket == "Both Ranked":
        df = df[(df["team1_rank"].fillna(0).astype(int) > 0) & (df["team2_rank"].fillna(0).astype(int) > 0)]
    elif rank_bucket == "Any Top 10":
        df = df[(df["team1_top_10"].fillna(0).astype(int) == 1) | (df["team2_top_10"].fillna(0).astype(int) == 1)]
    elif rank_bucket == "Unranked":
        df = df[(df["team1_rank"].fillna(0).astype(int) <= 0) & (df["team2_rank"].fillna(0).astype(int) <= 0)]

    rows = []
    sorted_df = df.sort_values(["viewers", "date"], ascending=[False, False])
    for idx, (_, row) in enumerate(sorted_df.iterrows(), start=1):
        team1_rank = int(row["team1_rank"]) if pd.notna(row.get("team1_rank")) and int(row.get("team1_rank") or 0) > 0 else 0
        team2_rank = int(row["team2_rank"]) if pd.notna(row.get("team2_rank")) and int(row.get("team2_rank") or 0) > 0 else 0
        rows.append({
            "rank": idx,
            "date": short_date_label(row.get("date")),
            "season": row.get("season_label"),
            "matchup": row.get("matchup_label") or row.get("matchup"),
            "team1": row.get("team1"),
            "team2": row.get("team2"),
            "team1_conference": CBB_TEAM_CONFERENCES.get(row.get("team1"), "Unknown"),
            "team2_conference": CBB_TEAM_CONFERENCES.get(row.get("team2"), "Unknown"),
            "team1_rank": team1_rank,
            "team2_rank": team2_rank,
            "network": row.get("network"),
            "time_slot": row.get("time_slot"),
            "stage": row.get("stage"),
            "viewers": float(row.get("viewers", 0)),
            "expected_viewers": float(row["expected_neutral_viewers"]) if pd.notna(row.get("expected_neutral_viewers")) else None,
            "actual_minus_expected": float(row["actual_minus_expected"]) if pd.notna(row.get("actual_minus_expected")) else None,
            "actual_vs_expected_pct": float(row["actual_vs_expected_pct"]) if pd.notna(row.get("actual_vs_expected_pct")) else None,
        })

    filters = cbb_filter_options()
    filters["teams"] = filters.get("opponents", ["all"])
    filters["rank_buckets"] = ["all", "Any Ranked", "Both Ranked", "Any Top 10", "Unranked"]
    return clean_nan({
        "rows": rows,
        "available_filters": filters,
        "filters": {
            "network": network,
            "time_slot": time_slot,
            "stage": stage,
            "season": season,
            "conference": conference,
            "team": team,
            "rank_bucket": rank_bucket,
            "include_tournament": include_tournament,
        },
    })


def expected_summary_fields(df, expected_column="expected_neutral_viewers", delta_column="actual_minus_expected"):
    expected_df = df.dropna(subset=[expected_column]).copy()
    if expected_df.empty:
        return {
            "expected_average_viewers": None,
            "average_minus_expected": None,
            "overperformance_pct": None,
        }
    expected_sum = expected_df[expected_column].sum()
    actual_sum = expected_df["viewers"].sum()
    return {
        "expected_average_viewers": float(round(expected_df[expected_column].mean(), 1)),
        "average_minus_expected": float(round(expected_df[delta_column].mean(), 1)),
        "overperformance_pct": float(round(((actual_sum / expected_sum) - 1) * 100, 1)) if expected_sum > 0 else None,
    }


def cbb_team_profile(team):
    team = canonical_team_name(team)
    df = enriched_games()
    team_df = df[(df["team1"] == team) | (df["team2"] == team)].copy()
    if team_df.empty:
        return {"error": f"No basketball profile data found for {team}."}

    team_df["opponent"] = np.where(team_df["team1"] == team, team_df["team2"], team_df["team1"])
    team_df = add_focal_expected_fields(team_df, team)
    team_df = team_df.sort_values(["viewers", "date"], ascending=[False, False])

    yearly_rows = []
    for season, season_df in team_df.groupby("season_label"):
        expected_fields = expected_summary_fields(
            season_df,
            expected_column="expected_team_viewers",
            delta_column="team_actual_minus_expected",
        )
        yearly_rows.append({
            "season": season,
            "games": int(len(season_df)),
            "average_viewers": float(round(season_df["viewers"].mean(), 1)),
            "median_viewers": float(round(season_df["viewers"].median(), 1)),
            **expected_fields,
            "peak_viewers": float(round(season_df["viewers"].max(), 1)),
        })

    top_games = []
    for idx, (_, row) in enumerate(team_df.head(10).iterrows(), start=1):
        top_games.append({
            "rank": idx,
            "season": row["season_label"],
            "date": row["date"].date().isoformat() if pd.notna(row["date"]) else None,
            "matchup": row["matchup_label"],
            "opponent": row["opponent"],
            "network": row["network"],
            "time_slot": row["time_slot"],
            "stage": row["stage"],
            "viewers": float(round(row["viewers"], 1)),
            "expected_viewers": float(round(row["expected_team_viewers"], 1)),
            "actual_minus_expected": float(round(row["team_actual_minus_expected"], 1)),
            "actual_vs_expected_pct": float(round(row["team_actual_vs_expected_pct"], 1)),
        })

    games = []
    for _, row in team_df.iterrows():
        games.append({
            "season": row["season_label"],
            "date": row["date"].date().isoformat() if pd.notna(row["date"]) else None,
            "matchup": row["matchup_label"],
            "opponent": row["opponent"],
            "network": row["network"],
            "time_slot": row["time_slot"],
            "stage": row["stage"],
            "is_tournament": int(row["is_tournament"]) if pd.notna(row.get("is_tournament")) else 0,
            "viewers": float(round(row["viewers"], 1)),
            "expected_viewers": float(round(row["expected_team_viewers"], 1)),
            "actual_minus_expected": float(round(row["team_actual_minus_expected"], 1)),
            "actual_vs_expected_pct": float(round(row["team_actual_vs_expected_pct"], 1)),
        })

    summary = {
        "games": int(len(team_df)),
        "average_viewers": float(round(team_df["viewers"].mean(), 1)),
        "median_viewers": float(round(team_df["viewers"].median(), 1)),
        **expected_summary_fields(
            team_df,
            expected_column="expected_team_viewers",
            delta_column="team_actual_minus_expected",
        ),
        "peak_viewers": float(round(team_df["viewers"].max(), 1)),
        "peak_matchup": team_df.iloc[0]["matchup_label"],
    }

    return clean_nan({
        "team": team,
        "summary": summary,
        "seasons": sorted(yearly_rows, key=lambda row: row["season"], reverse=True),
        "top_games": top_games,
        "games": games,
    })
