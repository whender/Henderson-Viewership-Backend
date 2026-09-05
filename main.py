from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
import os
import json
import math
import random
import re
import importlib.util
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from functools import lru_cache

from weekly_predictions_fs import (
    generate_pregame_prediction,
    generate_postgame_prediction,
    calc_error,
    build_features,
    parse_viewership,
    pregame_prediction_warnings,
)
from pregame_context_features import build_pregame_context_features
from pregame_ensemble import predict_pregame_points_000s

from firestore_client import db
from cbb_viewership import (
    CBBGameInput,
    cbb_filter_options,
    cbb_game_viewership_rankings,
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
    FEUD_END,
    FEUD_START,
    format_viewers,
    is_black_friday_slot,
    is_power_football_team,
    is_week_one_power_game,
    is_week_zero_power_game,
    rank_to_coefs,
    resolve_competing_games_score,
    resolve_game_record_features,
    team_conferences,
)

class GameInput(BaseModel):
    team1: str
    team2: str
    rank1: int
    rank2: int
    network: str
    time_slot: str
    date: str | None = None
    season_week: int | None = None
    team1_games_before: float | None = None
    team2_games_before: float | None = None
    team1_win_pct_to_date: float | None = Field(default=None, ge=0.0, le=1.0)
    team2_win_pct_to_date: float | None = Field(default=None, ge=0.0, le=1.0)
    cfp_rank1: int | None = None
    cfp_rank2: int | None = None
    ranking_source: str = "auto"
    conf_champ: bool = False
    comp_tier1: int = 0
    competing_games_score: float | None = Field(default=None, ge=0.0)


class ArticleInput(BaseModel):
    slug: str
    title: str
    date: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    published: bool = False
    body: str = ""


class RealignmentSimulationInput(BaseModel):
    conference: str = "Big 10"
    expansion_teams: list[str] = Field(default_factory=lambda: ["Utah"])
    protected_opponents: list[str] = Field(default_factory=list)
    protected_opponents_by_team: dict[str, list[str]] = Field(default_factory=dict)
    games_per_team: int = 9
    network_policy: str = "big_ten_tv_mix"
    ranking_policy: str = "espn_2026_preseason"
    custom_rankings: dict[str, int] = Field(default_factory=dict)


class SuperleagueSimulationInput(BaseModel):
    teams: list[str] = Field(default_factory=list)
    games_per_team: int = 9
    ranking_policy: str = "espn_2026_preseason"
    custom_rankings: dict[str, int] = Field(default_factory=dict)


class LeagueRealignmentSimulationInput(BaseModel):
    memberships: dict[str, str] = Field(default_factory=dict)
    protected_matchups_by_team: dict[str, list[str]] = Field(default_factory=dict)
    games_per_team: int = 9
    ranking_policy: str = "espn_2026_preseason"
    custom_rankings: dict[str, int] = Field(default_factory=dict)

@app.get("/")
def root():
    return {"status": "running", "message": "Henderson Viewership Model API"}


ARTICLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "articles"))


def _article_slug(value):
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value or "").strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:90] or "untitled"


def _article_path(slug):
    safe_slug = _article_slug(slug)
    return os.path.join(ARTICLES_DIR, f"{safe_slug}.md")


def _parse_article_file(path):
    with open(path, "r", encoding="utf-8") as article_file:
        text = article_file.read()

    metadata = {}
    body = text
    if text.startswith("---\n"):
        _, frontmatter, body = text.split("---\n", 2)
        for line in frontmatter.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "tags":
                metadata[key] = [
                    item.strip()
                    for item in value.strip("[]").replace('"', "").split(",")
                    if item.strip()
                ]
            elif key == "published":
                metadata[key] = value.lower() == "true"
            else:
                metadata[key] = value.strip('"')

    slug = os.path.splitext(os.path.basename(path))[0]
    return {
        "slug": slug,
        "title": metadata.get("title") or slug.replace("-", " ").title(),
        "date": metadata.get("date") or "",
        "description": metadata.get("description") or "",
        "tags": metadata.get("tags") or [],
        "published": bool(metadata.get("published", False)),
        "body": body.lstrip("\n"),
    }


def _serialize_article(article):
    slug = _article_slug(article.slug)
    tags = [str(tag).strip() for tag in (article.tags or []) if str(tag).strip()]
    frontmatter = [
        "---",
        f'title: "{article.title.strip() or slug.replace("-", " ").title()}"',
        f'date: "{article.date.strip()}"',
        f'description: "{article.description.strip()}"',
        f"tags: [{', '.join(tags)}]",
        f"published: {'true' if article.published else 'false'}",
        "---",
        "",
    ]
    return "\n".join(frontmatter) + (article.body or "").rstrip() + "\n"


def _list_articles(include_drafts=False):
    if not os.path.isdir(ARTICLES_DIR):
        return []
    articles = [
        _parse_article_file(os.path.join(ARTICLES_DIR, filename))
        for filename in os.listdir(ARTICLES_DIR)
        if filename.endswith(".md")
    ]
    articles = [
        article for article in articles
        if include_drafts or article.get("published")
    ]
    return sorted(
        articles,
        key=lambda article: (article.get("date") or "", article.get("title") or ""),
        reverse=True,
    )


def _article_summary(article):
    return {
        key: article.get(key)
        for key in ["slug", "title", "date", "description", "tags", "published"]
    }


def _require_local_admin(request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Article admin is local-only.")


@app.get("/articles")
def list_published_articles():
    return {"articles": [_article_summary(article) for article in _list_articles(include_drafts=False)]}


@app.get("/articles/{slug}")
def get_published_article(slug: str):
    path = _article_path(slug)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Article not found.")
    article = _parse_article_file(path)
    if not article.get("published"):
        raise HTTPException(status_code=404, detail="Article not found.")
    return article


@app.get("/admin/articles")
def list_admin_articles(request: Request):
    _require_local_admin(request)
    return {"articles": [_article_summary(article) for article in _list_articles(include_drafts=True)]}


@app.get("/admin/articles/{slug}")
def get_admin_article(slug: str, request: Request):
    _require_local_admin(request)
    path = _article_path(slug)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Article not found.")
    return _parse_article_file(path)


@app.post("/admin/articles")
def save_admin_article(article: ArticleInput, request: Request):
    _require_local_admin(request)
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    slug = _article_slug(article.slug or article.title)
    normalized_article = article.copy(update={"slug": slug})
    with open(_article_path(slug), "w", encoding="utf-8") as article_file:
        article_file.write(_serialize_article(normalized_article))
    return _parse_article_file(_article_path(slug))


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


@app.get("/cbb/game-viewership-rankings")
def cbb_game_viewership_rankings_endpoint(
    network: str = "all",
    time_slot: str = "all",
    stage: str = "all",
    season: str = "all",
    conference: str = "all",
    team: str = "all",
    rank_bucket: str = "all",
    include_tournament: bool = True,
):
    return cbb_game_viewership_rankings(
        network=network,
        time_slot=time_slot,
        stage=stage,
        season=season,
        conference=conference,
        team=team,
        rank_bucket=rank_bucket,
        include_tournament=include_tournament,
    )


@app.get("/cbb/team-profile")
def cbb_profile(team: str):
    return cbb_team_profile(team)

# ======================================================
# 🏆 BRAND RANKINGS (POSTGAME MODEL)
# ======================================================

numeric_features_post = [
    "Competing_Games_Score","FOX","ESPN","ESPN2","ESPNU","FS1","FS2","NBC","CBS",
    "ABC","BTN","CW","NFLN","ESPNNEWS",
    "Sun","Monday","Weekday","Friday Power","Friday Non-Power","Black Friday","Week 0 Power","Week 1 Power",
    "Sat Early","Sat Mid","Sat Late","Top 10 Rankings","25-11 Rankings",
    "AnyDayKickoff9pmOrLater","EarlyRecordNeutralized","MinGamesBefore",
    "LateBothRanked","CFP_RankStrengthSum","CFP_RankDifference",
    "Avg_WinPct_ToDate","WinPct_Diff_ToDate",
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
    "OhioSt_PennSt","Alabama_LSU","Vanderbilt_Tennessee","Georgia_GeorgiaTech",
    "Alabama_Georgia","WestVirginia_Pittsburgh"
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

df_all = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "viewership_cleaned.csv"),
    low_memory=False,
)

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
    score = pd.to_numeric(row.get("Competing_Games_Score"), errors="coerce")
    if pd.notna(score):
        if score <= 0:
            return "None"
        if score < 3:
            return "Light"
        if score < 7:
            return "Moderate"
        return "Heavy"

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
        rows.append(attach_football_brand_uncertainty({
            "rank": i,
            "team": team,
            "conference": team_conferences.get(team, "Independent"),
            "viewership_lift_pct": float(round(lift, 1)),
            "games_used": int(counts[team])
        }))
    return attach_football_rank_ranges(rows)


def _confidence_label(score):
    if score is None:
        return "Insufficient data"
    if score >= 80:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def _sample_confidence(games, full_sample=60):
    games = max(float(games or 0), 0.0)
    if games <= 0:
        return 0.0
    return min(math.sqrt(games / full_sample), 1.0)


def _error_confidence(median_ape):
    if median_ape is None or pd.isna(median_ape):
        return None
    median_ape = max(float(median_ape), 0.0)
    if median_ape <= 0.15:
        return 1.0
    if median_ape >= 0.55:
        return 0.0
    return (0.55 - median_ape) / 0.40


def _brand_confidence_from_components(games, median_ape=None, sample_full=60):
    sample_score = _sample_confidence(games, full_sample=sample_full)
    error_score = _error_confidence(median_ape)
    if error_score is None:
        confidence = sample_score * 100
    else:
        confidence = sample_score * error_score * 100
    confidence = max(0.0, min(100.0, confidence))
    return {
        "confidence_score": float(round(confidence, 1)),
        "confidence_label": _confidence_label(confidence),
        "confidence_games_component": float(round(sample_score * 100, 1)),
        "confidence_error_component": float(round(error_score * 100, 1)) if error_score is not None else None,
        "median_abs_pct_error": float(round(median_ape * 100, 1)) if median_ape is not None and not pd.isna(median_ape) else None,
    }


def _brand_lift_uncertainty_range(lift_pct, games, median_ape=None, sample_full=60):
    if lift_pct is None or median_ape is None or pd.isna(median_ape):
        return {
            "brand_range_low_pct": None,
            "brand_range_high_pct": None,
            "brand_range_minus_pct": None,
            "brand_range_plus_pct": None,
            "median_abs_pct_error": None,
        }

    games = max(float(games or 0), 1.0)
    lift_pct = float(lift_pct)
    observed_coefficient = math.log(max((lift_pct / 100.0) + 1.0, 0.01))
    sample_multiplier = math.sqrt(sample_full / games)
    sample_multiplier = max(0.75, min(sample_multiplier, 2.25))
    observation_sigma = math.log1p(max(float(median_ape), 0.0)) * sample_multiplier
    observation_sigma = max(0.08, min(observation_sigma, 1.25))

    # Sparse/noisy estimates should regress toward average, not imply unlimited upside.
    prior_mean = 0.0
    prior_sigma = 0.50
    observation_precision = 1.0 / (observation_sigma ** 2)
    prior_precision = 1.0 / (prior_sigma ** 2)
    posterior_variance = 1.0 / (observation_precision + prior_precision)
    posterior_mean = posterior_variance * (
        observed_coefficient * observation_precision
        + prior_mean * prior_precision
    )
    posterior_sigma = math.sqrt(posterior_variance)

    low_lift = min((math.exp(posterior_mean - posterior_sigma) - 1.0) * 100.0, lift_pct)
    high_lift = max((math.exp(posterior_mean + posterior_sigma) - 1.0) * 100.0, lift_pct)

    return {
        "brand_range_low_pct": float(round(low_lift, 1)),
        "brand_range_high_pct": float(round(high_lift, 1)),
        "brand_range_minus_pct": float(round(max(lift_pct - low_lift, 0.0), 1)),
        "brand_range_plus_pct": float(round(max(high_lift - lift_pct, 0.0), 1)),
        "median_abs_pct_error": float(round(float(median_ape) * 100, 1)),
    }


def _football_prediction_stability_lookup():
    rows = []
    for _, row in df_all.iterrows():
        actual = pd.to_numeric(row.get("Persons 2+"), errors="coerce")
        if pd.isna(actual) or actual <= 0:
            continue
        for team_column, expected_column in [
            ("Team 1", "expected_viewers_team1"),
            ("Team 2", "expected_viewers_team2"),
        ]:
            team = row.get(team_column)
            expected = pd.to_numeric(row.get(expected_column), errors="coerce")
            if not team or pd.isna(expected) or expected <= 0:
                continue
            rows.append({
                "team": str(team),
                "ape": abs(float(actual) - float(expected)) / float(actual),
            })
    if not rows:
        return {}
    stability = pd.DataFrame(rows).groupby("team")["ape"].median()
    return stability.to_dict()


def _football_forward_chain_holdout_stability_lookup(exclude_colorado_deion=False):
    holdout_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "RatingsAndRegression",
        "holdout_viewership_tests.py",
    ))
    if not os.path.exists(holdout_path):
        return {}
    try:
        spec = importlib.util.spec_from_file_location("holdout_viewership_tests_confidence", holdout_path)
        holdout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(holdout)
        df = holdout.load_feature_frame().sort_values("ParsedDate").reset_index(drop=True)
        rows = []
        for year in sorted(df["Year"].dropna().unique()):
            train_index = df.index[df["Year"] < year]
            test_index = df.index[df["Year"] == year]
            if len(train_index) < 500:
                continue
            x_train, x_test = holdout.build_matrix(df, train_index, test_index)
            y_train = np.log(df.loc[train_index, "Persons 2+"].astype(float) + 1)
            actual = df.loc[test_index, "Persons 2+"].astype(float)
            model = sm.OLS(y_train.astype(float), x_train).fit()
            smearing_factor = np.mean(np.exp(model.resid))
            predicted = np.maximum(np.exp(model.predict(x_test)) * smearing_factor - 1, 0)
            scored = df.loc[test_index, ["Team 1", "Team 2", "Year"]].copy()
            scored["actual"] = actual.values
            scored["predicted"] = predicted.values if hasattr(predicted, "values") else predicted
            scored["ape"] = (scored["predicted"] - scored["actual"]).abs() / scored["actual"]
            for team_column in ["Team 1", "Team 2"]:
                team_rows = scored[[team_column, "ape"]].rename(columns={team_column: "team"})
                if exclude_colorado_deion:
                    team_rows = team_rows[
                        ~(
                            (scored[team_column] == "Colorado")
                            & scored["Year"].isin([2023, 2024, 2025])
                        )
                    ]
                rows.extend(team_rows.to_dict("records"))
        if not rows:
            return {}
        stability = pd.DataFrame(rows).dropna(subset=["ape"]).groupby("team")["ape"].median()
        return stability.to_dict()
    except Exception:
        return {}


def attach_football_brand_uncertainty(row, control_deion=False):
    team = str(row.get("team") or "")
    display_team_fn = globals().get("display_brand_team_name")
    if callable(display_team_fn):
        team = display_team_fn(team)
    stability_lookup = globals().get("football_prediction_stability_display", football_prediction_stability)
    if control_deion and team == "Colorado":
        stability_lookup = globals().get("football_prediction_stability_deion_controlled_display", stability_lookup)
    median_ape = stability_lookup.get(team) or football_prediction_stability.get(str(row.get("team") or ""))
    uncertainty = _brand_lift_uncertainty_range(
        row.get("viewership_lift_pct"),
        row.get("games_used", row.get("football_games", 0)),
        median_ape,
    )
    row.update(uncertainty)
    return row


def attach_football_rank_ranges(rows):
    point_lifts = [
        float(row.get("viewership_lift_pct") or 0.0)
        for row in rows
    ]
    for idx, row in enumerate(rows):
        low = pd.to_numeric(row.get("brand_range_low_pct"), errors="coerce")
        high = pd.to_numeric(row.get("brand_range_high_pct"), errors="coerce")
        if pd.isna(low) or pd.isna(high):
            row["brand_rank_ceiling"] = None
            row["brand_rank_floor"] = None
            continue
        other_lifts = [lift for pos, lift in enumerate(point_lifts) if pos != idx]
        row["brand_rank_ceiling"] = int(1 + sum(lift > float(high) for lift in other_lifts))
        row["brand_rank_floor"] = int(1 + sum(lift > float(low) for lift in other_lifts))
        row["brand_tier"], row["brand_tier_label"] = _football_brand_tier(row)
    return rows


def _football_brand_tier(row):
    rank = int(row.get("rank") or 999)
    ceiling = row.get("brand_rank_ceiling")
    floor = row.get("brand_rank_floor")
    team = row.get("team")

    if ceiling == 1 and rank <= 2:
        return 1, "No. 1 Contender"
    if team == "Michigan" or (rank == 3 and ceiling is not None and ceiling <= 3):
        return 2, "Michigan Tier"
    if rank <= 12 and ceiling is not None and ceiling <= 6 and floor is not None and floor <= 25:
        return 3, "Elite National"
    if rank <= 25 and ceiling is not None and ceiling <= 16:
        return 4, "High National"
    if rank <= 45:
        return 5, "Upper Middle"
    if rank <= 75:
        return 6, "Middle"
    if rank <= 110:
        return 7, "Lower Middle"
    return 8, "Low / Sparse"


football_prediction_stability = (
    _football_forward_chain_holdout_stability_lookup()
    or _football_prediction_stability_lookup()
)
football_prediction_stability_deion_controlled = (
    _football_forward_chain_holdout_stability_lookup(exclude_colorado_deion=True)
    or football_prediction_stability
)
football_prediction_stability_display = football_prediction_stability
football_prediction_stability_deion_controlled_display = football_prediction_stability_deion_controlled


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
                    rows.append(attach_football_brand_uncertainty({
                        "rank": idx + 1,
                        "team": team_name,
                        "conference": team_conferences.get(team_name, "Independent"),
                        "viewership_lift_pct": float(round(lift_pct, 1)),
                        "games_used": int(pd.to_numeric(row.get("Games Used"), errors="coerce") or 0),
                    }))

                if rows:
                    return attach_football_rank_ranges(rows)
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


football_prediction_stability_display = {
    display_brand_team_name(team): median_ape
    for team, median_ape in football_prediction_stability.items()
}
football_prediction_stability_deion_controlled_display = {
    display_brand_team_name(team): median_ape
    for team, median_ape in football_prediction_stability_deion_controlled.items()
}


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
        rows.append(attach_football_brand_uncertainty({
            "team": row["team"],
            "conference": row["football_conference"],
            "viewership_lift_pct": row["football_lift_pct"],
            "games_used": row["football_games"],
            "deion_controlled": bool(row.get("deion_controlled", False)),
            "deion_adjustment": row.get("deion_adjustment"),
        }, control_deion=control_deion))
    rows = sorted(rows, key=lambda row: (-row["viewership_lift_pct"], -row["games_used"], row["team"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return attach_football_rank_ranges(rows)


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
REALIGNMENT_PRIMARY_CONFERENCE_WEEKS = list(range(4, 15))
REALIGNMENT_EARLY_CONFERENCE_WEEKS = [1, 2, 3]
REALIGNMENT_EARLY_CONFERENCE_GAME_SHARE = 0.015

REALIGNMENT_NATIONAL_TV_SCORE_FLOOR = {
    "Big 10": 1.05,
    "SEC": 1.00,
    "Big 12": 0.75,
    "ACC": 0.55,
    "Group of 6": 0.0,
    "American Athletic": 0.0,
    "Conference USA": 0.0,
    "FBS Independents": 0.0,
    "Mid-American": 0.0,
    "Mountain West": 0.0,
    "Pac-12": 0.0,
    "Sun Belt": 0.0,
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

REALIGNMENT_GLOBAL_DRAFT_SLOT_FLOORS = {
    "premium": 1_000_000,
    "secondary": 900_000,
    "cable": 500_000,
    "deep_cable": 550_000,
}

REALIGNMENT_GLOBAL_WEEKLY_DRAFT_SLOTS = [
    # Fixed Saturday national TV windows. Slot order approximates draft priority,
    # while eligible_conferences keeps media-rights constraints in place.
    {"network": "ABC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["SEC"]},
    {"network": "FOX", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "ABC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["SEC"]},
    {"network": "FOX", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "CBS", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10"]},
    {"network": "NBC", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10"]},
    {"network": "NBC", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10"]},
    {"network": "FOX", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10", "Big 12"], "flex_slot_id": "fox_primetime", "season_count": 7},
    {"network": "ABC", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["SEC"]},
    {"network": "ESPN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "secondary", "eligible_conferences": ["SEC", "ACC", "Big 12"]},
    {"network": "ESPN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "secondary", "eligible_conferences": ["SEC", "ACC", "Big 12"]},
    {"network": "ESPN", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "secondary", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "ESPN2", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "ESPN2", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "ESPN2", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "FS1", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "FS1", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "FS1", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "BTN", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable", "eligible_conferences": ["Big 10"], "cadence": 2, "offset": 0},
    {"network": "BTN", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 2, "tv_tier": "cable", "eligible_conferences": ["Big 10"], "cadence": 2, "offset": 1},
    {"network": "CW", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "cable", "eligible_conferences": ["ACC"]},
    {"network": "ESPNU", "time_slot": "Sat Mid (2:30p-6:30p)", "comp_tier1": 2, "tv_tier": "deep_cable", "eligible_conferences": ["SEC", "ACC", "Big 12"]},
    {"network": "ESPNU", "time_slot": "Primetime (7:00p-9:00p)", "comp_tier1": 2, "tv_tier": "deep_cable", "eligible_conferences": ["SEC", "ACC", "Big 12"]},
    {"network": "ESPNU", "time_slot": "Sat Early (11:00a-2:00p)", "comp_tier1": 2, "tv_tier": "deep_cable", "eligible_conferences": ["SEC", "ACC", "Big 12"], "cadence": 2, "offset": 0},
    {"network": "ESPN", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "secondary", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "FS1", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "ESPN2", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "FOX", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10", "Big 12"], "min_viewers": 600_000},
    {"network": "CW", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 2, "tv_tier": "cable", "eligible_conferences": ["ACC"], "min_viewers": 650_000},
    {"network": "ESPNU", "time_slot": "Sat Late (9:30p-Later)", "comp_tier1": 2, "tv_tier": "deep_cable", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "ESPN", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "secondary", "eligible_conferences": ["SEC", "ACC", "Big 12"]},
    {"network": "ESPN2", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["ACC", "Big 12"], "cadence": 2, "offset": 0},
    {"network": "FS1", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["Big 10", "Big 12"], "cadence": 2, "offset": 1},
    {"network": "FOX", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10", "Big 12"], "cadence": 3, "offset": 0},
    {"network": "ABC", "time_slot": "Friday", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["SEC", "ACC", "Big 12"], "cadence": 4, "offset": 1, "min_viewers": 1_000_000},
    {"network": "ESPNU", "time_slot": "Friday", "comp_tier1": 2, "tv_tier": "deep_cable", "eligible_conferences": ["SEC", "ACC", "Big 12"], "cadence": 5, "offset": 2},
    {"network": "BTN", "time_slot": "Friday", "comp_tier1": 2, "tv_tier": "cable", "eligible_conferences": ["Big 10"], "cadence": 5, "offset": 0},
    {"network": "CW", "time_slot": "Friday", "comp_tier1": 2, "tv_tier": "cable", "eligible_conferences": ["ACC"], "cadence": 6, "offset": 2, "min_viewers": 650_000},
]

REALIGNMENT_GLOBAL_BLACK_FRIDAY_DRAFT_SLOTS = [
    {"network": "ABC", "time_slot": "Black Friday Early", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["SEC"]},
    {"network": "FOX", "time_slot": "Black Friday Early", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "CBS", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10"]},
    {"network": "ABC", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["ACC", "Big 12"], "min_viewers": 1_000_000},
    {"network": "FOX", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10", "Big 12"]},
    {"network": "NBC", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["Big 10"]},
    {"network": "ABC", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "premium", "eligible_conferences": ["SEC"]},
    {"network": "ESPN", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "secondary", "eligible_conferences": ["SEC", "ACC", "Big 12"]},
    {"network": "ESPN2", "time_slot": "Black Friday Primetime", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["ACC", "Big 12"]},
    {"network": "FS1", "time_slot": "Black Friday Mid", "comp_tier1": 1, "tv_tier": "cable", "eligible_conferences": ["Big 10", "Big 12"]},
]

REALIGNMENT_KNOWN_TV_ASSIGNMENTS = {
    "401856810": {
        "network": "FOX",
        "time_slot": "Sat Mid (2:30p-6:30p)",
        "comp_tier1": 1,
        "tv_tier": "premium",
        "assignment_source": "announced_2026_tv",
    },
    "401856795": {
        "network": "CBS",
        "time_slot": "Primetime (7:00p-9:00p)",
        "comp_tier1": 1,
        "tv_tier": "premium",
        "assignment_source": "announced_2026_tv",
    },
}

REALIGNMENT_CONFERENCE_NETWORK_MINIMUMS = {
    # 2024-2025 measured Big 12-involved broadcast exposure was driven mainly
    # by FOX. This keeps the draft from assigning every FOX broadcast window
    # to the Big Ten when the pure viewer-maximizing answer says it should.
    ("Big 12", "FOX"): 14,
}

REALIGNMENT_LATE_INVENTORY_TEAMS = {
    "Big 10": {"USC", "UCLA", "Oregon", "Washington"},
    "Big 12": {"Arizona", "Arizona St.", "BYU", "Colorado", "Utah"},
    "ACC": {"California", "Stanford"},
}

REALIGNMENT_LATE_SLOT_SEASON_CAPS = {
    "ESPN": 11,
    "FS1": 8,
    "ESPN2": 2,
    "FOX": 2,
    "CW": 1,
    "ESPNU": 1,
    "BTN": 1,
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


def _is_late_saturday_slot(slot):
    return "Sat Late" in str(slot.get("time_slot") or "")


def _late_slot_season_cap(slot):
    if not _is_late_saturday_slot(slot):
        return None
    explicit_cap = slot.get("season_cap")
    if explicit_cap is not None:
        return int(explicit_cap)
    return REALIGNMENT_LATE_SLOT_SEASON_CAPS.get(slot.get("network"))


def _late_slot_usage_key(slot, conference=None):
    return (
        conference,
        slot.get("network"),
        slot.get("time_slot"),
    )


def _late_inventory_teams_for_slot(slot, conference):
    teams = slot.get("late_inventory_teams")
    if teams is not None:
        return {normalize_team(team) for team in teams if normalize_team(team)}
    return set(REALIGNMENT_LATE_INVENTORY_TEAMS.get(conference, set()))


def _slot_late_inventory_eligible(slot, game, conference):
    if not _is_late_saturday_slot(slot):
        return True
    inventory_teams = _late_inventory_teams_for_slot(slot, conference)
    if not inventory_teams:
        return False
    return normalize_team(game.get("team1")) in inventory_teams or normalize_team(game.get("team2")) in inventory_teams


def _late_slot_cap_available(slot, late_slot_usage, conference=None):
    cap = _late_slot_season_cap(slot)
    if cap is None:
        return True
    return late_slot_usage[_late_slot_usage_key(slot, conference)] < cap


def _record_late_slot_usage(slot, late_slot_usage, conference=None):
    if _late_slot_season_cap(slot) is not None:
        late_slot_usage[_late_slot_usage_key(slot, conference)] += 1


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
            if week is None or _slot_available_in_week(slot, week) or _is_late_saturday_slot(slot)
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


@lru_cache(maxsize=None)
def _football_team_coefficient(team):
    normalized_team = normalize_team(team)
    try:
        return float(pregame_model.params.get(normalized_team, 0.0))
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def _worst_modeled_fbs_team_coefficient():
    coefficients = [
        _football_team_coefficient(team)
        for team in MODEL_TEAM_NAMES
        if team_conferences.get(team) is not None
    ]
    return min(coefficients) if coefficients else 0.0


def _realignment_team_coefficient(team, conference=None):
    normalized_team = normalize_team(team)
    if conference and conference not in REALIGNMENT_FBS_CONFERENCES:
        return _worst_modeled_fbs_team_coefficient() - REALIGNMENT_FCS_TEAM_COEFFICIENT_MARGIN
    if normalized_team in MODEL_TEAM_NAMES:
        return _football_team_coefficient(normalized_team)
    return _worst_modeled_fbs_team_coefficient()


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
REALIGNMENT_HIDDEN_FBS_SLOT_NETWORKS = {"CBS", "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "CW"}
REALIGNMENT_HIDDEN_FBS_CANDIDATES_PER_WEEK = 14
REALIGNMENT_FBS_CONFERENCES = {
    "ACC",
    "Big 10",
    "Big 12",
    "SEC",
    "Pac-12",
    "AAC",
    "American Athletic",
    "Conference USA",
    "FBS Independents",
    "Independent",
    "MAC",
    "Mid-American",
    "Mountain West",
    "Sun Belt",
}
REALIGNMENT_FCS_TEAM_COEFFICIENT_MARGIN = 0.15
REALIGNMENT_ACTUAL_SCHEDULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "cfb_2026_power4_conference_schedule.json",
)


def _default_league_memberships():
    return {
        team: conference
        for team, conference in team_conferences.items()
        if conference in REALIGNMENT_EDITABLE_CONFERENCES and team in MODEL_TEAM_NAMES
    }


def _memberships_match_default(memberships):
    return dict(sorted((memberships or {}).items())) == dict(sorted(_default_league_memberships().items()))


@lru_cache(maxsize=1)
def _actual_fbs_schedule_payload():
    try:
        with open(REALIGNMENT_ACTUAL_SCHEDULE_PATH, "r", encoding="utf-8") as schedule_file:
            return json.load(schedule_file)
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def _actual_power4_conference_schedule():
    payload = _actual_fbs_schedule_payload()
    if not payload:
        return {}

    schedules = {conference: [] for conference in REALIGNMENT_EDITABLE_CONFERENCES}
    for raw_game in payload.get("games", []):
        owner_conference = raw_game.get("owner_conference") or raw_game.get("conference")
        team_1 = normalize_team(raw_game.get("team1"))
        team_2 = normalize_team(raw_game.get("team2"))
        team_1_conference = raw_game.get("team1_conference")
        team_2_conference = raw_game.get("team2_conference")
        involved_conferences = {
            conference
            for conference in [team_1_conference, team_2_conference]
            if conference in schedules
        }
        if not involved_conferences and owner_conference in schedules:
            involved_conferences.add(owner_conference)
        if (
            not involved_conferences
            or team_1 == team_2
        ):
            continue
        schedule_row = {
            "event_id": str(raw_game.get("event_id") or ""),
            "date": raw_game.get("date"),
            "week": raw_game.get("week"),
            "owner_conference": owner_conference,
            "team1": team_1,
            "team2": team_2,
            "team1_conference": team_1_conference,
            "team2_conference": team_2_conference,
            "neutral_site": bool(raw_game.get("neutral_site", False)),
            "conference_game": bool(raw_game.get("conference_game", True)),
        }
        for key in [
            "cfbd_media_type",
            "cfbd_tv_outlet",
            "tv_network",
            "tv_time_slot",
            "tv_start_time",
            "tv_start_time_tbd",
            "tv_comp_tier1",
            "tv_tier",
            "tv_assignment_source",
        ]:
            if key in raw_game:
                schedule_row[key] = raw_game[key]
        for conference in involved_conferences:
            schedules[conference].append(dict(schedule_row))
    return schedules


@lru_cache(maxsize=1)
def _actual_fbs_schedule_rows():
    payload = _actual_fbs_schedule_payload()
    rows = []
    for raw_game in payload.get("games", []):
        team_1 = normalize_team(raw_game.get("team1"))
        team_2 = normalize_team(raw_game.get("team2"))
        if not team_1 or not team_2 or team_1 == team_2:
            continue
        rows.append({
            "event_id": str(raw_game.get("event_id") or ""),
            "date": raw_game.get("date"),
            "week": raw_game.get("week"),
            "owner_conference": raw_game.get("owner_conference") or raw_game.get("conference"),
            "team1": team_1,
            "team2": team_2,
            "team1_conference": raw_game.get("team1_conference"),
            "team2_conference": raw_game.get("team2_conference"),
            "neutral_site": bool(raw_game.get("neutral_site", False)),
            "conference_game": bool(raw_game.get("conference_game", True)),
        })
        for key in [
            "cfbd_media_type",
            "cfbd_tv_outlet",
            "tv_network",
            "tv_time_slot",
            "tv_start_time",
            "tv_start_time_tbd",
            "tv_comp_tier1",
            "tv_tier",
            "tv_assignment_source",
        ]:
            if key in raw_game:
                rows[-1][key] = raw_game[key]
    return rows


REALIGNMENT_CFBD_TV_KEYS = [
    "cfbd_media_type",
    "cfbd_tv_outlet",
    "tv_network",
    "tv_time_slot",
    "tv_start_time",
    "tv_start_time_tbd",
    "tv_comp_tier1",
    "tv_tier",
    "tv_assignment_source",
]


REALIGNMENT_UNRATED_TV_NETWORKS = {"ACCN", "CBSSN", "SECN", "USA Net"}


def _copy_cfbd_tv_fields(source, target):
    for key in REALIGNMENT_CFBD_TV_KEYS:
        if key in source:
            target[key] = source[key]
    return target


def _has_unrated_realignment_tv_assignment(row):
    return (
        row.get("tv_network") in REALIGNMENT_UNRATED_TV_NETWORKS
        or str(row.get("cfbd_tv_outlet") or "").strip() in {
            "ACC Network",
            "CBS Sports Network",
            "SEC Network",
            "USA Net",
        }
        or row.get("tv_assignment_source") == "cfbd_games_media_unrated"
    )


@app.get("/realignment/memberships")
def realignment_memberships():
    memberships = _default_league_memberships()
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


@lru_cache(maxsize=1)
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

REALIGNMENT_RANKING_POLICIES = ["espn_2026_preseason", "final_ap_2021_2025", "custom", "unranked"]


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


def _normalized_custom_rankings(custom_rankings):
    normalized = {}
    used_ranks = set()
    for raw_team, raw_rank in (custom_rankings or {}).items():
        team = normalize_team(raw_team)
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            continue
        if not team or rank < 1 or rank > 25 or rank in used_ranks:
            continue
        normalized[team] = rank
        used_ranks.add(rank)
    return normalized


def _build_rank_map(teams, ranking_policy, custom_rankings=None):
    if ranking_policy == "custom":
        rankings = _normalized_custom_rankings(custom_rankings)
        return {
            team: rankings.get(team, 0)
            for team in teams
        }
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


def _matchup_score(team_1, team_2, rank_map, team_1_conference=None, team_2_conference=None):
    rank_1 = rank_map.get(team_1, 0)
    rank_2 = rank_map.get(team_2, 0)
    ranking_score = 0.0
    for rank in [rank_1, rank_2]:
        if 1 <= rank <= 10:
            ranking_score += 0.45
        elif 11 <= rank <= 25:
            ranking_score += 0.25

    return (
        _realignment_team_coefficient(team_1, team_1_conference)
        + _realignment_team_coefficient(team_2, team_2_conference)
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
    max_games = requested_games

    def add_pair(team_1, team_2, protected=False, allow_over_target=False):
        pair_key = tuple(sorted([team_1, team_2]))
        if pair_key in selected_keys:
            return False
        if (
            not protected
            and not allow_over_target
            and (counts[team_1] >= max_games or counts[team_2] >= max_games)
        ):
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

    for team_1, team_2 in sorted(protected_pair_keys):
        add_pair(team_1, team_2, protected=True)

    max_games = max(requested_games, max(counts.values(), default=requested_games))
    protected_selected = list(selected)
    protected_selected_keys = set(selected_keys)
    protected_counts = dict(counts)
    max_attempts = max(1, len(teams) * max_games * 2)

    def exact_fill_attempt(seed):
        rng = random.Random(seed)
        local_selected = [dict(game) for game in protected_selected]
        local_selected_keys = set(protected_selected_keys)
        local_counts = dict(protected_counts)
        if any(count > max_games for count in local_counts.values()):
            return None

        for _ in range(max_attempts):
            deficit_teams = [team for team in teams if local_counts[team] < max_games]
            if not deficit_teams:
                return local_selected, local_selected_keys, local_counts

            team_1 = sorted(
                deficit_teams,
                key=lambda team: (
                    -(max_games - local_counts[team]),
                    local_counts[team],
                    rng.random(),
                    team,
                ),
            )[0]
            candidates = [
                team_2
                for team_2 in deficit_teams
                if team_2 != team_1
                and tuple(sorted([team_1, team_2])) not in local_selected_keys
            ]
            if not candidates:
                return None

            team_2 = sorted(
                candidates,
                key=lambda opponent: (
                    -(max_games - local_counts[opponent]),
                    -_matchup_score(team_1, opponent, rank_map),
                    rng.random(),
                    opponent,
                ),
            )[0]
            pair_key = tuple(sorted([team_1, team_2]))
            local_selected.append({
                "team1": team_1,
                "team2": team_2,
                "score": _matchup_score(team_1, team_2, rank_map),
                "protected": False,
                "rivalry_week_priority": _rivalry_week_priority(team_1, team_2),
            })
            local_selected_keys.add(pair_key)
            local_counts[team_1] += 1
            local_counts[team_2] += 1

        if all(count == max_games for count in local_counts.values()):
            return local_selected, local_selected_keys, local_counts
        return None

    remaining_team_games = sum(max_games - count for count in protected_counts.values())
    if remaining_team_games % 2 == 0:
        for seed in range(250):
            exact_fill = exact_fill_attempt(seed)
            if exact_fill:
                return exact_fill[0], exact_fill[2]

    all_pairs = [
        (team_1, team_2)
        for i, team_1 in enumerate(teams)
        for team_2 in teams[i + 1:]
    ]

    # Fill each team's target count as a bounded b-matching problem. The previous
    # round-robin-first approach could trap dense slates with a few teams short
    # of the requested count even though valid unused pairings still existed.
    for _ in range(max_attempts):
        deficit_teams = [team for team in teams if counts[team] < max_games]
        if not deficit_teams:
            break

        team_1 = sorted(
            deficit_teams,
            key=lambda team: (
                -(max_games - counts[team]),
                counts[team],
                -_football_team_coefficient(team),
                team,
            ),
        )[0]
        candidates = [
            team_2
            for team_2 in deficit_teams
            if team_2 != team_1 and tuple(sorted([team_1, team_2])) not in selected_keys
        ]
        if not candidates:
            break

        team_2 = sorted(
            candidates,
            key=lambda opponent: (
                -(max_games - counts[opponent]),
                -_matchup_score(team_1, opponent, rank_map),
                abs(_football_team_coefficient(team_1) - _football_team_coefficient(opponent)),
                opponent,
            ),
        )[0]
        add_pair(team_1, team_2, protected=False)

    if any(count < max_games for count in counts.values()):
        for team_1, team_2 in sorted(
            all_pairs,
            key=lambda pair: (
                -(max_games - counts[pair[0]]) - (max_games - counts[pair[1]]),
                -_matchup_score(pair[0], pair[1], rank_map),
                pair[0],
                pair[1],
            ),
        ):
            if counts[team_1] >= max_games and counts[team_2] >= max_games:
                continue
            if tuple(sorted([team_1, team_2])) in selected_keys:
                continue
            add_pair(team_1, team_2, protected=False, allow_over_target=True)
            if all(count >= max_games for count in counts.values()):
                break

    while any(count > max_games for count in counts.values()):
        removable_game = next(
            (
                game for game in sorted(
                    selected,
                    key=lambda row: (
                        row.get("protected", False),
                        row["score"],
                        row["team1"],
                        row["team2"],
                    ),
                )
                if not game.get("protected", False)
                and counts[game["team1"]] > max_games
                and counts[game["team2"]] > max_games
            ),
            None,
        )
        if not removable_game:
            break
        selected.remove(removable_game)
        selected_keys.remove(tuple(sorted([removable_game["team1"], removable_game["team2"]])))
        counts[removable_game["team1"]] -= 1
        counts[removable_game["team2"]] -= 1

    for _ in range(max_attempts):
        if not any(count > max_games for count in counts.values()):
            break
        removable_game = next(
            (
                game for game in sorted(
                    selected,
                    key=lambda row: (
                        not (
                            counts[row["team1"]] > max_games
                            or counts[row["team2"]] > max_games
                        ),
                        row.get("protected", False),
                        row["score"],
                        row["team1"],
                        row["team2"],
                    ),
                )
                if not game.get("protected", False)
                and (
                    counts[game["team1"]] > max_games
                    or counts[game["team2"]] > max_games
                )
            ),
            None,
        )
        if not removable_game:
            break
        selected.remove(removable_game)
        selected_keys.remove(tuple(sorted([removable_game["team1"], removable_game["team2"]])))
        counts[removable_game["team1"]] -= 1
        counts[removable_game["team2"]] -= 1

        for _ in range(max_attempts):
            deficit_teams = [team for team in teams if counts[team] < max_games]
            if not deficit_teams:
                break
            candidates = [
                (team_1, team_2)
                for i, team_1 in enumerate(deficit_teams)
                for team_2 in deficit_teams[i + 1:]
                if tuple(sorted([team_1, team_2])) not in selected_keys
            ]
            if not candidates:
                break
            team_1, team_2 = sorted(
                candidates,
                key=lambda pair: (
                    -(max_games - counts[pair[0]]) - (max_games - counts[pair[1]]),
                    -_matchup_score(pair[0], pair[1], rank_map),
                    pair[0],
                    pair[1],
                ),
            )[0]
            add_pair(team_1, team_2, protected=False)

    return selected, counts


def _assign_schedule_weeks(schedule, teams, games_per_team, unavailable_team_weeks=None):
    unavailable_team_weeks = set(unavailable_team_weeks or set())
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

    def teams_available_in_week(game, week_number):
        return (
            (game["team1"], int(week_number)) not in unavailable_team_weeks
            and (game["team2"], int(week_number)) not in unavailable_team_weeks
        )

    remaining_games = []
    for game in schedule:
        pair_key = tuple(sorted([game["team1"], game["team2"]]))
        if pair_key in early_pair_keys:
            early_candidates = [
                week for week in weeks
                if week["number"] in early_week_numbers
                and game["team1"] not in week["teams"]
                and game["team2"] not in week["teams"]
                and teams_available_in_week(game, week["number"])
            ]
            if early_candidates:
                add_game_to_week(
                    game,
                    sorted(early_candidates, key=lambda week: (len(week["games"]), week["number"]))[0],
                )
                continue
        remaining_games.append(game)

    primary_weeks = [week for week in weeks if week["number"] in primary_week_numbers]

    def schedule_with_week_coloring(games_to_schedule, assignment_weeks):
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
        week_loads = Counter({week["number"]: len(week["games"]) for week in assignment_weeks})
        week_scores = Counter({week["number"]: week["score"] for week in assignment_weeks})
        assignments = {}

        def available_weeks(game):
            return [
                week["number"]
                for week in assignment_weeks
                if (game["team1"], week["number"]) not in used_team_weeks
                and (game["team2"], week["number"]) not in used_team_weeks
                and teams_available_in_week(game, week["number"])
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
        weeks_by_number = {week["number"]: week for week in assignment_weeks}
        for game_index, game in enumerate(games_to_schedule):
            add_game_to_week(game, weeks_by_number[assignments[game_index]])
        return True

    use_exact_primary_coloring = len(remaining_games) <= 160
    if use_exact_primary_coloring and schedule_with_week_coloring(remaining_games, primary_weeks):
        return [
            game
            for week in sorted(weeks, key=lambda row: row["number"])
            for game in week["games"]
        ]
    if use_exact_primary_coloring and schedule_with_week_coloring(remaining_games, weeks):
        return [
            game
            for week in sorted(weeks, key=lambda row: row["number"])
            for game in week["games"]
        ]

    rivalry_week = max(primary_week_numbers)
    rivalry_week_row = next(
        (week for week in primary_weeks if week["number"] == rivalry_week),
        None,
    )
    if rivalry_week_row:
        for game in sorted(
            [game for game in remaining_games if int(game.get("rivalry_week_priority") or 0) > 0],
            key=lambda row: (
                -int(row.get("rivalry_week_priority") or 0),
                -row["score"],
                row["team1"],
                row["team2"],
            ),
        ):
            if game["team1"] in rivalry_week_row["teams"] or game["team2"] in rivalry_week_row["teams"]:
                continue
            if not teams_available_in_week(game, rivalry_week_row["number"]):
                continue
            add_game_to_week(game, rivalry_week_row)
            remaining_games.remove(game)

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
                    and teams_available_in_week(game, week["number"])
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
                int(not teams_available_in_week(game, week["number"])),
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
                        and teams_available_in_week(scheduled_game, candidate_week["number"])
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
    late_slot_usage = Counter()
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
            if not _slot_late_inventory_eligible(slot, game, conference):
                continue
            if not _late_slot_cap_available(slot, late_slot_usage, conference):
                continue
            selected_slots[pair_key] = dict(slot)
            used_weekly_slot_indexes[week].add(slot_idx)
            _record_late_slot_usage(slot, late_slot_usage, conference)
            break

    return selected_slots


def _global_tv_slot_key(slot):
    return (
        slot.get("network"),
        slot.get("time_slot"),
    )


def _realignment_slot_start_hour(time_slot):
    value = str(time_slot or "").lower()
    if "early" in value:
        return 11.0
    if "mid" in value:
        return 14.5
    if "late" in value:
        return 21.5
    if "primetime" in value:
        return 19.0
    if "friday" in value:
        return 19.0
    return None


REALIGNMENT_LOCAL_TIMEZONE = "America/New_York"


def _realignment_local_game_date(date_value):
    if not date_value:
        return None
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        return None
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(REALIGNMENT_LOCAL_TIMEZONE)
    return parsed.date()


def _realignment_effective_game_date(date_value, time_slot=None):
    game_date = _realignment_local_game_date(date_value)
    if game_date is None:
        return None
    if "Friday" in str(time_slot or ""):
        return game_date - timedelta(days=1)
    return game_date


def _is_realignment_black_friday_slot(time_slot):
    return "Black Friday" in str(time_slot or "")


def _mark_realignment_row_unrated(row):
    row["network"] = "Not nationally rated"
    row["time_slot"] = "No national TV window"
    row["comp_tier1"] = None
    row["competing_games_score"] = 0.0
    row["tv_tier"] = "unrated"
    row["nationally_rated"] = False
    row["predicted_viewers"] = 0.0
    row["predicted_viewers_formatted"] = "Not rated"


INTRINSIC_COMPETITION_FALLBACK_WARNING = (
    "The pregame artifact does not include both an intrinsic competition model "
    "and its smearing factor. Slate competition was estimated with the final "
    "pregame model as a backward-compatible fallback."
)


def _append_row_warning(row, warning):
    existing = row.get("warnings", [])
    if isinstance(existing, str):
        existing = [existing]
    elif not isinstance(existing, list):
        existing = list(existing) if existing else []
    if warning not in existing:
        existing.append(warning)
    row["warnings"] = existing


def _batch_predict_realignment_rows(row_contexts, compute_competition=True, relative_competition=False):
    if not row_contexts:
        return

    model_columns = list(pregame_model.params.index)
    now = datetime.now()
    feud_active = (now >= FEUD_START) and (FEUD_END is None or now <= FEUD_END)
    slot_starts = []
    game_dates = []

    def build_feature_rows(competing_scores=None):
        feature_rows = []
        if competing_scores is None:
            competing_scores = [
                resolve_competing_games_score(context["row"], pregame_model)[0]
                for context in row_contexts
            ]

        for context, competing_score in zip(row_contexts, competing_scores):
            row = context["row"]
            conference_overrides = context.get("conference_overrides", {}) or {}
            team1 = normalize_team(row["team1"])
            team2 = normalize_team(row["team2"])
            rank1 = int(row.get("rank1") or 0)
            rank2 = int(row.get("rank2") or 0)
            network = row.get("network") or ""
            time_slot = row.get("time_slot") or ""
            comp_tier1 = row.get("comp_tier1") or 0
            game_date = _realignment_effective_game_date(row.get("date"), time_slot) or row.get("date")
            conf1 = conference_overrides.get(team1, team_conferences.get(team1, "Group of 6"))
            conf2 = conference_overrides.get(team2, team_conferences.get(team2, "Group of 6"))
            is_black_friday = _is_realignment_black_friday_slot(time_slot)
            is_friday = ("Friday" in str(time_slot)) and not is_black_friday
            is_power_friday = (
                is_friday
                and is_power_football_team(team1, conf1)
                and is_power_football_team(team2, conf2)
            )
            is_non_power_friday = is_friday and not is_power_friday
            both_ranked = rank1 > 0 and rank2 > 0
            same_conf = conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"]
            t1_top10, t1_25_11 = rank_to_coefs(rank1)
            t2_top10, t2_25_11 = rank_to_coefs(rank2)
            team1_win_pct, team2_win_pct, avg_win_pct, win_pct_diff = (
                resolve_game_record_features(row)
            )
            context_features, _ = build_pregame_context_features(
                time_slot=time_slot,
                date_value=game_date,
                season_week=row.get("season_week", row.get("week")),
                team1_games_before=row.get("team1_games_before"),
                team2_games_before=row.get("team2_games_before"),
                rank1=rank1,
                rank2=rank2,
                cfp_rank1=row.get("cfp_rank1"),
                cfp_rank2=row.get("cfp_rank2"),
                ranking_source=row.get("ranking_source", "preseason"),
            )
            auto_rivalry = next(
                (r for r, (a, b) in rivalries.items() if {team1, team2} == {a, b}),
                None,
            )

            features = {
                "const": 1.0,
                "Competing Tier 1": comp_tier1,
                "Competing_Games_Score": float(competing_score or 0.0),
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
                "NFLN": int(network == "NFLN"),
                "CW": int(network == "CW"),
                "ESPNNEWS": int(network == "ESPNNEWS"),
                "Sun": int("Sunday" in time_slot),
                "Monday": int("Monday" in time_slot),
                "Weekday": int("Weekday" in time_slot),
                "Friday": int(is_friday),
                "Friday Power": int(is_power_friday),
                "Friday Non-Power": int(is_non_power_friday),
                "Black Friday": int(is_black_friday),
                "Week 0 Power": int(is_week_zero_power_game(team1, team2, conf1, conf2, game_date, time_slot)),
                "Week 1 Power": int(is_week_one_power_game(team1, team2, conf1, conf2, game_date, time_slot)),
                "Sat Early": int(not is_black_friday and "Early" in time_slot),
                "Sat Mid": int(not is_black_friday and "Mid" in time_slot),
                "Sat Late": int(not is_black_friday and "Late" in time_slot),
                "Team1_WinPct_ToDate": team1_win_pct,
                "Team2_WinPct_ToDate": team2_win_pct,
                "Avg_WinPct_ToDate": avg_win_pct,
                "WinPct_Diff_ToDate": win_pct_diff,
                "Top 10 Rankings": t1_top10 + t2_top10,
                "25-11 Rankings": t1_25_11 + t2_25_11,
                "SEC": int(conf1 == "SEC") + int(conf2 == "SEC"),
                "Big 10": int(conf1 == "Big 10") + int(conf2 == "Big 10"),
                "ACC": int(conf1 == "ACC") + int(conf2 == "ACC"),
                "Big 12": int(conf1 == "Big 12") + int(conf2 == "Big 12"),
                "SEC_PostseasonImplications": int(both_ranked and same_conf and conf1 == "SEC"),
                "Big10_PostseasonImplications": int(both_ranked and same_conf and conf1 == "Big 10"),
                "Big12_PostseasonImplications": int(both_ranked and same_conf and conf1 == "Big 12"),
                "ACC_PostseasonImplications": int(both_ranked and same_conf and conf1 == "ACC"),
                "YTTV_ABC": int(feud_active and network == "ABC"),
                "YTTV_ESPN": int(feud_active and network == "ESPN"),
            }
            features.update(context_features)

            for rivalry_key in rivalries:
                features[rivalry_key] = int(rivalry_key == auto_rivalry)
            for team_column in MODEL_TEAM_NAMES:
                features[team_column] = int(team_column in [team1, team2])
            if "OhioSt_BTN" in model_columns:
                features["OhioSt_BTN"] = int(("Ohio St." in [team1, team2]) and network == "BTN")

            feature_rows.append({column: features.get(column, 0.0) for column in model_columns})

        return pd.DataFrame.from_records(feature_rows, columns=model_columns)

    for context in row_contexts:
        row = context["row"]
        slot_starts.append(_realignment_slot_start_hour(row.get("time_slot")))
        effective_date = _realignment_effective_game_date(
            row.get("date"),
            row.get("time_slot"),
        )
        season_week = row.get("season_week", row.get("week"))
        game_dates.append(
            effective_date
            if effective_date is not None
            else (("week", season_week) if season_week is not None else None)
        )

    X = build_feature_rows()
    if compute_competition and "Competing_Games_Score" in X.columns:
        X["Competing_Games_Score"] = 0.0
        intrinsic_model = getattr(pregame_model, "intrinsic_model", None)
        intrinsic_smearing = getattr(
            pregame_model,
            "intrinsic_smearing_factor",
            None,
        )
        try:
            intrinsic_smearing = float(intrinsic_smearing)
        except (TypeError, ValueError):
            intrinsic_smearing = None
        use_persisted_intrinsic = (
            intrinsic_model is not None
            and intrinsic_smearing is not None
            and np.isfinite(intrinsic_smearing)
            and intrinsic_smearing > 0
        )

        if use_persisted_intrinsic:
            intrinsic_columns = list(
                getattr(getattr(intrinsic_model, "params", None), "index", model_columns)
            )
            intrinsic_X = X.reindex(columns=intrinsic_columns, fill_value=0.0)
            if "Competing_Games_Score" in intrinsic_X.columns:
                intrinsic_X["Competing_Games_Score"] = 0.0
            intrinsic_ln_pred = intrinsic_model.predict(intrinsic_X)
        else:
            intrinsic_ln_pred = pregame_model.predict(X)
            intrinsic_smearing = getattr(pregame_model, "smearing_factor", 1.0)
            for context in row_contexts:
                _append_row_warning(
                    context["row"],
                    INTRINSIC_COMPETITION_FALLBACK_WARNING,
                )

        intrinsic_predictions_millions = np.asarray(
            ((np.exp(intrinsic_ln_pred) - 1) * intrinsic_smearing) / 1000.0,
            dtype=float,
        ).reshape(-1)
        competing_scores = []

        for idx, (game_date, start_hour) in enumerate(zip(game_dates, slot_starts)):
            score = 0.0
            if game_date is not None and start_hour is not None:
                target_intrinsic = float(intrinsic_predictions_millions[idx])
                for other_idx, (other_date, other_start) in enumerate(zip(game_dates, slot_starts)):
                    if other_idx == idx or other_date != game_date or other_start is None:
                        continue
                    if abs(other_start - start_hour) <= 1.5:
                        other_intrinsic = float(intrinsic_predictions_millions[other_idx])
                        if relative_competition and target_intrinsic > 0:
                            relative_strength = min(1.0, other_intrinsic / target_intrinsic)
                            other_intrinsic *= relative_strength ** 5
                        score += other_intrinsic
            competing_scores.append(score)

        X = build_feature_rows(competing_scores)
        for context, score in zip(row_contexts, competing_scores):
            context["row"]["competing_games_score"] = float(score)

    predictions = predict_pregame_points_000s(
        pregame_model,
        X,
        [context["row"] for context in row_contexts],
    ) * 1000

    for context, prediction in zip(row_contexts, predictions):
        viewers = float(prediction)
        row = context["row"]
        row["predicted_viewers"] = viewers
        row["predicted_viewers_formatted"] = format_viewers(viewers)


def _apply_tv_slot_to_realignment_row(row, slot, conference, conference_teams):
    prediction = predict_viewership({
        "team1": row["team1"],
        "team2": row["team2"],
        "rank1": row.get("rank1", 0),
        "rank2": row.get("rank2", 0),
        "network": slot["network"],
        "time_slot": slot["time_slot"],
        "date": row.get("date"),
        "season_week": row.get("season_week", row.get("week")),
        "team1_games_before": row.get("team1_games_before"),
        "team2_games_before": row.get("team2_games_before"),
        "cfp_rank1": row.get("cfp_rank1"),
        "cfp_rank2": row.get("cfp_rank2"),
        "ranking_source": "preseason",
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
    for warning in prediction.get("warnings", []):
        _append_row_warning(row, warning)


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


def _realignment_summary_for_rows(rows, team_count=0):
    rows = list(rows or [])
    rated_rows = [row for row in rows if row.get("nationally_rated", True)]
    return {
        "teams": int(team_count or 0),
        "games": len(rated_rows),
        "scheduled_games": len(rows),
        "nationally_rated_games": len(rated_rows),
        "unrated_games": len(rows) - len(rated_rows),
        "total_viewers": _viewer_total(row["predicted_viewers"] for row in rated_rows),
        "average_viewers": _viewer_average(row["predicted_viewers"] for row in rated_rows),
    }


REALIGNMENT_DISTRIBUTION_METRICS = [
    ("p75", "75th Percentile Game", 75),
    ("p50", "50th Percentile Game", 50),
    ("p25", "25th Percentile Game", 25),
]


def _rated_viewer_values(rows):
    return [
        float(row.get("predicted_viewers") or 0)
        for row in rows
        if row.get("nationally_rated", True)
    ]


def _top_n_average(values, count=10):
    if not values:
        return 0.0
    top_values = sorted(values, reverse=True)[:count]
    return float(sum(top_values) / len(top_values))


def _distribution_metric_comparison(current_rows, baseline_rows, membership_pct_delta=None):
    current_values = _rated_viewer_values(current_rows)
    baseline_values = _rated_viewer_values(baseline_rows)
    comparison = []

    for key, label, percentile in REALIGNMENT_DISTRIBUTION_METRICS:
        current_value = float(np.percentile(current_values, percentile)) if current_values else 0.0
        baseline_value = float(np.percentile(baseline_values, percentile)) if baseline_values else 0.0
        delta = current_value - baseline_value
        pct_delta = None if baseline_value == 0 else (delta / baseline_value) * 100
        comparison.append({
            "key": key,
            "label": label,
            "current_viewers": current_value,
            "baseline_viewers": baseline_value,
            "viewer_delta": delta,
            "pct_delta": pct_delta,
            "membership_elasticity": None if membership_pct_delta in (None, 0) or pct_delta is None else pct_delta / membership_pct_delta,
        })

    current_top_10 = _top_n_average(current_values, 10)
    baseline_top_10 = _top_n_average(baseline_values, 10)
    top_10_delta = current_top_10 - baseline_top_10
    top_10_pct_delta = None if baseline_top_10 == 0 else (top_10_delta / baseline_top_10) * 100
    comparison.append({
        "key": "top_10_avg",
        "label": "Top 10 Avg.",
        "current_viewers": current_top_10,
        "baseline_viewers": baseline_top_10,
        "viewer_delta": top_10_delta,
        "pct_delta": top_10_pct_delta,
        "membership_elasticity": None if membership_pct_delta in (None, 0) or top_10_pct_delta is None else top_10_pct_delta / membership_pct_delta,
    })

    return comparison


def _is_flexible_global_slot(slot):
    return bool(slot.get("flex_slot_id"))


def _flexible_global_slot_available_in_week(slot, week, flex_slot_weeks=None):
    if not _is_flexible_global_slot(slot):
        return _slot_available_in_week(slot, week)
    selected_weeks = (flex_slot_weeks or {}).get(slot.get("flex_slot_id"), set())
    return int(week or 1) in selected_weeks


def _global_weekly_draft_slots(week, flex_slot_weeks=None):
    top_saturday_slots = REALIGNMENT_GLOBAL_WEEKLY_DRAFT_SLOTS[:9]
    remaining_slots = REALIGNMENT_GLOBAL_WEEKLY_DRAFT_SLOTS[9:]
    slots = [
        dict(slot)
        for slot in top_saturday_slots
        if _flexible_global_slot_available_in_week(slot, week, flex_slot_weeks)
    ]
    if int(week or 0) == REALIGNMENT_BLACK_FRIDAY_WEEK:
        slots.extend(dict(slot) for slot in REALIGNMENT_GLOBAL_BLACK_FRIDAY_DRAFT_SLOTS)
    slots.extend(
        dict(slot)
        for slot in remaining_slots
        if _flexible_global_slot_available_in_week(slot, week, flex_slot_weeks) or _is_late_saturday_slot(slot)
    )
    return slots


def _slot_minimum_projected_viewers(slot):
    return float(
        slot.get("min_viewers")
        or REALIGNMENT_GLOBAL_DRAFT_SLOT_FLOORS.get(
            slot.get("tv_tier"),
            REALIGNMENT_GLOBAL_DRAFT_SLOT_FLOORS["cable"],
        )
    )


def _known_realignment_tv_assignment(row):
    if _has_unrated_realignment_tv_assignment(row):
        return None
    actual_network = row.get("tv_network")
    if actual_network:
        return {
            "network": actual_network,
            "time_slot": row.get("tv_time_slot") or "No national TV window",
            "comp_tier1": row.get("tv_comp_tier1"),
            "tv_tier": row.get("tv_tier") or "cable",
            "assignment_source": row.get("tv_assignment_source") or "cfbd_games_media",
        }
    event_id = str(row.get("event_id") or "")
    assignment = REALIGNMENT_KNOWN_TV_ASSIGNMENTS.get(event_id)
    return dict(assignment) if assignment else None


def _row_is_notre_dame_home_nbc_package(row):
    return (
        normalize_team(row.get("team2")) == "Notre Dame"
        and row.get("owner_conference") in {"FBS Independents", "Independent"}
        and not bool(row.get("neutral_site", False))
    )


def _conference_network_minimum(conference, network):
    return int(REALIGNMENT_CONFERENCE_NETWORK_MINIMUMS.get((conference, network), 0))


def _record_realignment_network_usage(row, conference, network_usage):
    if row.get("nationally_rated", False):
        network_usage[(conference, row.get("network"))] += 1


def _assign_realignment_tv_slot(candidate, slot):
    row = candidate["row"]
    row["network"] = slot["network"]
    row["time_slot"] = slot["time_slot"]
    row["comp_tier1"] = slot["comp_tier1"]
    row["tv_tier"] = slot.get("tv_tier")
    row["nationally_rated"] = True
    if slot.get("assignment_source"):
        row["assignment_source"] = slot["assignment_source"]


def _slot_should_prioritize_conference(slot, conference, network_usage, weekly_candidates):
    target = _conference_network_minimum(conference, slot.get("network"))
    if target <= 0 or network_usage[(conference, slot.get("network"))] >= target:
        return False
    eligible_conferences = set(slot.get("eligible_conferences") or REALIGNMENT_EDITABLE_CONFERENCES)
    if conference not in eligible_conferences:
        return False
    return any(
        candidate["conference"] == conference
        and _slot_late_inventory_eligible(slot, candidate["row"], candidate["conference"])
        and float(candidate["row"].get("matchup_score") or 0.0) >= REALIGNMENT_NATIONAL_TV_SCORE_FLOOR.get(
            conference,
            REALIGNMENT_NATIONAL_TV_SCORE_FLOOR["SEC"],
        )
        for candidate in weekly_candidates
    )


def _conference_overrides_for_slate(conference, slate):
    return {
        team: conference
        for team in slate.get("teams", [])
    }


def _realignment_event_key(row):
    event_id = str(row.get("event_id") or "")
    if event_id:
        return ("event", event_id)
    return (
        "matchup",
        int(row.get("week") or 0),
        tuple(sorted([normalize_team(row.get("team1")), normalize_team(row.get("team2"))])),
    )


def _copy_realignment_tv_assignment(source, target):
    for key in [
        "network",
        "time_slot",
        "comp_tier1",
        "tv_tier",
        "nationally_rated",
        "predicted_viewers",
        "predicted_viewers_formatted",
        "competing_games_score",
        "assignment_source",
    ]:
        if key in source:
            target[key] = source[key]


def _better_realignment_assignment_source(current, candidate):
    if current is None:
        return candidate
    current_viewers = float(current.get("predicted_viewers") or 0.0)
    candidate_viewers = float(candidate.get("predicted_viewers") or 0.0)
    if candidate_viewers > current_viewers:
        return candidate
    if candidate_viewers < current_viewers:
        return current
    if candidate.get("assignment_source") and not current.get("assignment_source"):
        return candidate
    return current


def _propagate_realignment_duplicate_assignments(conference_slates):
    assignment_by_event = {}
    for slate in conference_slates.values():
        for row in slate.get("rows", []):
            if row.get("nationally_rated", False):
                event_key = _realignment_event_key(row)
                assignment_by_event[event_key] = _better_realignment_assignment_source(
                    assignment_by_event.get(event_key),
                    row,
                )
    for slate in conference_slates.values():
        for row in slate.get("rows", []):
            source = assignment_by_event.get(_realignment_event_key(row))
            if source is not None and source is not row:
                _copy_realignment_tv_assignment(source, row)


def _realignment_row_involves_conference(row, conference):
    row_conferences = set()
    for value in [
        row.get("team1_conference") or team_conferences.get(normalize_team(row.get("team1"))),
        row.get("team2_conference") or team_conferences.get(normalize_team(row.get("team2"))),
    ]:
        if isinstance(value, (list, tuple, set)):
            row_conferences.update(item for item in value if item)
        elif value:
            row_conferences.add(value)
    return conference in row_conferences


def _realignment_row_home_conference(row):
    return row.get("team2_conference") or row.get("owner_conference")


def _realignment_row_owned_by_conference(row, conference):
    return _realignment_row_home_conference(row) == conference


def _public_realignment_slate_for_conference(slate, conference):
    public_slate = {
        **slate,
        "rows": [
            row
            for row in slate.get("rows", [])
            if _realignment_row_owned_by_conference(row, conference)
        ],
    }
    public_slate["summary"] = _realignment_summary_for_rows(
        public_slate["rows"],
        team_count=slate.get("summary", {}).get("teams", len(slate.get("teams", []))),
    )
    return public_slate


def _candidate_eligible_for_realignment_slot(candidate, slot, eligible_conferences):
    if candidate["conference"] in eligible_conferences:
        return True
    if (
        candidate.get("hidden_fbs_competitor")
        and slot.get("network") in REALIGNMENT_HIDDEN_FBS_SLOT_NETWORKS
    ):
        return True
    return slot.get("network") == "NBC" and _row_is_notre_dame_home_nbc_package(candidate["row"])


def _realignment_candidate_score_floor(candidate):
    if candidate.get("hidden_fbs_competitor"):
        return REALIGNMENT_NATIONAL_TV_SCORE_FLOOR["Group of 6"]
    return REALIGNMENT_NATIONAL_TV_SCORE_FLOOR.get(
        candidate["conference"],
        REALIGNMENT_NATIONAL_TV_SCORE_FLOOR["SEC"],
    )


def _select_flexible_global_slot_weeks(candidates, weeks, assigned_row_ids, used_weekly_slot_keys, late_slot_usage):
    selected_weeks_by_slot = {}
    flexible_slots = [
        slot for slot in REALIGNMENT_GLOBAL_WEEKLY_DRAFT_SLOTS
        if _is_flexible_global_slot(slot)
    ]
    for slot in flexible_slots:
        slot_key = _global_tv_slot_key(slot)
        season_count = int(slot.get("season_count") or 0)
        if season_count <= 0:
            continue

        week_scores = []
        eligible_conferences = set(slot.get("eligible_conferences") or REALIGNMENT_EDITABLE_CONFERENCES)
        for week in weeks:
            if slot_key in used_weekly_slot_keys[week]:
                continue
            if not _late_slot_cap_available(slot, late_slot_usage):
                continue
            slot_candidates = [
                candidate for candidate in candidates
                if int(candidate["row"].get("week") or 1) == week
                and id(candidate["row"]) not in assigned_row_ids
                and _candidate_eligible_for_realignment_slot(candidate, slot, eligible_conferences)
                and _slot_late_inventory_eligible(slot, candidate["row"], candidate["conference"])
                and float(candidate["row"].get("matchup_score") or 0.0) >= _realignment_candidate_score_floor(candidate)
            ]
            if not slot_candidates:
                continue

            evaluation_contexts = []
            for candidate in slot_candidates:
                evaluation_row = {
                    **candidate["row"],
                    "network": slot["network"],
                    "time_slot": slot["time_slot"],
                    "comp_tier1": slot["comp_tier1"],
                    "tv_tier": slot.get("tv_tier"),
                    "nationally_rated": True,
                }
                evaluation_contexts.append({
                    "candidate": candidate,
                    "row": evaluation_row,
                    "conference_overrides": candidate.get("conference_overrides") or {
                        team: candidate["conference"]
                        for team in candidate["conference_teams"]
                    },
                })

            _batch_predict_realignment_rows(evaluation_contexts, compute_competition=False)
            best_context = max(
                evaluation_contexts,
                key=lambda context: (
                    float(context["row"].get("predicted_viewers") or 0.0),
                    float(context["candidate"]["row"].get("matchup_score") or 0.0),
                    context["candidate"]["row"].get("matchup", ""),
                ),
            )
            best_prediction = float(best_context["row"].get("predicted_viewers") or 0.0)
            if best_prediction < _slot_minimum_projected_viewers(slot):
                continue
            week_scores.append((
                best_prediction,
                float(best_context["candidate"]["row"].get("matchup_score") or 0.0),
                -int(week),
                int(week),
            ))

        week_scores.sort(reverse=True)
        selected_weeks_by_slot[slot.get("flex_slot_id")] = {
            item[-1] for item in week_scores[:season_count]
        }
    return selected_weeks_by_slot


def _hidden_fbs_conference_overrides(row):
    overrides = {}
    for team_key, conference_key in [("team1", "team1_conference"), ("team2", "team2_conference")]:
        team = normalize_team(row.get(team_key))
        conference = row.get(conference_key)
        if team and conference in REALIGNMENT_EDITABLE_CONFERENCES:
            overrides[team] = conference
    return overrides


def _build_hidden_fbs_tv_candidates(existing_event_keys, ranking_policy, custom_rankings=None, excluded_teams=None):
    excluded_teams = {normalize_team(team) for team in (excluded_teams or set())}
    raw_rows = [
        raw_row for raw_row in _actual_fbs_schedule_rows()
        if _realignment_event_key(raw_row) not in existing_event_keys
        and raw_row["team1"] not in excluded_teams
        and raw_row["team2"] not in excluded_teams
    ]
    if not raw_rows:
        return []

    involved_teams = sorted({
        team
        for raw_row in raw_rows
        for team in [raw_row["team1"], raw_row["team2"]]
    })
    rank_map = _build_rank_map(involved_teams, ranking_policy, custom_rankings)
    candidates = []
    for idx, raw_row in enumerate(raw_rows):
        owner_conference = raw_row.get("owner_conference") or "Group of 6"
        row = {
            "game_number": idx + 1,
            "event_id": raw_row.get("event_id"),
            "date": raw_row.get("date"),
            "week": raw_row.get("week"),
            "team1": raw_row["team1"],
            "team2": raw_row["team2"],
            "matchup": f"{raw_row['team1']} vs {raw_row['team2']}",
            "rank1": rank_map.get(raw_row["team1"], 0),
            "rank2": rank_map.get(raw_row["team2"], 0),
            "network": "Not nationally rated",
            "time_slot": "No national TV window",
            "comp_tier1": None,
            "tv_tier": "unrated",
            "nationally_rated": False,
            "matchup_score": float(round(_matchup_score(
                raw_row["team1"],
                raw_row["team2"],
                rank_map,
                raw_row.get("team1_conference"),
                raw_row.get("team2_conference"),
            ), 4)),
            "protected": False,
            "actual_schedule": True,
            "conference_game": bool(raw_row.get("conference_game", True)),
            "owner_conference": owner_conference,
            "team1_conference": raw_row.get("team1_conference"),
            "team2_conference": raw_row.get("team2_conference"),
            "neutral_site": bool(raw_row.get("neutral_site", False)),
            "predicted_viewers": 0.0,
            "predicted_viewers_formatted": "Not rated",
        }
        _copy_cfbd_tv_fields(raw_row, row)
        candidates.append({
            "conference": owner_conference if owner_conference in REALIGNMENT_FBS_CONFERENCES else "Group of 6",
            "row": row,
            "team_count": 0,
            "conference_teams": [],
            "conference_overrides": _hidden_fbs_conference_overrides(row),
            "hidden_fbs_competitor": True,
        })
    candidates_by_week = defaultdict(list)
    for candidate in candidates:
        candidates_by_week[int(candidate["row"].get("week") or 1)].append(candidate)

    selected = []
    for week_candidates in candidates_by_week.values():
        guaranteed = [
            candidate for candidate in week_candidates
            if _row_is_notre_dame_home_nbc_package(candidate["row"])
        ]
        remaining = [
            candidate for candidate in week_candidates
            if candidate not in guaranteed
        ]
        remaining = sorted(
            remaining,
            key=lambda candidate: (
                -float(candidate["row"].get("matchup_score") or 0.0),
                candidate["row"].get("matchup", ""),
            ),
        )[:REALIGNMENT_HIDDEN_FBS_CANDIDATES_PER_WEEK]
        selected.extend(guaranteed + remaining)
    return selected


def _assign_notre_dame_home_nbc_games(
    week,
    weekly_candidates,
    assigned_row_ids,
    used_weekly_slot_keys,
    late_slot_usage,
    network_usage,
    flex_slot_weeks=None,
):
    available_slots = [
        slot for slot in _global_weekly_draft_slots(week, flex_slot_weeks)
        if slot.get("network") == "NBC"
        and _global_tv_slot_key(slot) not in used_weekly_slot_keys[week]
        and _late_slot_cap_available(slot, late_slot_usage)
    ]
    if not available_slots:
        return

    nd_candidates = [
        candidate for candidate in weekly_candidates
        if id(candidate["row"]) not in assigned_row_ids
        and _row_is_notre_dame_home_nbc_package(candidate["row"])
    ]
    while available_slots and nd_candidates:
        evaluation_contexts = []
        for candidate in nd_candidates:
            for slot in available_slots:
                evaluation_row = {
                    **candidate["row"],
                    "network": slot["network"],
                    "time_slot": slot["time_slot"],
                    "comp_tier1": slot["comp_tier1"],
                    "tv_tier": slot.get("tv_tier"),
                    "nationally_rated": True,
                }
                evaluation_contexts.append({
                    "candidate": candidate,
                    "slot": slot,
                    "row": evaluation_row,
                    "conference_overrides": candidate.get("conference_overrides") or {
                        team: candidate["conference"]
                        for team in candidate["conference_teams"]
                    },
                })

        _batch_predict_realignment_rows(evaluation_contexts, compute_competition=False)
        best_context = max(
            evaluation_contexts,
            key=lambda context: (
                float(context["row"].get("predicted_viewers") or 0.0),
                float(context["candidate"]["row"].get("matchup_score") or 0.0),
                context["candidate"]["row"].get("matchup", ""),
            ),
        )
        target_row = best_context["candidate"]["row"]
        best_slot = best_context["slot"]
        _assign_realignment_tv_slot(best_context["candidate"], best_slot)
        target_row["assignment_source"] = "notre_dame_nbc_home_package"
        target_row["predicted_viewers"] = float(best_context["row"].get("predicted_viewers") or 0.0)
        target_row["predicted_viewers_formatted"] = best_context["row"]["predicted_viewers_formatted"]
        assigned_row_ids.add(id(target_row))
        used_weekly_slot_keys[week].add(_global_tv_slot_key(best_slot))
        _record_late_slot_usage(best_slot, late_slot_usage)
        _record_realignment_network_usage(target_row, best_context["candidate"]["conference"], network_usage)
        available_slots = [
            slot for slot in available_slots
            if _global_tv_slot_key(slot) not in used_weekly_slot_keys[week]
        ]
        nd_candidates = [
            candidate for candidate in nd_candidates
            if id(candidate["row"]) not in assigned_row_ids
        ]


def _apply_global_tv_slots_to_league_slates(
    conference_slates,
    transferred_slots_by_conference,
    removed_slots_by_conference,
    ranking_policy="espn_2026_preseason",
    custom_rankings=None,
):
    candidates = []
    visible_teams = set()
    existing_event_keys = set()
    for conference, slate in conference_slates.items():
        visible_teams.update(normalize_team(team) for team in slate.get("teams", []))
        for row in slate.get("rows", []):
            existing_event_keys.add(_realignment_event_key(row))
            _mark_realignment_row_unrated(row)
            if _has_unrated_realignment_tv_assignment(row):
                continue
            if (
                row.get("owner_conference")
                and row.get("owner_conference") != conference
                and row.get("owner_conference") in REALIGNMENT_EDITABLE_CONFERENCES
                and not _known_realignment_tv_assignment(row)
                and not _row_is_notre_dame_home_nbc_package(row)
            ):
                continue
            candidates.append({
                "conference": conference,
                "row": row,
                "team_count": len(slate.get("teams", [])),
                "conference_teams": slate.get("teams", []),
            })

    hidden_fbs_candidates = _build_hidden_fbs_tv_candidates(
        existing_event_keys,
        ranking_policy,
        custom_rankings=custom_rankings,
        excluded_teams=visible_teams,
    )
    hidden_fbs_candidates = [
        candidate
        for candidate in hidden_fbs_candidates
        if not _has_unrated_realignment_tv_assignment(candidate["row"])
    ]
    candidates.extend(hidden_fbs_candidates)

    assigned_row_ids = set()
    used_weekly_slot_keys = defaultdict(set)
    late_slot_usage = Counter()
    network_usage = Counter()
    weeks = sorted({
        int(candidate["row"].get("week") or 1)
        for candidate in candidates
    })

    for candidate in candidates:
        slot = _known_realignment_tv_assignment(candidate["row"])
        if not slot:
            continue
        week = int(candidate["row"].get("week") or 1)
        _assign_realignment_tv_slot(candidate, slot)
        assigned_row_ids.add(id(candidate["row"]))
        used_weekly_slot_keys[week].add(_global_tv_slot_key(slot))
        _record_late_slot_usage(slot, late_slot_usage)
        _record_realignment_network_usage(candidate["row"], candidate["conference"], network_usage)

    flex_slot_weeks = _select_flexible_global_slot_weeks(
        candidates,
        weeks,
        assigned_row_ids,
        used_weekly_slot_keys,
        late_slot_usage,
    )

    for week in weeks:
        weekly_candidates = [
            candidate for candidate in candidates
            if int(candidate["row"].get("week") or 1) == week
            and id(candidate["row"]) not in assigned_row_ids
        ]
        _assign_notre_dame_home_nbc_games(
            week,
            weekly_candidates,
            assigned_row_ids,
            used_weekly_slot_keys,
            late_slot_usage,
            network_usage,
            flex_slot_weeks,
        )
        for slot in _global_weekly_draft_slots(week, flex_slot_weeks):
            slot_key = _global_tv_slot_key(slot)
            if slot_key in used_weekly_slot_keys[week]:
                continue
            if not _late_slot_cap_available(slot, late_slot_usage):
                continue
            eligible_conferences = set(slot.get("eligible_conferences") or REALIGNMENT_EDITABLE_CONFERENCES)
            for conference in REALIGNMENT_EDITABLE_CONFERENCES:
                if _slot_should_prioritize_conference(slot, conference, network_usage, weekly_candidates):
                    eligible_conferences = {conference}
                    break
            slot_candidates = [
                candidate for candidate in weekly_candidates
                if id(candidate["row"]) not in assigned_row_ids
                and _candidate_eligible_for_realignment_slot(candidate, slot, eligible_conferences)
                and _slot_late_inventory_eligible(slot, candidate["row"], candidate["conference"])
                and float(candidate["row"].get("matchup_score") or 0.0) >= _realignment_candidate_score_floor(candidate)
            ]
            if not slot_candidates:
                continue

            evaluation_contexts = []
            for candidate in slot_candidates:
                evaluation_row = {
                    **candidate["row"],
                    "network": slot["network"],
                    "time_slot": slot["time_slot"],
                    "comp_tier1": slot["comp_tier1"],
                    "tv_tier": slot.get("tv_tier"),
                    "nationally_rated": True,
                }
                evaluation_contexts.append({
                    "candidate": candidate,
                    "row": evaluation_row,
                    "conference_overrides": candidate.get("conference_overrides") or {
                        team: candidate["conference"]
                        for team in candidate["conference_teams"]
                    },
                })

            _batch_predict_realignment_rows(evaluation_contexts, compute_competition=False)
            best_context = max(
                evaluation_contexts,
                key=lambda context: (
                    float(context["row"].get("predicted_viewers") or 0.0),
                    float(context["candidate"]["row"].get("matchup_score") or 0.0),
                    context["candidate"]["row"].get("matchup", ""),
                ),
            )
            best_prediction = float(best_context["row"].get("predicted_viewers") or 0.0)
            if best_prediction < _slot_minimum_projected_viewers(slot):
                continue

            target_row = best_context["candidate"]["row"]
            _assign_realignment_tv_slot(best_context["candidate"], slot)
            target_row["predicted_viewers"] = best_prediction
            target_row["predicted_viewers_formatted"] = best_context["row"]["predicted_viewers_formatted"]
            assigned_row_ids.add(id(target_row))
            used_weekly_slot_keys[week].add(slot_key)
            _record_late_slot_usage(slot, late_slot_usage)
            _record_realignment_network_usage(target_row, best_context["candidate"]["conference"], network_usage)

    _propagate_realignment_duplicate_assignments(conference_slates)
    hidden_rated_rows = [
        candidate["row"]
        for candidate in hidden_fbs_candidates
        if candidate["row"].get("nationally_rated", False)
    ]

    rated_contexts = []
    seen_rated_events = set()
    for conference, slate in conference_slates.items():
        for row in slate.get("rows", []):
            if not row.get("nationally_rated", True):
                continue
            event_key = _realignment_event_key(row)
            if event_key in seen_rated_events:
                continue
            seen_rated_events.add(event_key)
            rated_contexts.append({
                "row": row,
                "conference_overrides": _conference_overrides_for_slate(conference, slate),
            })
    for candidate in hidden_fbs_candidates:
        row = candidate["row"]
        if not row.get("nationally_rated", False):
            continue
        event_key = _realignment_event_key(row)
        if event_key in seen_rated_events:
            continue
        seen_rated_events.add(event_key)
        rated_contexts.append({
            "row": row,
            "conference_overrides": candidate.get("conference_overrides") or {},
        })

    _batch_predict_realignment_rows(
        rated_contexts,
        compute_competition=True,
        relative_competition=False,
    )
    hidden_assignment_by_event = {
        _realignment_event_key(row): row
        for row in hidden_rated_rows
    }
    for slate in conference_slates.values():
        for row in slate.get("rows", []):
            source = hidden_assignment_by_event.get(_realignment_event_key(row))
            if source is not None:
                _copy_realignment_tv_assignment(source, row)
    _propagate_realignment_duplicate_assignments(conference_slates)

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
    assign_preliminary_tv=True,
):
    normalized_teams = [normalize_team(team) for team in teams]
    rank_map = _build_rank_map(normalized_teams, sim.ranking_policy, sim.custom_rankings)
    schedule, counts = _generate_conference_schedule(
        normalized_teams,
        sim.games_per_team,
        rank_map,
        protected_pairs=protected_pairs,
    )
    if realistic_tv_inventory:
        schedule = _assign_schedule_weeks(schedule, normalized_teams, sim.games_per_team)
        scheduled_tv_slots = (
            _realistic_tv_slots_for_schedule(
                schedule,
                conference,
                team_count=len(normalized_teams),
                transferred_slots=transferred_slots,
                removed_slots=removed_slots,
            )
            if assign_preliminary_tv
            else {}
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
            viewers = 0.0
            predicted_viewers_formatted = "Pending slate prediction"
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
            "season_week": game.get("season_week", game.get("week")),
            "date": game.get("date"),
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
    _batch_predict_realignment_rows([
        {
            "row": row,
            "conference_overrides": conference_overrides,
        }
        for row in rated_rows
    ])
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


def _predict_actual_realignment_slate(conference, teams, sim, protected_pairs=None, memberships=None):
    normalized_teams = [normalize_team(team) for team in teams]
    team_set = set(normalized_teams)
    memberships = memberships or {}
    raw_schedule = [
        raw_game
        for raw_game in _actual_fbs_schedule_rows()
        if raw_game["team1"] in team_set or raw_game["team2"] in team_set
    ]
    involved_teams = sorted({
        team
        for raw_game in raw_schedule
        for team in [raw_game["team1"], raw_game["team2"]]
    })
    rank_map = _build_rank_map(
        sorted(set(normalized_teams) | set(involved_teams)),
        sim.ranking_policy,
        sim.custom_rankings,
    )
    protected_pair_keys = {
        tuple(sorted([team_1, team_2]))
        for team_1, team_2 in (protected_pairs or [])
        if team_1 in team_set and team_2 in team_set and team_1 != team_2
    }
    schedule = []
    counts = Counter()

    for raw_game in raw_schedule:
        team_1 = raw_game["team1"]
        team_2 = raw_game["team2"]
        team_1_conference = memberships.get(team_1) or raw_game.get("team1_conference")
        team_2_conference = memberships.get(team_2) or raw_game.get("team2_conference")
        if team_1 in team_set:
            counts[team_1] += 1
        if team_2 in team_set:
            counts[team_2] += 1
        pair_key = tuple(sorted([team_1, team_2]))
        schedule_game = {
            "event_id": raw_game.get("event_id"),
            "date": raw_game.get("date"),
            "week": raw_game.get("week"),
            "owner_conference": raw_game.get("owner_conference"),
            "team1": team_1,
            "team2": team_2,
            "score": _matchup_score(
                team_1,
                team_2,
                rank_map,
                team_1_conference,
                team_2_conference,
            ),
            "protected": pair_key in protected_pair_keys,
            "rivalry_week_priority": _rivalry_week_priority(team_1, team_2),
            "neutral_site": bool(raw_game.get("neutral_site", False)),
            "conference_game": bool(raw_game.get("conference_game", True)),
            "team1_conference": team_1_conference,
            "team2_conference": team_2_conference,
        }
        _copy_cfbd_tv_fields(raw_game, schedule_game)
        schedule.append(schedule_game)

    for team in normalized_teams:
        counts.setdefault(team, 0)

    rows = []
    for idx, game in enumerate(
        sorted(schedule, key=lambda row: (row.get("week") or 99, row.get("date") or "", row["team1"], row["team2"]))
    ):
        row = {
            "game_number": idx + 1,
            "event_id": game.get("event_id"),
            "date": game.get("date"),
            "week": game.get("week"),
            "team1": game["team1"],
            "team2": game["team2"],
            "matchup": f"{game['team1']} vs {game['team2']}",
            "rank1": rank_map.get(game["team1"], 0),
            "rank2": rank_map.get(game["team2"], 0),
            "network": "Not nationally rated",
            "time_slot": "No national TV window",
            "comp_tier1": None,
            "tv_tier": "unrated",
            "nationally_rated": False,
            "matchup_score": float(round(game["score"], 4)),
            "protected": bool(game.get("protected", False)),
            "actual_schedule": True,
            "conference_game": bool(game.get("conference_game", True)),
            "owner_conference": game.get("owner_conference"),
            "team1_conference": game.get("team1_conference"),
            "team2_conference": game.get("team2_conference"),
            "predicted_viewers": 0.0,
            "predicted_viewers_formatted": "Not rated",
        }
        _copy_cfbd_tv_fields(game, row)
        rows.append(row)

    slate = {
        "teams": normalized_teams,
        "team_game_counts": dict(counts),
        "rank_map": rank_map,
        "rows": rows,
        "schedule_source": "actual_2026_public_schedule",
    }
    _refresh_realignment_slate_summary(slate)
    return slate


def _actual_nonconference_games_for_members(
    members,
    memberships,
    rank_map,
    max_games_per_team=12,
    used_team_weeks=None,
):
    member_set = set(members)
    counts = Counter()
    selected = []
    selected_event_keys = set()
    used_team_weeks = set(used_team_weeks or set())

    raw_candidates = []
    for raw_game in _actual_fbs_schedule_rows():
        team_1 = raw_game["team1"]
        team_2 = raw_game["team2"]
        team_1_in = team_1 in member_set
        team_2_in = team_2 in member_set
        if not (team_1_in or team_2_in):
            continue
        team_1_conference = memberships.get(team_1) or raw_game.get("team1_conference")
        team_2_conference = memberships.get(team_2) or raw_game.get("team2_conference")
        if team_1_in and team_2_in and team_1_conference == team_2_conference:
            continue
        score = _matchup_score(
            team_1,
            team_2,
            rank_map,
            team_1_conference,
            team_2_conference,
        )
        candidate_game = {
            "event_id": raw_game.get("event_id"),
            "date": raw_game.get("date"),
            "week": raw_game.get("week"),
            "owner_conference": raw_game.get("owner_conference"),
            "team1": team_1,
            "team2": team_2,
            "score": score,
            "protected": False,
            "rivalry_week_priority": _rivalry_week_priority(team_1, team_2),
            "neutral_site": bool(raw_game.get("neutral_site", False)),
            "conference_game": False,
            "team1_conference": team_1_conference,
            "team2_conference": team_2_conference,
        }
        _copy_cfbd_tv_fields(raw_game, candidate_game)
        raw_candidates.append(candidate_game)

    for game in sorted(
        raw_candidates,
        key=lambda row: (
            int(row.get("week") or 99),
            str(row.get("date") or ""),
            -int(row.get("rivalry_week_priority") or 0),
            -float(row.get("score") or 0.0),
            row["team1"],
            row["team2"],
        ),
    ):
        involved_members = [team for team in [game["team1"], game["team2"]] if team in member_set]
        if any(counts[team] >= max_games_per_team for team in involved_members):
            continue
        game_week = int(game.get("week") or 0)
        if game_week and any((team, game_week) in used_team_weeks for team in involved_members):
            continue
        event_key = _realignment_event_key(game)
        if event_key in selected_event_keys:
            continue
        selected_event_keys.add(event_key)
        selected.append(game)
        for team in involved_members:
            counts[team] += 1
            if game_week:
                used_team_weeks.add((team, game_week))

    return selected, counts


def _filter_team_week_schedule_conflicts(schedule, member_set):
    selected = []
    used_team_weeks = set()
    for game in sorted(
        schedule,
        key=lambda row: (
            not bool(row.get("event_id")),
            int(row.get("week") or 99),
            -float(row.get("score") or 0.0),
            row.get("team1", ""),
            row.get("team2", ""),
        ),
    ):
        game_week = int(game.get("week") or 0)
        involved_members = [
            team for team in [game.get("team1"), game.get("team2")]
            if team in member_set
        ]
        if game_week and any((team, game_week) in used_team_weeks for team in involved_members):
            continue
        selected.append(game)
        if game_week:
            for team in involved_members:
                used_team_weeks.add((team, game_week))
    return selected


def _remove_cross_slate_team_week_conflicts(slates):
    entries = []
    team_targets = {}
    team_counts = Counter()
    for conference, slate in slates.items():
        for team in slate.get("teams", []):
            team_targets[team] = 12
        for row in slate.get("rows", []):
            entries.append((conference, row))

    kept_ids = set()
    used_team_weeks = {}
    schedule_weeks = [
        *REALIGNMENT_EARLY_CONFERENCE_WEEKS,
        *REALIGNMENT_PRIMARY_CONFERENCE_WEEKS,
    ]

    def row_team_week_keys(row, week=None):
        week = int(week if week is not None else (row.get("week") or 0))
        teams = [normalize_team(row.get("team1")), normalize_team(row.get("team2"))]
        return [(team, week) for team in teams if team and week]

    def row_has_conflict(row, week=None):
        return any(key in used_team_weeks for key in row_team_week_keys(row, week))

    def reserve_row(row):
        for key in row_team_week_keys(row):
            used_team_weeks.setdefault(key, row)
        for team in [normalize_team(row.get("team1")), normalize_team(row.get("team2"))]:
            if team in team_targets:
                team_counts[team] += 1

    def reassign_generated_row(row):
        if row.get("event_id"):
            return False
        original_week = int(row.get("week") or 99)
        candidate_weeks = sorted(
            schedule_weeks,
            key=lambda week: (
                abs(int(week) - original_week),
                int(week),
            ),
        )
        for week in candidate_weeks:
            if row_has_conflict(row, week):
                continue
            row["week"] = int(week)
            row["date"] = None
            return True
        return False

    dropped_generated_rows = []
    for conference, row in sorted(
        entries,
        key=lambda item: (
            not bool(item[1].get("event_id")),
            int(item[1].get("week") or 99),
            -float(item[1].get("matchup_score") or 0.0),
            item[1].get("matchup", ""),
        ),
    ):
        conflicting_rows = [
            used_team_weeks[key]
            for key in row_team_week_keys(row)
            if key in used_team_weeks
        ]
        same_actual_duplicate = (
            row.get("event_id")
            and conflicting_rows
            and all(conflict.get("event_id") == row.get("event_id") for conflict in conflicting_rows)
        )
        if conflicting_rows and not same_actual_duplicate:
            if not reassign_generated_row(row):
                if not row.get("event_id"):
                    dropped_generated_rows.append(row)
                continue
        kept_ids.add(id(row))
        reserve_row(row)

    def row_schedule_deficit(row):
        deficits = [
            team_targets.get(team, 0) - team_counts.get(team, 0)
            for team in [normalize_team(row.get("team1")), normalize_team(row.get("team2"))]
            if team in team_targets
        ]
        return min(deficits) if deficits else -99

    for _ in range(len(dropped_generated_rows)):
        made_progress = False
        dropped_generated_rows = sorted(
            dropped_generated_rows,
            key=lambda row: (
                -row_schedule_deficit(row),
                -float(row.get("matchup_score") or 0.0),
                row.get("matchup", ""),
            ),
        )
        still_dropped = []
        for row in dropped_generated_rows:
            teams = [
                team for team in [normalize_team(row.get("team1")), normalize_team(row.get("team2"))]
                if team in team_targets
            ]
            if not teams or any(team_counts.get(team, 0) >= team_targets.get(team, 12) for team in teams):
                still_dropped.append(row)
                continue
            if not reassign_generated_row(row):
                still_dropped.append(row)
                continue
            kept_ids.add(id(row))
            reserve_row(row)
            made_progress = True
        dropped_generated_rows = still_dropped
        if not made_progress:
            break

    for slate in slates.values():
        slate["rows"] = [
            row for row in slate.get("rows", [])
            if id(row) in kept_ids
        ]
        _refresh_realignment_slate_summary(slate)


def _fill_short_generated_conference_games(slates):
    used_team_weeks = set()
    for slate in slates.values():
        for row in slate.get("rows", []):
            week = int(row.get("week") or 0)
            if not week:
                continue
            for team in [normalize_team(row.get("team1")), normalize_team(row.get("team2"))]:
                if team:
                    used_team_weeks.add((team, week))

    schedule_weeks = [
        *REALIGNMENT_EARLY_CONFERENCE_WEEKS,
        *REALIGNMENT_PRIMARY_CONFERENCE_WEEKS,
    ]

    for conference, slate in slates.items():
        teams = list(slate.get("teams", []))
        if len(teams) < 2:
            continue
        rank_map = slate.get("rank_map", {})
        counts = Counter({team: 0 for team in teams})
        existing_pairs = set()
        for row in slate.get("rows", []):
            team_1 = normalize_team(row.get("team1"))
            team_2 = normalize_team(row.get("team2"))
            if team_1 in counts:
                counts[team_1] += 1
            if team_2 in counts:
                counts[team_2] += 1
            if team_1 and team_2:
                existing_pairs.add(tuple(sorted([team_1, team_2])))

        for _ in range(len(teams) * 12):
            possible_additions = []
            short_teams = [team for team in teams if counts[team] < 12]
            for idx, team_1 in enumerate(short_teams):
                for team_2 in short_teams[idx + 1:]:
                    pair_key = tuple(sorted([team_1, team_2]))
                    if pair_key in existing_pairs:
                        continue
                    available_weeks = [
                        week for week in schedule_weeks
                        if (team_1, week) not in used_team_weeks
                        and (team_2, week) not in used_team_weeks
                    ]
                    if not available_weeks:
                        continue
                    week = sorted(
                        available_weeks,
                        key=lambda number: (
                            int(number not in REALIGNMENT_PRIMARY_CONFERENCE_WEEKS),
                            number,
                        ),
                    )[0]
                    score = _matchup_score(team_1, team_2, rank_map, conference, conference)
                    possible_additions.append((
                        min(12 - counts[team_1], 12 - counts[team_2]),
                        (12 - counts[team_1]) + (12 - counts[team_2]),
                        score,
                        team_1,
                        team_2,
                        week,
                    ))
            if not possible_additions:
                for team_1 in short_teams:
                    for team_2 in teams:
                        if team_2 == team_1 or counts[team_2] >= 13:
                            continue
                        pair_key = tuple(sorted([team_1, team_2]))
                        if pair_key in existing_pairs:
                            continue
                        available_weeks = [
                            week for week in schedule_weeks
                            if (team_1, week) not in used_team_weeks
                            and (team_2, week) not in used_team_weeks
                        ]
                        if not available_weeks:
                            continue
                        week = sorted(
                            available_weeks,
                            key=lambda number: (
                                int(number not in REALIGNMENT_PRIMARY_CONFERENCE_WEEKS),
                                number,
                            ),
                        )[0]
                        score = _matchup_score(team_1, team_2, rank_map, conference, conference)
                        possible_additions.append((
                            12 - counts[team_1],
                            int(counts[team_2] < 12),
                            score,
                            team_1,
                            team_2,
                            week,
                        ))
            if not possible_additions:
                break
            _, _, score, team_1, team_2, week = sorted(
                possible_additions,
                key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4], item[5]),
            )[0]
            score = _matchup_score(team_1, team_2, rank_map, conference, conference)
            row = {
                "game_number": len(slate.get("rows", [])) + 1,
                "event_id": None,
                "date": None,
                "week": int(week),
                "team1": team_1,
                "team2": team_2,
                "matchup": f"{team_1} vs {team_2}",
                "rank1": rank_map.get(team_1, 0),
                "rank2": rank_map.get(team_2, 0),
                "network": "Not nationally rated",
                "time_slot": "No national TV window",
                "comp_tier1": None,
                "tv_tier": "unrated",
                "nationally_rated": False,
                "matchup_score": float(round(score, 4)),
                "protected": False,
                "actual_schedule": False,
                "generated_conference_game": True,
                "conference_game": True,
                "owner_conference": conference,
                "team1_conference": conference,
                "team2_conference": conference,
                "predicted_viewers": 0.0,
                "predicted_viewers_formatted": "Not rated",
            }
            slate.setdefault("rows", []).append(row)
            existing_pairs.add(tuple(sorted([team_1, team_2])))
            used_team_weeks.add((team_1, week))
            used_team_weeks.add((team_2, week))
            counts[team_1] += 1
            counts[team_2] += 1

        slate["rows"] = sorted(
            slate.get("rows", []),
            key=lambda row: (row.get("week") or 99, row.get("date") or "", row.get("team1") or "", row.get("team2") or ""),
        )
        for idx, row in enumerate(slate["rows"], start=1):
            row["game_number"] = idx
        _refresh_realignment_slate_summary(slate)


def _generate_conference_games_around_actual(members, conference, rank_map, protected_pairs, actual_games):
    member_set = set(members)
    schedule = list(actual_games)
    counts = Counter({team: 0 for team in members})

    for game in schedule:
        for team in [game.get("team1"), game.get("team2")]:
            if team in member_set:
                counts[team] += 1

    if sum(max(0, 12 - counts[team]) for team in members) % 2:
        removable_games = []
        for idx, game in enumerate(schedule):
            involved_members = [team for team in [game.get("team1"), game.get("team2")] if team in member_set]
            if not involved_members:
                continue
            score = _matchup_score(
                game.get("team1"),
                game.get("team2"),
                rank_map,
                game.get("team1_conference") or "",
                game.get("team2_conference") or "",
            )
            removable_games.append((score, int(game.get("week") or 99), idx, involved_members))
        if removable_games:
            _, _, remove_idx, involved_members = sorted(removable_games)[0]
            schedule.pop(remove_idx)
            for team in involved_members:
                counts[team] -= 1

    schedule_weeks = [
        *REALIGNMENT_PRIMARY_CONFERENCE_WEEKS,
        *REALIGNMENT_EARLY_CONFERENCE_WEEKS,
    ]
    protected_pair_keys = [
        tuple(sorted([team_1, team_2]))
        for team_1, team_2 in (protected_pairs or [])
        if team_1 in member_set and team_2 in member_set and team_1 != team_2
    ]

    def base_state():
        used_team_weeks = set()
        existing_pairs = set()
        base_counts = Counter({team: 0 for team in members})
        for game in schedule:
            week = int(game.get("week") or 0)
            team_1 = game.get("team1")
            team_2 = game.get("team2")
            for team in [team_1, team_2]:
                if team in member_set:
                    base_counts[team] += 1
                    if week:
                        used_team_weeks.add((team, week))
            if team_1 in member_set and team_2 in member_set:
                existing_pairs.add(tuple(sorted([team_1, team_2])))
        return list(schedule), base_counts, used_team_weeks, existing_pairs

    def available_week(team_1, team_2, used_team_weeks, preferred_late=False):
        weeks = sorted(
            schedule_weeks,
            key=lambda week: (
                -week if preferred_late else int(week not in REALIGNMENT_PRIMARY_CONFERENCE_WEEKS),
                week if not preferred_late else 0,
            ),
        )
        for week in weeks:
            if (team_1, week) not in used_team_weeks and (team_2, week) not in used_team_weeks:
                return week
        return None

    def add_generated_game(local_schedule, local_counts, used_team_weeks, existing_pairs, team_1, team_2, protected=False):
        pair_key = tuple(sorted([team_1, team_2]))
        if pair_key in existing_pairs:
            return False
        if local_counts[team_1] >= 12 or local_counts[team_2] >= 12:
            return False
        score = _matchup_score(team_1, team_2, rank_map, conference, conference)
        local_schedule.append({
            "event_id": None,
            "date": None,
            "week": None,
            "owner_conference": conference,
            "team1": team_1,
            "team2": team_2,
            "score": score,
            "protected": protected,
            "rivalry_week_priority": _rivalry_week_priority(team_1, team_2),
            "neutral_site": False,
            "conference_game": True,
            "team1_conference": conference,
            "team2_conference": conference,
        })
        existing_pairs.add(pair_key)
        local_counts[team_1] += 1
        local_counts[team_2] += 1
        return True

    def assign_generated_weeks(local_schedule, attempt):
        unavailable_team_weeks = set()
        generated_games = []
        for idx, game in enumerate(local_schedule):
            team_1 = game.get("team1")
            team_2 = game.get("team2")
            if game.get("conference_game") and game.get("event_id") is None:
                generated_games.append({**game})
                game["week"] = None
                continue
            week = int(game.get("week") or 0)
            if not week:
                continue
            for team in [team_1, team_2]:
                if team in member_set:
                    unavailable_team_weeks.add((team, week))
        if not generated_games:
            return True
        assigned_games = _assign_schedule_weeks(
            generated_games,
            members,
            9,
            unavailable_team_weeks=unavailable_team_weeks,
        )
        if len(assigned_games) != len(generated_games) or any(not game.get("week") for game in assigned_games):
            return False
        assigned_weeks = {
            tuple(sorted([game["team1"], game["team2"]])): int(game["week"])
            for game in assigned_games
        }
        used_team_weeks = set(unavailable_team_weeks)
        for game in local_schedule:
            if not (game.get("conference_game") and game.get("event_id") is None):
                continue
            pair_key = tuple(sorted([game["team1"], game["team2"]]))
            week = assigned_weeks.get(pair_key)
            if not week:
                return False
            if (game["team1"], week) in used_team_weeks or (game["team2"], week) in used_team_weeks:
                return False
            game["week"] = int(week)
            used_team_weeks.add((game["team1"], week))
            used_team_weeks.add((game["team2"], week))
        return True

    def viable_pairs(local_counts, used_team_weeks, existing_pairs):
        pairs = []
        short_teams = [team for team in members if local_counts[team] < 12]
        for idx, team_1 in enumerate(short_teams):
            for team_2 in short_teams[idx + 1:]:
                pair_key = tuple(sorted([team_1, team_2]))
                if pair_key in existing_pairs:
                    continue
                pairs.append((team_1, team_2))
        return pairs

    def add_generated_game_in_week(local_schedule, local_counts, used_team_weeks, existing_pairs, team_1, team_2, week, protected=False):
        pair_key = tuple(sorted([team_1, team_2]))
        if pair_key in existing_pairs:
            return False
        if local_counts[team_1] >= 12 or local_counts[team_2] >= 12:
            return False
        if (team_1, week) in used_team_weeks or (team_2, week) in used_team_weeks:
            return False
        score = _matchup_score(team_1, team_2, rank_map, conference, conference)
        local_schedule.append({
            "event_id": None,
            "date": None,
            "week": int(week),
            "owner_conference": conference,
            "team1": team_1,
            "team2": team_2,
            "score": score,
            "protected": protected,
            "rivalry_week_priority": _rivalry_week_priority(team_1, team_2),
            "neutral_site": False,
            "conference_game": True,
            "team1_conference": conference,
            "team2_conference": conference,
        })
        existing_pairs.add(pair_key)
        used_team_weeks.add((team_1, week))
        used_team_weeks.add((team_2, week))
        local_counts[team_1] += 1
        local_counts[team_2] += 1
        return True

    def repair_two_short_teams(local_schedule, local_counts, used_team_weeks, existing_pairs):
        short_teams = [team for team in members if local_counts[team] == 11]
        if len(short_teams) != 2:
            return False
        team_1, team_2 = short_teams
        if tuple(sorted([team_1, team_2])) in existing_pairs:
            return False

        def try_move_and_add(open_team, blocked_team):
            open_weeks = [
                week for week in schedule_weeks
                if (open_team, week) not in used_team_weeks
            ]
            blocked_open_weeks = [
                week for week in schedule_weeks
                if (blocked_team, week) not in used_team_weeks
            ]
            for add_week in open_weeks:
                blocker_indexes = [
                    idx for idx, game in enumerate(local_schedule)
                    if game.get("conference_game")
                    and game.get("event_id") is None
                    and int(game.get("week") or 0) == int(add_week)
                    and blocked_team in [game.get("team1"), game.get("team2")]
                ]
                for idx in blocker_indexes:
                    blocker = local_schedule[idx]
                    opponent = blocker["team2"] if blocker["team1"] == blocked_team else blocker["team1"]
                    for move_week in blocked_open_weeks:
                        if (opponent, move_week) in used_team_weeks:
                            continue
                        old_week = int(blocker["week"])
                        used_team_weeks.remove((blocked_team, old_week))
                        used_team_weeks.remove((opponent, old_week))
                        blocker["week"] = int(move_week)
                        used_team_weeks.add((blocked_team, move_week))
                        used_team_weeks.add((opponent, move_week))
                        if add_generated_game_in_week(
                            local_schedule,
                            local_counts,
                            used_team_weeks,
                            existing_pairs,
                            open_team,
                            blocked_team,
                            add_week,
                        ):
                            return True
                        used_team_weeks.remove((blocked_team, move_week))
                        used_team_weeks.remove((opponent, move_week))
                        blocker["week"] = old_week
                        used_team_weeks.add((blocked_team, old_week))
                        used_team_weeks.add((opponent, old_week))
            return False

        return try_move_and_add(team_1, team_2) or try_move_and_add(team_2, team_1)

    best_schedule = schedule
    best_counts = counts
    best_shortfall = sum(max(0, 12 - counts[team]) for team in members)
    for attempt in range(220):
        local_schedule, local_counts, used_team_weeks, existing_pairs = base_state()
        rng = random.Random(attempt)
        protected_order = sorted(
            protected_pair_keys,
            key=lambda pair: (-_rivalry_week_priority(pair[0], pair[1]), pair[0], pair[1]),
        )
        if attempt % 2:
            rng.shuffle(protected_order)
        for team_1, team_2 in protected_order:
            preferred_weeks = sorted(
                schedule_weeks,
                key=lambda week: (
                    (team_1, week) in used_team_weeks or (team_2, week) in used_team_weeks,
                    -week,
                ),
            )
            for week in preferred_weeks:
                if add_generated_game_in_week(local_schedule, local_counts, used_team_weeks, existing_pairs, team_1, team_2, week, protected=True):
                    break

        week_order = list(schedule_weeks)
        if attempt % 3 == 1:
            week_order = sorted(schedule_weeks)
        elif attempt % 3 == 2:
            rng.shuffle(week_order)

        for _ in range(3):
            made_progress = False
            for week in week_order:
                while True:
                    deficits = {team: max(0, 12 - local_counts[team]) for team in members}
                    if not any(deficits.values()):
                        return local_schedule, local_counts
                    available_teams = [
                        team for team in members
                        if deficits[team] > 0 and (team, week) not in used_team_weeks
                    ]
                    candidates = []
                    for idx, team_1 in enumerate(available_teams):
                        for team_2 in available_teams[idx + 1:]:
                            pair_key = tuple(sorted([team_1, team_2]))
                            if pair_key in existing_pairs:
                                continue
                            score = _matchup_score(team_1, team_2, rank_map, conference, conference)
                            candidates.append((
                                -(deficits[team_1] + deficits[team_2]),
                                -max(deficits[team_1], deficits[team_2]),
                                -min(deficits[team_1], deficits[team_2]),
                                -score,
                                rng.random() if attempt >= 12 else 0,
                                team_1,
                                team_2,
                            ))
                    if not candidates:
                        break
                    candidates.sort()
                    candidate_pool = candidates[: min(10, len(candidates))]
                    if attempt < 12:
                        _, _, _, _, _, team_1, team_2 = candidate_pool[0]
                    elif attempt >= 80:
                        _, _, _, _, _, team_1, team_2 = rng.choice(candidates)
                    else:
                        _, _, _, _, _, team_1, team_2 = rng.choice(candidate_pool)
                    if not add_generated_game_in_week(local_schedule, local_counts, used_team_weeks, existing_pairs, team_1, team_2, week):
                        break
                    made_progress = True
            if not made_progress:
                break

        shortfall = sum(max(0, 12 - local_counts[team]) for team in members)
        if shortfall == 2 and repair_two_short_teams(local_schedule, local_counts, used_team_weeks, existing_pairs):
            return local_schedule, local_counts
        if shortfall == 0:
            return local_schedule, local_counts
        if shortfall < best_shortfall:
            best_shortfall = shortfall
            best_schedule = local_schedule
            best_counts = local_counts

    return best_schedule or schedule, best_counts or counts


def _predict_custom_realignment_slate(teams, conference, sim, protected_pairs=None, memberships=None):
    normalized_teams = [normalize_team(team) for team in teams]
    memberships = memberships or {}
    member_set = set(normalized_teams)
    actual_opponents = {
        team
        for raw_game in _actual_fbs_schedule_rows()
        if raw_game["team1"] in member_set or raw_game["team2"] in member_set
        for team in [raw_game["team1"], raw_game["team2"]]
    }
    rank_map = _build_rank_map(
        sorted(member_set | actual_opponents),
        sim.ranking_policy,
        sim.custom_rankings,
    )
    protected_pair_keys = {
        tuple(sorted([team_1, team_2]))
        for team_1, team_2 in (protected_pairs or [])
        if team_1 in member_set and team_2 in member_set and team_1 != team_2
    }
    ooc_schedule, _ = _actual_nonconference_games_for_members(
        normalized_teams,
        memberships,
        rank_map,
        max_games_per_team=12,
    )
    schedule, counts = _generate_conference_games_around_actual(
        normalized_teams,
        conference,
        rank_map,
        protected_pair_keys,
        ooc_schedule,
    )

    rows = []
    for idx, game in enumerate(
        sorted(schedule, key=lambda row: (row.get("week") or 99, row.get("date") or "", row["team1"], row["team2"]))
    ):
        row = {
            "game_number": idx + 1,
            "event_id": game.get("event_id"),
            "date": game.get("date"),
            "week": game.get("week"),
            "team1": game["team1"],
            "team2": game["team2"],
            "matchup": f"{game['team1']} vs {game['team2']}",
            "rank1": rank_map.get(game["team1"], 0),
            "rank2": rank_map.get(game["team2"], 0),
            "network": "Not nationally rated",
            "time_slot": "No national TV window",
            "comp_tier1": None,
            "tv_tier": "unrated",
            "nationally_rated": False,
            "matchup_score": float(round(game["score"], 4)),
            "protected": bool(game.get("protected", False)),
            "actual_schedule": bool(game.get("event_id")),
            "generated_conference_game": bool(game.get("conference_game", False) and not game.get("event_id")),
            "conference_game": bool(game.get("conference_game", True)),
            "owner_conference": game.get("owner_conference"),
            "team1_conference": game.get("team1_conference"),
            "team2_conference": game.get("team2_conference"),
            "predicted_viewers": 0.0,
            "predicted_viewers_formatted": "Not rated",
        }
        _copy_cfbd_tv_fields(game, row)
        rows.append(row)

    slate = {
        "teams": normalized_teams,
        "team_game_counts": dict(counts),
        "rank_map": rank_map,
        "rows": rows,
        "schedule_source": "actual_nonconference_plus_generated_conference",
    }
    _refresh_realignment_slate_summary(slate)
    return slate


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
            "custom_rankings": _normalized_custom_rankings(sim.custom_rankings) if sim.ranking_policy == "custom" else {},
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
            "ranking_policies": REALIGNMENT_RANKING_POLICIES,
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
                "custom_rankings": _normalized_custom_rankings(sim.custom_rankings) if sim.ranking_policy == "custom" else {},
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
            "custom_rankings": _normalized_custom_rankings(sim.custom_rankings) if sim.ranking_policy == "custom" else {},
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
            "ranking_policies": REALIGNMENT_RANKING_POLICIES,
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
        normalized_memberships = _default_league_memberships()

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

    def build_league_slates(memberships, protected_pair_overrides=None):
        transferred_slots, removed_slots, transfer_rows = _conference_tv_slot_transfers(memberships)
        slates = {}
        protected_pair_overrides = protected_pair_overrides or set()
        for conference in REALIGNMENT_EDITABLE_CONFERENCES:
            members = sorted(
                [
                    team
                    for team, team_conference in memberships.items()
                    if team_conference == conference
                ],
                key=lambda team: (-_football_team_coefficient(team), team),
            )
            if len(members) < 2:
                slates[conference] = {
                    "teams": members,
                    "team_game_counts": {},
                    "rank_map": {},
                    "rows": [],
                    "summary": {
                        "teams": len(members),
                        "games": 0,
                        "scheduled_games": 0,
                        "nationally_rated_games": 0,
                        "unrated_games": 0,
                        "total_viewers": 0.0,
                        "average_viewers": 0.0,
                    },
                }
                continue

            protected_rivalry_pairs = _protected_rivalry_pairs_for_teams(members)
            protected_pairs = sorted(
                set(protected_rivalry_pairs) | {
                    pair for pair in protected_pair_overrides
                    if pair[0] in members and pair[1] in members
                }
            )
            slates[conference] = _predict_custom_realignment_slate(
                members,
                conference,
                sim,
                protected_pairs=protected_pairs,
                memberships=memberships,
            )

        _apply_global_tv_slots_to_league_slates(
            slates,
            transferred_slots,
            removed_slots,
            ranking_policy=sim.ranking_policy,
            custom_rankings=sim.custom_rankings,
        )
        return slates, transferred_slots, removed_slots, transfer_rows

    conference_slates, transferred_tv_slots, removed_tv_slots, tv_slot_transfer_rows = build_league_slates(
        normalized_memberships,
        user_protected_pairs,
    )
    baseline_memberships = _default_league_memberships()
    baseline_conference_slates, *_ = build_league_slates(
        baseline_memberships,
    )
    public_conference_slates = {
        conference: _public_realignment_slate_for_conference(slate, conference)
        for conference, slate in conference_slates.items()
    }
    public_baseline_conference_slates = {
        conference: _public_realignment_slate_for_conference(slate, conference)
        for conference, slate in baseline_conference_slates.items()
    }

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
        conference: _realignment_summary_for_rows(
            public_conference_slates.get(conference, {}).get("rows", []),
            team_count=public_conference_slates.get(conference, {}).get("summary", {}).get("teams", 0),
        )
        for conference in REALIGNMENT_EDITABLE_CONFERENCES
    }
    baseline_rows = [
        {
            **row,
            "conference": conference,
        }
        for conference, slate in public_baseline_conference_slates.items()
        for row in slate.get("rows", [])
    ]
    baseline_conference_summaries = {
        conference: _realignment_summary_for_rows(
            public_baseline_conference_slates.get(conference, {}).get("rows", []),
            team_count=public_baseline_conference_slates.get(conference, {}).get("summary", {}).get("teams", 0),
        )
        for conference in REALIGNMENT_EDITABLE_CONFERENCES
    }
    conference_membership_changes = {
        conference: {
            "current_teams": int(conference_slates.get(conference, {}).get("summary", {}).get("teams", 0)),
            "baseline_teams": int(baseline_conference_slates.get(conference, {}).get("summary", {}).get("teams", 0)),
            "team_delta": int(conference_slates.get(conference, {}).get("summary", {}).get("teams", 0))
            - int(baseline_conference_slates.get(conference, {}).get("summary", {}).get("teams", 0)),
            "pct_delta": None
            if int(baseline_conference_slates.get(conference, {}).get("summary", {}).get("teams", 0)) == 0
            else (
                (
                    int(conference_slates.get(conference, {}).get("summary", {}).get("teams", 0))
                    - int(baseline_conference_slates.get(conference, {}).get("summary", {}).get("teams", 0))
                )
                / int(baseline_conference_slates.get(conference, {}).get("summary", {}).get("teams", 0))
            ) * 100,
        }
        for conference in REALIGNMENT_EDITABLE_CONFERENCES
    }
    conference_distribution_metrics = {
        conference: _distribution_metric_comparison(
            public_conference_slates.get(conference, {}).get("rows", []),
            public_baseline_conference_slates.get(conference, {}).get("rows", []),
            conference_membership_changes[conference]["pct_delta"],
        )
        for conference in REALIGNMENT_EDITABLE_CONFERENCES
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
            "custom_rankings": _normalized_custom_rankings(sim.custom_rankings) if sim.ranking_policy == "custom" else {},
            "conferences": REALIGNMENT_EDITABLE_CONFERENCES,
            "tv_slot_transfers": tv_slot_transfer_rows,
            "schedule_source": "actual_nonconference_plus_generated_conference",
        },
        "memberships": normalized_memberships,
        "conference_slates": conference_slates,
        "conference_summaries": conference_summaries,
        "baseline_conference_summaries": baseline_conference_summaries,
        "conference_membership_changes": conference_membership_changes,
        "conference_distribution_metrics": conference_distribution_metrics,
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
            "Generates conference schedules for both the baseline and edited memberships "
            "by preserving actual non-conference games and generating the remaining "
            "conference games from the current conference membership. "
            "The national weekly "
            "TV draft board then selects games by projected viewership for each network/window. "
            "Slots are constrained by media-rights eligibility and can remain unused by "
            "conference games when the available matchups fall below that slot's TV threshold."
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


@app.get("/game-viewership-rankings")
def game_viewership_rankings(
    season: str = "all",
    conference: str = "all",
    network: str = "all",
    time_bucket: str = "all",
    team: str = "all",
    rank_bucket: str = "all",
    include_conf_champ: bool = True,
):
    team_aliases = {
        "Kennesaw St.": "Kennesaw State",
    }
    normalized_team = normalize_team(team) if team != "all" else "all"
    rows = []

    for _, row in df_all.iterrows():
        viewers = pd.to_numeric(row.get("Persons 2+"), errors="coerce")
        year = row.get("Year")
        if pd.isna(viewers) or pd.isna(year):
            continue

        if season != "all" and str(int(year)) != str(season):
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
        team_1_conference = team_conferences.get(team_1, "Independent")
        team_2_conference = team_conferences.get(team_2, "Independent")
        if conference != "all" and conference not in {team_1_conference, team_2_conference}:
            continue
        if normalized_team != "all" and normalized_team not in {team_1, team_2}:
            continue

        team_1_display = team_aliases.get(str(team_1), str(team_1)) if pd.notna(team_1) else ""
        team_2_display = team_aliases.get(str(team_2), str(team_2)) if pd.notna(team_2) else ""
        rows.append({
            "date": row.get("Date"),
            "season": int(year),
            "matchup": f"{team_1_display} vs {team_2_display}",
            "team1": team_1_display,
            "team2": team_2_display,
            "team1_conference": team_1_conference,
            "team2_conference": team_2_conference,
            "network": row.get("scenario_network"),
            "time_slot": row.get("scenario_time_bucket"),
            "raw_time_slot": row.get("Time Slot"),
            "rank_bucket": row.get("scenario_rank_detail"),
            "competing_bucket": row.get("scenario_competing_bucket"),
            "conference_championship": bool(row.get("scenario_conf_champ") == 1),
            "viewers": float(viewers),
        })

    rows = sorted(
        rows,
        key=lambda game: (
            -(game["viewers"] or 0),
            -int(game["season"] or 0),
            str(game["date"] or ""),
            game["matchup"],
        ),
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return clean_nan({
        "rows": rows,
        "filters": {
            "season": season,
            "conference": conference,
            "network": network,
            "time_bucket": time_bucket,
            "team": team,
            "rank_bucket": rank_bucket,
            "include_conf_champ": include_conf_champ,
        },
        "available_filters": {
            "seasons": ["all"] + [str(year) for year in sorted(pd.to_numeric(df_all["Year"], errors="coerce").dropna().astype(int).unique().tolist(), reverse=True)],
            "conferences": ["all"] + sorted(set(team_conferences.values()) | {"Independent"}),
            "networks": ["all", "ABC", "CBS", "NBC", "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "FS2", "BTN", "CW", "NFLN", "ESPNNEWS"],
            "time_buckets": ["all", "Primetime", "Sat Early", "Sat Mid", "Sat Late", "Friday", "Black Friday", "Monday", "Sunday", "Weekday", "Other"],
            "rank_buckets": rank_detail_options,
            "teams": ["all"] + sorted(
                team for team in {
                    *(team_aliases.get(str(value), str(value)) for value in df_all["Team 1"].dropna().unique()),
                    *(team_aliases.get(str(value), str(value)) for value in df_all["Team 2"].dropna().unique()),
                }
                if team and team not in excluded_viewership_ranking_teams
            ),
        },
    })

# WEEKLY PREDICTIONS

def _weekly_prediction_input(game, document_week, document_season_week=None):
    """Attach document-level week context to a prediction-only game copy."""
    prediction_input = dict(game)
    prediction_input["week"] = document_week
    prediction_input["season_week"] = (
        document_season_week
        if document_season_week is not None
        else document_week
    )
    return prediction_input

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
        season_week = data.get("season_week")
        year = data["year"]
        games = data["games"]
        response_games = []

        updated = False  # If anything changes, we write back

        for g in games:
            prediction_input = _weekly_prediction_input(g, week, season_week)
            prediction_warnings = pregame_prediction_warnings(prediction_input)

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
                g["predicted"] = generate_pregame_prediction(prediction_input)

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
                    post_pred_str = generate_postgame_prediction(prediction_input)
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

            response_game = dict(g)
            if prediction_warnings:
                existing_warnings = response_game.get("warnings", [])
                if isinstance(existing_warnings, str):
                    existing_warnings = [existing_warnings]
                elif not isinstance(existing_warnings, list):
                    existing_warnings = (
                        list(existing_warnings) if existing_warnings else []
                    )
                response_game["warnings"] = list(dict.fromkeys([
                    *existing_warnings,
                    *prediction_warnings,
                ]))
            response_games.append(response_game)

        # =====================================================
        # 🔥 WRITE BACK TO FIRESTORE IF ANY FIELD CHANGED
        # =====================================================
        if updated:
            db.collection("weekly-predictions").document(doc.id).set(data)

        weeks_output.append({
            "week": week,
            "year": year,
            "games": response_games
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

    weeks_output = sorted(
        weeks_output,
        key=lambda w: (int(w.get("year") or 0), int(w["week"])),
        reverse=True,
    )

    return clean_nan({
        "weeks": weeks_output,
        "metrics": metrics
    })
