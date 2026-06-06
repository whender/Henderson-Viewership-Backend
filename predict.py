import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from model_loader import load_viewership_model

# Load once
model = load_viewership_model()

# These must match your Streamlit code:
teams_list = {
    "Air Force": "Air Force", "Akron": "Akron", "Alabama": "Alabama",
    "App State": "Appalachian St.", "Arizona": "Arizona", "Arizona State": "Arizona St.",
    "Arkansas": "Arkansas", "Arkansas State": "Arkansas St.", "Army": "Army",
    "Auburn": "Auburn", "Ball State": "Ball St.", "Baylor": "Baylor",
    "Boise State": "Boise St.", "Boston College": "Boston College", "Bowling Green": "Bowling Green",
    "Buffalo": "Buffalo", "BYU": "BYU", "California": "California", "Central Michigan": "Central Michigan",
    "Charlotte": "Charlotte", "Cincinnati": "Cincinnati", "Clemson": "Clemson",
    "Coastal Carolina": "Coastal Carolina", "Colorado": "Colorado", "Colorado State": "Colorado St.",
    "Duke": "Duke", "East Carolina": "East Carolina", "Eastern Michigan": "Eastern Michigan",
    "Florida": "Florida", "Florida Atlantic": "FAU", "Florida International": "FIU",
    "Florida State": "Florida St.", "Fresno State": "Fresno St.", "Georgia": "Georgia",
    "Georgia Southern": "Georgia Southern", "Georgia State": "Georgia St.", "Georgia Tech": "Georgia Tech",
    "Hawai'i": "Hawaii", "Houston": "Houston", "Illinois": "Illinois", "Indiana": "Indiana",
    "Iowa": "Iowa", "Iowa State": "Iowa St.", "Kansas": "Kansas", "Kansas State": "Kansas St.",
    "Jacksonville State": "Jacksonville St.", "James Madison": "James Madison",
    "Kent State": "Kent St.", "Kentucky": "Kentucky", "Kennesaw State": "Kennesaw St.",
    "Liberty": "Liberty", "Louisiana": "Louisiana", "Louisiana Monroe": "Louisiana Monroe",
    "Louisiana Tech": "Louisiana Tech", "Louisville": "Louisville",
    "LSU": "LSU", "Marshall": "Marshall", "Maryland": "Maryland", "Memphis": "Memphis",
    "Miami": "Miami", "Miami Ohio": "Miami Ohio", "Michigan": "Michigan",
    "Michigan State": "Michigan St.", "Middle Tennessee": "Middle Tennessee St.",
    "Minnesota": "Minnesota", "Mississippi State": "Mississippi St.", "Missouri": "Missouri",
    "Missouri State": "Missouri State", "Navy": "Navy",
    "NC State": "North Carolina St.", "Nebraska": "Nebraska", "Nevada": "Nevada",
    "New Mexico": "New Mexico", "New Mexico State": "New Mexico St.", "Northern Illinois": "Northern Illinois",
    "North Carolina": "North Carolina", "North Texas": "North Texas", "Northwestern": "Northwestern",
    "Notre Dame": "Notre Dame", "Ohio": "Ohio", "Ohio State": "Ohio St.", "Old Dominion": "Old Dominion",
    "Oklahoma": "Oklahoma",
    "Oklahoma State": "Oklahoma St.", "Ole Miss": "Mississippi", "Oregon": "Oregon", "Oregon State": "Oregon St.",
    "Penn State": "Penn St.", "Pittsburgh": "Pittsburgh", "Purdue": "Purdue",
    "Rutgers": "Rutgers", "Sam Houston": "Sam Houston", "San Diego State": "San Diego St.",
    "San Jose State": "San Jose St.", "SMU": "SMU",
    "Rice": "Rice", "South Alabama": "South Alabama", "South Carolina": "South Carolina",
    "South Florida": "South Florida", "Southern Miss": "Southern Miss", "Stanford": "Stanford",
    "Syracuse": "Syracuse", "Temple": "Temple",
    "TCU": "TCU", "Tennessee": "Tennessee", "Texas": "Texas", "Texas A&M": "Texas A&M",
    "Texas State": "Texas St.", "Texas Tech": "Texas Tech", "Toledo": "Toledo", "Troy": "Troy",
    "Tulane": "Tulane", "Tulsa": "Tulsa", "UAB": "UAB", "UCF": "UCF", "UCLA": "UCLA",
    "UConn": "Connecticut", "UMass": "Massachusetts", "UNLV": "UNLV",
    "USC": "USC", "Utah": "Utah", "Utah State": "Utah St.", "Vanderbilt": "Vanderbilt",
    "Virginia": "Virginia", "Virginia Tech": "Virginia Tech", "Wake Forest": "Wake Forest",
    "Washington": "Washington", "Washington State": "Washington St.", "West Virginia": "West Virginia",
    "Western Kentucky": "Western Kentucky", "Western Michigan": "Western Michigan",
    "Wisconsin": "Wisconsin", "Wyoming": "Wyoming", "Delaware": "Delaware", "UTEP": "UTEP",
    "UTSA": "UTSA"
}

def normalize_team(name):
    """
    Convert frontend team names (e.g., 'Ohio State') into the exact
    strings used in the model (e.g., 'Ohio St.').
    """
    if not name:
        return name

    # If already a model name, keep it
    if name in teams_list.values():
        return name

    # If in the frontend dictionary, convert it
    if name in teams_list:
        return teams_list[name]

    return name

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

    # AAC
    "Charlotte": "AAC", "East Carolina": "AAC", "FAU": "AAC", "Memphis": "AAC",
    "North Texas": "AAC", "Rice": "AAC", "South Florida": "AAC", "Temple": "AAC",
    "Tulane": "AAC", "Tulsa": "AAC", "UAB": "AAC", "UTSA": "AAC", "Army": "AAC",
    "Navy": "AAC",

    # Sun Belt
    "Appalachian St.": "Sun Belt", "Arkansas St.": "Sun Belt", "Coastal Carolina": "Sun Belt",
    "Georgia Southern": "Sun Belt", "Georgia St.": "Sun Belt", "James Madison": "Sun Belt",
    "Louisiana": "Sun Belt", "Louisiana Monroe": "Sun Belt", "Marshall": "Sun Belt",
    "Old Dominion": "Sun Belt", "South Alabama": "Sun Belt", "Southern Miss": "Sun Belt",
    "Texas St.": "Sun Belt", "Troy": "Sun Belt",

    # Mountain West
    "Air Force": "Mountain West", "Boise St.": "Mountain West", "Colorado St.": "Mountain West",
    "Fresno St.": "Mountain West", "Hawaii": "Mountain West", "Nevada": "Mountain West",
    "New Mexico": "Mountain West", "San Diego St.": "Mountain West", "San Jose St.": "Mountain West",
    "UNLV": "Mountain West", "Utah St.": "Mountain West", "Wyoming": "Mountain West",

    # MAC
    "Akron": "MAC", "Ball St.": "MAC", "Bowling Green": "MAC", "Buffalo": "MAC",
    "Central Michigan": "MAC", "Eastern Michigan": "MAC", "Kent St.": "MAC",
    "Miami Ohio": "MAC", "Northern Illinois": "MAC", "Ohio": "MAC", "Toledo": "MAC",
    "Western Michigan": "MAC",

    # Conference USA
    "Delaware": "Conference USA", "FIU": "Conference USA", "Jacksonville St.": "Conference USA",
    "Kennesaw State": "Conference USA", "Liberty": "Conference USA", "Louisiana Tech": "Conference USA",
    "Middle Tennessee St.": "Conference USA", "New Mexico St.": "Conference USA",
    "Sam Houston": "Conference USA", "UTEP": "Conference USA", "Missouri State": "Conference USA",
    "Kennesaw St.": "Conference USA", "Western Kentucky": "Conference USA",

    # Independents / others
    "Massachusetts": "Independent", "Notre Dame": "Independent", "Connecticut": "Independent",
    "Washington St.": "Pac-12", "Oregon St.": "Pac-12"
}

rivalries = {
    "Michigan_OhioSt": ("Michigan", "Ohio St."), "Texas_Oklahoma": ("Texas", "Oklahoma"),
    "Alabama_Auburn": ("Alabama", "Auburn"), "Georgia_Florida": ("Georgia", "Florida"),
    "NotreDame_USC": ("Notre Dame", "USC"), "Florida_Tennessee": ("Florida", "Tennessee"),
    "Oregon_Washington": ("Oregon", "Washington"), "BYU_Utah": ("BYU", "Utah"),
    "Iowa_IowaSt": ("Iowa", "Iowa St."), "OleMiss_MississippiSt": ("Mississippi", "Mississippi St."),
    "Clemson_SouthCarolina": ("Clemson", "South Carolina"), "Arizona_ArizonaSt": ("Arizona", "Arizona St."),
    "Miami_FloridaSt": ("Miami", "Florida St."), "Texas_TexasA&M": ("Texas", "Texas A&M"),
    "Oregon_OregonSt": ("Oregon", "Oregon St."), "USC_UCLA": ("USC", "UCLA"),
    "Louisville_Kentucky": ("Louisville", "Kentucky"), "Washington_WashingtonSt": ("Washington", "Washington St."),
    "Kansas_KansasSt": ("Kansas", "Kansas St."), "Minnesota_Wisconsin": ("Minnesota", "Wisconsin"),
    "Army_Navy": ("Army", "Navy"), "Army_AirForce": ("Army", "Air Force"),
    "Navy_AirForce": ("Navy", "Air Force"), "OhioSt_PennSt": ("Ohio St.", "Penn St."),
    "Alabama_LSU": ("Alabama", "LSU"), "Vanderbilt_Tennessee": ("Vanderbilt", "Tennessee"),
    "Georgia_GeorgiaTech": ("Georgia", "Georgia Tech"), "Alabama_Georgia": ("Alabama", "Georgia")
}

FEUD_START = datetime(2025, 10, 30)
FEUD_END = datetime(2025, 11, 13)


def black_friday_date(year):
    first_day = datetime(int(year), 11, 1)
    first_thursday = first_day + timedelta(days=(3 - first_day.weekday()) % 7)
    return (first_thursday + timedelta(days=22)).date()


def labor_day_date(year):
    september_first = datetime(int(year), 9, 1)
    return (september_first + timedelta(days=(0 - september_first.weekday()) % 7)).date()


def is_black_friday_slot(time_slot, date_value=None):
    if "Black Friday" in str(time_slot):
        return True
    if not date_value:
        return False
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    return pd.notna(parsed_date) and parsed_date.date() == black_friday_date(parsed_date.year)

def rank_to_coefs(r):
    if 1 <= r <= 10: return (1, 0)
    if 11 <= r <= 25: return (0, 1)
    return (0, 0)

def format_viewers(val):
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if val >= 1_000:
        return f"{val/1_000:.0f}K"
    return f"{val:.0f}"

MODEL_TEAM_NAMES = set(teams_list.values())
POWER_FOOTBALL_CONFERENCES = {"SEC", "Big 10", "ACC", "Big 12"}


def is_power_football_team(team, conference=None):
    resolved_conference = conference if conference is not None else team_conferences.get(team, "Group of 6")
    return resolved_conference in POWER_FOOTBALL_CONFERENCES or team == "Notre Dame"


def is_week_zero_power_game(team1, team2, conf1, conf2, date_value=None, time_slot=None):
    if not (is_power_football_team(team1, conf1) and is_power_football_team(team2, conf2)):
        return False
    if "Week 0" in str(time_slot):
        return True
    if not date_value:
        return False
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    return (
        pd.notna(parsed_date)
        and parsed_date.month == 8
        and parsed_date.day <= 28
    )


def is_week_one_power_game(team1, team2, conf1, conf2, date_value=None, time_slot=None):
    if not (is_power_football_team(team1, conf1) and is_power_football_team(team2, conf2)):
        return False
    if "Week 1" in str(time_slot):
        return True
    if not date_value:
        return False
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    return (
        pd.notna(parsed_date)
        and parsed_date.date() >= datetime(parsed_date.year, 8, 29).date()
        and parsed_date.date() <= labor_day_date(parsed_date.year)
    )

def predict_viewership(p):
    # Normalize the incoming names
    team1 = normalize_team(p["team1"])
    team2 = normalize_team(p["team2"])

    rank1 = p["rank1"]
    rank2 = p["rank2"]
    network = p["network"]
    time_slot = p["time_slot"]
    comp_tier1 = p.get("comp_tier1", 0)
    is_black_friday = is_black_friday_slot(time_slot, p.get("date"))
    is_friday = ("Friday" in str(time_slot)) and not is_black_friday

    conference_overrides = p.get("conference_overrides", {}) or {}
    conf1 = conference_overrides.get(team1, team_conferences.get(team1, "Group of 6"))
    conf2 = conference_overrides.get(team2, team_conferences.get(team2, "Group of 6"))
    is_power_friday = is_friday and is_power_football_team(team1, conf1) and is_power_football_team(team2, conf2)
    is_non_power_friday = is_friday and not is_power_friday
    is_week_zero_power = is_week_zero_power_game(team1, team2, conf1, conf2, p.get("date"), time_slot)
    is_week_one_power = is_week_one_power_game(team1, team2, conf1, conf2, p.get("date"), time_slot)

    both_ranked = rank1 > 0 and rank2 > 0
    same_conf = (conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"])

    t1_top10, t1_25_11 = rank_to_coefs(rank1)
    t2_top10, t2_25_11 = rank_to_coefs(rank2)

    top10 = t1_top10 + t2_top10
    rank_25_11 = t1_25_11 + t2_25_11

    # ✔️ auto rivalry match
    auto_rivalry = next(
        (r for r, (a, b) in rivalries.items() if {team1, team2} == {a, b}),
        None
    )

    # Build feature vector
    features = {
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
        "Friday": int(is_friday),
        "Friday Power": int(is_power_friday),
        "Friday Non-Power": int(is_non_power_friday),
        "Black Friday": int(is_black_friday),
        "Week 0 Power": int(is_week_zero_power),
        "Week 1 Power": int(is_week_one_power),
        "Sat Early": int(not is_black_friday and "Early" in time_slot),
        "Sat Mid": int(not is_black_friday and "Mid" in time_slot),
        "Sat Late": int(not is_black_friday and "Late" in time_slot),

        "Top 10 Rankings": top10,
        "25-11 Rankings": rank_25_11,

        "SEC": (conf1 == "SEC") + (conf2 == "SEC"),
        "Big 10": (conf1 == "Big 10") + (conf2 == "Big 10"),
        "ACC": (conf1 == "ACC") + (conf2 == "ACC"),
        "Big 12": (conf1 == "Big 12") + (conf2 == "Big 12"),
    }

    # postseason implication flags
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

    # YTTV adjustment
    features["YTTV_ABC"] = 0
    features["YTTV_ESPN"] = 0
    now = datetime.now()
    if (now >= FEUD_START) and (FEUD_END is None or now <= FEUD_END):
        if network in ["ABC", "ESPN"]:
            features[f"YTTV_{network}"] = 1

    # team dummy columns
    for col in model.params.index:
        if col in MODEL_TEAM_NAMES:
            features[col] = int(col in [team1, team2])


    # ensure full alignment
    features["const"] = 1.0
    for c in model.params.index:
        if c not in features:
            features[c] = 0.0

    # BTN × Ohio State adjustment
    if "OhioSt_BTN" in model.params.index:
        features["OhioSt_BTN"] = int(("Ohio St." in [team1, team2]) and network == "BTN")

    X = pd.DataFrame([[features[c] for c in model.params.index]], columns=model.params.index)

    ln_pred = float(model.predict(X)[0])
    smearing = getattr(model, "smearing_factor", 1.0)

    # Model output is in THOUSANDS → convert to REAL VIEWERS
    pred_raw = (np.exp(ln_pred) - 1) * smearing
    pred = pred_raw * 1000

    return {
        "raw": float(pred),
        "formatted": format_viewers(pred)
    }
