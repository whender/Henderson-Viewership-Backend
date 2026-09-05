"""Pure builders for deployable pregame context features.

This module intentionally performs no file or model I/O.  The same inputs can
therefore be used by the API, weekly predictions, and simulation paths without
letting those feature definitions drift apart.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import re
from typing import Any


CONTEXT_FEATURE_NAMES = (
    "AnyDayKickoff9pmOrLater",
    "EarlyRecordNeutralized",
    "MinGamesBefore",
    "LateBothRanked",
    "CFP_RankStrengthSum",
    "CFP_RankDifference",
)

MISSING_RECORD_CONTEXT_WARNING = (
    "No season week, game date, or games-before records were supplied. "
    "Record-timing features use neutral deployment values "
    "(EarlyRecordNeutralized=0 and MinGamesBefore=6)."
)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None

    # Cover the formats used by the API, Firestore rows, and historical CSVs.
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso_text).date()
    except ValueError:
        pass

    for date_format in ("%m/%d/%y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    return None


def _labor_day(year: int) -> date:
    september_first = date(int(year), 9, 1)
    return september_first + timedelta(days=(7 - september_first.weekday()) % 7)


def resolve_season_week(season_week: Any = None, date_value: Any = None) -> int | None:
    """Resolve the football week, preferring an explicitly supplied week.

    Week 1 starts on the Thursday before Labor Day.  Earlier dates resolve to
    Week 0; subsequent seven-day windows increment from that Thursday.
    """

    explicit = _finite_number(season_week)
    if explicit is not None and explicit >= 0 and explicit.is_integer():
        return int(explicit)

    # Also accept labels such as "Week 10" without accepting partial floats.
    if isinstance(season_week, str):
        match = re.fullmatch(r"\s*week\s+(\d+)\s*", season_week, re.IGNORECASE)
        if match:
            return int(match.group(1))

    game_date = _coerce_date(date_value)
    if game_date is None:
        return None

    week_one_start = _labor_day(game_date.year) - timedelta(days=4)
    # January bowl/playoff games belong to the football season that began in
    # the prior calendar year.  Pre-Labor-Day August games remain Week 0.
    if game_date < week_one_start and game_date.month <= 2:
        week_one_start = _labor_day(game_date.year - 1) - timedelta(days=4)
    if game_date < week_one_start:
        return 0
    return 1 + (game_date - week_one_start).days // 7


def parse_kickoff_hour(time_slot: Any) -> float | None:
    """Parse a kickoff into a 24-hour decimal value when one is available."""

    text = str(time_slot or "").strip().lower()
    if not text:
        return None

    match = re.search(
        r"(?<!\d)(\d{1,2})(?::(\d{2}))?\s*(a\.?m?\.?|p\.?m?\.?)?(?!\d)",
        text,
    )
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").replace(".", "")
    if minute >= 60:
        return None

    if meridiem:
        if hour < 1 or hour > 12:
            return None
        hour %= 12
        if meridiem.startswith("p"):
            hour += 12
    elif hour > 23:
        return None
    elif 7 <= hour <= 11:
        # Football feeds occasionally omit the trailing "p" (for example,
        # "10:30"); their morning values include an explicit "a".
        hour += 12

    return hour + minute / 60.0


def _games_before(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def _rank_or_unranked(value: Any) -> float:
    number = _finite_number(value)
    if number is None or number < 1 or number > 25:
        return 26.0
    return number


def build_pregame_context_features(
    *,
    time_slot: Any = None,
    date_value: Any = None,
    season_week: Any = None,
    team1_games_before: Any = None,
    team2_games_before: Any = None,
    rank1: Any = None,
    rank2: Any = None,
    cfp_rank1: Any = None,
    cfp_rank2: Any = None,
    ranking_source: Any = "auto",
) -> tuple[dict[str, float | int], list[str]]:
    """Build the six shared context features and any deployment warnings."""

    week = resolve_season_week(season_week, date_value)
    warnings: list[str] = []

    team1_games = _games_before(team1_games_before)
    team2_games = _games_before(team2_games_before)
    if week is not None:
        inferred_games = float(max(week - 1, 0))
        if team1_games is None:
            team1_games = inferred_games
        if team2_games is None:
            team2_games = inferred_games
    elif team1_games is None and team2_games is None:
        # A neutral fallback avoids treating an unknown game as early season.
        team1_games = team2_games = 6.0
        warnings.append(MISSING_RECORD_CONTEXT_WARNING)
    else:
        # Preserve the known side while keeping an unknown opponent neutral.
        if team1_games is None:
            team1_games = 6.0
        if team2_games is None:
            team2_games = 6.0

    source = str(ranking_source or "auto").strip().lower()
    generic_is_cfp = source == "cfp" or (source == "auto" and week is not None and week >= 10)

    # Explicit CFP fields take precedence one side at a time.  A supplied 0 or
    # out-of-range value explicitly means that side is unranked.
    explicit_cfp1 = _finite_number(cfp_rank1)
    explicit_cfp2 = _finite_number(cfp_rank2)
    if explicit_cfp1 is not None:
        resolved_cfp1 = _rank_or_unranked(explicit_cfp1)
    elif generic_is_cfp:
        resolved_cfp1 = _rank_or_unranked(rank1)
    else:
        resolved_cfp1 = 26.0

    if explicit_cfp2 is not None:
        resolved_cfp2 = _rank_or_unranked(explicit_cfp2)
    elif generic_is_cfp:
        resolved_cfp2 = _rank_or_unranked(rank2)
    else:
        resolved_cfp2 = 26.0

    kickoff_hour = parse_kickoff_hour(time_slot)
    if kickoff_hour is not None:
        # A concrete clock time always wins so labels such as "late afternoon"
        # cannot override the training definition's exact 9 p.m. threshold.
        kickoff_is_late = kickoff_hour >= 21.0
    else:
        # Simulation inventories also use a clock-free "Late" bucket whose
        # documented window begins at 9:30 p.m.
        kickoff_is_late = (
            re.search(r"\blate\b", str(time_slot or ""), re.IGNORECASE)
            is not None
        )
    generic_rank1 = _rank_or_unranked(rank1)
    generic_rank2 = _rank_or_unranked(rank2)

    features: dict[str, float | int] = {
        "AnyDayKickoff9pmOrLater": int(kickoff_is_late),
        "EarlyRecordNeutralized": int(team1_games < 5 or team2_games < 5),
        "MinGamesBefore": min(team1_games, team2_games),
        "LateBothRanked": int(
            week is not None
            and week >= 10
            and generic_rank1 <= 25
            and generic_rank2 <= 25
        ),
        "CFP_RankStrengthSum": max(26.0 - resolved_cfp1, 0.0)
        + max(26.0 - resolved_cfp2, 0.0),
        "CFP_RankDifference": abs(resolved_cfp1 - resolved_cfp2),
    }
    return features, warnings
