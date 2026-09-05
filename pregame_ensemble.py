"""Shared serving contract for the optional pregame point ensemble.

The primary model remains authoritative for the competition score.  When a
versioned ensemble is present in its artifact metadata, the challenger reuses
that exact feature row, adds only its fixed schedule/rank terms, and blends the
two point forecasts in thousands-of-viewers units.
"""

from __future__ import annotations

from datetime import date, datetime
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from pregame_context_features import parse_kickoff_hour


EXACT_STATIONS = (
    "ABC", "BTN", "CBS", "CW", "ESPN", "ESPN2", "ESPNU", "ESPNNEWS",
    "FOX", "FS1", "FS2", "NBC", "NFLN",
)
EXACT_KICKOFF_WINDOWS = (
    "SatNoon", "SatAfternoon", "SatPrime", "SatLate", "NonSat",
)
EXACT_STATION_REFERENCE = "ESPN"
EXACT_KICKOFF_REFERENCE = "SatAfternoon"


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
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


def _day_is_saturday(*, day: Any, date_value: Any, time_slot: Any) -> bool:
    slot = str(time_slot or "").strip().lower()
    # Predictor/simulation labels carry useful day information even when their
    # date is a weekly anchor rather than the literal kickoff date.
    if re.search(r"\bsat(?:urday)?\b", slot):
        return True
    if re.search(r"\b(sun(?:day)?|mon(?:day)?|fri(?:day)?|weekday)\b", slot):
        return False
    if re.search(r"\b(tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?)\b", slot):
        return False

    explicit_day = str(day or "").strip().lower()
    if explicit_day:
        if explicit_day.startswith("sat"):
            return True
        if explicit_day.startswith(("sun", "mon", "tue", "wed", "thu", "fri")):
            return False

    game_date = _coerce_date(date_value)
    if game_date is not None:
        return game_date.weekday() == 5

    # Clock-only and predictor bucket inputs overwhelmingly describe the
    # standard Saturday inventory; explicit non-Saturday labels above win.
    return True


def exact_kickoff_window(
    *,
    time_slot: Any = None,
    date_value: Any = None,
    day: Any = None,
) -> str:
    """Resolve the benchmark's exact kickoff cell from live input formats."""

    if not _day_is_saturday(day=day, date_value=date_value, time_slot=time_slot):
        return "NonSat"

    kickoff = parse_kickoff_hour(time_slot)
    if kickoff is None:
        slot = str(time_slot or "").lower()
        if "late" in slot:
            kickoff = 21.0
        elif "primetime" in slot or re.search(r"\bprime\b", slot):
            kickoff = 19.0
        elif "mid" in slot or "afternoon" in slot:
            kickoff = 14.5
        else:
            kickoff = 12.0

    if kickoff < 14.5:
        return "SatNoon"
    if kickoff < 19.0:
        return "SatAfternoon"
    if kickoff < 21.0:
        return "SatPrime"
    return "SatLate"


def _ranked(rank: Any) -> bool:
    value = _finite_number(rank)
    return value is not None and 1 <= value <= 25


def _top_ten(rank: Any) -> bool:
    value = _finite_number(rank)
    return value is not None and 1 <= value <= 10


def rank_scope(rank1: Any, rank2: Any) -> str:
    """Resolve the narrow top-ten calibration scope before both-ranked."""

    if _top_ten(rank1) and _top_ten(rank2):
        return "BothTop10"
    if _ranked(rank1) and _ranked(rank2):
        return "BothRanked"
    return "Other"


def build_exact_pregame_features(
    *,
    network: Any = None,
    time_slot: Any = None,
    date_value: Any = None,
    day: Any = None,
    rank1: Any = None,
    rank2: Any = None,
) -> dict[str, float]:
    """Build the exact fixed feature set used by the promoted challenger."""

    station = str(network or "").strip()
    window = exact_kickoff_window(
        time_slot=time_slot,
        date_value=date_value,
        day=day,
    )
    features: dict[str, float] = {
        "BothRanked_Nonlinear": float(_ranked(rank1) and _ranked(rank2)),
        "BothTop10_Nonlinear": float(_top_ten(rank1) and _top_ten(rank2)),
    }
    for candidate_window in EXACT_KICKOFF_WINDOWS:
        if candidate_window == EXACT_KICKOFF_REFERENCE:
            continue
        features[f"ExactKickoffWindow_{candidate_window}"] = float(
            window == candidate_window
        )
    for candidate_station in EXACT_STATIONS:
        if candidate_station == EXACT_STATION_REFERENCE:
            continue
        for candidate_window in EXACT_KICKOFF_WINDOWS:
            if candidate_window == EXACT_KICKOFF_REFERENCE:
                continue
            features[
                f"ExactNetworkKickoff_{candidate_station}_{candidate_window}"
            ] = float(
                station == candidate_station and window == candidate_window
            )
    return features


def _context_value(context: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in context:
            return context.get(key)
    return None


def exact_feature_frame(
    contexts: Iterable[Mapping[str, Any]],
    *,
    index: pd.Index,
) -> pd.DataFrame:
    rows = []
    for context in contexts:
        rows.append(build_exact_pregame_features(
            network=_context_value(context, "network", "Station"),
            time_slot=_context_value(context, "time_slot", "Time Slot"),
            date_value=_context_value(context, "date", "Date", "ParsedDate"),
            day=_context_value(context, "day", "DoW"),
            rank1=_context_value(context, "rank1"),
            rank2=_context_value(context, "rank2"),
        ))
    if len(rows) != len(index):
        raise ValueError("One ensemble context is required for every feature row")
    return pd.DataFrame.from_records(rows, index=index).astype(float)


def model_point_predictions_000s(model: Any, matrix: pd.DataFrame, smear: Any) -> np.ndarray:
    """Back-transform log-model points using the established backend formula."""

    resolved_smear = _finite_number(smear)
    if resolved_smear is None or resolved_smear <= 0:
        resolved_smear = 1.0
    predicted_log = np.asarray(model.predict(matrix), dtype=float).reshape(-1)
    return np.maximum((np.exp(predicted_log) - 1.0) * resolved_smear, 0.0)


def _valid_ensemble(model: Any) -> dict[str, Any] | None:
    metadata = getattr(model, "pregame_ensemble", None)
    if not isinstance(metadata, dict):
        return None
    weight = _finite_number(metadata.get("weight"))
    challenger_smear = _finite_number(metadata.get("challenger_smearing_factor"))
    if (
        metadata.get("version") != 1
        or metadata.get("competition_mode") != "shared_primary"
        or metadata.get("challenger_model") is None
        or weight is None
        or not 0.0 <= weight <= 1.0
        or challenger_smear is None
        or challenger_smear <= 0
    ):
        return None
    return metadata


def predict_pregame_points_000s(
    primary_model: Any,
    primary_matrix: pd.DataFrame,
    contexts: Iterable[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Predict primary or its optional level blend plus scoped calibration."""

    primary_points = model_point_predictions_000s(
        primary_model,
        primary_matrix,
        getattr(primary_model, "smearing_factor", 1.0),
    )
    metadata = _valid_ensemble(primary_model)
    if metadata is None:
        return primary_points

    context_rows = list(contexts or ({} for _ in range(len(primary_matrix))))
    extras = exact_feature_frame(context_rows, index=primary_matrix.index)
    challenger = metadata["challenger_model"]
    challenger_columns = list(metadata.get("challenger_feature_columns") or challenger.params.index)
    if (len(challenger_columns) != len(challenger.params)
            or len(set(challenger_columns)) != len(challenger_columns)
            or not all(isinstance(column, str) for column in challenger_columns)
            or "const" not in challenger_columns):
        raise ValueError("Challenger feature names are missing or invalid; refusing an unaligned prediction")
    missing = set(challenger_columns) - set(primary_matrix.columns) - set(extras.columns)
    if missing:
        raise ValueError(f"Missing challenger features: {sorted(missing)}")
    challenger_matrix = primary_matrix.reindex(
        columns=challenger_columns,
        fill_value=0.0,
    ).copy()
    for column in extras.columns.intersection(challenger_matrix.columns):
        challenger_matrix[column] = extras[column]

    challenger_points = model_point_predictions_000s(
        challenger,
        challenger_matrix,
        metadata["challenger_smearing_factor"],
    )
    weight = float(metadata["weight"])
    blended = (1.0 - weight) * primary_points + weight * challenger_points

    adjustments = metadata.get("rank_scope_additive_adjustments_000s", {})
    if not isinstance(adjustments, dict):
        adjustments = {}
    scopes = [
        rank_scope(
            _context_value(context, "rank1"),
            _context_value(context, "rank2"),
        )
        for context in context_rows
    ]
    additive = np.asarray([
        _finite_number(adjustments.get(scope)) or 0.0
        for scope in scopes
    ], dtype=float)
    return np.maximum(blended + additive, 0.0)
