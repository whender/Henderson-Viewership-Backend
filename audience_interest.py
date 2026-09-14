"""Weekly pregame attention: fixed, complete UTC windows; offline source cache."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import math
import joblib
import numpy as np

FEATURES = ['InterestLogTotal7', 'InterestLogMax7']
MAJOR = {'ABC', 'CBS', 'FOX', 'NBC', 'ESPN'}
POLICY = 'weekly_tuesday_v1'


def interest_cutoff(game_date):
    """Tuesday–Monday slate: use data through Sunday, available by Tuesday.

    For early-week games retain the original two-calendar-day safety lag.
    Cutoff is exclusive (00:00 UTC); no game-day observations enter.
    """
    start = game_date - timedelta(days=(game_date.weekday() - 1) % 7)
    return min(game_date - timedelta(days=2), start - timedelta(days=1))


def feature_values(game_date, teams, daily, as_of=None):
    cutoff = interest_cutoff(game_date)
    if as_of is not None and as_of < datetime.combine(cutoff + timedelta(days=1), datetime.min.time(), timezone.utc):
        return None
    totals = []
    for team in teams:
        history = daily.get(team, {})
        values = [history.get((cutoff - timedelta(days=i)).isoformat()) for i in range(1, 8)]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            return None
        totals.append(sum(values))
    return {'InterestLogTotal7': math.log1p(sum(totals)), 'InterestLogMax7': math.log1p(max(totals)), 'InterestMissing': 0.}


def load_audience_interest(directory):
    base = Path(directory)
    path = base / 'audience_interest.joblib'
    if not path.exists():
        return None
    artifact = joblib.load(path)
    if artifact.get('version') != 1 or artifact.get('policy') != POLICY or not artifact.get('promotion_passed'):
        raise ValueError('Unvalidated audience-interest artifact')
    for name, expected in artifact['base_sha256'].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Audience-interest model must be revalidated for {name}')
    source = json.loads((base / 'audience_interest_data.json').read_text())
    if source.get('policy') != POLICY:
        raise ValueError('Audience-interest source cutoff differs from training')
    artifact['daily'] = source['daily']
    artifact['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifact


def row_features(artifact, row):
    from pregame_ensemble import _coerce_date
    from predict import normalize_team
    d = _coerce_date(row.get('date'))
    if d is None or d.year != artifact.get('prediction_year', artifact['training_max_year'] + 1):
        return None
    try:
        as_of = datetime.fromisoformat(row['feature_as_of'].replace('Z', '+00:00')) if row.get('feature_as_of') else datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            return None
        as_of = min(as_of, datetime.now(timezone.utc))
    except (ValueError, TypeError):
        return None
    return feature_values(d, [normalize_team(row.get('team1', '')), normalize_team(row.get('team2', ''))], artifact['daily'], as_of)


def apply_audience_interest(model, matrix, contexts, baseline, points, scopes, postgame=False):
    """Replace validated components only where both teams have complete history."""
    artifact = getattr(model, 'audience_interest', None)
    if artifact is None:
        return points
    from pregame_ensemble import exact_feature_frame, _coerce_date
    from nonlinear_pregame import context_values
    from aligned_postgame import score_difference
    config = artifact['postgame' if postgame else 'pregame']
    result = np.asarray(points, dtype=float).copy()
    for i, row in enumerate(contexts):
        values = row_features(artifact, row)
        if values is None:
            continue
        x = matrix.iloc[[i]].copy()
        extras = exact_feature_frame([row], index=x.index)
        for c in extras:
            x[c] = extras[c]
        for c, value in values.items():
            x[c] = value
        if postgame:
            margin = score_difference(row)
            if margin is None:
                raise ValueError('Final scores required for audience-interest postgame model')
            x['Score Diff'] = margin
        d = _coerce_date(row.get('date'))
        week_config = getattr(model, 'week1_major_days', {})
        opening = (week_config.get('start_date', '9999') <= d.isoformat() <= week_config.get('end_date', '')
                   and row.get('network') in MAJOR and (row.get('week') is None or str(row['week']) in ('1', '1.0')))
        branch = 'week1' if opening else 'regular'
        for label, days in [('Weekday', {1, 2, 3}), ('Friday', {4}), ('Saturday', {5}), ('Sunday', {6}), ('Monday', {0})]:
            x['Week1Major_' + label] = float(opening and d.weekday() in days)
        from fox_friday import FEATURE, fox_friday_night
        if FEATURE in artifact.get('additional_features', []):
            x[FEATURE] = fox_friday_night(row)
        raw = 0.
        for component in config['components'][branch]:
            logpred = float((x[component['columns']].to_numpy(dtype=float) @ component['coefficients'])[0])
            raw += .5 * max(np.exp(logpred) * component['smearing_factor'] - 1, 0)
        nonlinear = context_values(row)
        use_forest = baseline[i] >= 1000 and nonlinear is not None
        if use_forest:
            for key, value in nonlinear.items():
                x[key] = value
                x[key + '_missing'] = 0.
            forest = config['forest']
            fp = max(float(np.exp(forest['model'].predict(x[forest['feature_columns']]))[0]) * forest['smearing_factor'] - 1, 0)
            raw = .75 * raw + .25 * fp
        correction = config['corrections'][branch]['nonlinear' if use_forest else 'linear'].get(scopes[i], 0.)
        result[i] = max(raw - correction, 0.)
    return result


def interest_warning(model, row):
    artifact = getattr(model, 'audience_interest', None)
    if artifact is not None and row_features(artifact, row) is None:
        return 'Complete audience-interest history is unavailable at this forecast cutoff; using the previous model.'
    return None
