"""Schedule-conditioned average-brand expectations through the serving model.

These are retrospective fitted comparisons, not held-out forecast accuracy.
The neutral brand vector averages equally over FBS teams with >=5 rated games;
all observed non-brand context (including pregame attention) stays fixed.
"""
import json
from pathlib import Path

import pandas as pd
from pregame_ensemble import predict_pregame_points_000s


def design_matrix(model, df, team_names):
    columns = list(model.params.index)
    x = df.reindex(columns=columns).apply(pd.to_numeric, errors='coerce').fillna(0.).astype(float)
    x['const'] = 1.
    for team in set(columns) & set(team_names):
        x[team] = df['Team 1'].eq(team).astype(float) + df['Team 2'].eq(team).astype(float)
    return x[columns]


def neutral_brand_matrix(matrix, df, team_columns, reference_teams, focal=None):
    """Replace one/both brand vectors, consistently in every ensemble component."""
    if not reference_teams:
        raise ValueError('No supported FBS reference teams')
    x = matrix.copy()
    sides = [focal] if focal else ['Team 1', 'Team 2']
    for side in sides:
        for team in team_columns:
            x[team] -= df[side].eq(team).astype(float)
            x[team] += float(team in reference_teams) / len(reference_teams)
    colorado = pd.Series(True, index=df.index) if focal is None else df[focal].eq('Colorado')
    for flag in ['DeionEra', 'DeionEra25']:
        if flag in x:
            x.loc[colorado, flag] = 0.
    return x


def compute_expectations(model, df, team_names, fbs_teams, context_path=None):
    context_path = context_path or Path(__file__).with_name('expected_viewership_context.json')
    saved = json.loads(Path(context_path).read_text())
    contexts = []
    for _, row in df.iterrows():
        ctx = saved.get(str(int(row.source_index)))
        if ctx is None or ctx['team1'] != row['Team 1'] or ctx['team2'] != row['Team 2'] or pd.Timestamp(ctx['date']) != pd.Timestamp(row.ParsedDate):
            raise ValueError(f'Missing/mismatched expected-viewership context: {row.source_index}')
        contexts.append(ctx)
    matrix = design_matrix(model, df, team_names)
    counts = pd.concat([df['Team 1'], df['Team 2']]).value_counts()
    reference = set(counts[counts >= 5].index) & set(fbs_teams)
    team_columns = set(matrix.columns) & set(team_names)
    result = {}
    for name, focal in [('expected_viewers', None), ('expected_viewers_team1', 'Team 1'), ('expected_viewers_team2', 'Team 2')]:
        neutral = neutral_brand_matrix(matrix, df, team_columns, reference, focal)
        result[name] = predict_pregame_points_000s(model, neutral, contexts)
    return pd.DataFrame(result, index=df.index)
