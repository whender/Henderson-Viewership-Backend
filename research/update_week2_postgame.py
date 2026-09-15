"""Populate Week 2 finals and aligned postgame predictions, preserving pregame forecasts."""
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE.parent / 'RatingsAndRegression')]
from pull_cfbd_2026_power4_schedule import normalize_team as cfbd_team
from predict import normalize_team
from weekly_predictions_fs import generate_postgame_prediction, build_features, calc_error, aligned_postgame, pregame_model
from audience_interest import row_features
from firestore_client import db
from google.cloud import firestore

OUT = BASE.parent / 'week2_2026/postgame'
ref = db.collection('weekly-predictions').document('2_2026')
before = json.loads((OUT / 'before.json').read_text())
after = copy.deepcopy(before)
games = {g['id']: g for g in json.loads((OUT / 'games.json').read_text())}
now = datetime.now(timezone.utc).isoformat()
display = {g['cfbd_game_id']: g for g in before['games']}
updates = {}
for stored in before['full_slate_games']:
    row = copy.deepcopy(display.get(stored['cfbd_game_id'], stored))
    game = games[row['cfbd_game_id']]
    assert game['completed'] is True and game['week'] == 2 and game['season'] == 2026
    assert datetime.fromisoformat(row['kickoff_utc'].replace('Z', '+00:00')) == datetime.fromisoformat(game['startDate'].replace('Z', '+00:00'))
    scores = {normalize_team(cfbd_team(game[side + 'Team'])): game[side + 'Points'] for side in ['away', 'home']}
    assert set(scores) == {row['team1'], row['team2']}
    assert all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in scores.values())
    context = dict(row, week=2, season_week=before.get('season_week', 2))
    original_features = build_features(context)
    context.update(score1=scores[row['team1']], score2=scores[row['team2']])
    assert build_features(context) == original_features
    assert row_features(pregame_model.audience_interest, context) is not None
    prediction = generate_postgame_prediction(context)
    assert prediction is not None and prediction.endswith('M')
    updates[row['cfbd_game_id']] = {
        'score1': context['score1'], 'score2': context['score2'],
        'game_completed': True, 'score_source': 'CFBD /games, 2026 regular-season Week 2',
        'scores_recorded_at': now, 'post_predicted': prediction,
        'postgame_generated_at': now, 'postgame_model_revision': 'aligned_pregame_plus_final_margin',
        'postgame_artifact_sha256': aligned_postgame['artifact_sha256'],
        'postgame_audience_interest_artifact_sha256': pregame_model.audience_interest['artifact_sha256'],
    }

assert len(updates) == 34
for field in ['games', 'full_slate_games']:
    for row in after[field]:
        previous = copy.deepcopy(row)
        update = updates[row['cfbd_game_id']]
        if row.get('post_predicted') and row['post_predicted'] != update['post_predicted']:
            row.setdefault('post_prediction_history', []).append({
                'post_predicted': row['post_predicted'], 'score1': row.get('score1'),
                'score2': row.get('score2'), 'saved_at': now,
                'reason': 'Before current aligned postgame model refresh with verified finals'})
        row.update(update)
        error = calc_error(row['post_predicted'], row.get('actual'))
        row['post_percent_error'] = error
        row['post_accuracy'] = '' if error is None else '🟢🎯' if error < 5 else '🟢' if error < 25 else '🟡' if error < 35 else '🔴'
        for key in ['predicted', 'prediction_feature_override', 'actual', 'percent_error', 'accuracy', 'competing_games_score', 'forecast_timing']:
            assert row.get(key) == previous.get(key), key
after['postgame_refresh'] = {'updated_at': now, 'displayed_games': len(after['games']), 'full_slate_games': len(updates), 'score_source': 'CFBD /games', 'model': 'Current pregame architecture plus final score differential'}
(OUT / 'after.json').write_text(json.dumps(after, indent=2) + '\n')
if '--publish' in sys.argv:
    @firestore.transactional
    def commit(tx):
        if ref.get(transaction=tx).to_dict() != before:
            raise RuntimeError('Week 2 changed since backup; re-read and prepare again')
        tx.update(ref, {key: after[key] for key in ['games', 'full_slate_games', 'postgame_refresh']})
    commit(db.transaction())
    assert ref.get(timeout=30).to_dict() == after
    (BASE.parent / 'week2_2026/predictions.json').write_text(json.dumps(after, indent=2) + '\n')
    (BASE.parent / 'week2_2026/full_slate_predictions.json').write_text(json.dumps(after['full_slate_games'], indent=2) + '\n')
print(json.dumps({'published': '--publish' in sys.argv, 'displayed': len(after['games']), 'full_slate': len(updates), 'games': [{'matchup': g['matchup'], 'score1': g['score1'], 'score2': g['score2'], 'post_predicted': g['post_predicted']} for g in after['games']]}, indent=2))
