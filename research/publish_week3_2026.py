"""Publish the reviewed Week 3 slate without overwriting an existing week."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from firestore_client import db
from weekly_predictions_fs import generate_pregame_prediction

payload = json.loads(Path(__file__).with_name('week3_2026_predictions.json').read_text())
assert (payload['year'], payload['week'], payload['season_week']) == (2026, 3, 3)
display_networks = {'ABC', 'CBS', 'NBC', 'FOX', 'ESPN', 'ESPN2'}
full = payload['full_slate_games']
assert len(full) == 35 and len({g['cfbd_game_id'] for g in full}) == 35
assert len(payload['games']) == 19
assert {g['cfbd_game_id'] for g in payload['games']} == {g['cfbd_game_id'] for g in full if g['network'] in display_networks}
assert sum(g['network'] == 'ESPN2' for g in payload['games']) == 3
assert all(next(r for r in full if r['cfbd_game_id'] == g['cfbd_game_id']) == g for g in payload['games'])
for name, expected in payload['model_artifact_hashes'].items():
    assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected
assert hashlib.sha256((BASE / 'audience_interest_data.json').read_bytes()).hexdigest() == payload['audience_interest_source_sha256']
for game in full:
    assert game['audience_interest_features'] is not None
    assert datetime.fromisoformat(game['kickoff_utc'].replace('Z', '+00:00')) > datetime.now(timezone.utc)
    assert game['forecast_timing'] == 'pregame' and not game.get('actual')
    assert not any(game.get(k) is not None for k in ['score1', 'score2', 'post_predicted'])
    assert generate_pregame_prediction(game) == game['predicted']
    assert game['percent_error'] is None
if '--publish' in sys.argv:
    ref = db.collection('weekly-predictions').document('3_2026')
    ref.create(payload, timeout=30)
    assert ref.get(timeout=30).to_dict() == payload
    print('Published and verified Week 3: 19 displayed games, 35 full-slate forecasts.')
else:
    print('Validated Week 3: scope, uniqueness, cutoffs, model hashes, and serving parity.')
