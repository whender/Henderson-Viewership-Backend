"""Apply the approved 2025 Deion effect to Colorado's Week 3 forecast."""
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from firestore_client import db
from google.cloud import firestore
from weekly_predictions_fs import build_features, generate_pregame_prediction, calc_error

GAME_ID = 401856796
OUT = BASE.parent / 'week3_2026'
ref = db.collection('weekly-predictions').document('3_2026')
before = ref.get(timeout=30).to_dict()
after = copy.deepcopy(before)
now = datetime.now(timezone.utc).isoformat()

def revise(payload):
    for field in ['games', 'full_slate_games']:
        matches = [g for g in payload[field] if g['cfbd_game_id'] == GAME_ID]
        assert len(matches) == 1
        game = matches[0]
        assert {game['team1'], game['team2']} == {'Colorado', 'Northwestern'}
        assert datetime.fromisoformat(game['kickoff_utc'].replace('Z', '+00:00')) > datetime.now(timezone.utc)
        assert not game.get('actual') and not game.get('post_predicted')
        assert build_features(game)['DeionEra25'] == 0, 'Already applied'
        game.setdefault('prediction_history', []).append({
            'predicted': game['predicted'], 'saved_at': now,
            'reason': 'Before user-requested DeionEra25 override'})
        game['prediction_feature_override'] = {**(game.get('prediction_feature_override') or {}), 'DeionEra25': 1}
        assert build_features(game)['DeionEra25'] == 1
        assert build_features(game)['DeionEra'] == 0
        game['predicted'] = generate_pregame_prediction(game)
        assert game['predicted'] == '1.69M (1.13-2.48M)'
        game['percent_error'] = calc_error(game['predicted'], game.get('actual'))
        game['accuracy'] = ''
        game.update(forecast_timing='pregame', forecast_updated_at=now,
                    revision_reason='User requested the 2025 Deion effect for Colorado.',
                    prediction_revision='week3_deion25_override')

revise(after)
for field in ['games', 'full_slate_games']:
    assert all(a == b for a, b in zip(before[field], after[field]) if a['cfbd_game_id'] != GAME_ID)
(OUT / 'before_deion_override.json').write_text(json.dumps(before, indent=2) + '\n')
if '--publish' in sys.argv:
    @firestore.transactional
    def commit(tx):
        if ref.get(transaction=tx).to_dict() != before:
            raise RuntimeError('Week changed during review')
        tx.update(ref, {field: after[field] for field in ['games', 'full_slate_games']})
    commit(db.transaction())
    assert ref.get(timeout=30).to_dict() == after
    path = Path(__file__).with_name('week3_2026_predictions.json')
    snapshot = json.loads(path.read_text())
    revise(snapshot)
    path.write_text(json.dumps(snapshot, indent=2) + '\n')
    (OUT / 'predictions.json').write_text(json.dumps(after, indent=2) + '\n')
print(json.dumps({'published': '--publish' in sys.argv, 'forecast': next(g['predicted'] for g in after['games'] if g['cfbd_game_id'] == GAME_ID), 'other_forecasts_unchanged': True}))
