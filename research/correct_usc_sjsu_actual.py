"""Apply the user's September 2026 USC–SJSU actual correction, preserving forecasts."""
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from firestore_client import db
from weekly_predictions_fs import calc_error
from google.cloud import firestore

OUT = BASE.parent / 'tmp/usc_actual_correction'
GAME_ID = 401864494
ref = db.collection('weekly-predictions').document('0_2026')
before = ref.get(timeout=30).to_dict()
after = copy.deepcopy(before)
changes = []
for field in ['games', 'full_slate_games']:
    for game in after.get(field, []):
        if game.get('cfbd_game_id') != GAME_ID:
            continue
        assert {game['team1'], game['team2']} == {'USC', 'San Jose St.'}
        assert game['actual'] == '1.862M', 'Already corrected or actual changed'
        game.setdefault('actual_history', []).append({
            'actual': game['actual'], 'corrected_at': datetime.now(timezone.utc).isoformat(),
            'reason': 'User corrected actual audience to 2.00M viewers.'})
        game['actual'] = '2.000M'
        game['actual_source'] = 'User-provided correction to 2.00M viewers, September 2026'
        for prediction, error, badge in [('predicted', 'percent_error', 'accuracy'), ('post_predicted', 'post_percent_error', 'post_accuracy')]:
            value = calc_error(game.get(prediction), game['actual'])
            game[error] = value
            game[badge] = '' if value is None else '🟢🎯' if value < 5 else '🟢' if value < 25 else '🟡' if value < 35 else '🔴'
        changes.append({'field': field, 'actual': game['actual'], 'predicted': game['predicted'], 'percent_error': game['percent_error']})
assert changes
OUT.mkdir(exist_ok=True)
for name, data in [('publish_before', before), ('publish_after', after)]:
    (OUT / (name + '.json')).write_text(json.dumps(data, indent=2, default=str) + '\n')
if '--publish' in sys.argv:
    @firestore.transactional
    def commit(transaction):
        if ref.get(transaction=transaction).to_dict() != before:
            raise RuntimeError('Weekly predictions changed during correction')
        transaction.set(ref, after)
    commit(db.transaction())
    assert ref.get(timeout=30).to_dict() == after
print(json.dumps({'published': '--publish' in sys.argv, 'changes': changes}, indent=2))
