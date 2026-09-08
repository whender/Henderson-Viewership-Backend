"""Review or publish the saved Monday calibration revision without replacing a slate."""
import argparse
import json
from pathlib import Path
from google.cloud import firestore
from firestore_client import db
from weekly_predictions_fs import calc_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    update = json.loads((Path(__file__).parent/'prediction_updates/monday_2026.json').read_text())
    revision = update['games'][0]
    reference = db.collection('weekly-predictions').document('1_2026')
    fields = ['predicted','predicted_before_monday_calibration',
              'forecast_timing_before_monday_calibration','forecast_timing',
              'prediction_revised_at','prediction_revision']

    @firestore.transactional
    def revise(transaction):
        snapshot = reference.get(transaction=transaction, timeout=20, retry=None)
        data = snapshot.to_dict()
        if not data:
            raise ValueError('Expected existing Week 1 document')
        matches = [g for g in data['games'] if g.get('cfbd_game_id') == revision['cfbd_game_id']]
        if len(matches) != 1:
            raise ValueError('Expected exactly one matching game')
        game = matches[0]
        if game.get('predicted') == revision['predicted']:
            return 'Already updated'
        if game.get('predicted') != revision['predicted_before_monday_calibration']:
            raise ValueError('Stored prediction changed since review; refusing overwrite')
        print(f"{game['matchup']}: {game['predicted']} -> {revision['predicted']}")
        if not args.publish:
            return 'Dry run; no changes written'
        for field in fields:
            game[field] = revision[field]
        game['warnings'] = list(dict.fromkeys(game.get('warnings', []) + revision['warnings']))
        error = calc_error(game['predicted'], game.get('actual'))
        game['percent_error'] = error
        game['accuracy'] = '' if error is None else '🟢🎯' if error < 5 else '🟢' if error < 25 else '🟡' if error < 35 else '🔴'
        transaction.update(reference, {'games': data['games']})
        return 'Published'

    print(revise(db.transaction()))
    if args.publish:
        saved = reference.get(timeout=20, retry=None).to_dict()
        game = next(g for g in saved['games'] if g.get('cfbd_game_id') == revision['cfbd_game_id'])
        assert game['predicted'] == revision['predicted']
        assert game['predicted_before_monday_calibration'] == revision['predicted_before_monday_calibration']
        print('Verified stored revision and preserved original forecast')


if __name__ == '__main__':
    main()
