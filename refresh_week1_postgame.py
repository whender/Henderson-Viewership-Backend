"""Refresh Week 1 postgame outputs without changing pregame forecasts or scores."""
import argparse,copy,csv,json
from pathlib import Path
from datetime import datetime,timezone
from google.cloud import firestore
from firestore_client import db
from weekly_predictions_fs import generate_postgame_prediction,calc_error,aligned_postgame

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    ref=db.collection('weekly-predictions').document('1_2026');before=ref.get(timeout=30).to_dict()
    assert before['week']==1 and before['year']==2026 and len(before['games'])==18
    after=copy.deepcopy(before);stamp=datetime.now(timezone.utc).isoformat()
    for row in after['games']:
        pred=generate_postgame_prediction(row);assert pred and pred.endswith('M')
        if row.get('postgame_model_sha256')!=aligned_postgame['artifact_sha256']:
            row['post_predicted_before_alignment']=row.get('post_predicted')
            row['post_percent_error_before_alignment']=row.get('post_percent_error')
        row['post_predicted']=pred;row['postgame_model_sha256']=aligned_postgame['artifact_sha256'];row['postgame_revised_at']=stamp
        error=calc_error(pred,row.get('actual'));row['post_percent_error']=error
        row['post_accuracy']='' if error is None else '🟢🎯' if error<5 else '🟢' if error<25 else '🟡' if error<35 else '🔴'
        print(row['matchup'],row.get('post_predicted_before_alignment'),'->',pred)
    for old,new in zip(before['games'],after['games']):
        assert all(old.get(k)==new.get(k) for k in ['predicted','actual','score1','score2','prediction_feature_override','percent_error'])
    if not args.publish:print('Dry run complete.');return
    out=Path(__file__).resolve().parent.parent/'week1_2026';suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (out/f'firestore_before_postgame_alignment_{suffix}.json').write_text(json.dumps(before,indent=2)+'\n')
    @firestore.transactional
    def update(t):
        if ref.get(transaction=t).to_dict()!=before:raise RuntimeError('Week changed during review; rerun.')
        t.update(ref,{'games':after['games']})
    update(db.transaction());saved=ref.get(timeout=30).to_dict();assert saved==after
    local=out/'predictions.json'
    if local.exists():(out/f'predictions_before_postgame_alignment_{suffix}.json').write_bytes(local.read_bytes())
    local.write_text(json.dumps(saved,indent=2)+'\n')
    with (out/'predictions.csv').open(newline='') as f:fields=next(csv.reader(f))
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(saved['games'])
    print('Verified 18 aligned postgame forecasts. Pregame forecasts, actuals and scores unchanged.')

if __name__=='__main__':main()
