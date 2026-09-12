"""Back up and revise only the 18 retained Week 1 forecasts, preserving actuals."""
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import numpy as np
from google.cloud import firestore
from firestore_client import db
from weekly_predictions_fs import pregame_model,build_features,generate_pregame_prediction,calc_error
from pregame_ensemble import predict_pregame_points_000s

BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    ref=db.collection('weekly-predictions').document('1_2026')
    before=ref.get(timeout=30).to_dict()
    assert before['week']==1 and before['year']==2026 and len(before['games'])==18
    digest=hashlib.sha256((BASE/'week1_major_days.joblib').read_bytes()).hexdigest()
    if before.get('week1_day_model_sha256')==digest:
        print('This Week 1 revision is already published.');return
    expected=pd.read_csv(BASE/'prediction_updates/week1_day_2026.csv').set_index('cfbd_game_id')
    after=json.loads(json.dumps(before));stamp=datetime.now(timezone.utc).isoformat()
    for row in after['games']:
        assert row['network'] in {'NBC','CBS','ABC','FOX','ESPN'}
        x=pd.DataFrame([build_features(row)]).reindex(columns=pregame_model.params.index,fill_value=0.)
        point=float(predict_pregame_points_000s(pregame_model,x,[row])[0])
        assert abs(point-expected.loc[row['cfbd_game_id'],'candidate_000s'])<1e-5
        revised=generate_pregame_prediction(row)
        assert not revised.startswith('Error:')
        row['predicted_before_week1_day_interactions']=row['predicted']
        row['forecast_timing_before_week1_day_interactions']=row.get('forecast_timing')
        row['percent_error_before_week1_day_interactions']=row.get('percent_error')
        row['prediction_revision_before_week1_day_interactions']=row.get('prediction_revision')
        row['predicted']=revised+' [retrospective]'
        row['forecast_timing']='retrospective'
        row['prediction_revised_at']=stamp
        row['prediction_revision']='week1_major_network_day_interactions_v1'
        row['warnings']=list(dict.fromkeys(row.get('warnings',[])+['Forecast revised after the game using a model trained through 2025; original forecast retained.']))
        error=calc_error(row['predicted'],row.get('actual'));row['percent_error']=error
        row['accuracy']='' if error is None else '🟢🎯' if error<5 else '🟢' if error<25 else '🟡' if error<35 else '🔴'
        print(f"{row['matchup']}: {row['predicted_before_week1_day_interactions']} -> {row['predicted']}")
    assert [g.get('actual') for g in before['games']]==[g.get('actual') for g in after['games']]
    if not args.publish:print('Dry run; no records changed.');return
    out=BASE.parent/'week1_2026';out.mkdir(exist_ok=True)
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (out/f'firestore_before_day_interactions_{suffix}.json').write_text(json.dumps(before,indent=2))
    @firestore.transactional
    def update(transaction):
        if ref.get(transaction=transaction).to_dict()!=before:raise RuntimeError('Week 1 changed after review; rerun before updating.')
        transaction.update(ref,{'games':after['games'],'week1_day_model_sha256':digest,'week1_day_revised_at':stamp})
    update(db.transaction())
    actual=ref.get(timeout=30).to_dict()
    assert actual=={**before,'games':after['games'],'week1_day_model_sha256':digest,'week1_day_revised_at':stamp}
    local=out/'predictions.json'
    if local.exists():(out/f'predictions_before_day_interactions_{suffix}.json').write_bytes(local.read_bytes())
    local.write_text(json.dumps(actual,indent=2)+'\n')
    with (out/'predictions.csv').open('w',newline='') as f:
        fields=['cfbd_game_id','date','time_slot','matchup','network','spread','predicted','actual','percent_error','forecast_timing','competing_games_score']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(actual['games'])
    print('Published and verified all 18 revised forecasts; actuals and previous forecasts preserved.')

if __name__=='__main__':main()
