"""Refresh Week 3 in place after the Week 2 refit, including full-slate competition."""
import copy,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from firestore_client import db
from google.cloud import firestore
from weekly_predictions_fs import pregame_model as model,build_features,generate_pregame_prediction,parse_viewership
from audience_interest import row_features
O=B.parent/'tmp/week3_refit_refresh';O.mkdir(parents=True,exist_ok=True)
ref=db.collection('weekly-predictions').document('3_2026')
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 if '--publish' in sys.argv:
  before=json.loads((O/'before.json').read_text());after=json.loads((O/'after.json').read_text())
  for n,d in after['model_artifact_hashes'].items():assert hashlib.sha256((B/n).read_bytes()).hexdigest()==d
  @firestore.transactional
  def commit(tx):
   if ref.get(transaction=tx).to_dict()!=before:raise RuntimeError('Week changed after preview')
   tx.set(ref,after)
  commit(db.transaction());assert ref.get(timeout=30).to_dict()==after
  folder=B.parent/'week3_2026'
  write(folder/'predictions.json',after);write(folder/'full_slate_predictions.json',after['full_slate_games'])
  write(B/'research/week3_2026_predictions.json',after)
  print('Published and verified 18 displayed / 35 total predictions.');return
 before=ref.get(timeout=30).to_dict();after=copy.deepcopy(before);now=datetime.now(timezone.utc).isoformat()
 full=after['full_slate_games'];assert len(full)==35 and len(after['games'])==18
 assert all(not r.get('actual') and not r.get('post_predicted') for r in full)
 hashes={n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in ['viewership_model_log.joblib','audience_interest.joblib','nonlinear_pregame.joblib','week1_major_days.joblib','aligned_postgame.joblib','marquee_calibration.json']}
 x=pd.DataFrame([build_features(r) for r in full]).reindex(columns=model.intrinsic_model.params.index,fill_value=0.)
 audiences=np.maximum((np.exp(model.intrinsic_model.predict(x))-1)*model.intrinsic_smearing_factor/1000,0)
 starts=[datetime.fromisoformat(r['kickoff_utc'].replace('Z','+00:00')) for r in full]
 assert all(s>datetime.now(timezone.utc) for s in starts)
 audit=[];oldmap={r['cfbd_game_id']:r for r in before['full_slate_games']}
 for i,r in enumerate(full):
  r['competing_games_score']=float(sum(audiences.iloc[j] for j,s in enumerate(full) if i!=j and r['date']==s['date'] and abs((starts[i]-starts[j]).total_seconds())<=5400))
  r['predicted']=generate_pregame_prediction(r);v=parse_viewership(r['predicted']);assert v and np.isfinite(v) and v<40
  r.update(forecast_updated_at=now,audience_interest_artifact_sha256=hashes['audience_interest.joblib'],marquee_calibration_sha256=hashes['marquee_calibration.json'],audience_interest_features=row_features(model.audience_interest,r),percent_error=None,accuracy='')
  assert r['audience_interest_features'] is not None
  r.pop('revised_predicted',None)
  old=oldmap[r['cfbd_game_id']]
  for k in ['prediction_feature_override','prediction_history','rank1','rank2','spread_home','feature_as_of']:assert r.get(k)==old.get(k)
  audit.append(dict(game_id=r['cfbd_game_id'],matchup=r['matchup'],before=old['predicted'],after=r['predicted'],competition_before=old['competing_games_score'],competition_after=r['competing_games_score']))
 byid={r['cfbd_game_id']:r for r in full};after['games']=[copy.deepcopy(byid[r['cfbd_game_id']]) for r in before['games']]
 assert [g['cfbd_game_id'] for g in after['games']]==[g['cfbd_game_id'] for g in before['games']]
 assert 401864507 not in {g['cfbd_game_id'] for g in after['games']}
 assert byid[401856796]['prediction_feature_override']==oldmap[401856796]['prediction_feature_override']
 after.update(model_artifact_hashes=hashes,model_sha256=hashes['viewership_model_log.joblib'],forecast_updated_at=now,audience_interest_source_sha256=hashlib.sha256((B/'audience_interest_data.json').read_bytes()).hexdigest(),marquee_calibration={'revision':model.marquee_calibration['revision'],'artifact_sha256':hashes['marquee_calibration.json'],'factor':model.marquee_calibration['factor'],'updated_at':now})
 after['refit_refresh']={'updated_at':now,'training_rows':2453,'training_through_date':'2026-09-12','competition_games':35,'displayed_games':18}
 write(O/'before.json',before);write(O/'after.json',after);write(O/'audit.json',audit)
 for r in after['games']:print(r['matchup'],oldmap[r['cfbd_game_id']]['predicted'],'->',r['predicted'])
if __name__=='__main__':main()
