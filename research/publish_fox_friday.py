"""Prepare or compare-and-set the approved Kansas–Missouri forecast revision."""
import copy,json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from firestore_client import db
from weekly_predictions_fs import generate_pregame_prediction,generate_postgame_prediction,pregame_model,calc_error
from fox_friday import fox_friday_night
O=B.parent/'tmp/fox_friday_deploy';O.mkdir(exist_ok=True)
ref=db.collection('weekly-predictions').document('2_2026');gid=401856678
if '--publish' in sys.argv:
 from google.cloud import firestore
 before=json.loads((O/'firestore_before.json').read_text());after=json.loads((O/'firestore_after.json').read_text())
 assert after['fox_friday_revision']['artifact_sha256']==pregame_model.audience_interest['artifact_sha256']
 @firestore.transactional
 def commit(tx):
  if ref.get(transaction=tx).to_dict()!=before:raise RuntimeError('Week changed since preview')
  tx.set(ref,after)
 commit(db.transaction());assert ref.get(timeout=30).to_dict()==after
 print('Published and verified Kansas–Missouri in displayed and full-slate forecasts.')
else:
 before=ref.get(timeout=30).to_dict();after=copy.deepcopy(before)
 original=next(r for r in before['games'] if r['cfbd_game_id']==gid);r=copy.deepcopy(original);assert fox_friday_night(r)==1
 now=datetime.now(timezone.utc).isoformat();pre=generate_pregame_prediction(r);post=generate_postgame_prediction(r)
 assert pre and not pre.startswith('Error') and post and not post.startswith('Error')
 old=r.get('revised_predicted') or r['predicted'];r.setdefault('prediction_history',[]).append({'predicted':old,'post_predicted':r.get('post_predicted'),'percent_error':r.get('percent_error'),'saved_at':now,'reason':'Before FOX Friday-night interaction'})
 r['predicted']=pre+' [retrospective]';r.pop('revised_predicted',None);r['post_predicted']=post
 r['percent_error']=calc_error(pre,r.get('actual'));r['post_percent_error']=calc_error(post,r.get('actual'))
 r.update(prediction_revision='fox_friday_night_v1',prediction_revised_at=now,forecast_updated_at=now,revised_at=now,revision_timing='retrospective',forecast_timing='retrospective',revision_reason='Approved FOX regular Friday-night interaction; excludes Black Friday and conference championships.',audience_interest_artifact_sha256=pregame_model.audience_interest['artifact_sha256'])
 if r['percent_error'] is None:r['accuracy']=''
 if r['post_percent_error'] is None:r['post_accuracy']=''
 for field in ['games','full_slate_games']:
  assert sum(g['cfbd_game_id']==gid for g in after[field])==1
  after[field]=[copy.deepcopy(r) if g['cfbd_game_id']==gid else g for g in after[field]]
  assert all(a==b for a,b in zip(before[field],after[field]) if a['cfbd_game_id']!=gid)
 after['fox_friday_revision']={'game_id':gid,'created_at':now,'artifact_sha256':pregame_model.audience_interest['artifact_sha256'],'training_through_date':pregame_model.audience_interest['training_through_date']}
 for name,data in [('firestore_before.json',before),('firestore_after.json',after)]:
  (O/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'matchup':r['matchup'],'old':old,'new':r['predicted'],'postgame':post,'artifact_sha256':pregame_model.audience_interest['artifact_sha256'],'other_games_unchanged':True},indent=2))
