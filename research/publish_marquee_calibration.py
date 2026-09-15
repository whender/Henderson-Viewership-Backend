"""Regenerate weeks 2/3 in place with approved calibration; no revision entry.

Prepare backups and review report first; --publish atomically compare-and-sets
both documents. Source features, displayed membership and competition preserved.
"""
import copy,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from firestore_client import db
from weekly_predictions_fs import pregame_model,generate_pregame_prediction,generate_postgame_prediction,calc_error,build_features
from google.cloud import firestore
from pregame_ensemble import predict_pregame_points_000s
from marquee_calibration import production_factors
import pandas as pd
O=B.parent/'tmp/marquee_production';O.mkdir(parents=True,exist_ok=True)
FIELDS=['games','full_slate_games'];IDS=['2_2026','3_2026']
def write(name,value): (O/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def accuracy(error):return '' if error is None else '🟢🎯' if error<5 else '🟢' if error<25 else '🟡' if error<35 else '🔴'
def main():
 cfg=pregame_model.marquee_calibration;assert cfg['revision']=='marquee_hybrid_v1'
 if '--publish' in sys.argv:
  before=json.loads((O/'before.json').read_text());after=json.loads((O/'after.json').read_text())
  assert all(w['marquee_calibration']['artifact_sha256']==cfg['artifact_sha256'] for w in after.values())
  refs={key:db.collection('weekly-predictions').document(key) for key in IDS}
  @firestore.transactional
  def commit(tx):
   for key,ref in refs.items():
    if ref.get(transaction=tx).to_dict()!=before[key]:raise RuntimeError('Week changed since preview: '+key)
   for key,ref in refs.items():tx.set(ref,after[key])
  commit(db.transaction())
  for key,ref in refs.items():assert ref.get(timeout=30).to_dict()==after[key]
  for week in [2,3]:
   w=after[f'{week}_2026'];folder=B.parent/f'week{week}_2026'
   (folder/'predictions.json').write_text(json.dumps(w,indent=2)+'\n')
   (folder/'full_slate_predictions.json').write_text(json.dumps(w['full_slate_games'],indent=2)+'\n')
  (B/'research/week3_2026_predictions.json').write_text(json.dumps(after['3_2026'],indent=2)+'\n')
  print('Published and verified both weeks, with no additional revision entries.')
  return
 before={key:db.collection('weekly-predictions').document(key).get(timeout=30).to_dict() for key in IDS};after=copy.deepcopy(before);now=datetime.now(timezone.utc).isoformat();audit=[]
 for key,w in after.items():
  full=w['full_slate_games'];assert len(full)==(34 if key=='2_2026' else 35)
  byid={};week=int(key[0])
  for original in full:
   r=copy.deepcopy(original);r['week']=week
   x=pd.DataFrame([build_features(r)]).reindex(columns=pregame_model.params.index,fill_value=0.)
   raw=float(predict_pregame_points_000s(pregame_model,x,[r],apply_marquee=False)[0]);factor=float(production_factors(pregame_model,[r],[raw])[0])
   predicted=generate_pregame_prediction(r);assert predicted and not predicted.startswith('Error')
   suffix=' [retrospective]' if '[retrospective]' in original.get('predicted','') else ''
   r['predicted']=predicted+suffix;r.pop('revised_predicted',None)
   r['percent_error']=calc_error(r['predicted'],r.get('actual'));r['accuracy']=accuracy(r['percent_error'])
   if original.get('post_predicted'):
    r['post_predicted']=generate_postgame_prediction(r);assert r['post_predicted']
    r['post_percent_error']=calc_error(r['post_predicted'],r.get('actual'));r['post_accuracy']=accuracy(r['post_percent_error'])
   r['forecast_updated_at']=now;r['marquee_calibration_sha256']=cfg['artifact_sha256']
   byid[r['cfbd_game_id']]=r
   audit.append(dict(week=week,cfbd_game_id=r['cfbd_game_id'],team1=r['team1'],team2=r['team2'],network=r['network'],displayed=any(g['cfbd_game_id']==r['cfbd_game_id'] for g in w['games']),before=original['predicted'],after=r['predicted'],uncalibrated_current_000s=raw,factor=factor,post_before=original.get('post_predicted'),post_after=r.get('post_predicted')))
  # Update only forecast fields in displayed rows, preserving display-only metadata.
  changes=['predicted','percent_error','accuracy','post_predicted','post_percent_error','post_accuracy','forecast_updated_at','marquee_calibration_sha256']
  for field in FIELDS:
   for g in w[field]:
    new=byid[g['cfbd_game_id']]
    for c in changes:
     if c in new:g[c]=new[c]
    g.pop('revised_predicted',None)
  w['marquee_calibration']={'revision':cfg['revision'],'artifact_sha256':cfg['artifact_sha256'],'factor':cfg['factor'],'updated_at':now}
  allowed=set(changes)|{'revised_predicted'}
  for field in FIELDS:
   old=before[key][field];new=w[field];assert [r['cfbd_game_id'] for r in old]==[r['cfbd_game_id'] for r in new]
   for a,b in zip(old,new):assert {k:v for k,v in a.items() if k not in allowed}=={k:v for k,v in b.items() if k not in allowed}
  assert len(w['games'])==(13 if week==2 else 18)
 write('before.json',before);write('after.json',after);write('audit.json',audit)
 (B/'research/marquee_calibration_results/published_preview.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(pd.DataFrame(audit).query('displayed')[['week','team1','team2','before','after','factor']].to_string(index=False))
if __name__=='__main__':main()
