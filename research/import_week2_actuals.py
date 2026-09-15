"""Apply user-supplied Week 2 actuals without regenerating saved forecasts.

Prepare: python research/import_week2_actuals.py
Publish: python research/import_week2_actuals.py --publish
Requires ../tmp/week2_actuals/source.json; records source and compare-and-set backup.
"""
import copy,json,sys
from datetime import datetime,timezone
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from firestore_client import db
from weekly_predictions_fs import calc_error
from google.cloud import firestore
O=B.parent/'tmp/week2_actuals';ref=db.collection('weekly-predictions').document('2_2026')
def write(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def accuracy(e):return '' if e is None else '🟢🎯' if e<5 else '🟢' if e<25 else '🟡' if e<35 else '🔴'
def main():
 if '--publish' in sys.argv:
  before=json.loads((O/'weekly_before.json').read_text());after=json.loads((O/'weekly_after.json').read_text())
  @firestore.transactional
  def commit(tx):
   if ref.get(transaction=tx).to_dict()!=before:raise RuntimeError('Week changed since preview')
   tx.set(ref,after)
  commit(db.transaction());assert ref.get(timeout=30).to_dict()==after
  folder=B.parent/'week2_2026'
  (folder/'predictions.json').write_text(json.dumps(after,indent=2)+'\n')
  (folder/'full_slate_predictions.json').write_text(json.dumps(after['full_slate_games'],indent=2)+'\n')
  print('Published actuals; all saved forecasts preserved.')
  return
 before=ref.get(timeout=30).to_dict();after=copy.deepcopy(before)
 source=json.loads((O/'source.json').read_text());actuals={r['cfbd_game_id']:r for r in source if r['broadcast_type']=='game'}
 now=datetime.now(timezone.utc).isoformat();counts={}
 for field in ['games','full_slate_games']:
  count=0
  for row in after[field]:
   a=actuals.get(row['cfbd_game_id'])
   if a is None:continue
   assert {a['team1'],a['team2']}=={row['team1'],row['team2']}
   assert a['network']==row['network']
   row.update(actual=f"{a['viewers_000s']/1000:.2f}M",actual_source=a['source'],actual_recorded_at=now)
   for prediction,error,icon in [('predicted','percent_error','accuracy'),('post_predicted','post_percent_error','post_accuracy')]:
    row[error]=calc_error(row.get(prediction),row['actual']);row[icon]=accuracy(row[error])
   count+=1
  counts[field]=count
  for a,b in zip(before[field],after[field]):
   for key in ['predicted','post_predicted','prediction_history','competing_games_score']:
    assert a.get(key)==b.get(key)
 assert counts=={'games':13,'full_slate_games':30},counts
 after['actuals_import']={'recorded_at':now,'source':'User-provided official Week 2 table','matched':counts,'reported_broadcasts':35,'individual_games':32,'excluded_from_individual_actuals':['ESPN2 Oregon–Oklahoma State partial','BTN regional coverage 7:15p','BTN regional coverage 3:30p']}
 write('weekly_before.json',before);write('weekly_after.json',after)
 import numpy as np
 rows=after['games'];errors=np.array([r['percent_error'] for r in rows]);stats={'games':len(rows),'mape_pct':float(errors.mean()),'median_ape_pct':float(np.median(errors)),'within_20':int(sum(errors<=20)),'within_30':int(sum(errors<=30)),'over_50':int(sum(errors>50))}
 write('accuracy.json',stats);print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
