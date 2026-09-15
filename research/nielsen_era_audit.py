"""Audit measurement-era identifiability and Michigan–Oklahoma sensitivity."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from nielsen_measurement import measurement_era_flags, measurement_status
from weekly_predictions_fs import pregame_model,aligned_postgame,build_features
from pregame_ensemble import predict_pregame_points_000s
from aligned_postgame import predict_postgame_points_000s
x=pd.read_csv(BASE/'viewership_cleaned.csv');flags=pd.DataFrame([measurement_era_flags(v) for v in x.ParsedDate]);d=pd.to_datetime(x.ParsedDate)
old=x.OldNielsenSystem.to_numpy();ooh=flags.NielsenOOHFullCoverageEra.to_numpy()
assert np.array_equal(ooh,1-old)
rank=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(x)),old,ooh])))
post=flags.Nielsen2026RevisionEra.eq(1)
assert not post[d.dt.year.le(2025)].any()
r=next(g for g in json.loads((BASE.parent/'week2_2026/postgame/after.json').read_text())['games'] if g['cfbd_game_id']==401856679)
def predict(row,old_flag=None):
 matrix=pd.DataFrame([build_features(row)]).reindex(columns=pregame_model.params.index,fill_value=0.)
 if old_flag is not None:matrix['OldNielsenSystem']=old_flag
 return {'pregame_millions':float(predict_pregame_points_000s(pregame_model,matrix,[row])[0])/1000,'postgame_millions':float(predict_postgame_points_000s(pregame_model,aligned_postgame,matrix,[row])[0])/1000}
no_comp=copy.deepcopy(r);no_comp['competing_games_score']=0
ranked=copy.deepcopy(r);ranked['rank2']=20
report={'measurement':measurement_status(),'rows':len(x),'OOH_is_exact_inverse_of_existing_flag':True,'intercept_oldflag_OOH_design_rank':rank,'column_count':3,'post_2026_change_games':int(post.sum()),'post_2026_change_min_date':str(d[post].min().date()),'post_2026_change_max_date':str(d[post].max().date()),'pre2026_positive_examples_for_new_revision':int(post[d.dt.year.le(2025)].sum()),'additional_fitted_uplift':None,'validation_conclusion':'No independent post-change validation period: all 29 positives are opening-week 2026 games. Randomly splitting these games would not validate transport to Week 2 or isolate measurement from seasonal changes. Retain current forecast calibration and track the new era until more actuals are available.','michigan_oklahoma':{'published_pregame':r['predicted'],'published_postgame':r['post_predicted'],'current_model':predict(r),'old_measurement_counterfactual':predict(r,1),'no_competition_counterfactual':predict(no_comp),'Michigan_rank20_counterfactual':predict(ranked),'competition_score':r['competing_games_score']}}
records=[{**row,**flag} for row,flag in zip(x[['source_index','ParsedDate']].to_dict('records'),flags.to_dict('records'))]
(BASE/'research/nielsen_era_flags.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
(BASE/'research/nielsen_era_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='measurement'},indent=2))
