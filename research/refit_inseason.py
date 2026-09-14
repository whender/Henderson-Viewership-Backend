"""Refit an existing validated architecture with appended, audited actuals.

Usage: python research/refit_inseason.py INPUT_DIRECTORY OUTPUT_DIRECTORY
Inputs: before/ artifact bundle, historical.joblib (pre-append research frame),
append_payload.json, rows.json, attention/cache/*.json. The historical frame's
source_index order must match the primary model's training arrays. No model
selection or OOS recalibration is performed on these newly observed outcomes.
"""
import copy, hashlib, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone

B=Path(__file__).resolve().parents[1];R=B.parent/'RatingsAndRegression'
sys.path[:0]=[str(B),str(R)]
from audience_interest import FEATURES,feature_values
from weekly_predictions_fs import build_features
from pregame_ensemble import exact_feature_frame

def main(inputs, output):
 inputs=Path(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 old=inputs/'before';payload=json.loads((inputs/'append_payload.json').read_text());rows=json.loads((inputs/'rows.json').read_text())
 load=lambda name:joblib.load(old/name)
 digest=lambda name:hashlib.sha256((output/name).read_bytes()).hexdigest()
 primary=load('viewership_model_log.joblib');h=joblib.load(inputs/'historical.joblib')
 np.testing.assert_allclose(np.log1p(h['Persons 2+']),primary['model'].model.endog)
 h.index=h.source_index.astype(int)
 x=primary['model'].model.data.orig_exog.copy();x.index=h.index
 fresh=pd.DataFrame(payload['clean']['rows']).set_index('source_index',drop=False)
 fresh['Station']=[r['network'] for r in rows];fresh['period']=['Week'+str(r['week']) for r in rows]
 fresh['ParsedDate']=pd.to_datetime(fresh['ParsedDate']);fresh['rank_scope']=fresh.RankResidualScope
 newx=pd.DataFrame([build_features(r) for r in rows],index=fresh.index).reindex(columns=x.columns)
 newx['OldNielsenSystem']=0.
 assert newx.notna().all().all(),newx.columns[newx.isna().any()].tolist()
 h=pd.concat([h,fresh],axis=0);x=pd.concat([x,newx],axis=0).astype(float)
 assert len(h)==2423 and h.source_index.is_unique
 assert not h.duplicated(['Date','Team 1','Team 2']).any()
 y=np.log1p(h['Persons 2+']).astype(float)
 z=x.copy()
 # Exact terms already exist in the historical frame and are produced by the
 # same serving extractor for the appended games.
 for c in primary['pregame_ensemble']['challenger_feature_columns']:
  if c not in z:z[c]=pd.to_numeric(h[c])
 for label,days in [('Weekday',['Tue','Wed','Thu']),('Friday',['Fri']),('Saturday',['Sat']),('Sunday',['Sun']),('Monday',['Mon'])]:
  z['Week1Major_'+label]=(h.period.eq('Week1')&h.Station.isin(['ABC','CBS','FOX','NBC','ESPN'])&h.DoW.isin(days)).astype(float)
 z['Score Diff']=pd.to_numeric(h['Winner Score'])-pd.to_numeric(h['Loser Score'])
 assert z['Score Diff'].ge(0).all()
 context=pd.concat([pd.read_csv(old/'cfbd_pregame_context.csv'),pd.DataFrame(payload['context']['rows'])]).set_index('source_index').reindex(h.source_index)
 for key,col in [('elo_average','pregame_elo_average'),('elo_difference','pregame_elo_difference'),('spread','abs_spread'),('neutral','neutral_site')]:z['Add_'+key]=pd.to_numeric(context[col])
 z['Add_close_game']=np.where(z.Add_spread.notna(),z.Add_spread.le(7).astype(float),np.nan)
 for c in [c for c in z if c.startswith('Add_')]:
  z[c+'_missing']=z[c].isna().astype(float);median=z[c].median();z[c]=z[c].fillna(0 if pd.isna(median) else median)
 daily={}
 for directory in [R/'audience_interest/cache',inputs/'attention/cache']:
  for path in directory.glob('*.json'):
   data=json.loads(path.read_text());daily.setdefault(data['team'],{}).update({pd.to_datetime(v['timestamp'],format='%Y%m%d%H').date().isoformat():v['views'] for v in data['items']})
 attention=[feature_values(pd.Timestamp(r.ParsedDate).date(),[r['Team 1'],r['Team 2']],daily) for _,r in h.iterrows()]
 assert all(v is not None for v in attention),'Incomplete pregame attention history'
 for c in FEATURES+['InterestMissing']:z[c]=[v[c] for v in attention]
 metadata=dict(training_max_year=2026,prediction_year=2026,training_through_date='2026-09-07',training_rows=len(h),
               refit_policy='fixed_validated_architecture; frozen pre-2026 OOS calibrations and intrinsic serving scale')
 def ols(cols):
  xx=z[cols].astype(float);assert np.isfinite(xx.to_numpy()).all()
  fitted=sm.OLS(y,xx).fit();return fitted,float(np.exp(fitted.resid).mean())
 def component(config):
  fitted,smear=ols(config['columns']);config.update(coefficients=np.asarray(fitted.params),smearing_factor=smear)
 def forest(config):
  xx=z[config['feature_columns']].astype(float);assert np.isfinite(xx.to_numpy()).all()
  fitted=clone(config['model']).fit(xx,y);config.update(model=fitted,smearing_factor=float(np.exp(y-fitted.predict(xx)).mean()))
 counts=pd.concat([h['Team 1'],h['Team 2']]).value_counts().to_dict()
 primary['model'],primary['smearing_factor']=ols(list(x.columns));primary['team_counts']=counts
 intrinsic,smear=ols(list(primary['intrinsic_model'].params.index));assert intrinsic.model.exog_names is not None;intrinsic.remove_data()
 primary.update(intrinsic_model=intrinsic,intrinsic_raw_smearing_factor=smear,intrinsic_smearing_factor=smear*primary['intrinsic_crossfit_scale_factor'],intrinsic_team_counts=counts,**metadata)
 ensemble=primary['pregame_ensemble'];challenger,smear=ols(ensemble['challenger_feature_columns']);assert challenger.model.exog_names is not None;challenger.remove_data()
 ensemble.update(challenger_model=challenger,challenger_smearing_factor=smear,challenger_team_counts=counts)
 joblib.dump(primary,output/'viewership_model_log.joblib',compress=3)
 nonlinear=load('nonlinear_pregame.joblib');forest(nonlinear);nonlinear.update(primary_sha256=digest('viewership_model_log.joblib'),**metadata)
 nonlinear['description']='Same validated nonlinear blend, refitted through September 7, 2026; historical OOS calibration retained.'
 joblib.dump(nonlinear,output/'nonlinear_pregame.joblib',compress=3)
 week=load('week1_major_days.joblib')
 for c in week['components']:component(c)
 week.update(primary_sha256=digest('viewership_model_log.joblib'),**metadata);week['description']='Validated Week 1 day interactions refitted with 2026 opening results.'
 joblib.dump(week,output/'week1_major_days.joblib',compress=3)
 post=load('aligned_postgame.joblib')
 for c in post['components']:component(c)
 for c in post['week1_major_days']['components']:component(c)
 forest(post['nonlinear_pregame']);post['week1_major_days'].update(**metadata)
 post.update(pregame_sha256={n:digest(n) for n in post['pregame_sha256']},**metadata)
 post['description']='Full pregame architecture plus final absolute score differential; refitted through September 7, 2026.'
 joblib.dump(post,output/'aligned_postgame.joblib',compress=3)
 interest=load('audience_interest.joblib')
 for mode in ['pregame','postgame']:
  for components in interest[mode]['components'].values():
   for c in components:component(c)
  forest(interest[mode]['forest'])
 interest.update(base_sha256={n:digest(n) for n in interest['base_sha256']},**metadata)
 for branch in ['regular','week1']:
  for pre,po in zip(interest['pregame']['components'][branch],interest['postgame']['components'][branch]):assert set(po['columns'])==set(pre['columns'])|{'Score Diff'}
 assert set(interest['postgame']['forest']['feature_columns'])==set(interest['pregame']['forest']['feature_columns'])|{'Score Diff'}
 joblib.dump(interest,output/'audience_interest.joblib',compress=3)
 legacy=load('viewership_postgame_model.joblib');legacy['model'],legacy['smearing_factor']=ols(list(legacy['model'].params.index));legacy.update(team_counts=counts,**metadata)
 joblib.dump(legacy,output/'viewership_postgame_model.joblib',compress=3)
 for name in ['opening_week_calibration.json','monday_calibration.json']:
  config=json.loads((old/name).read_text());config['model_sha256']=digest('viewership_model_log.joblib');config['refit_note']='Historical OOS-validated calibration held fixed during September 2026 coefficient refit.'
  (output/name).write_text(json.dumps(config,indent=2)+'\n')
 source=json.loads((old/'audience_interest_data.json').read_text())
 for team,history in daily.items():source['daily'].setdefault(team,{}).update({d:v for d,v in history.items() if d.startswith('2026-')})
 (output/'audience_interest_data.json').write_text(json.dumps(source,separators=(',',':'))+'\n')
 report=dict(**metadata,added_games=len(rows),added_week0=5,added_week1=29,actual_viewers_000s=sum(r['Persons 2+'] for r in payload['raw']['rows']),
             historical_rows_preserved=2389,attention_training_coverage=len(attention),primary_features=len(x.columns),
             new_data_evaluation='Training data: not an out-of-sample accuracy claim',artifact_sha256={p.name:digest(p.name) for p in output.glob('*.joblib')})
 (output/'refit_report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
