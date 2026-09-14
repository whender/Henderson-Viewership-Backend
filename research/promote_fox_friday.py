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
 from fox_friday import FEATURE,fox_friday_night
 from strict_rolling_origin_challenger import fit_rank_calibrator
 source_hist=joblib.load(inputs/'historical.joblib')
 historical_flag=(source_hist.Station.eq('FOX') & source_hist.DoW.eq('Fri') & source_hist.Time_float.ge(18.5) & source_hist['Black Friday'].eq(0) & source_hist['Conf Champ'].eq(0)).astype(float)
 z[FEATURE]=list(historical_flag)+[fox_friday_night(r) for r in rows]
 assert sum(historical_flag)==13 and len(z)==2423
 artifact=joblib.load(B/'audience_interest.joblib')
 assert artifact['training_rows']==len(z)
 for mode in ['pregame','postgame']:
  for components in artifact[mode]['components'].values():
   for c in components:
    cols=c['columns']+[FEATURE];xx=z[cols].astype(float);fit=sm.OLS(y,xx).fit()
    c.update(columns=cols,coefficients=np.asarray(fit.params),smearing_factor=float(np.exp(fit.resid).mean()))
  c=artifact[mode]['forest'];cols=c['feature_columns']+[FEATURE];xx=z[cols].astype(float);fit=clone(c['model']).fit(xx,y)
  c.update(feature_columns=cols,model=fit,smearing_factor=float(np.exp(y-fit.predict(xx)).mean()))
 hist=joblib.load(B/'research/fox_friday_results/raw_histories.joblib')
 for branch in ['regular','week1']:
  for kind in ['linear','nonlinear']:
   history=pd.concat(hist['fox_friday_night',branch,kind]);assert history.Year.max()==2025
   artifact['pregame']['corrections'][branch][kind]=fit_rank_calibrator(history,2026,'fox_friday_night').set_index('rank_scope').correction_000s.to_dict()
 artifact.update(additional_features=[FEATURE],feature_revision='fox_friday_night_v1',
   feature_training_matches=int(z[FEATURE].sum()),feature_promotion='User approved September 2026; exploratory holdout evidence documented in research/fox_friday_results/report.md',
   postgame_calibration_policy='Preserve existing pre-2026 postgame calibration; paired feature refit, not separately validated for accuracy')
 for branch in ['regular','week1']:
  for pre,post in zip(artifact['pregame']['components'][branch],artifact['postgame']['components'][branch]):assert set(post['columns'])==set(pre['columns'])|{'Score Diff'}
 assert set(artifact['postgame']['forest']['feature_columns'])==set(artifact['pregame']['forest']['feature_columns'])|{'Score Diff'}
 for name,sha in artifact['base_sha256'].items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==sha
 joblib.dump(artifact,output/'audience_interest.joblib',compress=3)
 report={'training_rows':len(z),'training_through_date':artifact['training_through_date'],'slot_training_matches':int(z[FEATURE].sum()),'historical_matches':13,'opening_2026_matches':int(z[FEATURE].sum()-13),'artifact_sha256':hashlib.sha256((output/'audience_interest.joblib').read_bytes()).hexdigest(),'postgame_note':artifact['postgame_calibration_policy']}
 (output/'promotion_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main(*sys.argv[1:])
