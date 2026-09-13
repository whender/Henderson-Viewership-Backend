"""Pregame Wikipedia football-page attention experiment; no result-day data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import broadcast_mismatch_experiment as prior
from holdout_viewership_tests import build_matrix
from viewership_competition import cross_fitted_intrinsic_predictions,fit_intrinsic_audience_model,predict_intrinsic_viewers,competing_scores_from_predictions
from viewership_feature_contract import BASE_NUMERIC_FEATURES,EXACT_NETWORK_RANK_CHALLENGER_FEATURES
from week1_major_day_interactions import design
from strict_rolling_origin_challenger import fit_rank_calibrator,apply_rank_calibrator
from team_half_life_experiment import metrics
B=Path(__file__).resolve().parents[2]/'RatingsAndRegression';OUT=B/'audience_interest_weekly';OUT.mkdir(exist_ok=True);prior.OUT=OUT
MAJOR=['ABC','CBS','FOX','NBC','ESPN'];SPECS={'current':None,'interest_level':'level','interest_momentum':'momentum','interest_level_momentum':'both'}
INTEREST_COLUMNS=['InterestLogTotal7','InterestLogMax7','InterestMomentumMax','InterestMomentumMean']

def interest_features(frame):
 import re
 cache={};coverage=[]
 for path in sorted((B/'audience_interest'/'cache').glob('*.json')):
  data=json.loads(path.read_text());cache[data['team']]=pd.Series([v['views'] for v in data['items']],index=pd.to_datetime([v['timestamp'] for v in data['items']],format='%Y%m%d%H')).sort_index()
 result=pd.DataFrame(index=frame.index,columns=INTEREST_COLUMNS,dtype=float)
 for idx,r in frame.iterrows():
  # Cutoff is 00:00 UTC two calendar days before the listed local game date.
  # Last included UTC date is game-date minus three; game-day traffic excluded.
  cutoff=pd.Timestamp(r.ParsedDate).normalize()-pd.Timedelta(days=2)
  week_start=pd.Timestamp(r.ParsedDate).normalize()-pd.Timedelta(days=(pd.Timestamp(r.ParsedDate).weekday()-1)%7)
  cutoff=min(cutoff,week_start-pd.Timedelta(days=1))
  recent_dates=pd.date_range(cutoff-pd.Timedelta(days=7),periods=7)
  normal_dates=pd.date_range(cutoff-pd.Timedelta(days=35),periods=28)
  levels=[];mom=[];missing=[]
  for team in [r['Team 1'],r['Team 2']]:
   series=cache.get(team,pd.Series(dtype=float));recent=series.reindex(recent_dates);normal=series.reindex(normal_dates)
   ok=recent.notna().all()&normal.notna().all();missing.append(not ok)
   levels.append(float(recent.sum()) if ok else np.nan)
   mom.append(float(np.log1p(recent.mean())-np.log1p(normal.mean())) if ok else np.nan)
  if not any(missing):result.loc[idx]=[np.log1p(sum(levels)),np.log1p(max(levels)),max(mom),np.mean(mom)]
  coverage.append(dict(source_index=r.source_index,year=r.Year,team1=r['Team 1'],team2=r['Team 2'],network=r.Station,game_date=str(r.ParsedDate.date()),cutoff_utc=str(cutoff),last_included_utc_date=str(recent_dates[-1].date()),team1_missing=missing[0],team2_missing=missing[1],team1_views7=levels[0],team2_views7=levels[1],team1_log_momentum=mom[0],team2_log_momentum=mom[1]))
 audit=pd.DataFrame(coverage,index=frame.index)
 assert pd.to_datetime(audit.last_included_utc_date).lt(pd.to_datetime(audit.cutoff_utc)).all()
 assert pd.to_datetime(audit.cutoff_utc).lt(pd.to_datetime(audit.game_date)).all()
 return result,audit

def checks():
 print('Interest windows: seven complete UTC days before Monday 00:00 UTC for a Tuesday–Monday slate, capped at game date minus two days.',flush=True)

def nonlinear(a,b,train,test,context):
 a=a.drop(columns='const').copy();b=b.drop(columns='const').copy()
 for frame,x in [(train,a),(test,b)]:
  c=context.reindex(frame.source_index).set_axis(frame.index)
  for key,source in [('elo_average','pregame_elo_average'),('elo_difference','pregame_elo_difference'),('spread','abs_spread'),('neutral','neutral_site')]:x['Add_'+key]=c[source]
  x['Add_close_game']=np.where(x.Add_spread.notna(),x.Add_spread.le(7).astype(float),np.nan)
 for c in ['Add_elo_average','Add_elo_difference','Add_spread','Add_close_game','Add_neutral']:
  a[c+'_missing']=a[c].isna().astype(float);b[c+'_missing']=b[c].isna().astype(float);median=a[c].median();median=0 if pd.isna(median) else median;a[c]=a[c].fillna(median);b[c]=b[c].fillna(median)
 model=HistGradientBoostingRegressor(max_iter=200,max_depth=5,max_leaf_nodes=15,min_samples_leaf=25,learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=11)
 y=np.log1p(train['Persons 2+']);model.fit(a,y);return np.maximum(np.exp(model.predict(b))*np.exp(y-model.predict(a)).mean()-1,0)

def main():
 checks();h=prior.prepare();interest,interest_audit=interest_features(h);interest_audit.to_csv(OUT/'interest_source_audit.csv',index=False);print('Rows with complete attention history',int(interest.notna().all(axis=1).sum()),'of',len(h),flush=True);context=pd.read_csv(B/'cfbd_pregame_context.csv').set_index('source_index');complete=context[['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']].notna().all(axis=1)
 ref=pd.read_csv(B/'week1_major_day_predictions.csv');ref=ref[ref.variant.eq('week1_major_days')].set_index('source_index');rawref=pd.read_csv(B/'targeted_improvements_raw.csv')
 hist={(n,d,br):[] for n in SPECS for d in ['regular','week1'] for br in ['linear','nonlinear']};outputs=[];audit=[];coefs=[]
 for year in [2019,2021,2022,2023,2024,2025]:
  train,test=h[h.Year.lt(year)],h[h.Year.eq(year)];assert train.Year.max()<year
  a,b=build_matrix(h,train.index,test.index);y=np.log1p(train['Persons 2+']).to_numpy()
  tr_intr=cross_fitted_intrinsic_predictions(train,BASE_NUMERIC_FEATURES);fit=fit_intrinsic_audience_model(train,BASE_NUMERIC_FEATURES);te_intr=predict_intrinsic_viewers(fit,test)
  np.testing.assert_allclose(a.Competing_Games_Score,competing_scores_from_predictions(train,tr_intr),atol=1e-8)
  np.testing.assert_allclose(b.Competing_Games_Score,competing_scores_from_predictions(test,te_intr),atol=1e-8)
  exact=[c for c in EXACT_NETWORK_RANK_CHALLENGER_FEATURES if c not in a and train[c].nunique()>1];preds={};forest={}
  for name,mode in SPECS.items():
   aa=a.copy();bb=b.copy()
   columns={'level':INTEREST_COLUMNS[:2],'momentum':INTEREST_COLUMNS[2:],'both':INTEREST_COLUMNS}.get(mode,[])
   for c in columns:
    rawtrain=interest.loc[train.index,c];rawtest=interest.loc[test.index,c];median=rawtrain.median();median=0. if pd.isna(median) else float(median)
    aa[c]=rawtrain.fillna(median);bb[c]=rawtest.fillna(median)
   if columns:
    aa['InterestMissing']=interest.loc[train.index].isna().any(axis=1).astype(float);bb['InterestMissing']=interest.loc[test.index].isna().any(axis=1).astype(float)
   for i,r in test.iterrows():
    audit.append(dict(year=year,source_index=r.source_index,variant=name,**interest.loc[i].to_dict()))
   forest[name]=nonlinear(aa,bb,train,test,context)
   for branch in ['regular','week1']:
    xa,added=design(aa,train,'current' if branch=='regular' else 'week1_major_days');xb,_=design(bb,test,'current' if branch=='regular' else 'week1_major_days');total=np.zeros(len(test))
    for comp,x,z in [('primary',xa,xb),('exact',pd.concat([xa,train[exact]],axis=1),pd.concat([xb,test[exact]],axis=1))]:
     pred,coeff=prior.fit(x,z,y,added);total+=.5*pred
     for c in INTEREST_COLUMNS:
      if c in coeff:coefs.append(dict(year=year,variant=name,branch=branch,component=comp,feature=c,coefficient=coeff[c]))
    preds[name,branch]=total
  old=rawref[rawref.Year.eq(year)&rawref.variant.eq('baseline')].set_index('source_index').reindex(test.source_index).predicted_viewers_000s.to_numpy();blend=rawref[rawref.Year.eq(year)&rawref.variant.eq('nonlinear_depth5_blend25')].set_index('source_index').reindex(test.source_index).predicted_viewers_000s.to_numpy()
  np.testing.assert_allclose(preds['current','regular'],old,atol=1e-5)
  np.testing.assert_allclose(.75*old+.25*forest['current'],blend,atol=1e-5)
  cal={}
  for name in SPECS:
   for branch in ['regular','week1']:
    for br,pr in [('linear',preds[name,branch]),('nonlinear',.75*preds[name,branch]+.25*forest[name])]:
     row=test[['source_index','Year','rank_scope']].copy();row['actual_viewers_000s']=test['Persons 2+'];row['predicted_viewers_000s']=pr
     if year>=2021:
      previous=pd.concat(hist[name,branch,br]);assert previous.Year.max()<year
      cal[name,branch,br]=apply_rank_calibrator(pr,row.rank_scope,fit_rank_calibrator(previous,year,name))
     else:cal[name,branch,br]=pr
     hist[name,branch,br].append(row)
  eligible=(cal['current','regular','linear']>=1000)&test.source_index.map(complete).fillna(False).to_numpy();opening=test.period.eq('Week1')&test.Station.isin(MAJOR)
  for name in SPECS:
   regular=np.where(eligible,cal[name,'regular','nonlinear'],cal[name,'regular','linear']);week=np.where(eligible,cal[name,'week1','nonlinear'],cal[name,'week1','linear'])
   row=test[['source_index','Year','Team 1','Team 2','Station','period']].copy();row['actual']=test['Persons 2+'];row['variant']=name;row['pred']=np.where(opening,week,regular)
   assert np.isfinite(row.pred).all() and row.pred.ge(0).all()
   if year>=2021 and name=='current':np.testing.assert_allclose(row.pred,ref.reindex(row.source_index).pred,atol=1e-5)
   outputs.append(row)
  print('Completed',year,flush=True)
 import joblib
 joblib.dump(hist,OUT/'raw_histories.joblib')
 allp=pd.concat(outputs,ignore_index=True);allp.to_csv(OUT/'rolling_predictions.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'feature_audit.csv',index=False);pd.DataFrame(coefs).to_csv(OUT/'coefficients.csv',index=False)
 chosen=[];nested=[]
 for year in [2021,2022,2023,2024,2025]:
  previous=allp[allp.Year.lt(year)&allp.Station.isin(MAJOR)].copy();previous['error']=(previous.pred-previous.actual).abs();loss=previous.groupby('variant').error.mean();name=loss.idxmin();row=allp[allp.Year.eq(year)&allp.variant.eq(name)].copy();row['variant']='past_selected';nested.append(row);chosen.append(dict(year=year,prior_max_year=int(previous.Year.max()),selected=name,prior_major_mae_000s=loss[name]))
 p=pd.concat([allp[allp.Year.ge(2021)],*nested],ignore_index=True);assert all(len(g)==1821 and g.source_index.is_unique for _,g in p.groupby('variant'));p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(chosen).to_csv(OUT/'past_only_selection.csv',index=False)
 summary=[];annual=[]
 for scope,mask in {'all':p.Year.gt(0),'five_major':p.Station.isin(MAJOR),'four_broadcast':p.Station.isin(MAJOR[:-1]),'week1_major':p.period.eq('Week1')&p.Station.isin(MAJOR),'later_major':p.period.eq('later')&p.Station.isin(MAJOR),'complete_history_major':p.Station.isin(MAJOR)&p.source_index.isin(interest_audit.loc[~interest_audit.team1_missing&~interest_audit.team2_missing,'source_index'])}.items():
  q=p[mask];base=q[q.variant.eq('current')].set_index('source_index');be=(base.pred-base.actual).abs()
  for name,g in q.groupby('variant'):
   t=g.set_index('source_index');delta=(t.pred-t.actual).abs()-be;years=pd.DataFrame({'delta':delta,'year':t.Year}).groupby('year').delta.agg(['sum','size']);rng=np.random.default_rng(93);ix=rng.integers(0,len(years),(20000,len(years)));boot=years['sum'].to_numpy()[ix].sum(1)/years['size'].to_numpy()[ix].sum(1)
   summary.append(dict(scope=scope,variant=name,**metrics(g),mae_improvement_pct=-100*delta.mean()/be.mean(),year_wins=int(years['sum'].lt(-1e-6).sum()),ci05=np.quantile(boot,.05),ci95=np.quantile(boot,.95)))
   for year,gg in g.groupby('Year'):annual.append(dict(scope=scope,variant=name,year=year,**metrics(gg)))
 s=pd.DataFrame(summary);s.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'by_year.csv',index=False)
 print(s[s.scope.isin(['all','five_major','week1_major'])][['scope','variant','n','mae_000s','mape_pct','mae_improvement_pct','year_wins','ci05','ci95']].round(3).to_string(index=False),flush=True);print(pd.DataFrame(chosen).to_string(index=False),flush=True)
if __name__=='__main__':main()
