"""Onsite pregame show ablation against audience interest plus FOX Friday.
Strict rolling-season refits, historical appearances (announcement timing not verified).
No production model or forecast writes.
"""
import json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];ROOT=B.parent;R=ROOT/'RatingsAndRegression';I=ROOT/'tmp/week01_retrain'
sys.path[:0]=[str(B/'research'),str(B),str(R)]
import train_audience_interest as training
import audience_interest_weekly_experiment as exp
from strict_rolling_origin_challenger import fit_rank_calibrator,apply_rank_calibrator
from aligned_postgame_experiment import forest_fit
from audience_interest import FEATURES,feature_values
from team_half_life_experiment import metrics
from weekly_predictions_fs import build_features
from pregame_ensemble import exact_feature_frame,model_point_predictions_000s,rank_scope,parse_kickoff_hour
O=B/'research/pregame_show_results';O.mkdir(exist_ok=True)
MAJOR=['ABC','CBS','FOX','NBC','ESPN']
SPECS={'current':(), 'gameday':('CollegeGameDayOnsite',), 'bignoon':('BigNoonKickoffOnsite',), 'both':('CollegeGameDayOnsite','BigNoonKickoffOnsite')}
from pregame_shows import flags,load_data,FEATURES as SHOW_FEATURES
SHOW_DATA=load_data()
FEATURE='FOXFridayNight'
def interaction(frame,x):
 return (frame.Station.eq('FOX') & frame.DoW.eq('Fri') & frame.Time_float.ge(18.5) & x['Black Friday'].eq(0) & frame['Conf Champ'].eq(0)).astype(float)
def fit_variant(a,b,tr,te,context,y,variant,base):
 if variant=='current':return base,[]
 aa=a.copy();bb=b.copy()
 for c in SPECS[variant]:aa[c]=tr[c].astype(float);bb[c]=te[c].astype(float)
 result=training.fit(aa,bb,tr,te,context,y);audit=[]
 for branch,cs in result[1].items():
  for i,c in enumerate(cs):
   for feature in SPECS[variant]:
    if feature in c['columns']:
     coef=float(c['coefficients'][c['columns'].index(feature)])
     audit.append(dict(feature=feature,branch=branch,component=i,coefficient=coef,multiplier=float(np.exp(coef)),train_matches=int(aa[feature].sum()),test_matches=int(bb[feature].sum())))
 return result,audit

def apply_cal(raw,fp,test,hist,year,eligible,variant):
 cal={}
 for branch,pred in raw.items():
  for kind,values in [('linear',pred),('nonlinear',.75*pred+.25*fp)]:
   row=test[['source_index','Year','rank_scope']].copy();row['actual_viewers_000s']=test['Persons 2+'];row['predicted_viewers_000s']=values
   if year>=2021:
    prior=pd.concat(hist[variant,branch,kind]);assert prior.Year.max()<year
    cal[branch,kind]=apply_rank_calibrator(values,row.rank_scope,fit_rank_calibrator(prior,year,variant))
   if year<2026:hist[variant,branch,kind].append(row)
 if year<2021:return None
 reg=np.where(eligible,cal['regular','nonlinear'],cal['regular','linear']);week=np.where(eligible,cal['week1','nonlinear'],cal['week1','linear'])
 return np.where(test.period.eq('Week1')&test.Station.isin(MAJOR),week,reg)

def main():
 h=joblib.load(I/'historical.joblib');assert h.Year.max()==2025 and len(h)==2389
 show=pd.DataFrame([flags(r.ParsedDate,r['Team 1'],r['Team 2'],SHOW_DATA) for _,r in h.iterrows()],index=h.index)
 for c in SHOW_FEATURES:h[c]=show[c]
 assert h[list(SHOW_FEATURES)].notna().all().all()
 classification=h[['source_index','Year','Date','Team 1','Team 2','Station','DoW','Time_float','Black Friday','Conf Champ']].copy()
 classification[FEATURE]=interaction(h,h)
 for c in SHOW_FEATURES:classification[c]=h[c]
 classification.to_csv(O/'classification.csv',index=False)
 attention,_=exp.interest_features(h);assert attention[FEATURES].notna().all().all()
 context=pd.read_csv(R/'cfbd_pregame_context.csv').set_index('source_index');complete=context[['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']].notna().all(axis=1)
 gate=pd.read_csv(R/'targeted_improvements_predictions.csv');gate=gate[gate.variant.eq('baseline')].set_index('source_index')
 ref=pd.read_csv(B/'research/fox_friday_results/predictions.csv');ref=ref[ref.variant.eq('fox_friday_night')].set_index('source_index')
 hist={(v,b,k):[] for v in SPECS for b in ['regular','week1'] for k in ['linear','nonlinear']};outputs=[];coeffs=[]
 def add(x,frame,interest):
  x=x.copy()
  for c in FEATURES:x[c]=interest.loc[frame.index,c]
  x['InterestMissing']=0.;x[FEATURE]=interaction(frame,x);return x
 for year in [2019,2021,2022,2023,2024,2025]:
  tr,te=h[h.Year.lt(year)],h[h.Year.eq(year)];a,b=exp.build_matrix(h,tr.index,te.index);a=add(a,tr,attention);b=add(b,te,attention);y=np.log1p(tr['Persons 2+']).to_numpy()
  eligible=(gate.reindex(te.source_index).pred.to_numpy()>=1000)&te.source_index.map(complete).fillna(False).to_numpy();base=training.fit(a,b,tr,te,context,y)
  for variant in SPECS:
   (raw,_,fp,_),audit=fit_variant(a,b,tr,te,context,y,variant,base)
   coeffs.extend(dict(holdout_year=year,variant=variant,**r) for r in audit)
   pred=apply_cal(raw,fp,te,hist,year,eligible,variant)
   if pred is not None:
    if variant=='current':np.testing.assert_allclose(pred,ref.reindex(te.source_index).pred,atol=1e-5)
    row=te[['source_index','Year','Date','Team 1','Team 2','Station','period']].copy();row['actual']=te['Persons 2+'];row['pred']=pred;row['variant']=variant;outputs.append(row)
  print('Completed holdout',year,flush=True)
 p=pd.concat(outputs,ignore_index=True)
 # Choose a configuration using only previously scored seasons. No evidence
 # before 2021 means retaining current; ties likewise favor the simpler baseline.
 selected=[];selection=[]
 for year in sorted(p.Year.unique()):
  prior=p[p.Year.lt(year)&p.Station.isin(MAJOR)].copy()
  winner='current' if prior.empty else prior.assign(loss=(prior.pred-prior.actual).abs()).groupby('variant').loss.mean().idxmin()
  selection.append(dict(year=int(year),selected=winner,prior_max_year=None if prior.empty else int(prior.Year.max())))
  row=p[p.Year.eq(year)&p.variant.eq(winner)].copy();row['variant']='past_selected';selected.append(row)
 p=pd.concat([p,*selected],ignore_index=True);p.to_csv(O/'predictions.csv',index=False);pd.DataFrame(selection).to_csv(O/'selection.csv',index=False)
 summary=[];annual=[];ci=[]
 scopes={'all':p.Year.gt(0),'major':p.Station.isin(MAJOR),'FOX':p.Station.eq('FOX'),'target':p.source_index.isin(classification.loc[classification[list(SHOW_FEATURES)].max(axis=1).eq(1),'source_index']),'gameday_games':p.source_index.isin(classification.loc[classification.CollegeGameDayOnsite.eq(1),'source_index']),'bignoon_games':p.source_index.isin(classification.loc[classification.BigNoonKickoffOnsite.eq(1),'source_index']),'major_2024_2025':p.Station.isin(MAJOR)&p.Year.ge(2024),'week1_major':p.Station.isin(MAJOR)&p.period.eq('Week1')}
 for scope,mask in scopes.items():
  sub=p[mask];base=sub[sub.variant.eq('current')].set_index('source_index');bm=metrics(base)
  for variant,g in sub.groupby('variant'):
   mm=metrics(g);summary.append(dict(scope=scope,variant=variant,**mm,mae_improvement_pct=100*(1-mm['mae_000s']/bm['mae_000s'])))
   if variant!='current':
    g=g.set_index('source_index').reindex(base.index);d=pd.DataFrame({'date':base.Date,'year':base.Year,'delta':abs(g.pred-g.actual)-abs(base.pred-base.actual)})
    interval={}
    for grouping in ['date','year']:
     clusters=d.groupby(grouping).delta.agg(['sum','count']);rng=np.random.default_rng(711);draw=rng.integers(0,len(clusters),size=(5000,len(clusters)));loss=clusters['sum'].to_numpy()[draw].sum(1)/clusters['count'].to_numpy()[draw].sum(1)
     interval.update({grouping+'_ci025':float(np.quantile(loss,.025)),grouping+'_ci975':float(np.quantile(loss,.975))})
    ci.append(dict(scope=scope,variant=variant,delta_mae_000s=float(d.delta.mean()),**interval))
  for (year,variant),g in sub.groupby(['Year','variant']):annual.append(dict(scope=scope,year=int(year),variant=variant,**metrics(g)))
 s=pd.DataFrame(summary);s.to_csv(O/'summary.csv',index=False);pd.DataFrame(annual).to_csv(O/'by_year.csv',index=False);pd.DataFrame(ci).to_csv(O/'uncertainty.csv',index=False)
 joblib.dump(hist,O/'raw_histories.joblib',compress=3)
 # Week 2 preview: full fit through 2025, no 2026 outcomes or refit data.
 rows=json.loads((ROOT/'week2_2026/full_slate_predictions.json').read_text())
 frozen=joblib.load(I/'before/viewership_model_log.joblib');a=frozen['model'].model.data.orig_exog.copy();a.index=h.index
 idx=pd.Index(range(2000000,2000000+len(rows)),name='source_index')
 b=pd.DataFrame([build_features(r) for r in rows],index=idx).reindex(columns=a.columns,fill_value=0.);b['OldNielsenSystem']=0
 te=b.copy();extra=exact_feature_frame(rows,index=idx)
 for col in extra:te[col]=extra[col]
 te['Conf Champ']=[int(r.get('conf_champ',False)) for r in rows];te['source_index']=idx;te['Year']=2026;te['Station']=[r['network'] for r in rows];te['period']='Week2';te['Persons 2+']=np.nan
 te['ParsedDate']=pd.to_datetime([r['date'] for r in rows],format='%m/%d/%y');te['DoW']=te.ParsedDate.dt.strftime('%a');te['Time_float']=[parse_kickoff_hour(r['time_slot']) for r in rows]
 te['Team 1']=[r['team1'] for r in rows];te['Team 2']=[r['team2'] for r in rows];te['rank_scope']=[rank_scope(r['rank1'],r['rank2']) for r in rows]
 # 2026 preview is conditional on the currently listed visits. Unknown other visits
 # remain explicitly a preview assumption, never used for publication.
 for c in SHOW_FEATURES:te[c]=[flags(r['date'],r['team1'],r['team2'],SHOW_DATA)[c] or 0. for r in rows]
 for i,r in zip(idx,rows):
  e1=r.get('team1_pregame_elo');e2=r.get('team2_pregame_elo')
  context.loc[i,['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']]=[(e1+e2)/2 if e1 is not None and e2 is not None else np.nan,abs(e1-e2) if e1 is not None and e2 is not None else np.nan,abs(r['spread_home']) if r.get('spread_home') is not None else np.nan,float(r.get('neutral_site',False))]
 complete=context[['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']].notna().all(axis=1)
 daily=json.loads((B/'audience_interest_data.json').read_text())['daily'];ni=pd.DataFrame([feature_values(r.ParsedDate.date(),[r['Team 1'],r['Team 2']],daily) for _,r in te.iterrows()],index=te.index)
 ens=frozen['pregame_ensemble'];extra=exact_feature_frame(rows,index=b.index);xx=pd.concat([b,extra[[c for c in extra if c not in b]]],axis=1)
 first=model_point_predictions_000s(frozen['model'],b,frozen['smearing_factor']);second=model_point_predictions_000s(ens['challenger_model'],xx[ens['challenger_feature_columns']],ens['challenger_smearing_factor'])
 eligibility=.5*(first+second)+te.rank_scope.map(ens['rank_scope_additive_adjustments_000s']).fillna(0).to_numpy();eligible=(eligibility>=1000)&te.source_index.map(complete).fillna(False).to_numpy()
 a=add(a,h,attention);b=add(b,te,ni);y=np.log1p(h['Persons 2+']).to_numpy();base=training.fit(a,b,h,te,context,y);previews=[]
 for variant in SPECS:
  (raw,_,fp,_),audit=fit_variant(a,b,h,te,context,y,variant,base);coeffs.extend(dict(holdout_year=2026,variant=variant,**r) for r in audit)
  pred=apply_cal(raw,fp,te,hist,2026,eligible,variant);row=te[['source_index','Team 1','Team 2','Station']].copy();row['actual']=te['Persons 2+'];row['pred']=pred;row['variant']=variant;previews.append(row)
 preview=pd.concat(previews,ignore_index=True);preview.to_csv(O/'week2_2026_preview.csv',index=False);pd.DataFrame(coeffs).to_csv(O/'coefficients.csv',index=False)
 assert p.groupby('variant').size().eq(1821).all() and not p.duplicated(['source_index','variant']).any()
 print(s[['scope','variant','n','mae_000s','median_ape_pct','mae_improvement_pct']].round(3).to_string(index=False),flush=True)
 print('Past-only configuration selections:',selection,flush=True)
 print('Michigan-Oklahoma:',flush=True);print(preview[(preview['Team 1'].eq('Michigan')|preview['Team 2'].eq('Michigan'))][['variant','pred','actual']].to_string(index=False),flush=True)

if __name__=='__main__':main()
