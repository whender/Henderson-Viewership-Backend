"""Marquee-only calibration; frozen OOS forecasts with earlier-season fitting."""
import hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];sys.path[:0]=[str(B),str(B/'research')]
from marquee_calibration import SPECS,mask,fit,apply,MAJOR
from recent_audience_calibration_experiment import metrics
O=B/'research/marquee_calibration_results';O.mkdir(exist_ok=True)
def main():
 source=B/'research/fox_friday_results/predictions.csv';h=pd.read_csv(source);h=h[h.variant.eq('fox_friday_night')].copy()
 history=joblib.load(B.parent/'tmp/week01_retrain/historical.joblib').set_index('source_index')
 h['BothRanked']=h.source_index.map(history.BothRanked);assert h.BothRanked.notna().all() and h.source_index.is_unique
 out=[];configs=[]
 for year in sorted(h.Year.unique()):
  te=h[h.Year.eq(year)]
  for v in ['current',*SPECS]:
   g=te.copy();g['base_pred']=g.pred;g['variant']=v
   if v!='current':
    c=fit(h,year,v);assert c['training_max_year'] is None or c['training_max_year']<year
    g['pred']=apply(te,c);configs.append(c)
    pd.testing.assert_series_equal(g.loc[~mask(te,c['definition']),'pred'],te.loc[~mask(te,c['definition']),'pred'])
   out.append(g)
 p=pd.concat(out,ignore_index=True);selected=[];selection=[]
 for year in sorted(h.Year.unique()):
  prior=p[p.Year.lt(year)&p.Station.isin(MAJOR)];choice='current'
  if len(prior):
   scores={v:metrics(g) for v,g in prior.groupby('variant')};base=scores['current'];eligible=[v for v,s in scores.items() if s['mae_000s']<base['mae_000s'] and abs(s['bias_000s'])<abs(base['bias_000s'])]
   if eligible:choice=min(eligible,key=lambda v:scores[v]['mae_000s'])
  selection.append(dict(year=int(year),choice=choice));g=p[p.Year.eq(year)&p.variant.eq(choice)].copy();g['variant']='past_selected';selected.append(g)
 p=pd.concat([p,*selected],ignore_index=True);summary=[];annual=[]
 scopes={'major':p.Station.isin(MAJOR),'marquee':p.Station.isin(MAJOR)&p.base_pred.ge(5000),'marquee_2022_2025':p.Station.isin(MAJOR)&p.base_pred.ge(5000)&p.Year.ge(2022),'marquee_2024_2025':p.Station.isin(MAJOR)&p.base_pred.ge(5000)&p.Year.ge(2024),'both_ranked':p.Station.isin(MAJOR)&p.BothRanked.eq(1),'nonmarquee':p.Station.isin(MAJOR)&p.base_pred.lt(5000)}
 for scope,m in scopes.items():
  sub=p[m];base=sub[sub.variant.eq('current')].set_index('source_index');bm=metrics(base)
  for v,g in sub.groupby('variant'):
   mm=metrics(g);row=dict(scope=scope,variant=v,**mm,mae_improvement_pct=100*(1-mm['mae_000s']/bm['mae_000s']))
   q=g.set_index('source_index').reindex(base.index);delta=pd.DataFrame({'year':base.Year,'date':base.Date,'delta':abs(q.pred-q.actual)-abs(base.pred-base.actual)})
   for grouping in ['year','date']:
    c=delta.groupby(grouping).delta.agg(['sum','count']);rng=np.random.default_rng(123);draw=rng.integers(0,len(c),(10000,len(c)));boot=c['sum'].to_numpy()[draw].sum(1)/c['count'].to_numpy()[draw].sum(1)
    row[grouping+'_ci025']=float(np.quantile(boot,.025));row[grouping+'_ci975']=float(np.quantile(boot,.975))
   summary.append(row)
   for year,gg in g.groupby('Year'):annual.append(dict(scope=scope,year=int(year),variant=v,**metrics(gg)))
 p.to_csv(O/'predictions.csv',index=False);pd.DataFrame(summary).to_csv(O/'summary.csv',index=False);pd.DataFrame(annual).to_csv(O/'by_year.csv',index=False)
 for name,value in [('fold_configs',configs),('selection',selection),('2026_configs',[fit(h,2026,v) for v in SPECS]),('source_hash',{'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})]:(O/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 print(pd.DataFrame(summary).query('scope in ["major","marquee","marquee_2024_2025"]')[['scope','variant','n','mae_000s','aggregate_under_pct','mae_improvement_pct','year_ci025','year_ci975','date_ci025','date_ci975']].round(3).to_string(index=False));print('Selection',selection);print('2026',[fit(h,2026,v) for v in SPECS])
if __name__=='__main__':main()
