"""Past-only calibration ablation; frozen OOS base predictions, no model writes."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from recent_audience_calibration import fit,apply,MAJOR
O=B/'research/recent_audience_calibration_results';O.mkdir(exist_ok=True)
def metrics(g):
 e=g.pred-g.actual;a=100*abs(e)/g.actual
 return dict(n=len(g),mae_000s=float(abs(e).mean()),bias_000s=float(e.mean()),aggregate_under_pct=float(100*(1-g.pred.sum()/g.actual.sum())),median_ape_pct=float(a.median()),mape_pct=float(a.mean()),underpredicted_pct=float(100*(e<0).mean()))
def main():
 source=B/'research/fox_friday_results/predictions.csv';h=pd.read_csv(source);h=h[h.variant.eq('fox_friday_night')].copy();assert h.source_index.is_unique
 variants=['current','global_mean','band_mean','band_median'];out=[];configs=[]
 for year in sorted(h.Year.unique()):
  test=h[h.Year.eq(year)]
  for variant in variants:
   g=test.copy();g['base_pred']=g.pred;g['variant']=variant
   if variant!='current':
    config=fit(h,year,variant);assert config['training_max_year'] is None or config['training_max_year']<year
    g['pred']=apply(g.pred,g.Station,config);configs.append(config)
   out.append(g)
 p=pd.concat(out,ignore_index=True);selection=[];selected=[]
 for year in sorted(p.Year.unique()):
  prior=p[p.Year.lt(year)&p.Station.isin(MAJOR)]
  # Require lower pooled prior MAE and smaller absolute bias; otherwise no change.
  choice='current'
  if len(prior):
   scores={v:metrics(g) for v,g in prior.groupby('variant')};base=scores['current']
   eligible=[v for v,s in scores.items() if s['mae_000s']<base['mae_000s'] and abs(s['bias_000s'])<abs(base['bias_000s'])]
   if eligible:choice=min(eligible,key=lambda v:scores[v]['mae_000s'])
  selection.append(dict(year=int(year),choice=choice));g=p[p.Year.eq(year)&p.variant.eq(choice)].copy();g['variant']='past_selected';selected.append(g)
 p=pd.concat([p,*selected],ignore_index=True);summary=[];annual=[]
 masks={'all':p.Year.ge(2021),'major':p.Station.isin(MAJOR),'major_2022_2025':p.Station.isin(MAJOR)&p.Year.ge(2022),'major_2024_2025':p.Station.isin(MAJOR)&p.Year.ge(2024),'major_pred_ge5M':p.Station.isin(MAJOR)&p.base_pred.ge(5000),'major_pred_ge3M':p.Station.isin(MAJOR)&p.base_pred.ge(3000)}
 for scope,mask in masks.items():
  sub=p[mask];base=sub[sub.variant.eq('current')].set_index('source_index');bm=metrics(base)
  for v,g in sub.groupby('variant'):
   m=metrics(g);entry=dict(scope=scope,variant=v,**m,mae_improvement_pct=100*(1-m['mae_000s']/bm['mae_000s']))
   q=g.set_index('source_index').reindex(base.index);d=pd.DataFrame({'year':base.Year,'delta':abs(q.pred-q.actual)-abs(base.pred-base.actual)})
   c=d.groupby('year').delta.agg(['sum','count']);rng=np.random.default_rng(23);draw=rng.integers(0,len(c),(10000,len(c)));boot=c['sum'].to_numpy()[draw].sum(1)/c['count'].to_numpy()[draw].sum(1)
   entry.update(year_ci025=float(np.quantile(boot,.025)),year_ci975=float(np.quantile(boot,.975)));summary.append(entry)
   for year,gg in g.groupby('Year'):annual.append(dict(scope=scope,year=int(year),variant=v,**metrics(gg)))
 p.to_csv(O/'predictions.csv',index=False);pd.DataFrame(summary).to_csv(O/'summary.csv',index=False);pd.DataFrame(annual).to_csv(O/'by_year.csv',index=False)
 (O/'fold_configs.json').write_text(json.dumps(configs,indent=2)+'\n');(O/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 current=[fit(h,2026,v) for v in variants[1:]];(O/'2026_candidate_configs.json').write_text(json.dumps(current,indent=2)+'\n')
 print(pd.DataFrame(summary).query('scope in ["major","major_2024_2025","major_pred_ge5M"]')[['scope','variant','n','mae_000s','bias_000s','aggregate_under_pct','mae_improvement_pct','year_ci025','year_ci975']].round(3).to_string(index=False))
 print('Selections',selection)
 for config in current:print('2026',config['method'],'Michigan 5.29M ->',apply([5290],['FOX'],config)[0]/1000,'factors',config['records'])
 (O/'source_hash.json').write_text(json.dumps({'source':str(source.relative_to(B)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
