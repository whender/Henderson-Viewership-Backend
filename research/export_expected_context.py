"""Export audited historical schedule/Elo/line context for dashboard expectations.
Run from the development workspace after appending rated games. Source inputs
are the existing CFBD research cache and audited 2026 opening-slate rows.
Only pregame fields are exported; no scores, outcomes, or credentials.
"""
import json,pandas as pd
from pathlib import Path
b=Path(__file__).resolve().parents[1]
x=pd.read_csv(b/'viewership_cleaned.csv');c=pd.read_csv(b.parent/'RatingsAndRegression/cfbd_pregame_context.csv').set_index('source_index')
new={int(r['source_index']):r for r in json.load(open(b.parent/'tmp/week01_retrain/rows.json'))}
out={}
for _,r in x.iterrows():
 k=int(r.source_index);d=c.loc[k] if k in c.index else {}
 ctx={'date':r.ParsedDate,'team1':r['Team 1'],'team2':r['Team 2'],'time_slot':r['Time Slot'],'day':r.DoW,'week':None,'network':next((n for n in ['ABC','CBS','NBC','FOX','ESPN2','ESPNU','FS1','FS2','BTN','CW','NFLN','ESPNNEWS'] if r.get(n)==1),'ESPN')}
 for i in [1,2]:ctx['rank'+str(i)]=5 if r.get(f'Team {i} Top 10?')==1 else 15 if r.get(f'Team {i} 11-25?')==1 else 0
 if len(d):
  ctx.update(team1_pregame_elo=d.get('pregame_elo_maximum'),team2_pregame_elo=d.get('pregame_elo_minimum'),spread_home=d.get('abs_spread'),neutral_site=bool(d.neutral_site) if pd.notna(d.neutral_site) else None)
 if k in new:
  ctx.update({a:new[k].get(a) for a in ctx})
 out[str(k)]={a:None if isinstance(v,float) and pd.isna(v) else v for a,v in ctx.items()}
(b/'expected_viewership_context.json').write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
