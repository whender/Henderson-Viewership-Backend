"""Import factual onsite dates/teams; never import scores or picker information.
Requires requests, beautifulsoup4 and lxml. Archives include retrospectively
recorded appearances; unknown announcement dates remain null, never invented.
"""
import hashlib,json,re
from datetime import datetime,timezone
from io import StringIO
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
B=Path(__file__).resolve().parents[1]
SOURCES={'CollegeGameDayOnsite':'List_of_College_GameDay_(football_TV_program)_locations','BigNoonKickoffOnsite':'Big_Noon_Kickoff'}
def clean(value):
 value=re.sub(r'\[[^]]*\]','',str(value)).strip()
 value=re.sub(r'^(?:No\.\s*)?\d+\s*(?:FCS|D-II|D-III)?\s*','',value).strip()
 return {'Miami (FL)':'Miami','Appalachian State':'App State'}.get(value,value)
def main():
 records=[];sources=[]
 for feature,title in SOURCES.items():
  r=requests.get('https://en.wikipedia.org/w/api.php',params={'action':'parse','page':title,'prop':'text|revid','format':'json'},headers={'User-Agent':'HendersonResearch/1.0'},timeout=60);r.raise_for_status();p=r.json()['parse'];body=p['text']['*']
  sources.append(dict(feature=feature,url='https://en.wikipedia.org/wiki/'+title,revision=p['revid'],sha256=hashlib.sha256(body.encode()).hexdigest()))
  soup=BeautifulSoup(body,'html.parser')
  for table in soup.select('table.wikitable'):
   heading=table.find_previous(['h2','h3']).get_text(' ',strip=True);m=re.search(r'^(20\d\d) season',heading)
   if not m or not 2018<=int(m[1])<=2026:continue
   season=int(m[1]);frame=pd.read_html(StringIO(str(table)))[0]
   if not {'Date','Visitor','Host'}.issubset(frame):continue
   for _,row in frame.iterrows():
    teams=[clean(row[c]) for c in ['Visitor','Host']]
    if any(t in ('nan','None','TBD','N/A') for t in teams):continue
    ds=re.sub(r'\[[^]]*\]','',str(row.Date)).strip()
    try:d=pd.to_datetime(ds if re.search(r'20\d\d',ds) else ds+', '+str(season)).date()
    except (ValueError,TypeError):continue
    if not re.match(r'^(August|September|October|November|December|January)',ds):continue
    records.append(dict(feature=feature,date=d.isoformat(),season=season,team1=teams[0],team2=teams[1],announcement_date=None,source_url=sources[-1]['url'],source_revision=p['revid'],verification='historical_archive'))
 for c in json.loads((B/'research/pregame_show_primary_checks.json').read_text()):
  rows=[r for r in records if all(r[k]==c[k] for k in ['feature','date','team1','team2'])]
  assert len(rows)==1, c
  rows[0].update(announcement_date=c['announcement_date'],verification_source=c['verification_source'],verification='primary_source_spot_check')
 data=dict(version=1,retrieved_at=datetime.now(timezone.utc).isoformat(),sources=sources,coverage={'CollegeGameDayOnsite':[2018,2025],'BigNoonKickoffOnsite':[2019,2025]},policy='Historical retrospective onsite experiment, not point-in-time announcement backtest. No-game studio/remote shows excluded. Future unlisted games are unknown.',records=records)
 (B/'research/pregame_show_locations.json').write_text(json.dumps(data,indent=2)+'\n')
 print(pd.DataFrame(records).groupby(['feature','season']).size().to_string())
if __name__=='__main__':main()
