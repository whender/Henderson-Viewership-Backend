"""Date-and-matchup joins for onsite pregame shows, independent of TV network.

Historical mode is retrospective. As-of mode requires a sourced announcement
on/before the forecast cutoff and never silently converts unknown to absent.
"""
import json,re
from datetime import date,datetime
from pathlib import Path
FEATURES=('CollegeGameDayOnsite','BigNoonKickoffOnsite')
ALIASES={'miami (fl)':'miami','miami (florida)':'miami','appalachian state':'app state','pitt':'pittsburgh','brigham young':'byu','southern california':'usc','mississippi':'ole miss','north carolina state':'nc state'}
def team_key(value):
 value=re.sub(r'^\s*(?:#|No\.\s*)?\d+\s*','',str(value),flags=re.I).strip().lower()
 value=re.sub(r"\bst\.$", "state", value)
 return ALIASES.get(value,value)
def game_key(day,team1,team2):
 if isinstance(day,datetime):day=day.date()
 elif not isinstance(day,date):
  for fmt in ('%Y-%m-%d','%m/%d/%y','%m/%d/%Y'):
   try:day=datetime.strptime(str(day),fmt).date();break
   except ValueError:pass
  else:raise ValueError('Unrecognized game date')
 return day.isoformat(),tuple(sorted((team_key(team1),team_key(team2))))
def load_data(path=None):
 return json.loads(Path(path or Path(__file__).parent/'research/pregame_show_locations.json').read_text())
def flags(day,team1,team2,data=None,as_of=None):
 data=load_data() if data is None else data;key=game_key(day,team1,team2);year=int(key[0][:4]);out={}
 for feature in FEATURES:
  matches=[r for r in data['records'] if r['feature']==feature and game_key(r['date'],r['team1'],r['team2'])==key]
  if as_of is not None:
   out[feature]=1. if any(r.get('announcement_date') and r['announcement_date']<=str(as_of)[:10] for r in matches) else None
  elif matches:out[feature]=1.
  elif feature=='BigNoonKickoffOnsite' and year<2019:out[feature]=0.
  elif data['coverage'][feature][0]<=year<=data['coverage'][feature][1]:out[feature]=0.
  else:out[feature]=None
 return out
