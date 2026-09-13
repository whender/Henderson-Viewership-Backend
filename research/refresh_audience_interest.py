"""Fetch audited public pageviews for a slate, then write the offline serving cache.

Usage: python research/refresh_audience_interest.py --slate ../week2_2026/predictions.json
Existing cache is retained. An incomplete slate fails before the serving file changes.
"""
import argparse,json,sys,time,urllib.request,urllib.parse,urllib.error,ssl
from datetime import datetime,timedelta,timezone,date
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from audience_interest import interest_cutoff,feature_values,POLICY
UA='HendersonViewershipResearch/1.0 (college football audience research; public pageview analysis)'
try:
    import certifi
    SSL_CONTEXT=ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT=ssl.create_default_context()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slate',type=Path,required=True);parser.add_argument('--cache-dir',type=Path,default=BASE/'research/pageview_cache');parser.add_argument('--output',type=Path,default=BASE/'audience_interest_data.json');args=parser.parse_args()
    payload=json.loads(args.slate.read_text());rows=payload.get('full_slate_games') or payload['games']
    mapping=json.loads((BASE/'research/audience_interest_page_mapping.json').read_text())
    requests={};now=datetime.now(timezone.utc);games=[]
    for row in rows:
        try:d=datetime.strptime(row['date'],'%m/%d/%y').date()
        except ValueError:d=date.fromisoformat(row['date'])
        cutoff=interest_cutoff(d)
        assert now.date()>=cutoff+timedelta(days=1), 'This slate is too early for the validated interest cutoff'
        games.append((row,d))
        for t in [row['team1'],row['team2']]:
            requests.setdefault(t,set()).update((cutoff-timedelta(days=i)).isoformat() for i in range(1,8))
    args.cache_dir.mkdir(parents=True,exist_ok=True);daily={};sources=[]
    # Accept saved raw per-team downloads as well as this script's dated downloads.
    for path in args.cache_dir.glob('*.json'):
        data=json.loads(path.read_text())
        if 'team' not in data or 'items' not in data:continue
        if data['title']!=mapping.get(data['team']):raise ValueError('Cached article mapping differs')
        daily.setdefault(data['team'],{}).update({datetime.strptime(v['timestamp'],'%Y%m%d%H').date().isoformat():v['views'] for v in data['items']})
        sources.append({k:data[k] for k in ['team','title','url','retrieved_at']})
    for team,dates in sorted(requests.items()):
        if dates.issubset(daily.get(team,{})):continue
        title=mapping[team];start=min(dates).replace('-','')+'00';end=max(dates).replace('-','')+'00'
        url='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/'+urllib.parse.quote(title.replace(' ','_'),safe='')+'/daily/'+start+'/'+end
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':UA}),timeout=30,context=SSL_CONTEXT) as response:data=json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code!=429 or attempt==4:raise
                time.sleep(int(error.headers.get('Retry-After','60'))+1)
        data.update(team=team,title=title,url=url,retrieved_at=now.isoformat());name=urllib.parse.quote(team,safe='')+'_'+start+'_'+end+'.json'
        (args.cache_dir/name).write_text(json.dumps(data));daily.setdefault(team,{}).update({datetime.strptime(v['timestamp'],'%Y%m%d%H').date().isoformat():v['views'] for v in data['items']});sources.append({k:data[k] for k in ['team','title','url','retrieved_at']});print('Fetched',team,flush=True);time.sleep(6)
    audit=[]
    for row,d in games:
        values=feature_values(d,[row['team1'],row['team2']],daily,now)
        assert values is not None, f"Incomplete history: {row['team1']} / {row['team2']}"
        audit.append(dict(game_id=row.get('cfbd_game_id'),date=d.isoformat(),cutoff=interest_cutoff(d).isoformat(),team1=row['team1'],team2=row['team2'],**values))
    prior=json.loads(args.output.read_text()) if args.output.exists() else {'daily':{}}
    for team,dates in requests.items():prior['daily'].setdefault(team,{}).update({d:daily[team][d] for d in sorted(dates)})
    prior.update(policy=POLICY,source='Wikimedia en.wikipedia all-access user daily pageviews',refreshed_at=now.isoformat(),sources=sources,slate_audit=audit)
    temp=args.output.with_suffix('.json.tmp');temp.write_text(json.dumps(prior,indent=2,allow_nan=False)+'\n');temp.replace(args.output)
    print('Validated and saved',len(games),'games,',len(requests),'teams')
if __name__=='__main__':main()
