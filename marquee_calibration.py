"""Experimental pregame-defined marquee calibration; no automatic serving use."""
import numpy as np
from recent_audience_calibration import MAJOR
SPECS={'marquee_mean':('audience','mean','target'),'marquee_median':('audience','median','target'), 'broad_to_marquee':('audience','mean','major'),'ranked_mean':('ranked','mean','target'),'hybrid_mean':('hybrid','mean','target')}
def mask(frame,definition):
 major=frame.Station.isin(MAJOR);audience=frame.pred.ge(5000);ranked=frame.BothRanked.eq(1)
 if definition=='audience':return major&audience
 if definition=='ranked':return major&ranked
 if definition=='hybrid':return major&(audience|(ranked&frame.pred.ge(3000)))
 raise ValueError('Unknown marquee definition')
def fit(history,year,variant):
 definition,method,population=SPECS[variant]
 h=history[history.Year.lt(year)&history.Year.ge(year-2)&history.Station.isin(MAJOR)]
 if population=='target':h=h[mask(h,definition)]
 if not np.isfinite(h[['pred','actual']]).all().all() or h.pred.le(0).any() or h.actual.le(0).any():raise ValueError('Positive finite audiences required')
 ratio=1. if h.empty else float(np.median(h.actual/h.pred) if method=='median' else h.actual.sum()/h.pred.sum())
 factor=float(np.clip(1+(ratio-1)*len(h)/(len(h)+30),.75,1.25))
 return dict(variant=variant,definition=definition,year=int(year),n=len(h),raw_ratio=ratio,factor=factor,training_max_year=None if h.empty else int(h.Year.max()))
def apply(frame,config):
 out=frame.pred.copy();selected=mask(frame,config['definition']);out.loc[selected]*=config['factor'];return out


def load_production(directory):
 import hashlib,json
 from pathlib import Path
 base=Path(directory);path=base/'marquee_calibration.json'
 if not path.exists():return None
 c=json.loads(path.read_text())
 if c.get('version')!=1 or c.get('variant')!='hybrid_mean' or not c.get('user_approved') or not .75<=c.get('factor',0)<=1.25:
  raise ValueError('Invalid marquee calibration configuration')
 for name,digest in c['base_sha256'].items():
  if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Marquee calibration must be revalidated for {name}')
 c['artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 return c


def production_factors(model,contexts,uncalibrated_points):
 from pregame_ensemble import _coerce_date,rank_scope
 config=getattr(model,'marquee_calibration',None);result=np.ones(len(uncalibrated_points))
 if config is None:return result
 for i,(row,point) in enumerate(zip(contexts,uncalibrated_points)):
  d=_coerce_date(row.get('date'))
  if d is None or d.year!=config['prediction_year'] or row.get('network') not in MAJOR:continue
  ranked=rank_scope(row.get('rank1'),row.get('rank2'))!='Other'
  if point>=5000 or (ranked and point>=3000):result[i]=config['factor']
 return result
