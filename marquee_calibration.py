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
