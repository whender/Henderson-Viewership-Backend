"""Experimental audience calibration learned only from earlier OOS forecasts.

Fixed bands in thousands of viewers, 30-game shrinkage toward no adjustment,
25% adjustment bounds. No actual audience enters band assignment.
"""
import numpy as np
import pandas as pd
MAJOR=('ABC','CBS','FOX','NBC','ESPN')
EDGES=np.array([1000.,3000.,5000.,8000.])
def band(values):return np.searchsorted(EDGES,np.asarray(values,dtype=float),side='right')
def fit(history,year,method='band_mean',window=2,shrink=30):
 if method not in ('band_mean','band_median','global_mean'):raise ValueError('Unknown calibration method')
 h=history[history.Year.lt(year)&history.Year.ge(year-window)&history.Station.isin(MAJOR)].copy()
 if not np.isfinite(h[['pred','actual']]).all().all() or h.pred.le(0).any() or h.actual.le(0).any():raise ValueError('Positive finite inputs required')
 h['band']=0 if method=='global_mean' else band(h.pred)
 records=[]
 for k,g in h.groupby('band'):
  ratio=float(np.median(g.actual/g.pred)) if method=='band_median' else float(g.actual.sum()/g.pred.sum())
  factor=float(np.clip(1+(ratio-1)*len(g)/(len(g)+shrink),.75,1.25))
  records.append(dict(band=int(k),n=len(g),raw_ratio=ratio,factor=factor))
 return dict(year=int(year),method=method,window=window,shrink=shrink,training_min_year=None if h.empty else int(h.Year.min()),training_max_year=None if h.empty else int(h.Year.max()),records=records)
def apply(pred,stations,config):
 values=np.asarray(pred,dtype=float);keys=np.zeros(len(values),dtype=int) if config['method']=='global_mean' else band(values)
 factors={r['band']:r['factor'] for r in config['records']}
 return np.array([p*factors.get(int(k),1.) if s in MAJOR else p for p,k,s in zip(values,keys,stations)])
