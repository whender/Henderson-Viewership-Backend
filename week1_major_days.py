"""Scoped Week 1 interaction model, applied before existing event overlays."""
import copy
from datetime import date
import numpy as np
from nonlinear_pregame import apply_nonlinear,context_values


def apply_week1_major_days(model,matrix,contexts,baseline,points,scopes):
    config=getattr(model,'week1_major_days',None)
    if config is None:return points
    from pregame_ensemble import _coerce_date,exact_feature_frame
    start=date.fromisoformat(config['start_date']);end=date.fromisoformat(config['end_date'])
    eligible=[];days=[]
    for i,row in enumerate(contexts):
        d=_coerce_date(row.get('date',row.get('Date',row.get('ParsedDate'))))
        week=row.get('week')
        if (d is not None and start<=d<=end and d.year==config['prediction_year']
                and row.get('network') in config['networks']
                and (week is None or str(week) in ('1','1.0'))):
            eligible.append(i);days.append(d.weekday())
    if not eligible:return points
    selected=[contexts[i] for i in eligible]
    x=matrix.iloc[eligible].copy()
    extras=exact_feature_frame(selected,index=x.index)
    for c in extras.columns:x[c]=extras[c]
    for label,dayset in [('Weekday',{1,2,3}),('Friday',{4}),('Saturday',{5}),('Sunday',{6}),('Monday',{0})]:
        x['Week1Major_'+label]=[float(d in dayset) for d in days]
    raw=np.zeros(len(eligible))
    for component in config['components']:
        missing=set(component['columns'])-set(x.columns)
        if missing:raise ValueError(f'Missing Week 1 features: {sorted(missing)}')
        logpred=x[component['columns']].to_numpy(dtype=float)@component['coefficients']
        raw+=.5*np.maximum((np.exp(logpred)-1)*component['smearing_factor'],0)
    selected_scopes=[scopes[i] for i in eligible]
    pred=np.maximum(raw-np.asarray([config['rank_corrections_000s'].get(s,0.) for s in selected_scopes]),0)
    nonlinear=getattr(model,'nonlinear_pregame',None)
    if nonlinear is not None:
        candidate=copy.copy(model);candidate.nonlinear_pregame=dict(nonlinear)
        candidate.nonlinear_pregame['rank_corrections_000s']=config['nonlinear_rank_corrections_000s']
        original=np.asarray(baseline)[eligible]
        blended=apply_nonlinear(candidate,matrix.iloc[eligible],selected,raw,original,selected_scopes)
        # Match the historical test: nonlinear eligibility uses the original baseline.
        use=[original[j]>=nonlinear['threshold_000s'] and context_values(row) is not None for j,row in enumerate(selected)]
        pred=np.where(use,blended,pred)
    result=np.asarray(points,dtype=float).copy();result[eligible]=pred
    return result
