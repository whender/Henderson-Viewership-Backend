"""Validated nonlinear point blend with explicit pregame-input requirements."""
import math
import numpy as np


def context_values(row):
    values=[]
    for key in ('team1_pregame_elo','team2_pregame_elo','spread_home'):
        value=row.get(key)
        if value is None or isinstance(value,bool): return None
        try: value=float(value)
        except (ValueError,TypeError): return None
        if not math.isfinite(value): return None
        values.append(value)
    neutral=row.get('neutral_site')
    if not isinstance(neutral,bool): return None
    e1,e2,spread=values
    if e1<=0 or e2<=0: return None
    return {'Add_elo_average':(e1+e2)/2,'Add_elo_difference':abs(e1-e2),
            'Add_spread':abs(spread),'Add_close_game':float(abs(spread)<=7),
            'Add_neutral':float(neutral)}


def apply_nonlinear(model,matrix,contexts,raw_blended,baseline,scopes):
    artifact=getattr(model,'nonlinear_pregame',None)
    if artifact is None: return baseline
    # Partial input rows use the exact existing forecast, never median guesses.
    eligible=[(i,context_values(row)) for i,row in enumerate(contexts)
              if baseline[i]>=artifact['threshold_000s']]
    eligible=[(i,values) for i,values in eligible if values is not None]
    if not eligible: return baseline
    indices=[i for i,_ in eligible]
    x=matrix.iloc[indices].copy()
    for i,values in eligible:
        for key,value in values.items():
            x.loc[matrix.index[i],key]=value
            x.loc[matrix.index[i],key+'_missing']=0.
    missing=set(artifact['feature_columns'])-set(x.columns)
    if missing: raise ValueError(f'Missing nonlinear model features: {sorted(missing)}')
    x=x.reindex(columns=artifact['feature_columns']).astype(float)
    prediction=np.maximum(np.exp(artifact['model'].predict(x))*artifact['smearing_factor']-1,0)
    result=np.asarray(baseline).copy()
    for i,p in zip(indices,prediction):
        correction=artifact['rank_corrections_000s'].get(scopes[i],0.)
        result[i]=max(.75*raw_blended[i]+.25*p-correction,0.)
    return result
