"""Postgame serving with the pregame architecture plus final score differential."""
import copy,hashlib,math
from pathlib import Path
import joblib
import numpy as np
from nonlinear_pregame import apply_nonlinear,context_values
from week1_major_days import apply_week1_major_days


def load_aligned_postgame(directory):
    base=Path(directory);artifact=joblib.load(base/'aligned_postgame.joblib')
    if artifact.get('version')!=1:raise ValueError('Unsupported aligned postgame artifact')
    for name,expected in artifact['pregame_sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Postgame model must be retrained for changed pregame artifact: {name}')
    artifact['artifact_sha256']=hashlib.sha256((base/'aligned_postgame.joblib').read_bytes()).hexdigest()
    return artifact


def score_difference(row):
    scores=[]
    for key in ['score1','score2']:
        value=row.get(key)
        if value is None or isinstance(value,bool):return None
        try:value=float(value)
        except (TypeError,ValueError):return None
        if not math.isfinite(value) or value<0 or not value.is_integer():return None
        scores.append(value)
    return abs(scores[0]-scores[1])


def predict_postgame_points_000s(model,artifact,matrix,contexts):
    from pregame_ensemble import (exact_feature_frame,model_point_predictions_000s,rank_scope,
        apply_opening_week_calibration,apply_monday_calibration)
    margins=[score_difference(row) for row in contexts]
    if any(v is None for v in margins):raise ValueError('Two valid final scores are required')
    x=matrix.copy();extras=exact_feature_frame(contexts,index=x.index)
    for c in extras.columns:x[c]=extras[c]
    x['Score Diff']=margins
    raw=np.zeros(len(x))
    for c in artifact['components']:
        missing=set(c['columns'])-set(x.columns)
        if missing:raise ValueError(f'Missing postgame features: {sorted(missing)}')
        raw+=.5*np.maximum((np.exp(x[c['columns']].to_numpy(dtype=float)@c['coefficients'])-1)*c['smearing_factor'],0)
    scopes=[rank_scope(row.get('rank1'),row.get('rank2')) for row in contexts]
    points=np.maximum(raw-np.asarray([artifact['rank_corrections_000s'].get(s,0.) for s in scopes]),0)
    # Retain pregame-only eligibility for the nonlinear branch.
    ens=model.pregame_ensemble
    primary=model_point_predictions_000s(model,matrix,model.smearing_factor)
    exact=model_point_predictions_000s(ens['challenger_model'],x[ens['challenger_feature_columns']],ens['challenger_smearing_factor'])
    pre_raw=(1-ens['weight'])*primary+ens['weight']*exact
    original=np.maximum(pre_raw+np.asarray([ens['rank_scope_additive_adjustments_000s'].get(s,0.) for s in scopes]),0)
    proxy=copy.copy(model);proxy.nonlinear_pregame=artifact['nonlinear_pregame'];proxy.week1_major_days=artifact['week1_major_days']
    nonlinear=apply_nonlinear(proxy,x,contexts,raw,original,scopes)
    eligible=[original[i]>=proxy.nonlinear_pregame['threshold_000s'] and context_values(row) is not None for i,row in enumerate(contexts)]
    points=np.where(eligible,nonlinear,points)
    points=apply_week1_major_days(proxy,x,contexts,original,points,scopes)
    points=apply_opening_week_calibration(proxy,matrix,contexts,points)
    return apply_monday_calibration(proxy,matrix,contexts,points)
