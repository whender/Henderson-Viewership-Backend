"""Pregame slate hierarchy from outcome-disjoint intrinsic audience forecasts.

Audiences are supplied in thousands. Competition keeps the current same-date,
90-minute kickoff window. It is not a game-duration overlap approximation.
"""
import numpy as np
import pandas as pd
FEATURES=('SlateOwnLogAudience','SlateLeader','SlateAudienceShare','CompetitionXLeader','CompetitionXShare','StrongerCompetition')
def slate_features(frame,intrinsic,window_hours=1.5):
 if not frame.index.is_unique:raise ValueError('Slate indexes must be unique')
 intrinsic=intrinsic.reindex(frame.index).astype(float)
 times=pd.to_numeric(frame.Time_float,errors='coerce');dates=pd.to_datetime(frame.ParsedDate,errors='coerce').dt.normalize()
 if not np.isfinite(intrinsic).all() or intrinsic.lt(0).any():raise ValueError('Finite nonnegative audience forecasts required')
 if times.isna().any() or dates.isna().any():raise ValueError('Date and kickoff required for every slate row')
 out=pd.DataFrame(0.,index=frame.index,columns=[*FEATURES,'SlateRank','CompetitorCount','CompetitionTotal','OwnAudience000s'])
 for _,ix in frame.groupby(dates).groups.items():
  ix=pd.Index(ix);p=intrinsic.loc[ix].to_numpy();t=times.loc[ix].to_numpy()
  for i,key in enumerate(ix):
   overlap=np.abs(t-t[i])<=window_hours;overlap[i]=False
   others=p[overlap];total=others.sum()/1000.;stronger=others[others>p[i]].sum()/1000.
   leader=float(len(others)>0 and (others<=p[i]).all());share=p[i]/(p[i]+others.sum()) if p[i]+others.sum()>0 else 0.
   out.loc[key]=[np.log1p(p[i]),leader,share,total*leader,total*share,stronger,1+sum(others>p[i]),len(others),total,p[i]]
 return out
