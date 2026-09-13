"""Reproduce weekly-cutoff promotion and fit paired pre/postgame artifacts.

Run from the workspace containing RatingsAndRegression; no production writes
beyond the explicit artifact file. All fits/calibrators stop at 2025.
"""
import hashlib, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
BACKEND=Path(__file__).resolve().parents[1];ROOT=BACKEND.parent;R=ROOT/'RatingsAndRegression'
sys.path[:0]=[str(BACKEND),str(R)]
import audience_interest_weekly_experiment as exp
from audience_interest import feature_values, FEATURES, POLICY
from aligned_postgame_experiment import forest_fit
from strict_rolling_origin_challenger import fit_rank_calibrator,apply_rank_calibrator
from model_loader import load_viewership_model
from team_half_life_experiment import metrics


def linear(a,b,y):
    keep=[c for c in a if c=='const' or a[c].nunique()>1]
    accepted=[c for c in keep if not c.startswith('Week1Major_')]
    for c in keep:
        if not c.startswith('Week1Major_'):continue
        residual=a[c].to_numpy()-a[accepted].to_numpy()@np.linalg.lstsq(a[accepted],a[c],rcond=None)[0]
        if np.linalg.norm(residual)>1e-7:accepted.append(c)
    x=a[accepted].to_numpy(dtype=float);z=b[accepted].to_numpy(dtype=float)
    fit=sm.OLS(y,x).fit();coef=np.asarray(fit.params);smear=float(np.exp(y-x@coef).mean())
    return np.maximum(np.exp(z@coef)*smear-1,0),dict(columns=accepted,coefficients=coef,smearing_factor=smear)


def fit(a,b,train,test,context,y,post=False):
    a=a.copy();b=b.copy()
    if post:
        a['Score Diff']=train['Score Diff'];b['Score Diff']=test['Score Diff']
    extra=[c for c in exp.EXACT_NETWORK_RANK_CHALLENGER_FEATURES if c not in a and train[c].nunique()>1]
    predictions={};components={}
    for branch in ['regular','week1']:
        aa,_=exp.design(a,train,'current' if branch=='regular' else 'week1_major_days')
        bb,_=exp.design(b,test,'current' if branch=='regular' else 'week1_major_days')
        p,c1=linear(aa,bb,y);q,c2=linear(pd.concat([aa,train[extra]],axis=1),pd.concat([bb,test[extra]],axis=1),y)
        predictions[branch]=.5*(p+q);components[branch]=[c1,c2]
    fp,forest=forest_fit(a,b,train,test,context,y)
    return predictions,components,fp,forest


def corrections(hist):
    return {branch:{kind:fit_rank_calibrator(pd.concat(hist[branch,kind]),2026,'audience_interest').set_index('rank_scope').correction_000s.to_dict()
                   for kind in ['linear','nonlinear']} for branch in ['regular','week1']}


def main():
    h=exp.prior.prepare();interest,audit=exp.interest_features(h)
    h['Score Diff']=(pd.to_numeric(h['Winner Score'])-pd.to_numeric(h['Loser Score'])).abs()
    assert h.Year.max()==2025 and h['Score Diff'].notna().all() and interest.notna().all().all()
    daily={}
    for path in (R/'audience_interest/cache').glob('*.json'):
        data=json.loads(path.read_text());daily[data['team']]={pd.to_datetime(v['timestamp'],format='%Y%m%d%H').date().isoformat():v['views'] for v in data['items']}
    # Shared serving extractor exactly reproduces every historical feature window.
    for idx,row in h.iterrows():
        actual=feature_values(row.ParsedDate.date(),[row['Team 1'],row['Team 2']],daily)
        np.testing.assert_allclose([actual[c] for c in FEATURES],interest.loc[idx,FEATURES],atol=1e-12)
    context=pd.read_csv(R/'cfbd_pregame_context.csv').set_index('source_index')
    complete=context[['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']].notna().all(axis=1)
    def add(x,frame):
        x=x.copy()
        for c in FEATURES:x[c]=interest.loc[frame.index,c]
        x['InterestMissing']=0.
        return x
    posthist={(b,k):[] for b in ['regular','week1'] for k in ['linear','nonlinear']};outputs=[]
    old=pd.read_csv(R/'targeted_improvements_predictions.csv');old=old[old.variant.eq('baseline')].set_index('source_index')
    for year in [2019,2021,2022,2023,2024,2025]:
        train,test=h[h.Year.lt(year)],h[h.Year.eq(year)];a,b=exp.build_matrix(h,train.index,test.index)
        a=add(a,train);b=add(b,test);y=np.log1p(train['Persons 2+']).to_numpy()
        raw,_,fp,_=fit(a,b,train,test,context,y,post=True);cal={}
        for branch,pred in raw.items():
            for kind,values in [('linear',pred),('nonlinear',.75*pred+.25*fp)]:
                row=test[['source_index','Year','rank_scope']].copy();row['actual_viewers_000s']=test['Persons 2+'];row['predicted_viewers_000s']=values
                if year>=2021:cal[branch,kind]=apply_rank_calibrator(values,row.rank_scope,fit_rank_calibrator(pd.concat(posthist[branch,kind]),year,'post_interest'))
                posthist[branch,kind].append(row)
        if year>=2021:
            eligible=(old.reindex(test.source_index).pred.to_numpy()>=1000)&test.source_index.map(complete).fillna(False).to_numpy()
            opening=test.period.eq('Week1')&test.Station.isin(exp.MAJOR)
            reg=np.where(eligible,cal['regular','nonlinear'],cal['regular','linear']);week=np.where(eligible,cal['week1','nonlinear'],cal['week1','linear'])
            row=test[['source_index','Year','Station','period']].copy();row['actual']=test['Persons 2+'];row['pred']=np.where(opening,week,reg);row['variant']='postgame_interest';outputs.append(row)
        print('Postgame interest fold',year,flush=True)
    p=pd.concat(outputs);oldpost=pd.read_csv(R/'aligned_postgame_predictions.csv');oldpost=oldpost[oldpost.variant.eq('aligned_postgame')]
    summary=[]
    for name,frame in [('current_postgame',oldpost),('postgame_interest',p)]:
        summary.append(dict(variant=name,scope='five_major',**metrics(frame[frame.Station.isin(exp.MAJOR)])))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    assert summary[1]['mae_000s']<summary[0]['mae_000s'], 'Postgame promotion gate failed'
    pd.DataFrame(summary).to_csv(exp.OUT/'postgame_summary.csv',index=False);p.to_csv(exp.OUT/'postgame_predictions.csv',index=False)
    m=load_viewership_model();x=m.model.data.orig_exog.copy();assert len(h)==len(x)
    np.testing.assert_allclose(np.log1p(h['Persons 2+']),m.model.endog)
    x.index=h.index;x=add(x,h)
    rawhist=joblib.load(exp.OUT/'raw_histories.joblib')
    prehist={(b,k):rawhist['interest_level',b,k] for b in ['regular','week1'] for k in ['linear','nonlinear']}
    paired={}
    for post,name,hist in [(False,'pregame',prehist),(True,'postgame',posthist)]:
        _,components,_,forest=fit(x,x,h,h,context,m.model.endog,post=post)
        paired[name]=dict(components=components,forest=forest,corrections=corrections(hist))
    for branch in ['regular','week1']:
        for pre,post in zip(paired['pregame']['components'][branch],paired['postgame']['components'][branch]):
            assert set(post['columns'])==set(pre['columns'])|{'Score Diff'}
    assert set(paired['postgame']['forest']['feature_columns'])==set(paired['pregame']['forest']['feature_columns'])|{'Score Diff'}
    names=['viewership_model_log.joblib','nonlinear_pregame.joblib','week1_major_days.joblib','aligned_postgame.joblib']
    artifact=dict(version=1,policy=POLICY,training_max_year=2025,promotion_passed=True,
        base_sha256={n:hashlib.sha256((BACKEND/n).read_bytes()).hexdigest() for n in names},**paired)
    joblib.dump(artifact,BACKEND/'audience_interest.joblib',compress=3)
    print('Saved paired pregame/postgame audience-interest artifact',flush=True)
if __name__=='__main__':main()
