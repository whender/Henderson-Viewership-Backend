"""Fit and backtest the current pregame architecture with final margin added."""
import copy,hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from holdout_viewership_tests import load_feature_frame,build_matrix
from opening_week_audit import attach_opening_features
from week1_major_day_interactions import design
from viewership_feature_contract import attach_exact_station_rank_features,EXACT_NETWORK_RANK_CHALLENGER_FEATURES
from strict_rolling_origin_challenger import fit_rank_calibrator,apply_rank_calibrator
from team_half_life_experiment import metrics

BASE=Path(__file__).resolve().parent;BACKEND=BASE.parent/'HendersonViewershipBackend'


def linear(a,b,y):
    keep=[c for c in a if c=='const' or a[c].nunique()>1]
    # Preserve the original columns, then remove only redundant new interactions.
    accepted=[c for c in keep if not c.startswith('Week1Major_')]
    for c in keep:
        if not c.startswith('Week1Major_'):continue
        r=a[c].to_numpy()-a[accepted].to_numpy()@np.linalg.lstsq(a[accepted],a[c],rcond=None)[0]
        if np.linalg.norm(r)>1e-7:accepted.append(c)
    m=sm.OLS(y,a[accepted]).fit();smear=float(np.exp(m.resid).mean())
    component={'columns':accepted,'coefficients':np.asarray(m.params),'smearing_factor':smear}
    return np.maximum((np.exp(m.predict(b[accepted]))-1)*smear,0).to_numpy(),component


def forest_fit(a,b,train,test,context,y):
    a=a.drop(columns='const').copy();b=b.drop(columns='const').copy()
    ta=context.reindex(train.source_index).copy();ta.index=train.index
    tb=context.reindex(test.source_index).copy();tb.index=test.index
    for key,source in [('elo_average','pregame_elo_average'),('elo_difference','pregame_elo_difference'),('spread','abs_spread'),('neutral','neutral_site')]:
        a['Add_'+key]=ta[source];b['Add_'+key]=tb[source]
    for f in [a,b]:f['Add_close_game']=np.where(f.Add_spread.notna(),f.Add_spread.le(7).astype(float),np.nan)
    for c in [c for c in a if c.startswith('Add_')]:
        a[c+'_missing']=a[c].isna().astype(float);b[c+'_missing']=b[c].isna().astype(float)
        median=a[c].median();median=0 if pd.isna(median) else median
        a[c]=a[c].fillna(median);b[c]=b[c].fillna(median)
    m=HistGradientBoostingRegressor(max_iter=200,max_depth=5,max_leaf_nodes=15,min_samples_leaf=25,
        learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=11).fit(a,y)
    smear=float(np.exp(y-m.predict(a)).mean())
    return np.maximum(np.exp(m.predict(b))*smear-1,0),{'model':m,'feature_columns':list(a),'smearing_factor':smear,'threshold_000s':1000.}


def fit_components(a,b,train,test,context,y):
    a=a.copy();b=b.copy();a['Score Diff']=train['Score Diff'];b['Score Diff']=test['Score Diff']
    extra=[c for c in EXACT_NETWORK_RANK_CHALLENGER_FEATURES if c not in a and train[c].nunique()>1]
    predictions={};components={};first=None
    for name in ['standard','week1']:
        aa,bb=a,b
        if name=='week1':aa,_=design(a,train,'week1_major_days');bb,_=design(b,test,'week1_major_days')
        pp,c1=linear(aa,bb,y)
        qq,c2=linear(pd.concat([aa,train[extra]],axis=1),pd.concat([bb,test[extra]],axis=1),y)
        predictions[name]=.5*pp+.5*qq;components[name]=[c1,c2]
        if name=='standard':first=pp
    fp,forest=forest_fit(a,b,train,test,context,y)
    return predictions,components,fp,forest,first


def main():
    h=attach_exact_station_rank_features(load_feature_frame())
    h,bounds=attach_opening_features(h,json.loads((BASE/'cfbd_raw_games_cache.json').read_text()))
    h['period']='later'
    for bound in bounds:
        end=pd.Timestamp(bound['opening_week_end']);mask=h.Year.eq(bound['year'])
        h.loc[mask&h.ParsedDate.gt(end)&h.ParsedDate.le(end+pd.Timedelta(days=7)),'period']='Week1'
    h['Score Diff']=(pd.to_numeric(h['Winner Score'])-pd.to_numeric(h['Loser Score'])).abs()
    assert h['Score Diff'].notna().all() and h.Year.max()==2025
    h['rank_scope']=np.select([h['Top 10 Rankings'].ge(2),h.BothRanked.eq(1)],['BothTop10','BothRanked'],default='Other')
    context=pd.read_csv(BASE/'cfbd_pregame_context.csv').set_index('source_index')
    complete=context[['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']].notna().all(axis=1)
    pre=pd.read_csv(BASE/'targeted_improvements_predictions.csv');pre=pre[pre.variant.eq('baseline')].set_index('source_index')
    histories={(variant,branch):[] for variant in ['standard','week1'] for branch in ['linear','nonlinear']}
    outputs=[]
    for year in [2019,2021,2022,2023,2024,2025]:
        train,test=h[h.Year.lt(year)],h[h.Year.eq(year)];assert train.Year.max()<year
        a,b=build_matrix(h,train.index,test.index);y=np.log1p(train['Persons 2+'])
        predictions,_,fp,_,old=fit_components(a,b,train,test,context,y)
        cal={}
        for variant,raw in predictions.items():
            for branch,pred in [('linear',raw),('nonlinear',.75*raw+.25*fp)]:
                row=test[['source_index','Year','rank_scope']].copy();row['actual_viewers_000s']=test['Persons 2+'];row['predicted_viewers_000s']=pred
                if year>=2021:
                    prior=pd.concat(histories[variant,branch]);assert prior.Year.max()<year
                    cal[variant,branch]=apply_rank_calibrator(pred,row.rank_scope,fit_rank_calibrator(prior,year,variant))
                histories[variant,branch].append(row)
        if year>=2021:
            eligible=pre.reindex(test.source_index).pred.to_numpy()>=1000
            eligible &= test.source_index.map(complete).fillna(False).to_numpy()
            standard=np.where(eligible,cal['standard','nonlinear'],cal['standard','linear'])
            week1=np.where(eligible,cal['week1','nonlinear'],cal['week1','linear'])
            target=test.period.eq('Week1')&test.Station.isin(['NBC','CBS','ABC','FOX','ESPN'])
            for name,pred in [('legacy_primary_plus_score',old),('aligned_postgame',np.where(target,week1,standard))]:
                row=test[['source_index','Year','Team 1','Team 2','Station','period','Score Diff']].copy();row['actual']=test['Persons 2+'];row['pred']=pred;row['variant']=name;outputs.append(row)
        print('Completed postgame holdout',year,flush=True)
    p=pd.concat(outputs,ignore_index=True);p.to_csv(BASE/'aligned_postgame_predictions.csv',index=False)
    summaries=[]
    major=p.Station.isin(['NBC','CBS','ABC','FOX','ESPN'])
    for scope,mask in [('all',p.Year.gt(0)),('major',major),('major_week1',major&p.period.eq('Week1'))]:
        for name,g in p[mask].groupby('variant'):
            summaries.append(dict(scope=scope,variant=name,**metrics(g)))
    s=pd.DataFrame(summaries);s.to_csv(BASE/'aligned_postgame_summary.csv',index=False);print(s.round(3).to_string(index=False))
    # Full fit: use the saved pregame design to preserve its complete feature contract.
    sys.path.insert(0,str(BACKEND))
    from model_loader import load_viewership_model
    m=load_viewership_model();x=m.model.data.orig_exog.copy();h.index=x.index
    np.testing.assert_allclose(np.log1p(h['Persons 2+']),m.model.endog)
    _,components,_,forest,_=fit_components(x,x,h,h,context,m.model.endog)
    corrections={}
    for key,prior in histories.items():
        stat=fit_rank_calibrator(pd.concat(prior),2026,'aligned_postgame')
        corrections[key]=stat.set_index('rank_scope').correction_000s.to_dict()
    forest['rank_corrections_000s']=corrections['standard','nonlinear']
    week=copy.deepcopy(m.week1_major_days);week['components']=components['week1'];week['rank_corrections_000s']=corrections['week1','linear'];week['nonlinear_rank_corrections_000s']=corrections['week1','nonlinear']
    bound=['viewership_model_log.joblib','nonlinear_pregame.joblib','week1_major_days.joblib']
    artifact={'version':1,'training_max_year':2025,'components':components['standard'],
        'rank_corrections_000s':corrections['standard','linear'],'nonlinear_pregame':forest,'week1_major_days':week,
        'pregame_sha256':{f:hashlib.sha256((BACKEND/f).read_bytes()).hexdigest() for f in bound},
        'description':'Current pregame feature architecture plus final absolute score differential; fits exclude 2026.'}
    for component in components['standard']:
        assert 'Score Diff' in component['columns']
    assert set(m.params.index)|{'Score Diff'}==set(components['standard'][0]['columns'])
    assert set(m.pregame_ensemble['challenger_feature_columns'])|{'Score Diff'}==set(components['standard'][1]['columns'])
    joblib.dump(artifact,BACKEND/'aligned_postgame.joblib',compress=3)
    return s


if __name__=='__main__':main()
