"""Test lean pregame additions and nonlinear challengers, with past-only fits."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from holdout_viewership_tests import load_feature_frame,build_matrix
from strict_rolling_origin_challenger import fit_rank_calibrator,apply_rank_calibrator
from viewership_feature_contract import attach_exact_station_rank_features,POWER_FOOTBALL_TEAMS
import week1_error_investigation as audit

BASE=Path(__file__).resolve().parent
ADDITIONS={
    'baseline':[],
    'elo_strength':['Add_elo_average','Add_elo_difference'],
    'market_competitiveness':['Add_spread','Add_close_game'],
    'lean_pregame':['Add_elo_average','Add_elo_difference','Add_spread','Add_close_game','Add_neutral'],
}
audit.SPECS.update(ADDITIONS)


def metrics(g):
    e=g.pred-g.actual;a=100*e.abs()/g.actual
    return dict(n=len(g),mae_000s=e.abs().mean(),mape_pct=a.mean(),within25_pct=100*a.lt(25).mean(),bias_000s=e.mean())


def main():
    h=attach_exact_station_rank_features(load_feature_frame())
    context=pd.read_csv(BASE/'cfbd_pregame_context.csv').set_index('source_index').reindex(h.source_index)
    context.index=h.index
    for short,column in [('elo_average','pregame_elo_average'),('elo_difference','pregame_elo_difference'),('spread','abs_spread'),('neutral','neutral_site')]:
        h['Add_'+short]=context[column]
    h['Add_close_game']=np.where(h.Add_spread.notna(),h.Add_spread.le(7).astype(float),np.nan)
    h['rank_scope']=np.select([h['Top 10 Rankings'].ge(2),h.BothRanked.eq(1)],['BothTop10','BothRanked'],default='Other')
    raw=[];outputs=[]
    for year in [2019,2021,2022,2023,2024,2025]:
        train=h[h.Year.lt(year)].copy();test=h[h.Year.eq(year)].copy()
        a,b=build_matrix(h,train.index,test.index);y=np.log1p(train['Persons 2+'])
        for column in ADDITIONS['lean_pregame']:
            missing=column+'_missing';train[missing]=train[column].isna().astype(float);test[missing]=test[column].isna().astype(float)
            median=train[column].median();median=0 if pd.isna(median) else median
            train[column]=train[column].fillna(median);test[column]=test[column].fillna(median)
        predictions={}
        for name,columns in ADDITIONS.items():
            audit.SPECS[name]=columns+[c+'_missing' for c in columns]
            predictions[name]=np.asarray(audit.predict(a,b,train,test,y,name))
        for depth in [3,5]:
            aa=a.drop(columns='const').copy();bb=b.drop(columns='const').copy()
            for column in ADDITIONS['lean_pregame']:
                aa[column]=train[column];bb[column]=test[column]
                aa[column+'_missing']=train[column+'_missing'];bb[column+'_missing']=test[column+'_missing']
            model=HistGradientBoostingRegressor(max_iter=200,max_depth=depth,max_leaf_nodes=15,
                     min_samples_leaf=25,learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=11)
            model.fit(aa,y)
            smear=np.exp(y-model.predict(aa)).mean()
            p=np.maximum(np.exp(model.predict(bb))*smear-1,0)
            predictions[f'nonlinear_depth{depth}']=p
            predictions[f'nonlinear_depth{depth}_blend25']=.75*predictions['baseline']+.25*p
            if depth == 5:
                missing_inputs=bb.copy()
                for column in ADDITIONS['lean_pregame']:
                    missing_inputs[column]=train[column].median()
                    missing_inputs[column+'_missing']=1.
                missing_prediction=np.maximum(np.exp(model.predict(missing_inputs))*smear-1,0)
                predictions['nonlinear_missing_context_blend25']=.75*predictions['baseline']+.25*missing_prediction
        for name,p in predictions.items():
            r=test[['source_index','Year','Team 1','Team 2','Station','season_week','rank_scope']].copy()
            r['variant']=name;r['actual_viewers_000s']=test['Persons 2+'];r['predicted_viewers_000s']=p
            if year>=2021:
                calibration_name='nonlinear_depth5_blend25' if name=='nonlinear_missing_context_blend25' else name
                prior=pd.concat([v for v in raw if v.variant.iloc[0]==calibration_name])
                g=r.copy();g['predicted_viewers_000s']=apply_rank_calibrator(p,r.rank_scope,fit_rank_calibrator(prior,year,name));outputs.append(g)
            raw.append(r)
        print('Completed',year,flush=True)
    pd.concat(raw,ignore_index=True).to_csv(BASE/'targeted_improvements_raw.csv',index=False)
    out=pd.concat(outputs,ignore_index=True).rename(columns={'actual_viewers_000s':'actual','predicted_viewers_000s':'pred'})
    # Follow-up diagnostic: protect small forecasts using predicted audience,
    # never the held-out actual. Threshold fixed at 1M; no threshold sweep.
    current=out[out.variant.eq('baseline')].set_index('source_index').pred
    gated=out[out.variant.eq('nonlinear_depth5_blend25')].copy()
    original=gated.source_index.map(current)
    gated.loc[original.lt(1000),'pred']=original[original.lt(1000)]
    gated['variant']='nonlinear_blend25_predicted_1m_plus'
    out=pd.concat([out,gated],ignore_index=True)
    missing=out[out.variant.eq('nonlinear_missing_context_blend25')].copy()
    original=missing.source_index.map(current)
    missing.loc[original.lt(1000),'pred']=original[original.lt(1000)]
    missing['variant']='nonlinear_missing_context_predicted_1m_plus'
    out=pd.concat([out,missing],ignore_index=True)
    saved=pd.read_csv(BASE/'strict_rolling_origin_challenger_predictions.csv')
    saved=saved[saved.model.eq('blend_current50_exact50_shared_competition_scoped_rank_calibration')]
    check=out[out.variant.eq('baseline')].merge(saved,on='source_index',validate='one_to_one')
    np.testing.assert_allclose(check.pred,check.predicted_viewers_000s,atol=1e-5)
    out['dashboard_like']=(out['Team 1'].isin(POWER_FOOTBALL_TEAMS)|out['Team 2'].isin(POWER_FOOTBALL_TEAMS))&~out.Station.isin(['CW','ESPNU'])
    summaries=[];annual=[];uncertainty=[];rng=np.random.default_rng(11)
    for scope,mask in [('all',out.Year.gt(0)),('dashboard_like',out.dashboard_like),('dashboard_week2_onward',out.dashboard_like&out.season_week.gt(1))]:
        base=out[mask&out.variant.eq('baseline')].set_index('source_index');be=(base.pred-base.actual).abs()
        for name,g in out[mask].groupby('variant'):
            summaries.append({'scope':scope,'variant':name,**metrics(g)})
            for year,v in g.groupby('Year'):annual.append({'scope':scope,'variant':name,'year':year,**metrics(v)})
            p=g.set_index('source_index');delta=(p.pred-p.actual).abs()-be
            block=pd.DataFrame({'delta':delta,'year':p.Year}).groupby('year').delta.agg(['sum','size'])
            sample=rng.integers(0,5,(20000,5));boot=block['sum'].to_numpy()[sample].sum(axis=1)/block['size'].to_numpy()[sample].sum(axis=1)
            uncertainty.append({'scope':scope,'variant':name,'mae_delta_000s':delta.mean(),'year_wins':int(block['sum'].lt(0).sum()),'ci05_delta_000s':np.quantile(boot,.05),'ci95_delta_000s':np.quantile(boot,.95)})
    for name,df in [('predictions',out),('summary',pd.DataFrame(summaries)),('by_year',pd.DataFrame(annual)),('uncertainty',pd.DataFrame(uncertainty))]:df.to_csv(BASE/f'targeted_improvements_{name}.csv',index=False)
    print(pd.DataFrame(summaries).round(3).to_string(index=False),flush=True)


if __name__=='__main__':main()
