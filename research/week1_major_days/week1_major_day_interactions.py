"""Week 1 x five major-network day groups; historical test and 2026 preview."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from holdout_viewership_tests import load_feature_frame, build_matrix
from opening_week_audit import attach_opening_features
from viewership_feature_contract import attach_exact_station_rank_features, EXACT_NETWORK_RANK_CHALLENGER_FEATURES
from strict_rolling_origin_challenger import fit_rank_calibrator, apply_rank_calibrator
from team_half_life_experiment import metrics
from large_miss_diagnosis import table

BASE=Path(__file__).resolve().parent
NAMES=['current','week1_major_days','week1_major_days_shrink10']
DAYS=['Mon','Tue','Wed','Thu','Fri','Sun']  # Saturday reference


def design(x, frame, name):
    x=x.copy(); added=[]
    if name=='current': return x,added
    major=frame.Station.isin(['NBC','CBS','ABC','FOX','ESPN'])
    week=frame.period.eq('Week1')
    for label,days in [('Weekday',['Tue','Wed','Thu']),('Friday',['Fri']),('Saturday',['Sat']),('Sunday',['Sun']),('Monday',['Mon'])]:
        c='Week1Major_'+label
        x[c]=(major & week & frame.DoW.isin(days)).astype(float)
        added.append(c)
    return x,added


def main():
    h=attach_exact_station_rank_features(load_feature_frame())
    h,bounds=attach_opening_features(h,json.loads((BASE/'cfbd_raw_games_cache.json').read_text()))
    h['period']='later'
    for bound in bounds:
        end=pd.Timestamp(bound['opening_week_end']);start=pd.Timestamp(bound['first_fbs_kickoff']).tz_localize(None).normalize()
        season=h.Year.eq(bound['year'])
        h.loc[season & h.ParsedDate.between(start,end),'period']='Week0'
        h.loc[season & h.ParsedDate.gt(end)&h.ParsedDate.le(end+pd.Timedelta(days=7)),'period']='Week1'
    assert h.period.eq('Week0').sum()==10
    assert h[h.period.eq('Week0')].DoW.eq('Sat').all()
    assert h[h.DoW.eq('Mon')].period.eq('Week1').all()
    h['rank_scope']=np.select([h['Top 10 Rankings'].ge(2),h.BothRanked.eq(1)],['BothTop10','BothRanked'],default='Other')
    h[h.period.ne('later')][['source_index','Year','Date','DoW','period','Team 1','Team 2','Station','Persons 2+']].to_csv(BASE/'week1_major_day_history.csv',index=False)
    ctx=pd.read_csv(BASE/'cfbd_pregame_context.csv').set_index('source_index')
    complete=ctx[['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']].notna().all(axis=1)
    saved=pd.read_csv(BASE/'targeted_improvements_raw.csv')
    reference=pd.read_csv(BASE/'team_half_life_predictions.csv');reference=reference[reference.variant.eq('current')].set_index('source_index')
    history={(name,branch):[] for name in NAMES for branch in ['linear','nonlinear']}
    output=[];audit=[]
    for year in [2019,2021,2022,2023,2024,2025]:
        train,test=h[h.Year.lt(year)],h[h.Year.eq(year)]
        a,b=build_matrix(h,train.index,test.index);y=np.log1p(train['Persons 2+']).to_numpy()
        extras=[c for c in EXACT_NETWORK_RANK_CHALLENGER_FEATURES if c not in a and train[c].nunique()>1]
        predictions={}
        for name in NAMES:
            pa,added=design(a,train,name);pb,_=design(b,test,name)
            total=np.zeros(len(test))
            for component,aa,bb in [('primary',pa,pb),('exact_network',pd.concat([pa,train[extras]],axis=1),pd.concat([pb,test[extras]],axis=1))]:
                keep=[c for c in aa if c=='const' or aa[c].nunique()>1]
                # Discard added interactions already represented by the base design.
                basecols=[c for c in keep if c not in added]
                accepted=list(basecols)
                for c in added:
                    status='no_training_variation'
                    if c in keep:
                        residual=aa[c].to_numpy()-aa[accepted].to_numpy()@np.linalg.lstsq(aa[accepted],aa[c],rcond=None)[0]
                        status='redundant' if np.linalg.norm(residual)<1e-7 else 'estimated'
                        if status=='estimated':accepted.append(c)
                    audit.append(dict(year=year,variant=name,component=component,feature=c,status=status,training_games=int(aa[c].sum())))
                xa=aa[accepted].to_numpy();xb=bb[accepted].to_numpy()
                if name=='week1_major_days_shrink10':
                    penalty=np.diag([np.sqrt(10.) if c in added else 0. for c in accepted])
                    coef=np.linalg.lstsq(np.vstack([xa,penalty]),np.r_[y,np.zeros(len(accepted))],rcond=None)[0]
                else:coef=sm.OLS(y,xa).fit().params
                smear=np.exp(y-xa@coef).mean()
                total+=.5*np.maximum(np.exp(xb@coef)*smear-1,0)
            predictions[name]=total
        rawbase=saved[saved.Year.eq(year)&saved.variant.eq('baseline')].set_index('source_index').reindex(test.source_index).predicted_viewers_000s.to_numpy()
        rawblend=saved[saved.Year.eq(year)&saved.variant.eq('nonlinear_depth5_blend25')].set_index('source_index').reindex(test.source_index).predicted_viewers_000s.to_numpy()
        np.testing.assert_allclose(predictions['current'],rawbase,atol=1e-5)
        forest=(rawblend-.75*rawbase)/.25
        calibrated={}
        for name in NAMES:
            for branch,pred in [('linear',predictions[name]),('nonlinear',.75*predictions[name]+.25*forest)]:
                row=test[['source_index','Year','rank_scope']].copy();row['actual_viewers_000s']=test['Persons 2+'];row['predicted_viewers_000s']=pred
                if year>=2021:
                    prior=pd.concat(history[name,branch]);assert prior.Year.max()<year
                    calibrated[name,branch]=apply_rank_calibrator(pred,row.rank_scope,fit_rank_calibrator(prior,year,name))
                history[name,branch].append(row)
        if year>=2021:
            eligible=(calibrated['current','linear']>=1000)&test.source_index.map(complete).fillna(False).to_numpy()
            for name in NAMES:
                row=test[['source_index','Year','ParsedDate','DoW','period','Team 1','Team 2','Station']].copy()
                row['actual']=test['Persons 2+'];row['pred']=np.where(eligible,calibrated[name,'nonlinear'],calibrated[name,'linear']);row['variant']=name
                baseline=np.where(eligible,calibrated['current','nonlinear'],calibrated['current','linear'])
                target=test.period.eq('Week1')&test.Station.isin(['NBC','CBS','ABC','FOX','ESPN'])
                row['pred']=np.where(target,row['pred'],baseline)
                if name=='current':np.testing.assert_allclose(row.pred,reference.reindex(row.source_index).pred,atol=1e-5)
                output.append(row)
        print('Completed',year,flush=True)
    p=pd.concat(output,ignore_index=True);assert all(len(g)==1821 and g.source_index.is_unique for _,g in p.groupby('variant'))
    p.to_csv(BASE/'week1_major_day_predictions.csv',index=False);pd.DataFrame(audit).to_csv(BASE/'week1_major_day_feature_audit.csv',index=False)
    major=p.Station.isin(['NBC','CBS','ABC','FOX','ESPN'])
    scopes={'all':p.Year.gt(0),'major':major,'opening':p.period.ne('later'),'major_week1':major&p.period.eq('Week1'),'major_later':major&p.period.eq('later'),'week0':p.period.eq('Week0'),'week1':p.period.eq('Week1')}
    summary=[];annual=[];days=[]
    for scope,mask in scopes.items():
        q=p[mask];base=q[q.variant.eq('current')].set_index('source_index');be=(base.pred-base.actual).abs()
        for name,g in q.groupby('variant'):
            t=g.set_index('source_index');delta=(t.pred-t.actual).abs()-be
            yearly=pd.DataFrame({'delta':delta,'year':t.Year}).groupby('year').delta.agg(['sum','size'])
            rng=np.random.default_rng(13);samples=rng.integers(0,len(yearly),(20000,len(yearly)))
            boot=yearly['sum'].to_numpy()[samples].sum(axis=1)/yearly['size'].to_numpy()[samples].sum(axis=1)
            summary.append(dict(scope=scope,variant=name,**metrics(g),mae_delta_000s=delta.mean(),year_wins=int(yearly['sum'].lt(-1e-6).sum()),season_ci05_000s=np.quantile(boot,.05),season_ci95_000s=np.quantile(boot,.95)))
            for year,gg in g.groupby('Year'):annual.append(dict(scope=scope,variant=name,year=year,**metrics(gg)))
    for (period,day,name),g in p[p.period.ne('later')].groupby(['period','DoW','variant']):days.append(dict(period=period,day=day,variant=name,**metrics(g)))
    s=pd.DataFrame(summary);s.to_csv(BASE/'week1_major_day_summary.csv',index=False)
    pd.DataFrame(annual).to_csv(BASE/'week1_major_day_by_year.csv',index=False);pd.DataFrame(days).to_csv(BASE/'week1_major_day_by_day.csv',index=False)
    print(s[s.scope.eq('major_week1')].round(3).to_string(index=False),flush=True)
    return h,history,p,s,pd.DataFrame(audit)


def preview(h,history,backtest,summary,audit,artifact_path=None):
    import sys,copy
    sys.path.insert(0,str(BASE.parent/'HendersonViewershipBackend'))
    from weekly_predictions_fs import pregame_model as model,build_features
    from pregame_ensemble import (exact_feature_frame,predict_pregame_points_000s,
        model_point_predictions_000s,rank_scope,apply_monday_calibration,apply_opening_week_calibration)
    from nonlinear_pregame import apply_nonlinear
    model=copy.copy(model)
    if hasattr(model,'week1_major_days'):del model.week1_major_days
    rows=json.loads((BASE.parent/'week1_2026/predictions.json').read_text())['games']
    assert len(rows)==18 and all(g['network'] in ['NBC','CBS','ABC','FOX','ESPN'] for g in rows)
    x=model.model.data.orig_exog.copy();h=h.copy();h.index=x.index
    np.testing.assert_allclose(np.log1p(h['Persons 2+']),model.model.endog)
    assert h.Year.max()==2025
    serve=pd.DataFrame([build_features(g) for g in rows]).reindex(columns=x.columns,fill_value=0.)
    ctx=exact_feature_frame(rows,index=serve.index)
    live=pd.DataFrame({'Station':[g['network'] for g in rows], 'DoW':[g['day'] for g in rows], 'period':'Week1'})
    dates=pd.to_datetime([g['date'] for g in rows],format='%m/%d/%y')
    assert dates.min()>=pd.Timestamp('2026-09-01') and dates.max()<=pd.Timestamp('2026-09-07')
    current=predict_pregame_points_000s(model,serve,rows)
    ens=model.pregame_ensemble;challenger=ens['challenger_model']
    ex=x.reindex(columns=ens['challenger_feature_columns'],fill_value=0.).copy()
    for c in ex.columns.difference(x.columns):ex[c]=h[c]
    exlive=serve.reindex(columns=ex.columns,fill_value=0.).copy()
    for c in ctx.columns.intersection(exlive.columns):exlive[c]=ctx[c]
    for trained,trainx,testx in [(model,x,serve),(challenger,ex,exlive)]:
        reconstructed=sm.OLS(model.model.endog,trainx).fit()
        np.testing.assert_allclose(reconstructed.predict(testx),trained.predict(testx),atol=1e-7)
    outputs=[];coefficients=[]
    for name in NAMES[1:]:
        points=[];components=[]
        for component,trainx,testx in [('primary',x,serve),('exact_network',ex,exlive)]:
            aa,added=design(trainx,h,name);bb,_=design(testx,live,name)
            accepted=list(trainx.columns)
            for c in added:
                residual=aa[c].to_numpy()-aa[accepted].to_numpy()@np.linalg.lstsq(aa[accepted],aa[c],rcond=None)[0]
                status='redundant' if np.linalg.norm(residual)<1e-7 else 'estimated'
                if status=='estimated':accepted.append(c)
                coefficients.append(dict(variant=name,component=component,feature=c,status=status,training_games=int(aa[c].sum())))
            a=aa[accepted].to_numpy();b=bb[accepted].to_numpy();y=model.model.endog
            if name.endswith('shrink10'):
                penalty=np.diag([np.sqrt(10.) if c in added else 0. for c in accepted])
                coef=np.linalg.lstsq(np.vstack([a,penalty]),np.r_[y,np.zeros(len(accepted))],rcond=None)[0]
            else:coef=sm.OLS(y,a).fit().params
            smear=np.exp(y-a@coef).mean()
            components.append({'columns':accepted,'coefficients':np.asarray(coef),'smearing_factor':float(smear)})
            points.append(np.maximum((np.exp(b@coef)-1)*smear,0))
            for entry in coefficients:
                if entry['variant']==name and entry['component']==component:
                    entry['log_coefficient']=float(coef[accepted.index(entry['feature'])]) if entry['feature'] in accepted else None
        raw=.5*points[0]+.5*points[1]
        scopes=pd.Series([rank_scope(g['rank1'],g['rank2']) for g in rows])
        stats=fit_rank_calibrator(pd.concat(history[name,'linear']),2026,name)
        pred=apply_rank_calibrator(raw,scopes,stats)
        candidate=copy.copy(model);candidate.nonlinear_pregame=dict(model.nonlinear_pregame)
        nonlinear_stats=fit_rank_calibrator(pd.concat(history[name,'nonlinear']),2026,name)
        candidate.nonlinear_pregame['rank_corrections_000s']=nonlinear_stats.set_index('rank_scope').correction_000s.to_dict()
        pred=apply_nonlinear(candidate,serve,rows,raw,pred,scopes.tolist())
        pred=apply_opening_week_calibration(candidate,serve,rows,pred)
        pred=apply_monday_calibration(candidate,serve,rows,pred)
        if artifact_path is not None and name=='week1_major_days':
            import joblib,hashlib
            artifact={'version':1,'training_max_year':2025,'prediction_year':2026,
                      'start_date':'2026-09-01','end_date':'2026-09-07',
                      'networks':['NBC','CBS','ABC','FOX','ESPN'],
                      'primary_sha256':hashlib.sha256((BASE.parent/'HendersonViewershipBackend/viewership_model_log.joblib').read_bytes()).hexdigest(),
                      'components':components,
                      'rank_corrections_000s':stats.set_index('rank_scope').correction_000s.to_dict(),
                      'nonlinear_rank_corrections_000s':nonlinear_stats.set_index('rank_scope').correction_000s.to_dict(),
                      'description':'User-approved Week 1 major-network day interactions, fitted through 2025; other games retain the existing model.'}
            joblib.dump(artifact,artifact_path,compress=3)
        for i,g in enumerate(rows):
            published=float(g['predicted'].split('M')[0])*1000
            outputs.append(dict(variant=name,cfbd_game_id=g['cfbd_game_id'],date=g['date'],day=g['day'],network=g['network'],matchup=g['matchup'],published_000s=published,current_replay_000s=current[i],candidate_000s=pred[i],change_000s=pred[i]-current[i],change_pct=100*(pred[i]/current[i]-1)))
    out=pd.DataFrame(outputs)
    assert np.isfinite(out.candidate_000s).all() and out.candidate_000s.ge(0).all()
    assert (out.current_replay_000s-out.published_000s).abs().max()<5.01
    untouched=backtest[~(backtest.period.eq('Week1')&backtest.Station.isin(['NBC','CBS','ABC','FOX','ESPN']))].pivot(index='source_index',columns='variant',values='pred')
    for name in NAMES[1:]:np.testing.assert_allclose(untouched[name],untouched.current,atol=1e-8)
    out.to_csv(BASE/'week1_major_day_2026_preview.csv',index=False)
    pd.DataFrame(coefficients).to_csv(BASE/'week1_major_day_full_fit_coefficients.csv',index=False)
    print(out.round(3).to_string(index=False),flush=True)
    review=out[out.variant.eq('week1_major_days')][['date','day','network','matchup']].copy()
    chosen=out[out.variant.eq('week1_major_days')]
    review['Published M']=(chosen.published_000s/1000).map(lambda x:f'{x:.2f}')
    review['Candidate M']=(chosen.candidate_000s/1000).map(lambda x:f'{x:.2f}')
    review['Change %']=chosen.change_pct
    year=pd.read_csv(BASE/'week1_major_day_by_year.csv')
    report=['# Week 1 major-network day interactions — September 12, 2026',
        '''## Requested specification

Five indicators: Week 1 × major network × weekday (Tuesday–Thursday), Friday, Saturday, Sunday, and Monday. Major means exactly NBC, CBS, ABC, FOX and ESPN; ESPN2 is excluded. Week 1 uses the second FBS schedule block, following the initial Week 0 block and ending on Monday. No new Week 0 features, corrected power flags, team decay or additional opening main effects are added in this experiment.

The five indicators cover all requested day categories; there is no extra major-Week-1 intercept. Redundant columns are removed using training data only. The Monday interaction is exactly the existing Monday indicator in this historical sample, so it cannot supply a separate coefficient. Monday predictions can still move slightly when the remaining model parameters and calibration are refitted.

## Historical result

For 74 Week 1 games on the five major networks in the 2021–2025 season-ahead backtest, the unpenalized specification reduces average absolute error from 893K to 840K (5.94%). Average percentage error falls from 26.71% to 25.72%; within-30% improves from 67.57% to 71.62%; >50% misses fall from 13 to 10. Within-20% declines from 48.65% to 47.30%. MAE improves in all five seasons. The exploratory 90% season-block interval for the MAE change is approximately −84K to −23K; it does not account for the historical candidate-selection process.

A sensitivity variant penalizing the new coefficients with lambda=10 gives a smaller 893K to 886K improvement and wins three seasons. Both variants are supplied in the preview CSV. The main preview uses the stronger historical candidate, selected before considering any 2026 actuals.''',
        table(summary[summary.scope.eq('major_week1')][['variant','n','mae_000s','mape_pct','within20_pct','within30_pct','over50_pct','year_wins','season_ci05_000s','season_ci95_000s']]),
        table(year[year.scope.eq('major_week1')&year.variant.isin(['current','week1_major_days'])][['variant','year','n','mae_000s','mape_pct','over50_pct']]),
        '## 2026 Week 1 forecast preview',
        'All audiences are millions. Published forecasts are the saved rounded values; percentage changes use the exact current-model replay, which agrees within rounding. The existing 2026 Monday pooling adjustment is retained for both replay and candidate. These are point forecasts; old uncertainty intervals are not reused for the candidate.',
        table(review),
        '''These are model refits, not flat day multipliers. Adding an interaction changes the other fitted coefficients and candidate-specific rank calibration too, so games on the same day can move by different amounts or even in opposite directions. The full-model Sunday interaction coefficients should not be interpreted as stand-alone audience multipliers: the existing Sunday coefficient shifts simultaneously.

## Method and limitations

Historical training uses strictly earlier seasons. The 2019 fold seeds rank calibration for the 2021–2025 holdouts. The interaction is added to both primary and exact-network log regressions; competition inputs and the nonlinear component remain fixed to isolate the change. Candidate-specific rank corrections use earlier rolling predictions only. Predictions outside Week 1 on the five major networks are explicitly held at the current baseline; this restriction is part of the tested proposal. Assertions verify every held-out game, current-model reconstruction and unchanged out-of-scope predictions.

The 2026 preview refits through 2025 on the saved model's original design matrix, reuses the saved 2026 pregame competition inputs and reconstructs both existing component predictions before adding features. Full-history candidate rank corrections come only from 2019/2021–2025 rolling predictions. Existing nonlinear and opening/Monday serving adjustments are retained in the preview. The retained weekly rows lack neutral-site context, so their nonlinear branch remains bypassed, as in current serving. No 2026 actual audience is used in fitting, calibration, selection or preview generation.

The historical benchmark excludes the later serving-only opening/Monday overlays, matching earlier analyses; the 2026 preview includes them. Therefore the historical 5.94% improvement is not a measured incremental gain on top of those deployed overlays. The Sunday interaction has only nine historical major-network Week 1 games, and there is just one Sunday outside opening week in the full historical sample. Monday adds no independent information. These support limitations still require care before deployment, notwithstanding the consistent historical directional improvement.

Published predictions and their actuals were not changed. No production artifact, Firestore record or deployment was modified.''',
        '## Reproduce',
        'Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 RatingsAndRegression/week1_major_day_interactions.py`. Outputs use the `week1_major_day_` prefix: predictions.csv, summary.csv, by_year.csv, by_day.csv, history.csv, feature_audit.csv, full_fit_coefficients.csv, 2026_preview.csv and this report.']
    (BASE/'week1_major_day_report.md').write_text('\n\n'.join(report)+'\n')
    return out


if __name__=='__main__':
    result=main()
    preview(*result)
