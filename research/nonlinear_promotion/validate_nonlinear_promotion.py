"""Apply the existing promotion rules to the fixed nonlinear candidate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import strict_rolling_origin_challenger as gate

BASE=Path(__file__).resolve().parent
NAMES=['nonlinear_blend25_predicted_1m_plus','nonlinear_missing_context_predicted_1m_plus']


def main():
    p=pd.read_csv(BASE/'targeted_improvements_predictions.csv')
    meta=pd.read_csv(BASE/'strict_rolling_origin_challenger_predictions.csv')
    meta=meta[meta.model.eq('blend_current50_exact50_shared_competition_scoped_rank_calibration')]
    p=p[p.variant.isin(['baseline',*NAMES])].rename(columns={'variant':'model','Year':'holdout_year','actual':'actual_viewers_000s','pred':'predicted_viewers_000s'})
    context=pd.read_csv(BASE/'cfbd_pregame_context.csv').set_index('source_index')
    complete=context[['pregame_elo_average','pregame_elo_difference','abs_spread','neutral_site']].notna().all(axis=1)
    scoped=p[p.model.eq(NAMES[0])].copy()
    base=p[p.model.eq('baseline')].set_index('source_index').predicted_viewers_000s
    incomplete=~scoped.source_index.map(complete).fillna(False)
    scoped.loc[incomplete,'predicted_viewers_000s']=scoped.loc[incomplete,'source_index'].map(base)
    scoped['model']='nonlinear_complete_context_only'
    p=pd.concat([p,scoped],ignore_index=True)
    names=[*NAMES,'nonlinear_complete_context_only']
    p['model']=p.model.replace({'baseline':'served_current'})
    p=p.merge(meta[['source_index','game_date','conference_champ_flag','predicted_decile']],on='source_index',validate='many_to_one')
    p['signed_error_000s']=p.predicted_viewers_000s-p.actual_viewers_000s
    p['absolute_error_000s']=p.signed_error_000s.abs()
    p['absolute_percent_error_pct']=100*p.absolute_error_000s/p.actual_viewers_000s
    gate.FINAL_MODEL_ORDER=['served_current',*names]
    # Do not mark a new model deployable just because its statistical gate passes.
    gate.DEPLOYABLE_NOW_MODELS=set()
    summary,annual=gate.summarize_predictions(p)
    boot=gate.date_block_bootstrap(p,reps=gate.BOOTSTRAP_REPS)
    result=gate.apply_practical_gates(summary,boot)
    for name,frame in [('gates',result),('by_year',annual),('bootstrap',boot)]:
        frame.to_csv(BASE/f'nonlinear_promotion_{name}.csv',index=False)
    print(result[['model','mae_000s','mape_pct','worst_year_mae_delta_000s','mae_year_wins_vs_served','mae_improvement_probability','mae_delta_ci95_000s','practical_gate_pass']].round(3).to_string(index=False))
    import sys
    sys.path.insert(0,str(BASE.parent/'HendersonViewershipBackend'))
    # Audit source declarations without importing the web application or doing DB I/O.
    import ast
    module=ast.parse((BASE.parent/'HendersonViewershipBackend/main.py').read_text())
    definition=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='GameInput')
    fields={n.target.id for n in definition.body if isinstance(n,ast.AnnAssign)}
    required={'team1_pregame_elo','team2_pregame_elo','spread_home','neutral_site'}
    missing=sorted(required-fields)
    weekly=json.loads((BASE.parent/'week1_2026/predictions.json').read_text())['games']
    completeness={field:sum(g.get(field) is not None for g in weekly) for field in sorted(required)}
    ready=not missing and bool(result.set_index('model').loc['nonlinear_complete_context_only','practical_gate_pass'])
    audit={'api_fields_missing':missing,'weekly_games':len(weekly),'weekly_nonmissing_counts':completeness,
           'deployment_decision':'conditional_rollout' if ready else 'hold',
           'reason':'Require complete pregame inputs for nonlinear predictions; preserve current model exactly otherwise. Existing weekly rows without neutral_site keep their current forecast.'}
    (BASE/'nonlinear_promotion_input_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
