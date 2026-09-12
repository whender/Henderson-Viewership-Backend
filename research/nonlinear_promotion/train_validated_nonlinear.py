"""Build the separately versioned nonlinear artifact after promotion validation."""
import hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from holdout_viewership_tests import load_feature_frame
from strict_rolling_origin_challenger import fit_rank_calibrator

BASE=Path(__file__).resolve().parent;BACKEND=BASE.parent/'HendersonViewershipBackend'
gates=pd.read_csv(BASE/'nonlinear_promotion_gates.csv').set_index('model')
assert bool(gates.loc['nonlinear_complete_context_only','practical_gate_pass'])
sys.path.insert(0,str(BACKEND))
from model_loader import load_viewership_model
primary=load_viewership_model();x=primary.model.data.orig_exog.drop(columns='const').copy()
h=load_feature_frame();h.index=x.index
np.testing.assert_allclose(np.log1p(h['Persons 2+']),primary.model.endog)
assert h.Year.max()==2025
ctx=pd.read_csv(BASE/'cfbd_pregame_context.csv').set_index('source_index').reindex(h.source_index);ctx.index=x.index
for key,source in [('elo_average','pregame_elo_average'),('elo_difference','pregame_elo_difference'),('spread','abs_spread'),('neutral','neutral_site')]:x['Add_'+key]=ctx[source]
x['Add_close_game']=np.where(x.Add_spread.notna(),x.Add_spread.le(7).astype(float),np.nan)
for c in ['Add_elo_average','Add_elo_difference','Add_spread','Add_close_game','Add_neutral']:
    x[c+'_missing']=x[c].isna().astype(float);x[c]=x[c].fillna(x[c].median())
model=HistGradientBoostingRegressor(max_iter=200,max_depth=5,max_leaf_nodes=15,min_samples_leaf=25,
    learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=11)
y=np.log1p(h['Persons 2+']);model.fit(x,y)
raw=pd.read_csv(BASE/'targeted_improvements_raw.csv')
stats=fit_rank_calibrator(raw[raw.variant.eq('nonlinear_depth5_blend25')],2026,'nonlinear_depth5_blend25')
artifact={'version':1,'model':model,'feature_columns':list(x.columns),
          'smearing_factor':float(np.exp(y-model.predict(x)).mean()),'threshold_000s':1000.,
          'rank_corrections_000s':stats.set_index('rank_scope').correction_000s.to_dict(),
          'primary_sha256':hashlib.sha256((BACKEND/'viewership_model_log.joblib').read_bytes()).hexdigest(),
          'promotion_passed':True,'training_max_year':2025,
          'gate':gates.loc['nonlinear_complete_context_only'].to_dict(),
          'description':'25% nonlinear blend only at baseline predictions >=1M and complete pregame Elo/spread/neutral inputs.'}
joblib.dump(artifact,BACKEND/'nonlinear_pregame.joblib',compress=3)
print('Saved validated nonlinear_pregame.joblib; primary model untouched.')
