import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from marquee_calibration import production_factors,load_production
B=Path(__file__).parent
class MarqueeProductionTest(unittest.TestCase):
 def test_scope_and_threshold(self):
  m=SimpleNamespace(marquee_calibration={'prediction_year':2026,'factor':1.06})
  base={'date':'09/12/26','network':'ABC','rank1':0,'rank2':0}
  rows=[base,base,{**base,'rank1':12,'rank2':20},{**base,'rank1':12,'rank2':20},{**base,'network':'ESPN2'},{**base,'date':'09/12/25'}]
  np.testing.assert_equal(production_factors(m,rows,[4999,5000,3000,2999,9000,9000]),[1,1.06,1.06,1,1,1])
 def test_changed_model_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'base').write_text('changed')
   (p/'marquee_calibration.json').write_text(json.dumps({'version':1,'variant':'hybrid_mean','user_approved':True,'factor':1.06,'base_sha256':{'base':'wrong'}}))
   with self.assertRaises(ValueError):load_production(p)
 def test_paired_integration_and_no_compounding(self):
  from weekly_predictions_fs import pregame_model,aligned_postgame,build_features
  from pregame_ensemble import predict_pregame_points_000s
  from aligned_postgame import predict_postgame_points_000s
  rows=json.loads((B/'research/audience_interest_week2_fixture.json').read_text())
  r=next(r for r in rows if r['cfbd_game_id']==401856679);r=copy.deepcopy(r);r.update(score1=10,score2=17)
  x=pd.DataFrame([build_features(r)]).reindex(columns=pregame_model.params.index,fill_value=0.)
  uncal=copy.copy(pregame_model);uncal.marquee_calibration=None
  a=predict_pregame_points_000s(uncal,x,[r]);b=predict_pregame_points_000s(pregame_model,x,[r]);factor=pregame_model.marquee_calibration['factor']
  np.testing.assert_allclose(b,a*factor)
  np.testing.assert_allclose(predict_postgame_points_000s(pregame_model,aligned_postgame,x,[r]),predict_postgame_points_000s(uncal,aligned_postgame,x,[r])*factor)
  r['predicted']='100M';np.testing.assert_allclose(predict_pregame_points_000s(pregame_model,x,[r]),b)
if __name__=='__main__':unittest.main()
