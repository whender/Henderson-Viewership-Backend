"""Protect source totals, units, exclusions, and training-target alignment."""
import json,unittest
from pathlib import Path
import numpy as np,pandas as pd,joblib
B=Path(__file__).parent;D=B/'research/week2_2026_import'
class Week2ActualsTests(unittest.TestCase):
 def test_source_and_individual_game_matching(self):
  source=json.loads((D/'source.json').read_text());self.assertEqual(len(source),35)
  games=[r for r in source if r['broadcast_type']=='game'];self.assertEqual(len(games),32)
  self.assertEqual(len({r['cfbd_game_id'] for r in games}),32)
  self.assertEqual(sum(r['viewers_000s'] for r in games),56320)
  excluded=[r for r in source if not r['training_eligible']]
  self.assertEqual(len(excluded),5)
  self.assertEqual(sorted(r['network'] for r in excluded),['BTN','BTN','ESPN2','TNT','USA'])
  clean=pd.read_csv(B/'viewership_cleaned.csv').set_index('source_index')
  ctx=json.loads((B/'expected_viewership_context.json').read_text())
  for r in games:
   if not r['training_eligible']:
    self.assertNotIn(r['source_index'],clean.index);continue
   row=clean.loc[r['source_index']]
   self.assertEqual(row['Persons 2+'],r['viewers_000s'])
   self.assertEqual({row['Team 1'],row['Team 2']},{r['team1'],r['team2']})
   self.assertEqual(ctx[str(r['source_index'])]['cfbd_game_id'],r['cfbd_game_id'])
 def test_fitted_targets_include_correction_and_exact_new_actuals(self):
  data=pd.read_csv(B/'viewership_cleaned.csv');model=joblib.load(B/'viewership_model_log.joblib')['model']
  np.testing.assert_allclose(model.model.endog,np.log1p(data['Persons 2+']))
  self.assertEqual(len(data),2453)
  interest=joblib.load(B/'audience_interest.joblib')
  self.assertIn('FOXFridayNight',interest['additional_features'])
  self.assertEqual(interest['feature_training_matches'],15)
  cfg=json.loads((B/'marquee_calibration.json').read_text())
  self.assertAlmostEqual(cfg['factor'],1.060880650203136)
if __name__=='__main__':unittest.main()
