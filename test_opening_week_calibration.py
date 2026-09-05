import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pregame_ensemble import apply_opening_week_calibration

class OpeningWeekCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.model=SimpleNamespace(opening_week_calibration={
            'version':1,'weight':.5,'event_mean_000s':4600,
            'training_max_year':2025,'prediction_year':2026,'networks':['ESPN','FOX']})
        self.row={'week':0,'date':'08/29/26','network':'ESPN'}

    def test_scoped_pool_ignores_actuals_and_preserves_other_games(self):
        contexts=[self.row,dict(self.row,week=1),dict(self.row,network='NBC'),
                  dict(self.row,date='08/24/24'),dict(self.row,date='08/28/27'),
                  dict(self.row,week=None),dict(self.row)]
        matrix=pd.DataFrame({'Week 0 Power':[1,0,1,1,1,1,0]})
        points=np.array([3000.,3000.,3000.,3000.,3000.,3000.,3000.])
        revised=apply_opening_week_calibration(self.model,matrix,contexts,points)
        np.testing.assert_array_equal(revised,[3800,3000,3000,3000,3000,3000,3000])
        np.testing.assert_array_equal(points,[3000]*7)
        contexts[0]=dict(self.row,actual='99M',score1=99,score2=0)
        np.testing.assert_array_equal(apply_opening_week_calibration(self.model,matrix,contexts,points),revised)

    def test_missing_configuration_is_exact_noop(self):
        points=np.array([3000.])
        result=apply_opening_week_calibration(SimpleNamespace(),pd.DataFrame(),[],points)
        np.testing.assert_array_equal(result,points)

    def test_invalid_configuration_fails_closed(self):
        self.model.opening_week_calibration['weight']=1.5
        with self.assertRaisesRegex(ValueError,'Invalid opening-week'):
            apply_opening_week_calibration(self.model,pd.DataFrame({'Week 0 Power':[1]}),[self.row],[3000])

if __name__=='__main__':unittest.main()
