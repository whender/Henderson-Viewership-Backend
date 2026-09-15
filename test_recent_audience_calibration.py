import unittest
import numpy as np,pandas as pd
from recent_audience_calibration import fit,apply,band
class CalibrationTest(unittest.TestCase):
 def frame(self):return pd.DataFrame({'Year':[2021,2023,2024,2025,2026],'Station':['FOX']*5,'pred':[5000.]*5,'actual':[100000.,5500.,6000.,7000.,90000.]})
 def test_future_and_old_rows_excluded(self):
  h=self.frame();c=fit(h,2025);self.assertEqual(c['training_min_year'],2023);self.assertEqual(c['training_max_year'],2024);self.assertEqual(c['records'][0]['n'],2)
  h.loc[h.Year.ge(2025),'actual']=1;self.assertEqual(fit(h,2025),c)
 def test_no_history_and_other_network_noop(self):
  c=fit(self.frame(),2020);np.testing.assert_equal(apply([5000],['FOX'],c),[5000])
  c=fit(self.frame(),2025);np.testing.assert_equal(apply([5000],['ESPN2'],c),[5000])
 def test_band_is_prediction_only(self):
  np.testing.assert_equal(band([999,1000,2999,3000,4999,5000,7999,8000]),[0,1,1,2,2,3,3,4])
 def test_shrink_and_cap(self):
  h=self.frame();c=fit(h,2025);self.assertAlmostEqual(c['records'][0]['factor'],1+.15*2/32)
  h.loc[h.Year.lt(2025),'actual']=1e8;c=fit(h,2025);self.assertEqual(c['records'][0]['factor'],1.25)
if __name__=='__main__':unittest.main()
