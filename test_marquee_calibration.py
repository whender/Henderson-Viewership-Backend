import unittest
import pandas as pd
from marquee_calibration import fit,apply,mask
class MarqueeCalibrationTest(unittest.TestCase):
 def data(self):return pd.DataFrame({'Year':[2023,2024,2025,2026],'Station':['FOX']*4,'pred':[5000.,6000.,8000.,10000.],'actual':[6000.,7000.,12000.,20000.],'BothRanked':[0,1,1,0]})
 def test_cutoff(self):
  h=self.data();a=fit(h,2025,'marquee_mean');h.loc[h.Year.ge(2025),'actual']=1
  self.assertEqual(a,fit(h,2025,'marquee_mean'));self.assertEqual(a['training_max_year'],2024)
 def test_prediction_not_actual_selects(self):
  h=self.data();h.loc[0,'pred']=4999;h.loc[0,'actual']=100000
  self.assertFalse(mask(h,'audience').iloc[0]);self.assertTrue(mask(h,'audience').iloc[1])
 def test_nonmarquee_unchanged(self):
  h=self.data();c=fit(h,2025,'marquee_mean');h.loc[0,'pred']=2000;h.loc[1,'Station']='ESPN2'
  out=apply(h,c);self.assertEqual(out.iloc[0],2000);self.assertEqual(out.iloc[1],6000)
 def test_empty_history_noop(self):
  h=self.data();pd.testing.assert_series_equal(apply(h,fit(h,2020,'marquee_mean')),h.pred)
if __name__=='__main__':unittest.main()
