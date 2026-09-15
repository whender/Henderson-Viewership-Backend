import unittest
import numpy as np
import pandas as pd
from slate_position import slate_features
class SlatePositionTest(unittest.TestCase):
 def test_hierarchy_and_boundaries(self):
  f=pd.DataFrame({'ParsedDate':['2026-09-12']*4+['2026-09-13'],'Time_float':[12,12,13.5,14,12]},index=[10,11,12,13,14]);p=pd.Series([5000,3000,1000,2000,9000],index=f.index);r=slate_features(f,p)
  self.assertEqual(r.loc[10,'CompetitionTotal'],4);self.assertEqual(r.loc[10,'SlateRank'],1);self.assertEqual(r.loc[10,'SlateLeader'],1)
  self.assertAlmostEqual(r.loc[10,'SlateAudienceShare'],5/9);self.assertEqual(r.loc[11,'StrongerCompetition'],5)
  self.assertEqual(r.loc[14,'CompetitionTotal'],0);self.assertEqual(r.loc[14,'CompetitionXLeader'],0)
  pd.testing.assert_frame_equal(slate_features(f.iloc[::-1],p).sort_index(),r.sort_index())
 def test_ties_and_zero(self):
  f=pd.DataFrame({'ParsedDate':['2026-09-12']*2,'Time_float':[12,12]});r=slate_features(f,pd.Series([1000,1000]))
  self.assertTrue(r.SlateLeader.eq(1).all());self.assertTrue(r.StrongerCompetition.eq(0).all())
  self.assertTrue(np.isfinite(slate_features(f,pd.Series([0,0]))).all().all())
 def test_outcomes_do_not_enter(self):
  f=pd.DataFrame({'ParsedDate':['2026-09-12']*2,'Time_float':[12,12],'Persons 2+':[500,9000]});p=pd.Series([1000,2000]);a=slate_features(f,p)
  f['Persons 2+']=[900000,1];f['Score Diff']=[60,0]
  pd.testing.assert_frame_equal(a,slate_features(f,p))
 def test_missing_fails(self):
  f=pd.DataFrame({'ParsedDate':['2026-09-12'],'Time_float':[np.nan]})
  with self.assertRaises(ValueError):slate_features(f,pd.Series([1000]))
if __name__=='__main__':unittest.main()
