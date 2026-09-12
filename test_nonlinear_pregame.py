import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from nonlinear_pregame import context_values,apply_nonlinear


class ConstantModel:
    def predict(self,x):
        assert x['Add_elo_average'].eq(1550).all()
        return np.full(len(x),np.log(4001))


class NonlinearTests(unittest.TestCase):
    def setUp(self):
        self.row={'team1_pregame_elo':1500,'team2_pregame_elo':1600,'spread_home':-7,'neutral_site':False}
        self.model=SimpleNamespace(nonlinear_pregame={'threshold_000s':1000.,'feature_columns':['Add_elo_average'],
                'model':ConstantModel(),'smearing_factor':1.,'rank_corrections_000s':{'Other':0,'BothRanked':100}})

    def test_complete_inputs_threshold_rank_and_no_mutation(self):
        contexts=[self.row]*3;matrix=pd.DataFrame({'x':[1,2,3]});base=np.array([999.,2000.,2000.])
        result=apply_nonlinear(self.model,matrix,contexts,base,base,['Other','Other','BothRanked'])
        np.testing.assert_allclose(result,[999,2500,2400])
        np.testing.assert_array_equal(base,[999,2000,2000])
        self.assertEqual(list(matrix),['x'])

    def test_missing_invalid_and_absent_artifact_are_noop(self):
        bad=[dict(self.row,neutral_site=None),dict(self.row,spread_home=None),
             dict(self.row,team1_pregame_elo=float('nan')),dict(self.row,team2_pregame_elo=0),
             dict(self.row,neutral_site='false')]
        p=np.full(len(bad),2000.)
        np.testing.assert_array_equal(apply_nonlinear(self.model,pd.DataFrame(index=range(len(bad))),bad,p,p,['Other']*len(bad)),p)
        self.assertIs(apply_nonlinear(SimpleNamespace(),None,[],p,p,[]),p)

    def test_team_order_and_actuals_do_not_change_features(self):
        swapped=dict(self.row,team1_pregame_elo=1600,team2_pregame_elo=1500,spread_home=7,actual='99M',score1=90)
        self.assertEqual(context_values(swapped),context_values(self.row))

    def test_api_accepts_optional_context(self):
        from main import GameInput
        row=GameInput(team1='A',team2='B',rank1=0,rank2=0,network='ABC',time_slot='7:30p',**self.row)
        self.assertEqual(context_values(row.model_dump()),context_values(self.row))


if __name__=='__main__':unittest.main()
