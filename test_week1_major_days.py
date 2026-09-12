import copy
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from week1_major_days import apply_week1_major_days

class Forest:
    def predict(self,x):return np.full(len(x),np.log(4001.))


class Week1DayTests(unittest.TestCase):
    def setUp(self):
        component={'columns':['const','Week1Major_Sunday'],'coefficients':np.array([np.log(2001),np.log(2)]),'smearing_factor':1.}
        self.model=SimpleNamespace(week1_major_days={'version':1,'prediction_year':2026,'start_date':'2026-09-01','end_date':'2026-09-07',
            'networks':['ABC','CBS','FOX','NBC','ESPN'],'components':[component,component],
            'rank_corrections_000s':{'Other':10},'nonlinear_rank_corrections_000s':{}})

    def test_scope_and_no_mutation(self):
        rows=[{'date':'09/06/26','network':'NBC','week':1},
              {'date':'09/06/26','network':'ESPN2','week':1},
              {'date':'08/29/26','network':'ABC','week':0},
              {'date':'09/12/26','network':'ABC','week':2},
              {'date':'09/06/27','network':'ABC','week':1},
              {'date':'09/06/26','network':'ABC','week':0},
              {'date':None,'network':'ABC'}]
        x=pd.DataFrame({'const':np.ones(len(rows))});p=np.full(len(rows),1000.)
        old=copy.deepcopy(rows)
        result=apply_week1_major_days(self.model,x,rows,p,p,['Other']*len(rows))
        np.testing.assert_allclose(result,[3991,1000,1000,1000,1000,1000,1000])
        np.testing.assert_array_equal(p,np.full(len(rows),1000.));self.assertEqual(rows,old);self.assertEqual(list(x),['const'])

    def test_all_five_day_groups(self):
        names=['Weekday','Friday','Saturday','Sunday','Monday']
        component={'columns':['const']+['Week1Major_'+n for n in names],
            'coefficients':np.r_[0.,np.log(np.array([101,201,301,401,501]))],'smearing_factor':1.}
        self.model.week1_major_days['components']=[component,component]
        self.model.week1_major_days['rank_corrections_000s']={}
        rows=[{'date':f'09/0{d}/26','network':'ESPN'} for d in [3,4,5,6,7]]
        p=np.zeros(5);x=pd.DataFrame({'const':np.ones(5)})
        np.testing.assert_allclose(apply_week1_major_days(self.model,x,rows,p,p,['Other']*5),[100,200,300,400,500])

    def test_absent_artifact_and_postgame_fields(self):
        p=np.array([1000.]);x=pd.DataFrame({'const':[1.]})
        self.assertIs(apply_week1_major_days(SimpleNamespace(),x,[],p,p,[]),p)
        row={'date':'09/06/26','network':'NBC'}
        a=apply_week1_major_days(self.model,x,[row],p,p,['Other'])
        b=apply_week1_major_days(self.model,x,[dict(row,actual='99M',score1=60,score2=0)],p,p,['Other'])
        np.testing.assert_array_equal(a,b)

    def test_nonlinear_uses_original_eligibility(self):
        self.model.nonlinear_pregame={'threshold_000s':1000.,'feature_columns':['Add_elo_average'],
            'model':Forest(),'smearing_factor':1.,'rank_corrections_000s':{}}
        row={'date':'09/05/26','network':'ABC','team1_pregame_elo':1500,'team2_pregame_elo':1600,'spread_home':7,'neutral_site':False}
        x=pd.DataFrame({'const':[1.,1.]});base=np.array([999.,1000.])
        result=apply_week1_major_days(self.model,x,[row,row],base,base,['Other','Other'])
        np.testing.assert_allclose(result,[1990.,2500.])


if __name__=='__main__':unittest.main()
