import copy,json,unittest
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from audience_interest import feature_values,interest_cutoff,row_features,apply_audience_interest,FEATURES


class WindowTests(unittest.TestCase):
    def test_fixed_weekly_cutoff_and_future_exclusion(self):
        d=date(2026,9,12);cut=interest_cutoff(d)
        self.assertEqual(cut,date(2026,9,7))
        daily={t:{(cut-timedelta(days=i)).isoformat():value for i in range(1,8)} for t,value in [('A',10),('B',20)]}
        got=feature_values(d,['A','B'],daily)
        self.assertAlmostEqual(got['InterestLogTotal7'],np.log1p(210))
        self.assertAlmostEqual(got['InterestLogMax7'],np.log1p(140))
        for t in daily:daily[t][cut.isoformat()]=100000000
        self.assertEqual(got,feature_values(d,['A','B'],daily))
        self.assertEqual(got,feature_values(d,['B','A'],daily))
        self.assertIsNone(feature_values(d,['A','B'],daily,datetime(2026,9,7,tzinfo=timezone.utc)))
        self.assertEqual(got,feature_values(d,['A','B'],daily,datetime(2026,9,8,tzinfo=timezone.utc)))
        del daily['A']['2026-09-06']
        self.assertIsNone(feature_values(d,['A','B'],daily))

    def test_early_week_never_uses_game_day(self):
        for i in range(7):
            d=date(2026,9,8)+timedelta(days=i)
            self.assertLessEqual(interest_cutoff(d),d-timedelta(days=2))


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from weekly_predictions_fs import pregame_model,aligned_postgame,build_features
        cls.model=pregame_model;cls.post=aligned_postgame;cls.build=staticmethod(build_features)
        cls.rows=json.loads((Path(__file__).parent/'research/audience_interest_week2_fixture.json').read_text())

    def matrix(self,rows):
        return pd.DataFrame([self.build(r) for r in rows]).reindex(columns=self.model.params.index,fill_value=0.)

    def test_complete_week2_coverage_and_api_weekly_parity(self):
        from predict import predict_viewership
        from weekly_predictions_fs import generate_pregame_prediction,parse_viewership
        from pregame_ensemble import predict_pregame_points_000s
        from main import GameInput
        rows=self.rows;x=self.matrix(rows);batch=predict_pregame_points_000s(self.model,x,rows)
        self.assertEqual(len(rows),34)
        for i,r in enumerate(rows):
            self.assertIsNotNone(row_features(self.model.audience_interest,r))
            self.assertTrue(np.isfinite(batch[i]) and batch[i]>=0)
            one=predict_pregame_points_000s(self.model,x.iloc[[i]],[r])[0]
            self.assertAlmostEqual(batch[i],one,places=7)
            api=predict_viewership(GameInput(**r).model_dump())['raw']/1000
            self.assertAlmostEqual(one,api,places=7)
            self.assertAlmostEqual(round(one/1000,2),parse_viewership(generate_pregame_prediction(r)),places=7)

    def test_missing_or_premature_data_exact_fallback(self):
        from pregame_ensemble import predict_pregame_points_000s
        old=copy.copy(self.model);old.audience_interest=None
        for row in [dict(self.rows[0],feature_as_of='2026-08-01T00:00:00+00:00'),dict(self.rows[0],date='09/12/27'),dict(self.rows[0],team1='No history')]:
            x=self.matrix([row]);np.testing.assert_array_equal(predict_pregame_points_000s(self.model,x,[row]),predict_pregame_points_000s(old,x,[row]))

    def test_pre_post_feature_contract_and_no_score_leak(self):
        from aligned_postgame import predict_postgame_points_000s
        from pregame_ensemble import predict_pregame_points_000s
        a=self.model.audience_interest
        for branch in ['regular','week1']:
            for pre,post in zip(a['pregame']['components'][branch],a['postgame']['components'][branch]):
                self.assertEqual(set(post['columns']),set(pre['columns'])|{'Score Diff'})
                self.assertTrue(set(FEATURES).issubset(pre['columns']))
        self.assertEqual(set(a['postgame']['forest']['feature_columns']),set(a['pregame']['forest']['feature_columns'])|{'Score Diff'})
        r=dict(self.rows[0],score1=21,score2=20);s=dict(r,score1=63,score2=0,actual='100M')
        x=self.matrix([r,s]);pre=predict_pregame_points_000s(self.model,x,[r,s])
        self.assertEqual(pre[0],pre[1])
        post=predict_postgame_points_000s(self.model,self.post,x,[r,s]);self.assertNotEqual(post[0],post[1])
        no_actual=predict_postgame_points_000s(self.model,self.post,x.iloc[[1]],[dict(s,actual='')])
        self.assertAlmostEqual(no_actual[0],post[1])

    def test_serialized_components_match_manual_prediction(self):
        from nonlinear_pregame import context_values
        from pregame_ensemble import exact_feature_frame,rank_scope
        row=next(r for r in self.rows if context_values(r) is not None);x=self.matrix([row]);a=self.model.audience_interest
        f=x.copy();extra=exact_feature_frame([row],index=x.index)
        for c in extra:f[c]=extra[c]
        for c,v in row_features(a,row).items():f[c]=v
        cfg=a['pregame'];scope=rank_scope(row['rank1'],row['rank2'])
        raw=sum(.5*max(float(np.exp(f[c['columns']].to_numpy()@c['coefficients'])[0])*c['smearing_factor']-1,0) for c in cfg['components']['regular'])
        expected=max(raw-cfg['corrections']['regular']['linear'].get(scope,0),0)
        actual=apply_audience_interest(self.model,x,[row],np.array([0.]),np.array([1.]),[scope])[0]
        self.assertAlmostEqual(expected,actual,places=7)
        values=context_values(row);self.assertIsNotNone(values)
        for c,v in values.items():f[c]=v;f[c+'_missing']=0.
        forest=cfg['forest'];fp=max(float(np.exp(forest['model'].predict(f[forest['feature_columns']]))[0])*forest['smearing_factor']-1,0)
        expected=max(.75*raw+.25*fp-cfg['corrections']['regular']['nonlinear'].get(scope,0),0)
        actual=apply_audience_interest(self.model,x,[row],np.array([2000.]),np.array([1.]),[scope])[0]
        self.assertAlmostEqual(expected,actual,places=7)

if __name__=='__main__':unittest.main()
