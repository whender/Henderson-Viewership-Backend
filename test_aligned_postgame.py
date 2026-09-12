import copy,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from aligned_postgame import score_difference,predict_postgame_points_000s
from weekly_predictions_fs import aligned_postgame,build_features,pregame_model,generate_postgame_prediction
from pregame_ensemble import predict_pregame_points_000s


class AlignedPostgameTests(unittest.TestCase):
    def row(self):
        return dict(team1='Colorado',team2='Georgia Tech',rank1=0,rank2=0,network='ESPN',time_slot='Weekday 8:00p',
            date='09/03/26',week=1,season_week=1,score1=14,score2=13,competing_games_score=.5)

    def test_complete_feature_contract(self):
        a=aligned_postgame
        self.assertEqual(set(a['components'][0]['columns']),set(pregame_model.params.index)|{'Score Diff'})
        self.assertEqual(set(a['components'][1]['columns']),set(pregame_model.pregame_ensemble['challenger_feature_columns'])|{'Score Diff'})
        self.assertEqual(set(a['nonlinear_pregame']['feature_columns']),set(pregame_model.nonlinear_pregame['feature_columns'])|{'Score Diff'})
        for post,pre in zip(a['week1_major_days']['components'],pregame_model.week1_major_days['components']):
            self.assertEqual(set(post['columns']),set(pre['columns'])|{'Score Diff'})

    def test_scores_are_required_and_valid(self):
        for value in [None,-1,float('nan'),float('inf'),True,1.5,'bad']:
            self.assertIsNone(score_difference(dict(score1=value,score2=0)))
        self.assertEqual(score_difference(dict(score1=0,score2=39)),39)
        self.assertEqual(score_difference(dict(score1=39,score2=0)),39)
        self.assertIsNone(generate_postgame_prediction(dict(self.row(),score1=None)))

    def test_margin_changes_only_postgame_and_no_actual_leak(self):
        row=self.row();other=dict(row,score1=52,score2=0,actual='99M')
        x=pd.DataFrame([build_features(row)]).reindex(columns=pregame_model.params.index,fill_value=0.)
        y=pd.DataFrame([build_features(other)]).reindex(columns=pregame_model.params.index,fill_value=0.)
        pd.testing.assert_frame_equal(x,y)
        np.testing.assert_array_equal(predict_pregame_points_000s(pregame_model,x,[row]),predict_pregame_points_000s(pregame_model,y,[other]))
        a=predict_postgame_points_000s(pregame_model,aligned_postgame,x,[row])
        b=predict_postgame_points_000s(pregame_model,aligned_postgame,y,[other])
        self.assertGreater(a[0],b[0])
        np.testing.assert_array_equal(a,predict_postgame_points_000s(pregame_model,aligned_postgame,x,[dict(row,actual='99M')]))
        self.assertNotIn('Score Diff',x)

    def test_saved_deion_override_shared_by_both_pipelines(self):
        row=dict(self.row(),prediction_feature_override={'DeionEra25':1})
        self.assertEqual(build_features(row)['DeionEra25'],1)
        self.assertEqual(build_features(self.row())['DeionEra25'],0)
        self.assertEqual(build_features(dict(row,team1='Baylor'))['DeionEra25'],0)
        self.assertNotEqual(generate_postgame_prediction(row),generate_postgame_prediction(self.row()))

    def test_complete_nonlinear_context_and_mixed_rows(self):
        rows=[self.row(),dict(self.row(),date='09/12/26',week=2,team1_pregame_elo=1500,team2_pregame_elo=1600,spread_home=7,neutral_site=False)]
        x=pd.DataFrame([build_features(r) for r in rows]).reindex(columns=pregame_model.params.index,fill_value=0.)
        result=predict_postgame_points_000s(pregame_model,aligned_postgame,x,rows)
        self.assertTrue(np.isfinite(result).all());self.assertTrue((result>0).all())
        for i,row in enumerate(rows):
            np.testing.assert_allclose(result[i],predict_postgame_points_000s(pregame_model,aligned_postgame,x.iloc[[i]],[row])[0])


if __name__=='__main__':unittest.main()
