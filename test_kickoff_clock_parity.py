import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import predict
from weekly_predictions_fs import build_features
from pregame_context_features import saturday_kickoff_features

class KickoffClockTests(unittest.TestCase):
    def test_training_boundaries_and_non_saturdays(self):
        for clock,expected in [('12:00p',(1,0,0)),('3:30p',(0,1,0)),('7:30p',(0,0,0)),('10:15p',(0,0,1)),('6:30p',(0,0,0))]:
            self.assertEqual(tuple(saturday_kickoff_features(clock,'09/12/26').values()),expected)
        for text in ['Sunday 3:30p','Monday 12:00p','Friday 10:15p','Weekday 12:00p']:
            self.assertEqual(sum(saturday_kickoff_features(text,'09/12/26').values()),0)
        self.assertEqual(saturday_kickoff_features('Sat Early')['Sat Early'],1)

    def test_api_and_weekly_matrices_for_complete_inputs(self):
        for slot in ['12:00p','3:30p','7:30p','10:15p']:
            row=dict(team1='Oklahoma',team2='Michigan',rank1=11,rank2=0,network='FOX',time_slot=slot,date='09/12/26',week=2,season_week=2,
                team1_games_before=1,team2_games_before=1,team1_win_pct_to_date=1.,team2_win_pct_to_date=1.,competing_games_score=4.5,
                team1_pregame_elo=1799,team2_pregame_elo=1686,spread_home=5.5,neutral_site=False)
            captured=[]
            def capture(model,x,rows):captured.append(x.copy());return np.array([1000.])
            with patch.object(predict,'predict_pregame_points_000s',side_effect=capture):predict.predict_viewership(row)
            weekly=pd.DataFrame([build_features(row)]).reindex(columns=predict.model.params.index,fill_value=0.)
            np.testing.assert_array_equal(captured[0].to_numpy(),weekly.to_numpy())

if __name__=='__main__':unittest.main()
