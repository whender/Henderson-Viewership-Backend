import copy,unittest
from unittest.mock import patch
import main
from weekly_forecast_state import refresh_latest_forecast
from test_weekly_week_propagation import _FakeDB,_FakeDocument,_FakeCollection

class LatestPredictionTests(unittest.TestCase):
    def test_latest_forecast_drives_rows_and_aggregate_and_full_slate(self):
        row=dict(team1='Ohio State',predicted='1.00M',revised_predicted='1.80M',actual='2.00M',percent_error=50,accuracy='🔴',revision_timing='retrospective',revised_at='2026-09-13T18:00:00Z')
        data=dict(year=2026,week=2,games=[row],full_slate_games=[copy.deepcopy(row)])
        collection=_FakeCollection(_FakeDocument(data))
        with patch.object(main,'db',_FakeDB(collection)),patch.object(main,'pregame_prediction_warnings',return_value=[]):
            response=main.weekly_predictions()
            again=main.weekly_predictions()
        for g in [response['weeks'][0]['games'][0],data['full_slate_games'][0]]:
            self.assertEqual(g['predicted'],'1.80M')
            self.assertNotIn('revised_predicted',g)
            self.assertAlmostEqual(g['percent_error'],10.)
            self.assertEqual(g['prediction_history'][0]['predicted'],'1.00M')
            self.assertEqual(len(g['prediction_history']),1)
            self.assertEqual(g['forecast_timing'],'retrospective')
        self.assertAlmostEqual(response['metrics']['pregame']['mean_error'],10.)
        self.assertEqual(response['accuracy_basis'],'latest_predictions')
        self.assertEqual(response['metrics'],again['metrics'])

    def test_corrects_stale_error_even_without_a_new_revision(self):
        row=dict(predicted='3M',actual='2M',percent_error=0,accuracy='🟢🎯')
        self.assertTrue(refresh_latest_forecast(row));self.assertEqual(row['percent_error'],50)
        self.assertFalse(refresh_latest_forecast(row))
        row['actual']='';refresh_latest_forecast(row)
        self.assertIsNone(row['percent_error']);self.assertEqual(row['accuracy'],'')

    def test_blank_revision_keeps_current(self):
        row=dict(predicted='3M',actual='2M',revised_predicted='')
        refresh_latest_forecast(row);self.assertEqual(row['predicted'],'3M')

if __name__=='__main__':unittest.main()
