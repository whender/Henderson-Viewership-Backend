import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pregame_ensemble import apply_monday_calibration


class MondayCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(monday_calibration={
            'version': 1, 'weight': .25, 'event_mean_000s': 4859.142857142857,
            'training_max_year': 2025, 'prediction_year': 2026})
        self.row = {'date': '09/07/26', 'time_slot': 'Monday 7:30p'}

    def test_date_time_and_model_day_scope_preserve_other_games(self):
        rows = [self.row, dict(self.row, date='09/06/26'),
                dict(self.row, time_slot='Monday 12:00p'),
                dict(self.row, date='09/01/25'), dict(self.row, date='09/06/27'),
                dict(self.row, date='09/21/26'), dict(self.row, date=''),
                dict(self.row, time_slot='Monday'), self.row]
        matrix = pd.DataFrame({'Monday': [1]*8+[0]})
        points = np.full(len(rows), 3861.762)
        result = apply_monday_calibration(self.model, matrix, rows, points)
        self.assertAlmostEqual(result[0], 4111.107214285715)
        np.testing.assert_array_equal(result[1:], points[1:])
        np.testing.assert_array_equal(points, np.full(len(rows), 3861.762))
        rows[0] = dict(self.row, actual='100M', score1=99, score2=0)
        np.testing.assert_array_equal(apply_monday_calibration(self.model, matrix, rows, points), result)

    def test_missing_and_invalid_configuration(self):
        points = np.array([3000.])
        self.assertIs(apply_monday_calibration(SimpleNamespace(), None, [], points), points)
        self.model.monday_calibration['weight'] = 2
        with self.assertRaisesRegex(ValueError, 'Invalid Monday'):
            apply_monday_calibration(self.model, None, [], points)


if __name__ == '__main__':
    unittest.main()
