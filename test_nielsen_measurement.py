import unittest
from nielsen_measurement import measurement_era_flags, measurement_status, REGIMES
from datetime import date, timedelta


class NielsenMeasurementTests(unittest.TestCase):
    def test_each_effective_boundary(self):
        for name, config in REGIMES.items():
            day = date.fromisoformat(config['effective_date'])
            self.assertEqual(measurement_era_flags(day - timedelta(days=1))[name], 0)
            self.assertEqual(measurement_era_flags(day)[name], 1)

    def test_opening_weekends_straddle_2026_revision(self):
        self.assertEqual(measurement_era_flags('08/29/26')['Nielsen2026RevisionEra'], 0)
        self.assertEqual(measurement_era_flags('09/03/26')['Nielsen2026RevisionEra'], 1)
        self.assertTrue(all(v == 1 for v in measurement_era_flags('2026-09-12').values()))

    def test_unknown_date_is_not_old_era(self):
        self.assertTrue(all(v is None for v in measurement_era_flags('unknown').values()))
        self.assertFalse(measurement_status()['additional_adjustment_enabled'])


if __name__ == '__main__':
    unittest.main()
