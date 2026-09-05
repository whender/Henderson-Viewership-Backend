import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import main
import predict
import weekly_predictions_fs
from pregame_ensemble import (
    build_exact_pregame_features,
    exact_kickoff_window,
    predict_pregame_points_000s,
    rank_scope,
)


class _LinearLogModel:
    def __init__(self, coefficients, smear=1.0):
        self.params = pd.Series(coefficients, dtype=float)
        self.smearing_factor = smear
        self.competition_score_by_tier = {}
        self.pregame_ensemble = None
        self.calls = []

    def predict(self, matrix):
        aligned = matrix.reindex(columns=self.params.index, fill_value=0.0)
        self.calls.append(aligned.copy())
        return aligned.dot(self.params)


def _ensemble_models():
    primary = _LinearLogModel({
        "const": 7.0,
        "Competing_Games_Score": -0.03,
        "ABC": 0.12,
    }, smear=1.04)
    challenger = _LinearLogModel({
        "const": 6.9,
        "Competing_Games_Score": -0.025,
        "ABC": 0.10,
        "BothRanked_Nonlinear": 0.15,
        "BothTop10_Nonlinear": 0.20,
        "ExactKickoffWindow_SatPrime": 0.08,
        "ExactNetworkKickoff_ABC_SatPrime": 0.06,
    }, smear=1.08)
    primary.pregame_ensemble = {
        "name": "test_shared_primary_ensemble",
        "version": 1,
        "weight": 0.5,
        "challenger_model": challenger,
        "challenger_smearing_factor": challenger.smearing_factor,
        "challenger_team_counts": {},
        "competition_mode": "shared_primary",
        "rank_scope_additive_adjustments_000s": {
            "BothRanked": 30.0,
            "BothTop10": 50.0,
            "Other": 0.0,
        },
    }
    return primary, challenger


class ExactEnsembleFeatureTests(unittest.TestCase):
    def test_missing_challenger_names_fail_instead_of_halving_prediction(self):
        primary, challenger = _ensemble_models()
        challenger.params.index = pd.RangeIndex(len(challenger.params))
        frame = pd.DataFrame([{"const": 1.0, "Competing_Games_Score": 0.0, "ABC": 1.0}])
        with self.assertRaisesRegex(ValueError, "feature names"):
            predict_pregame_points_000s(primary, frame, [{"network": "ABC"}])

    def test_stripped_statsmodels_roundtrip_preserves_blend(self):
        import io
        import joblib
        import statsmodels.api as sm
        primary, _ = _ensemble_models()
        training = pd.DataFrame({"const": [1.0]*4, "ABC": [0.0, 1.0, 0.0, 1.0]})
        challenger = sm.OLS([7.0, 7.3, 7.1, 7.4], training).fit()
        primary.pregame_ensemble["challenger_model"] = challenger
        primary.pregame_ensemble["challenger_feature_columns"] = list(training.columns)
        frame = pd.DataFrame([{"const": 1.0, "Competing_Games_Score": 0.0, "ABC": 1.0}])
        expected = predict_pregame_points_000s(primary, frame, [{"network": "ABC"}])
        challenger.remove_data()
        buf = io.BytesIO()
        joblib.dump(primary, buf)
        buf.seek(0)
        restored = joblib.load(buf)
        np.testing.assert_allclose(predict_pregame_points_000s(restored, frame, [{"network": "ABC"}]), expected)

    def test_exact_window_boundaries_and_non_saturday(self):
        self.assertEqual(
            exact_kickoff_window(
                time_slot="2:29p",
                date_value="2026-10-03",
            ),
            "SatNoon",
        )
        self.assertEqual(
            exact_kickoff_window(
                time_slot="2:30p",
                date_value="2026-10-03",
            ),
            "SatAfternoon",
        )
        self.assertEqual(
            exact_kickoff_window(
                time_slot="7:00p",
                date_value="2026-10-03",
            ),
            "SatPrime",
        )
        self.assertEqual(
            exact_kickoff_window(
                time_slot="9:00p",
                date_value="2026-10-03",
            ),
            "SatLate",
        )
        self.assertEqual(
            exact_kickoff_window(
                time_slot="Friday Primetime",
                date_value="2026-10-03",
                day="Sat",
            ),
            "NonSat",
        )

    def test_reference_cells_and_top_ten_scope(self):
        reference = build_exact_pregame_features(
            network="ESPN",
            time_slot="Sat Mid (2:30p-6:30p)",
            rank1=3,
            rank2=8,
        )
        self.assertEqual(reference["BothRanked_Nonlinear"], 1.0)
        self.assertEqual(reference["BothTop10_Nonlinear"], 1.0)
        self.assertFalse(any(
            value
            for key, value in reference.items()
            if key.startswith("Exact")
        ))
        self.assertEqual(rank_scope(3, 8), "BothTop10")
        self.assertEqual(rank_scope(3, 18), "BothRanked")
        self.assertEqual(rank_scope(3, 0), "Other")

    def test_nonreference_station_window_interaction(self):
        features = build_exact_pregame_features(
            network="ABC",
            time_slot="Sat Prime (7:00p-9:00p)",
            rank1=3,
            rank2=18,
        )
        self.assertEqual(features["ExactKickoffWindow_SatPrime"], 1.0)
        self.assertEqual(features["ExactNetworkKickoff_ABC_SatPrime"], 1.0)
        self.assertEqual(features["BothRanked_Nonlinear"], 1.0)
        self.assertEqual(features["BothTop10_Nonlinear"], 0.0)


class PregameEnsemblePointTests(unittest.TestCase):
    def test_manual_level_blend_shared_score_and_additive_calibration(self):
        primary, challenger = _ensemble_models()
        matrix = pd.DataFrame([{
            "const": 1.0,
            "Competing_Games_Score": 4.25,
            "ABC": 1.0,
        }])
        context = {
            "network": "ABC",
            "time_slot": "Sat Prime (7:00p-9:00p)",
            "date": "2026-10-03",
            "rank1": 3,
            "rank2": 8,
        }

        actual = predict_pregame_points_000s(primary, matrix, [context])[0]
        primary_log = 7.0 - 0.03 * 4.25 + 0.12
        challenger_log = (
            6.9 - 0.025 * 4.25 + 0.10 + 0.15 + 0.20 + 0.08 + 0.06
        )
        primary_point = (np.exp(primary_log) - 1.0) * 1.04
        challenger_point = (np.exp(challenger_log) - 1.0) * 1.08
        expected = 0.5 * primary_point + 0.5 * challenger_point + 50.0

        self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(
            challenger.calls[-1]["Competing_Games_Score"].iloc[0],
            4.25,
        )

    def test_legacy_artifact_is_exact_primary_noop(self):
        primary, _ = _ensemble_models()
        primary.pregame_ensemble = None
        matrix = pd.DataFrame([{
            "const": 1.0,
            "Competing_Games_Score": 4.25,
            "ABC": 1.0,
        }])
        actual = predict_pregame_points_000s(primary, matrix, [{}])[0]
        expected = (np.exp(7.0 - 0.03 * 4.25 + 0.12) - 1.0) * 1.04
        self.assertEqual(actual, expected)

    def test_direct_weekly_and_batch_points_are_consistent(self):
        primary, _ = _ensemble_models()
        game = {
            "team1": "Alabama",
            "team2": "Georgia",
            "rank1": 3,
            "rank2": 8,
            "network": "ABC",
            "time_slot": "Sat Prime (7:00p-9:00p)",
            "date": "2026-10-03",
            "season_week": 6,
            "team1_games_before": 5,
            "team2_games_before": 5,
            "team1_win_pct_to_date": 0.8,
            "team2_win_pct_to_date": 0.8,
            "competing_games_score": 4.25,
        }

        with (
            patch.object(predict, "model", primary),
            patch.object(weekly_predictions_fs, "pregame_model", primary),
            patch.object(main, "pregame_model", primary),
        ):
            direct_000s = predict.predict_viewership(game)["raw"] / 1000.0
            weekly = weekly_predictions_fs.generate_pregame_prediction(game)
            contexts = [{"row": dict(game)}]
            main._batch_predict_realignment_rows(
                contexts,
                compute_competition=False,
            )
            batch_000s = contexts[0]["row"]["predicted_viewers"] / 1000.0

        weekly_000s = float(weekly.split("M", 1)[0]) * 1000.0
        self.assertAlmostEqual(direct_000s, batch_000s, places=10)
        self.assertAlmostEqual(direct_000s, weekly_000s, delta=5.01)


if __name__ == "__main__":
    unittest.main()
