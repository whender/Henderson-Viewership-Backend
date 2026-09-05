import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import main


class _FakeModel:
    def __init__(self, columns, prediction_thousands):
        self.params = pd.Series(
            np.zeros(len(columns)),
            index=pd.Index(columns),
            dtype=float,
        )
        self.prediction_thousands = prediction_thousands
        self.calls = []
        self.smearing_factor = 1.0
        self.intrinsic_model = None
        self.intrinsic_smearing_factor = None

    def predict(self, frame):
        self.calls.append(frame.copy())
        values = np.asarray(self.prediction_thousands, dtype=float).reshape(-1)
        if values.size == 1:
            values = np.repeat(values[0], len(frame))
        return pd.Series(
            np.log1p(values),
            index=frame.index,
        )


def _contexts():
    return [
        {
            "row": {
                "team1": "Alabama",
                "team2": "Auburn",
                "rank1": 5,
                "rank2": 0,
                "network": "ABC",
                "time_slot": "Sat Early",
                "date": "2025-09-06",
            }
        },
        {
            "row": {
                "team1": "Michigan",
                "team2": "Ohio State",
                "rank1": 4,
                "rank2": 2,
                "network": "FOX",
                "time_slot": "Sat Early",
                "date": "2025-09-06",
            }
        },
    ]


class SlateCompetitionInferenceTests(unittest.TestCase):
    def test_batch_honors_explicit_cfp_ranking_source(self):
        model = _FakeModel(
            ["const", "CFP_RankStrengthSum", "CFP_RankDifference"],
            prediction_thousands=1500,
        )
        contexts = [{
            "row": {
                "team1": "Alabama",
                "team2": "Georgia",
                "rank1": 3,
                "rank2": 8,
                "network": "ABC",
                "time_slot": "Sat Early",
                "season_week": 12,
                "ranking_source": "cfp",
            }
        }]

        with patch.object(main, "pregame_model", model):
            main._batch_predict_realignment_rows(
                contexts,
                compute_competition=False,
            )

        features = model.calls[-1].iloc[0]
        self.assertEqual(features["CFP_RankStrengthSum"], 41.0)
        self.assertEqual(features["CFP_RankDifference"], 5.0)

    def test_generated_full_slate_is_finalized_by_batch_predictor(self):
        schedule = [
            {
                "team1": "Alabama",
                "team2": "Auburn",
                "score": 1.0,
                "protected": False,
                "week": 5,
            },
            {
                "team1": "Michigan",
                "team2": "Ohio St.",
                "score": 0.9,
                "protected": False,
                "week": 5,
            },
        ]
        sim = SimpleNamespace(
            ranking_policy="custom",
            custom_rankings={},
            games_per_team=1,
        )

        def finalize(contexts):
            for context in contexts:
                context["row"]["predicted_viewers"] = 1_000_000.0
                context["row"]["predicted_viewers_formatted"] = "1.00M"

        with (
            patch.object(main, "_build_rank_map", return_value={}),
            patch.object(
                main,
                "_generate_conference_schedule",
                return_value=(schedule, {}),
            ),
            patch.object(
                main,
                "_select_realignment_tv_slot",
                return_value={
                    "network": "ABC",
                    "time_slot": "Sat Early",
                    "comp_tier1": 0,
                    "tv_tier": "broadcast",
                },
            ),
            patch.object(
                main,
                "_batch_predict_realignment_rows",
                side_effect=finalize,
            ) as batch_predict,
        ):
            slate = main._predict_realignment_slate(
                ["Alabama", "Auburn", "Michigan", "Ohio St."],
                "Big 10",
                sim,
            )

        batch_predict.assert_called_once()
        self.assertEqual(len(batch_predict.call_args.args[0]), 2)
        self.assertEqual(slate["summary"]["total_viewers"], 2_000_000.0)

    def test_full_slate_uses_unweighted_persisted_intrinsic_sum(self):
        final_model = _FakeModel(
            ["const", "Competing_Games_Score"],
            prediction_thousands=1500,
        )
        intrinsic_model = _FakeModel(
            ["const"],
            prediction_thousands=[1000, 3000],
        )
        final_model.intrinsic_model = intrinsic_model
        final_model.intrinsic_smearing_factor = 1.0
        contexts = _contexts()

        with patch.object(main, "pregame_model", final_model):
            main._batch_predict_realignment_rows(contexts)

        self.assertEqual(len(intrinsic_model.calls), 1)
        self.assertEqual(list(intrinsic_model.calls[0].columns), ["const"])
        self.assertEqual(len(final_model.calls), 1)
        self.assertAlmostEqual(contexts[0]["row"]["competing_games_score"], 3.0)
        self.assertAlmostEqual(contexts[1]["row"]["competing_games_score"], 1.0)
        for context in contexts:
            self.assertNotIn("warnings", context["row"])

    def test_legacy_slate_fallback_surfaces_warning(self):
        legacy_model = _FakeModel(
            ["const", "Competing_Games_Score"],
            prediction_thousands=1500,
        )
        contexts = _contexts()

        with patch.object(main, "pregame_model", legacy_model):
            main._batch_predict_realignment_rows(contexts)

        self.assertEqual(len(legacy_model.calls), 2)
        for context in contexts:
            self.assertIn(
                main.INTRINSIC_COMPETITION_FALLBACK_WARNING,
                context["row"]["warnings"],
            )


if __name__ == "__main__":
    unittest.main()
