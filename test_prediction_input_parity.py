import unittest
from types import SimpleNamespace

import pandas as pd

from predict import (
    MISSING_COMPETITION_TIER_MAPPING_WARNING,
    resolve_competing_games_score,
    resolve_game_record_features,
)
from weekly_predictions_fs import build_features
from main import GameInput


def _competition_model(mapping=None):
    return SimpleNamespace(
        params=pd.Series(
            [0.0],
            index=pd.Index(["Competing_Games_Score"]),
            dtype=float,
        ),
        competition_score_by_tier=mapping or {},
    )


class PredictionInputParityTests(unittest.TestCase):
    def test_weekly_ohio_state_btn_adjustment(self):
        row = dict(team1="Ohio St.", team2="Ball St.", rank1=1, rank2=0,
                   network="BTN", time_slot="12:30p", date="09/05/26",
                   competing_games_score=0, week=1)
        self.assertEqual(build_features(row)["OhioSt_BTN"], 1)
        self.assertEqual(build_features(dict(row, network="ABC"))["OhioSt_BTN"], 0)
        self.assertEqual(build_features(dict(row, team1="Oregon"))["OhioSt_BTN"], 0)

    def test_api_schema_accepts_and_propagates_bounded_team_records(self):
        game = GameInput(
            team1="Alabama",
            team2="Auburn",
            rank1=5,
            rank2=0,
            network="ABC",
            time_slot="Sat Early",
            team1_win_pct_to_date=0.0,
            team2_win_pct_to_date=0.8,
        )

        self.assertEqual(game.dict()["team1_win_pct_to_date"], 0.0)
        self.assertEqual(game.dict()["team2_win_pct_to_date"], 0.8)
        with self.assertRaises(ValueError):
            GameInput(
                team1="Alabama",
                team2="Auburn",
                rank1=5,
                rank2=0,
                network="ABC",
                time_slot="Sat Early",
                team1_win_pct_to_date=1.1,
            )

    def test_explicit_competition_precedes_persisted_tier_median(self):
        score, warnings = resolve_competing_games_score(
            {"competing_games_score": 4.5, "comp_tier1": 2},
            _competition_model({2: 1.25}),
        )

        self.assertEqual(score, 4.5)
        self.assertEqual(warnings, [])

    def test_tier_uses_persisted_training_median(self):
        score, warnings = resolve_competing_games_score(
            {"comp_tier1": 2},
            _competition_model({"2": 1.25}),
        )

        self.assertEqual(score, 1.25)
        self.assertEqual(warnings, [])

    def test_unseen_tier_uses_persisted_overall_training_median(self):
        score, warnings = resolve_competing_games_score(
            {"comp_tier1": 8},
            _competition_model({0: 0.5, "overall": 3.25}),
        )

        self.assertEqual(score, 3.25)
        self.assertEqual(warnings, [])

    def test_legacy_tier_fallback_is_never_silent(self):
        score, warnings = resolve_competing_games_score(
            {"comp_tier1": 2},
            _competition_model(),
        )

        self.assertEqual(score, 0.0)
        self.assertEqual(warnings, [MISSING_COMPETITION_TIER_MAPPING_WARNING])

    def test_zero_win_percentage_is_preserved_and_weekly_features_match(self):
        values = {
            "team1_games_before": 5,
            "team2_games_before": 7,
            "team1_win_pct_to_date": 0.0,
            "team2_win_pct_to_date": 0.8,
        }
        self.assertEqual(resolve_game_record_features(values), (0.0, 0.8, 0.4, 0.8))

        features = build_features({
            "team1": "Alabama",
            "team2": "Auburn",
            "rank1": 0,
            "rank2": 0,
            "network": "ABC",
            "time_slot": "Sat Early",
            "date": "09/06/25",
            "competing_games_score": 0.0,
            **values,
        })

        self.assertEqual(features["Team1_WinPct_ToDate"], 0.0)
        self.assertEqual(features["Team2_WinPct_ToDate"], 0.8)
        self.assertEqual(features["Avg_WinPct_ToDate"], 0.4)
        self.assertEqual(features["WinPct_Diff_ToDate"], 0.8)

    def test_early_records_are_neutralized_before_average_and_difference(self):
        values = {
            "season_week": 4,
            "team1_win_pct_to_date": 0.0,
            "team2_win_pct_to_date": 1.0,
        }

        self.assertEqual(resolve_game_record_features(values), (0.5, 0.5, 0.5, 0.0))

        features = build_features({
            "team1": "Alabama",
            "team2": "Auburn",
            "rank1": 0,
            "rank2": 0,
            "network": "ABC",
            "time_slot": "Sat Early",
            "season_week": 4,
            "competing_games_score": 0.0,
            **values,
        })

        self.assertEqual(features["Team1_WinPct_ToDate"], 0.5)
        self.assertEqual(features["Team2_WinPct_ToDate"], 0.5)
        self.assertEqual(features["Avg_WinPct_ToDate"], 0.5)
        self.assertEqual(features["WinPct_Diff_ToDate"], 0.0)


if __name__ == "__main__":
    unittest.main()
