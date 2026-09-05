import unittest

from pregame_context_features import (
    build_pregame_context_features,
    parse_kickoff_hour,
    resolve_season_week,
)


class PregameContextFeatureTests(unittest.TestCase):
    def test_kickoff_threshold_and_clock_free_late_bucket(self):
        self.assertEqual(parse_kickoff_hour("Friday 9:00p"), 21.0)
        self.assertEqual(parse_kickoff_hour("10:30"), 22.5)

        at_threshold, _ = build_pregame_context_features(time_slot="Friday 9:00p")
        before_threshold, _ = build_pregame_context_features(time_slot="Late afternoon 4:00p")
        labeled_bucket, _ = build_pregame_context_features(time_slot="Sat Late")

        self.assertEqual(at_threshold["AnyDayKickoff9pmOrLater"], 1)
        self.assertEqual(before_threshold["AnyDayKickoff9pmOrLater"], 0)
        self.assertEqual(labeled_bucket["AnyDayKickoff9pmOrLater"], 1)

    def test_explicit_week_precedes_date_and_january_stays_in_prior_season(self):
        self.assertEqual(resolve_season_week(7, "2025-11-15"), 7)
        self.assertGreaterEqual(resolve_season_week(date_value="2026-01-01"), 18)

    def test_records_prefer_actual_then_infer_then_use_neutral_warning(self):
        actual, actual_warnings = build_pregame_context_features(
            season_week=11,
            team1_games_before=8,
            team2_games_before=9,
        )
        inferred, inferred_warnings = build_pregame_context_features(season_week=4)
        neutral, neutral_warnings = build_pregame_context_features()

        self.assertEqual(actual["EarlyRecordNeutralized"], 0)
        self.assertEqual(actual["MinGamesBefore"], 8)
        self.assertEqual(actual_warnings, [])
        self.assertEqual(inferred["EarlyRecordNeutralized"], 1)
        self.assertEqual(inferred["MinGamesBefore"], 3)
        self.assertEqual(inferred_warnings, [])
        self.assertEqual(neutral["EarlyRecordNeutralized"], 0)
        self.assertEqual(neutral["MinGamesBefore"], 6)
        self.assertEqual(len(neutral_warnings), 1)

    def test_cfp_source_rules_and_late_both_ranked(self):
        automatic, _ = build_pregame_context_features(
            season_week=11,
            rank1=4,
            rank2=12,
            ranking_source="auto",
        )
        media_poll, _ = build_pregame_context_features(
            season_week=11,
            rank1=4,
            rank2=12,
            ranking_source="ap",
        )
        explicit, _ = build_pregame_context_features(
            season_week=11,
            rank1=4,
            rank2=12,
            cfp_rank1=2,
            cfp_rank2=0,
            ranking_source="ap",
        )

        self.assertEqual(automatic["LateBothRanked"], 1)
        self.assertEqual(automatic["CFP_RankStrengthSum"], 36)
        self.assertEqual(automatic["CFP_RankDifference"], 8)
        self.assertEqual(media_poll["CFP_RankStrengthSum"], 0)
        self.assertEqual(media_poll["CFP_RankDifference"], 0)
        self.assertEqual(explicit["CFP_RankStrengthSum"], 24)
        self.assertEqual(explicit["CFP_RankDifference"], 24)


if __name__ == "__main__":
    unittest.main()
