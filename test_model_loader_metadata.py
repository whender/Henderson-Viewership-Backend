import unittest

from model_loader import _attach_pregame_artifact_metadata


class _Model:
    pass


class PregameArtifactMetadataTests(unittest.TestCase):
    def test_attaches_intrinsic_contract_and_tier_mapping(self):
        model = _Model()
        intrinsic_model = _Model()
        artifact = {
            "team_counts": {"Alabama": 12},
            "smearing_factor": 1.04,
            "intrinsic_model": intrinsic_model,
            "intrinsic_smearing_factor": 1.08,
            "intrinsic_team_counts": {"Alabama": 11},
            "competition_score_by_tier": {0: 0.25, 1: 1.75},
            "pregame_ensemble": {"version": 1, "weight": 0.5},
        }

        loaded = _attach_pregame_artifact_metadata(model, artifact)

        self.assertIs(loaded, model)
        self.assertIs(loaded.intrinsic_model, intrinsic_model)
        self.assertEqual(loaded.intrinsic_smearing_factor, 1.08)
        self.assertEqual(loaded.intrinsic_team_counts, {"Alabama": 11})
        self.assertEqual(loaded.competition_score_by_tier[1], 1.75)
        self.assertIs(loaded.pregame_ensemble, artifact["pregame_ensemble"])

    def test_legacy_artifact_gets_explicit_empty_optional_metadata(self):
        loaded = _attach_pregame_artifact_metadata(
            _Model(),
            {"smearing_factor": 1.02},
        )

        self.assertIsNone(loaded.intrinsic_model)
        self.assertIsNone(loaded.intrinsic_smearing_factor)
        self.assertEqual(loaded.intrinsic_team_counts, {})
        self.assertEqual(loaded.competition_score_by_tier, {})
        self.assertIsNone(loaded.pregame_ensemble)


if __name__ == "__main__":
    unittest.main()
