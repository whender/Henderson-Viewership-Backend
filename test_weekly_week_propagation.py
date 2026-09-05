import unittest
from unittest.mock import patch

import main


class _FakeDocument:
    id = "week-doc"

    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


class _FakeCollection:
    def __init__(self, document):
        self._document = document
        self.writes = []

    def stream(self):
        return [self._document]

    def document(self, document_id):
        self.document_id = document_id
        return self

    def set(self, data):
        self.writes.append(data)


class _FakeDB:
    def __init__(self, collection):
        self._collection = collection

    def collection(self, _name):
        return self._collection


class WeeklyWeekPropagationTest(unittest.TestCase):
    def test_weeks_sort_by_year_before_week_number(self):
        documents = [_FakeDocument({"year": year, "week": week, "games": []})
                     for year, week in [(2025, 1), (2025, 14), (2026, 1), (2025, 11)]]
        collection = _FakeCollection(documents[0])
        with patch.object(collection, "stream", return_value=documents), patch.object(main, "db", _FakeDB(collection)):
            response = main.weekly_predictions()
        self.assertEqual([(w["year"], w["week"]) for w in response["weeks"]],
                         [(2026, 1), (2025, 14), (2025, 11), (2025, 1)])

    def test_document_week_reaches_both_predictors_without_adding_it_to_game(self):
        game = {
            "team1": "Ohio State",
            "team2": "Michigan",
            "rank1": 2,
            "rank2": 5,
            "network": "FOX",
            "time_slot": "Saturday Late",
            "score1": 24,
            "score2": 17,
            "custom_marker": "unchanged",
        }
        document = _FakeDocument({
            "week": 7,
            "season_week": 8,
            "year": 2025,
            "games": [game],
        })
        fake_db = _FakeDB(_FakeCollection(document))
        pregame_inputs = []
        postgame_inputs = []

        def fake_pregame(row):
            pregame_inputs.append(dict(row))
            return "1.00M"

        def fake_postgame(row):
            postgame_inputs.append(dict(row))
            return "0.90M"

        with (
            patch.object(main, "db", fake_db),
            patch.object(main, "generate_pregame_prediction", fake_pregame),
            patch.object(main, "generate_postgame_prediction", fake_postgame),
        ):
            response = main.weekly_predictions()

        self.assertEqual(pregame_inputs[0]["week"], 7)
        self.assertEqual(pregame_inputs[0]["season_week"], 8)
        self.assertEqual(postgame_inputs[0]["week"], 7)
        self.assertEqual(postgame_inputs[0]["season_week"], 8)
        self.assertNotIn("week", game)
        self.assertNotIn("season_week", game)
        self.assertNotIn("warnings", game)
        self.assertTrue(response["weeks"][0]["games"][0]["warnings"])
        self.assertEqual(game["custom_marker"], "unchanged")

    def test_document_week_is_the_season_week_fallback(self):
        game = {"team1": "Ohio State"}
        prediction_input = main._weekly_prediction_input(game, document_week=6)

        self.assertEqual(prediction_input["week"], 6)
        self.assertEqual(prediction_input["season_week"], 6)
        self.assertEqual(game, {"team1": "Ohio State"})


if __name__ == "__main__":
    unittest.main()
