import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from expected_viewership import compute_expectations, neutral_brand_matrix


class ExpectedViewershipTests(unittest.TestCase):
    def test_neutralization_preserves_opponent_and_all_context(self):
        df = pd.DataFrame({'Team 1': ['Colorado'], 'Team 2': ['Virginia']})
        x = pd.DataFrame({'Colorado': [1.], 'Virginia': [1.], 'ABC': [1.], 'DeionEra25': [1.], 'Competing_Games_Score': [4.]})
        got = neutral_brand_matrix(x, df, {'Colorado', 'Virginia'}, {'Colorado', 'Virginia'}, 'Team 1')
        self.assertEqual(got.iloc[0].to_dict(), {'Colorado': .5, 'Virginia': 1.5, 'ABC': 1., 'DeionEra25': 0., 'Competing_Games_Score': 4.})
        self.assertEqual(x.at[0, 'Colorado'], 1.)

    def test_all_expectations_use_shared_pipeline_with_observed_context(self):
        df = pd.DataFrame({'Team 1': ['A']*5, 'Team 2': ['B']*5, 'source_index': range(5), 'ParsedDate': ['2026-09-04']*5, 'FOX': [1.]*5})
        model = SimpleNamespace(params=pd.Series([0.]*4,index=['const','A','B','FOX']))
        contexts = {str(i): {'team1':'A','team2':'B','date':'2026-09-04','network':'FOX','week':1,'spread_home':7,'team1_pregame_elo':1500,'team2_pregame_elo':1600,'neutral_site':False} for i in range(5)}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'context.json';path.write_text(json.dumps(contexts))
            with patch('expected_viewership.predict_pregame_points_000s', return_value=np.full(5,1234.)) as predict:
                result=compute_expectations(model,df,{'A','B'},{'A','B'},path)
                self.assertEqual(predict.call_count,3)
                self.assertTrue((result==1234.).all().all())
                for call in predict.call_args_list:
                    self.assertIs(call.args[0],model)
                    self.assertEqual(call.args[2],list(contexts.values()))
                    self.assertTrue(call.args[1].FOX.eq(1).all())
            contexts['0']['team1']='wrong';path.write_text(json.dumps(contexts))
            with self.assertRaisesRegex(ValueError,'mismatched'):
                compute_expectations(model,df,{'A','B'},{'A','B'},path)


if __name__ == '__main__':
    unittest.main()
