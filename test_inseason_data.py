"""Protect the imported actuals, units, and paired in-season model bundle."""
import json,unittest
from pathlib import Path
import joblib
import pandas as pd

B=Path(__file__).parent

class InseasonDataTests(unittest.TestCase):
    def test_actuals_are_unique_and_available_for_both_teams(self):
        data=pd.read_csv(B/'viewership_cleaned.csv');new=data[data.Year.eq(2026)]
        self.assertEqual(len(data),2423);self.assertEqual(len(new),34)
        self.assertTrue(data.source_index.is_unique)
        self.assertFalse(new.duplicated(['Date','Team 1','Team 2']).any())
        # USC–SJSU actual corrected from 1,862K to 2,000K by the user.
        self.assertAlmostEqual(new['Persons 2+'].sum(),83406)
        self.assertEqual(new.loc[new.source_index.eq(1044001), 'Persons 2+'].iloc[0],2000)
        from main import team_profile
        for _,game in new.iterrows():
            for team,other in [('Team 1','Team 2'),('Team 2','Team 1')]:
                found=[g for g in team_profile(game[team])['games']
                       if g['year']==2026 and g['date']==game.Date and g['opponent']==game[other]]
                self.assertEqual(len(found),1)
                self.assertEqual(found[0]['viewers'],game['Persons 2+'])

    def test_paired_bundle_includes_all_new_observations(self):
        from model_loader import load_viewership_model
        from aligned_postgame import load_aligned_postgame
        model=load_viewership_model();post=load_aligned_postgame(B)
        self.assertEqual(model.nobs,2423)
        self.assertEqual(model.audience_interest['training_rows'],2423)
        self.assertEqual(model.audience_interest['prediction_year'],2026)
        self.assertEqual(post['training_through_date'],'2026-09-07')
        for pre,po in zip([model.params.index,model.pregame_ensemble['challenger_feature_columns']],post['components']):
            self.assertEqual(set(po['columns']),set(pre)|{'Score Diff'})

if __name__=='__main__':unittest.main()
