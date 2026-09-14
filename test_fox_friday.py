import unittest
from fox_friday import fox_friday_night

class FoxFridayTests(unittest.TestCase):
    def test_window_and_exclusions(self):
        row=dict(date='09/11/26',network='FOX',time_slot='Friday 8:00p',conf_champ=False)
        self.assertEqual(fox_friday_night(row),1)
        for changes in [dict(network='FS1'),dict(time_slot='5:00p'),dict(conf_champ=True),dict(conf_champ='true'),dict(date='11/27/26'),dict(date='09/12/26',time_slot='8:00p')]:
            with self.subTest(changes=changes):self.assertEqual(fox_friday_night(dict(row,**changes)),0)
        self.assertEqual(fox_friday_night(dict(row,conf_champ='false')),1)
        self.assertEqual(fox_friday_night(dict(row,time_slot='6:30p')),1)
        self.assertEqual(fox_friday_night(dict(row,date='09/12/26')),1) # hypothetical Friday label with weekly date anchor

    def test_historical_definition_parity(self):
        import csv
        from pathlib import Path
        from datetime import datetime,timedelta
        with (Path(__file__).parent/'research/fox_friday_results/classification.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        for r in rows:
            hour=float(r['Time_float']);hour24=int(hour);minute=round((hour-hour24)*60)
            row=dict(date=r['Date'],network=r['Station'],time_slot=f'{hour24%12 or 12}:{minute:02d}{"p" if hour24>=12 else "a"}',conf_champ=r['Conf Champ'])
            self.assertEqual(fox_friday_night(row),float(r['FOXFridayNight']),r)

if __name__=='__main__':unittest.main()
