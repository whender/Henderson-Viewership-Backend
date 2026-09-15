import unittest
from pregame_shows import flags,game_key,load_data
class PregameShowsTest(unittest.TestCase):
 def test_alias_order_and_exact_date(self):
  self.assertEqual(game_key('9/11/21','Ohio St.','Oregon'),game_key('2021-09-11','Oregon','Ohio State'))
  self.assertEqual(flags('2021-09-11','Ohio St.','Oregon')['BigNoonKickoffOnsite'],1)
  self.assertEqual(flags('2021-09-18','Ohio St.','Oregon')['BigNoonKickoffOnsite'],0)
 def test_cross_network_dual_show(self):
  self.assertEqual(flags('2023-09-16','Colorado','Colorado St.'),{'CollegeGameDayOnsite':1.,'BigNoonKickoffOnsite':1.})
 def test_unknown_not_zero(self):
  self.assertIsNone(flags('2026-10-10','Michigan','Indiana')['BigNoonKickoffOnsite'])
  self.assertIsNone(flags('2023-09-16','Colorado','Colorado St.',as_of='2023-09-12')['BigNoonKickoffOnsite'])
 def test_announced_cutoff(self):
  d=load_data();r=next(r for r in d['records'] if r['feature']=='BigNoonKickoffOnsite' and r['date']=='2026-09-12');r['announcement_date']='2026-05-27'
  self.assertIsNone(flags(r['date'],'Michigan','Oklahoma',d,as_of='2026-05-26')['BigNoonKickoffOnsite'])
  self.assertEqual(flags(r['date'],'Michigan','Oklahoma',d,as_of='2026-05-27')['BigNoonKickoffOnsite'],1)
 def test_no_duplicate_matchups(self):
  d=load_data();keys=[(r['feature'],game_key(r['date'],r['team1'],r['team2'])) for r in d['records']]
  self.assertEqual(len(keys),len(set(keys)))
if __name__=='__main__':unittest.main()
