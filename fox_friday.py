"""Shared definition for the ordinary FOX Friday-night broadcast slot."""
from datetime import timedelta
import re
from pregame_ensemble import _coerce_date, parse_kickoff_hour

FEATURE = 'FOXFridayNight'

def fox_friday_night(row):
    day = _coerce_date(row.get('date'))
    hour = parse_kickoff_hour(row.get('time_slot'))
    if day is None or hour is None:
        return 0.
    first = day.replace(month=11, day=1)
    thanksgiving = first + timedelta(days=(3-first.weekday()) % 7 + 21)
    championship = row.get('conf_champ', False)
    championship = str(championship).strip().lower() in {'true', '1', '1.0', 'yes'}
    friday = day.weekday() == 4 or str(row.get('day', '')).lower() == 'fri' or bool(re.search(r'\bfri(?:day)?\b', str(row.get('time_slot', '')).lower()))
    return float(str(row.get('network', '')).upper() == 'FOX'
                 and friday and hour >= 18.5
                 and day != thanksgiving + timedelta(days=1)
                 and 'black friday' not in str(row.get('time_slot', '')).lower()
                 and not championship)
