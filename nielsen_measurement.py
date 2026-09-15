"""Date-based measurement-era indicators, separate from fitted adjustments.

These describe the measurement calendar, not a verified rating's publication
basis. Preliminary panel-only and broadcaster streaming totals still require
source-specific auditing. Do not add these on top of OldNielsenSystem blindly.
"""
from datetime import date
from pregame_ensemble import _coerce_date

REGIMES = {
    'NielsenOOHFullCoverageEra': {
        'effective_date': '2025-01-27',
        'description': 'Out-of-home contiguous US coverage expanded from 66% to 100%.',
        'source': 'https://www.nielsen.com/news-center/2025/nielsen-out-of-home-measurement-now-covers-100-of-the-united-states/',
    },
    'NielsenBigDataPressPolicyEra': {
        'effective_date': '2025-09-01',
        'description': 'Big Data + Panel press-claims policy, with preliminary-data exceptions.',
        'source': 'https://www.nielsen.com/news-center/2025/nielsen-begins-updated-era-of-tv-ratings-with-big-data-panel-for-this-falls-tv-season/',
    },
    'Nielsen2026RevisionEra': {
        'effective_date': '2026-08-31',
        'description': 'Co-viewing, weighting, demographic assignment and universe-estimate revisions.',
        'source': 'https://www.nielsen.com/news-center/2026/nielsen-incorporates-new-enhancements-to-improve-its-data-measurement-leading-into-the-new-fall-tv-season/',
    },
}


def measurement_era_flags(value):
    day = _coerce_date(value)
    return {name: None if day is None else int(day >= date.fromisoformat(regime['effective_date']))
            for name, regime in REGIMES.items()}


def measurement_status():
    return {
        'regimes': REGIMES,
        'flags_are': 'Calendar indicators; not confirmation of an individual reported audience basis.',
        'fitted_adjustment': 'OldNielsenSystem (pre-2025 versus 2025 onward)',
        'additional_adjustment_enabled': False,
        'reason': 'OOH flag duplicates the fitted era split in current data. The August 2026 change lacks an independent post-change validation period.',
    }
