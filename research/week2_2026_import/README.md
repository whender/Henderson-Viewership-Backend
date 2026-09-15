# Week 2 actuals and in-season refit, September 15, 2026

The user's official table supplies 35 broadcast measurements: 32 individually
identified games, one partial ESPN2 Oregon–Oklahoma State broadcast, and two BTN
regional totals. `reported_broadcasts.csv` preserves all 35 entries and original
reported dates/times. `source.json` joins the individual games to CFBD IDs.
Audience values are thousands of viewers. The 32 individual broadcasts sum to
56,320K; the 30 modeled broadcasts sum to 55,720K.

All 32 individual games were appended to the canonical local spreadsheet
`RatingsAndRegression/CollegeFootballViewershipWithSpreads.csv` and context/
record companions, preserving existing physical source indices and bytes.
The separate broadcast ledger preserves regional and partial measurements,
which cannot be treated as additional full-game ratings or assigned to teams.
USA (Memphis–Boise State) and TNT (Washington State–Kansas State) are retained in
the spreadsheet but excluded from model training and fitted dashboard comparisons:
this architecture has no separate network terms for them. Encoding them as the
omitted ESPN category would be incorrect. Local full-retrain loaders also exclude
these channels. The 30 eligible games were appended to `viewership_cleaned.csv`
and their full serving contexts added to `expected_viewership_context.json`.

Source dates are September 10/11, one day before CFBD's September 11/12 schedule.
Matchups and IDs determine the joins; canonical game dates and kickoff features
use the saved CFBD schedule. Reported broadcast start times remain in the ledger.
They are not substituted for scheduled kickoff times (for example the Oregon
ESPN broadcast starts at 1:30p while the saved game kickoff is noon).

The Firestore update matched all 13 displayed predictions and 30 of 34 full-slate
predictions. Four BTN regional games retain missing individual actuals. Saved
pregame/postgame forecasts, competition and histories remain unchanged. Latest
saved forecast accuracy: MAPE 19.1845%, median APE 13.5989%, 9/13 within 20%,
10/13 within 30%, 1/13 over 50%. These stats are calculated before the refit.

The paired model bundle was refitted with 2,453 games through September 12.
This includes the previously corrected USC–San Jose State value of 2,000K,
which was not yet present in the old fitted targets. All 2,453 observations have
complete pregame attention windows. Architecture and pre-2026 held-out rank,
opening-event, intrinsic serving-scale and marquee calibrations are unchanged.
The FOX Friday feature is retained and now has 15 training observations.
All dependency hashes were rebuilt together. Refit performance on these newly
trained games is not used to replace the published weekly accuracy results.

`append_payload.json` contains the exact appended rows. `rows.json` contains
the 30 training contexts. `refit_report.json` records bundle hashes and corrections.
