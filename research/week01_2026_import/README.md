# September 14, 2026 actuals import and refit

Imported all 34 rated games from the user's Sports Media Watch Weeks 0–1
image: five games on August 29 and 29 games on September 3–7. Audience values
are stored in **thousands of viewers**, matching the existing dataset.
The new rows total 83,268 thousand viewers (sum across broadcasts).

The canonical local source is `RatingsAndRegression/CollegeFootballViewershipWithSpreads.csv`.
Its previous bytes and physical row indices were preserved. New source indices
1044000–1044033 are appended consistently to that file, the records/context
companions, and the deployed `viewership_cleaned.csv`. There are now 2,423 rated
games. `appended_data.json` records every appended value; `games.json` records
the CFBD IDs, final scores, pregame ranks/records/Elo/spreads, and slate context.

## Source reconciliation

- Actuals: the user-provided Sports Media Watch image, previously transcribed
  in `week1_2026/record_image_actuals.py` and saved in the weekly forecasts.
- Preserve the previously recorded, more precise North Carolina–TCU audience
  of 4.908 million and San Jose State–USC audience of 1.862 million. The image
  rounds them to 4.91 and 1.86 million.
- San Jose State–USC and Memphis–UNLV are dated August 29 in the saved CFBD
  schedule, correcting the September 5 labels in the screenshot transcription.
- Final scores, kickoff dates, conference/neutral-site and pregame Elo:
  saved CFBD schedule `week2_2026/games.json`, retrieved September 12.
- Broadcasts, betting lines and AP ranks: saved Week 1 CFBD media/lines/rankings.
- Records count only completed games before each kickoff. Competition uses all
  51 supported opening-slate broadcasts, including games without known ratings;
  scores use the frozen pre-update intrinsic model and existing 90-minute window.
- Audience interest uses the same seven-day Wikipedia football-page view
  windows as serving. Each window ends before its game's weekly cutoff. All
  2,423 training games have complete attention coverage; no game-day traffic
  or actual viewership is included in these pregame features.

## Training and validation

`research/refit_inseason.py` refits the existing 203-column primary design,
exact-network challenger, intrinsic model, nonlinear component, Week 1 day
interactions, paired audience-interest components, and aligned postgame model.
The legacy postgame fallback is also refitted. Each postgame branch has all its
pregame features plus final absolute score differential.

The model structure, blend weights, pre-2026 out-of-sample rank corrections,
event priors, and intrinsic serving-scale calibration are held fixed. This is
an in-season coefficient refit, not a new model-selection experiment.
Training now includes results through September 7, 2026; prediction year remains
2026. Model dependency hashes are rebuilt as one bundle.

Validation: 63 backend tests passed, including all 34 game values under both
teams, model row counts, attention cutoffs, input parity, missing-feature
fallbacks, and paired pre/post feature contracts. All 34 Week 2 fixture forecasts
are finite and nonnegative. Median absolute forecast change is 3.66%, with a
maximum of 16.07%. These changes are not out-of-sample accuracy improvements.
Published weekly forecasts and their accuracy histories were not overwritten.

`refit_report.json` contains bundle hashes and training metadata. The separate
brand-ranking model and plots were not changed by this refit.
