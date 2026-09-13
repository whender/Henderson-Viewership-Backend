# Audience-interest deployment, September 13, 2026

## Decision

Promote the **level-only weekly-cutoff** Wikipedia attention model. Train through 2025; no 2026 audiences or target-game results enter pregame fitting or features. Retain existing intrinsic competition scores, nonlinear eligibility, Week 1 day interactions, and event overlays. No momentum, CatBoost, lead-in, or overlap experiment is bundled into this promotion.

| Season-ahead evaluation, 2021–2025 | Previous model | Weekly attention | Improvement |
|---|---:|---:|---:|
| Five major networks: MAE, 987 games | 765,467 viewers | 741,889 viewers | 3.08% |
| Five major networks: MAPE | 24.01% | 23.09% | 0.92 percentage points |
| All networks: MAE, 1,821 games | 479,430 viewers | 465,770 viewers | 2.85% |
| Week 1 major networks: MAE, 74 games | 840,187 viewers | 817,907 viewers | 2.65% |
| Aligned postgame, five majors: MAE | 707,159 viewers | 676,901 viewers | 4.28% |

The pregame candidate improves major-network and overall MAE in all five held-out seasons. A chooser using only earlier seasons chooses level-only in every fold (2019 seeds the 2021 choice). Models and rank calibrations are refit using earlier seasons only. This is repeated historical evaluation, not a new untouched prospective test; five seasons provide limited uncertainty estimates. The previously reported 3.29% used a later game-specific cutoff; **3.08% is the correct deployment benchmark**.

## Feature and data contract

For a Tuesday–Monday slate, the exclusive cutoff is Monday 00:00 UTC before the slate, capped at two calendar days before each game's local date. Use exactly the seven UTC dates immediately before this cutoff (normally Monday–Sunday of the prior week). Allow at least one further complete day before serving the observations, making them available by Tuesday. Do not substitute a shorter window or impute missing days.

The two symmetric features are `log1p(sum of both teams' seven-day counts)` and `log1p(larger team's seven-day count)`. Counts are public English Wikipedia football-program pageviews, all-access, user category. They measure attention rather than U.S. unique viewers. Article mapping is fixed and audited; all 2,389 historical game windows are complete.

Both linear components and the existing gradient-boosted component are refit with the two features. Postgame is a separate paired refit of every pregame component with absolute final score difference added. Missing interest history or a publication time before the cutoff uses the previous model and emits a warning; no network requests occur inside prediction requests. Future slates must refresh the cache before publishing.

Artifact hashes bind this paired model to the existing primary, nonlinear, Week 1, and aligned-postgame artifacts. Serving validates these bindings on load. Numerical checks compare the shared serving extractor to all historical feature windows. Source cache entries have article URLs, retrieval timestamps, and per-game cutoff audits.

## Reproduction and publication

Use the pinned backend requirements (Python 3.13, pandas 3.0.1, scikit-learn 1.8.0). Research prerequisites are the existing sibling `RatingsAndRegression` historical data, feature-building modules, archived Wikimedia histories, and previous rolling-baseline outputs.

1. Run `research/audience_interest_weekly_experiment.py` to reproduce candidate comparisons and save raw fold histories.
2. Run `research/train_audience_interest.py` to compare postgame accuracy and create `audience_interest.joblib`.
3. Run `research/refresh_audience_interest.py --slate ../week2_2026/predictions.json --cache-dir ../week2_2026/audience_interest/cache` to validate the full slate and refresh `audience_interest_data.json`. Public API failures or incomplete histories stop before changing the cache. For later weeks supply the new slate and cache.
4. Run backend unit tests, then deploy the explicitly reviewed files. Verify `/model-status` and prediction parity against the deployed service before writing the weekly document.
5. Preserve original pregame forecasts and accuracy fields. For Week 2, publish new forecasts under `revised_predicted`, marked retrospective when kickoff has passed; retain all 34 hidden supported-network games and only the 13 major-network display games. Recalculate existing scored games with the aligned postgame model, preserving earlier estimates in audit fields.

Existing interval widths remain the legacy primary model's shifted intervals; this experiment validates point accuracy, not a new interval-coverage claim. Previously published Week 1 values are not regenerated.

Sources: [Wikimedia pageview API](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html). Detailed fold results and the Week 2 feature fixture are stored alongside this report.
