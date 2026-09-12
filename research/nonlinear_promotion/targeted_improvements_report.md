# Targeted model improvements

The strongest exploratory candidate is a limited nonlinear ensemble: use the
existing model below a predicted audience of 1M; otherwise use a 75/25 blend
with a shallow gradient-boosted model. On 1,352 dashboard-like holdout games,
MAE falls from 623.45K to 590.69K (5.25%); MAPE falls from 27.27% to 26.43%;
the fraction within 25% rises from 58.14% to 61.02%. MAE improves in all five
held-out seasons, including after excluding Weeks 0–1.

This is a promising research candidate, not a promoted production model.

## What the tests imply

The current log-linear ensemble explains substantial audience variation, but
its fixed effects and hand-written interactions do not capture all useful
relationships. The nonlinear model can allow a network, team, rankings, and
matchup strength to interact differently across games. The test demonstrates
complementary predictive value; it does not identify a single causal missing
factor or establish which nonlinear interaction matters most.

Pregame Elo captures team strength beyond ranked/unranked buckets and improves
both dashboard MAE and MAPE modestly. Betting spread alone adds little. Adding
Elo, spread, a close-game flag, and neutral-site status together improves
average audience error about 2.3%, below the nonlinear blend's gain.

A nonlinear replacement alone performs worse. A modest blend works better
because the errors of the two models differ. Applying it to very small
forecasts worsens percentage errors; the 1M eligibility rule avoids that problem.

Known source-definition inconsistencies in NBC measurements remain a separate
issue. No audience labels were corrected or assumed to have a particular
streaming definition in this experiment.

## Method

- Same 1,821 outer-holdout games from 2021–2025 as the strict production-family
  benchmark. Each fit uses strictly earlier seasons; 2019 seeds calibration.
- Baseline predictions reproduce the saved 50/50 primary/exact-network ensemble
  with scoped rank calibration to numerical tolerance. Later Week 0/Monday
  overlays are excluded, consistent with prior broad-model benchmarks.
- Leakage-safe competition inputs are held fixed within each fold. All new
  context is keyed by source_index to the historical pregame context file.
- New features: pregame Elo average and difference; absolute pregame spread;
  spread at most 7 points; neutral-site status. Missing values are filled from
  training medians, with missing-value indicators. No scores or 2026 outcomes
  are used.
- Linear candidates append these features to both current ensemble components.
- Nonlinear challengers use the base feature matrix plus the five new context
  terms and missingness flags. HistGradientBoostingRegressor uses 200 trees,
  learning rate .05, depths 3 or 5, at most 15 leaves, minimum 25 rows per leaf,
  L2 penalty 10, seed 11, and no early stopping. Models fit log1p audiences and
  use training-residual smearing to return to audience units.
- Fixed blends use 25% nonlinear and 75% existing raw ensemble. Rank calibration
  for each candidate uses its own earlier rolling predictions.
- The 1M eligibility rule compares the calibrated baseline prediction with 1000
  in thousands-of-viewers units. Below that point it preserves baseline exactly;
  above it uses the calibrated blend. Actual audience never determines eligibility.

## Dashboard-like results

At least one power-team-list member, excluding CW/ESPNU; MAE and bias are
thousands of viewers. Negative bias means underprediction.

| variant | n | mae_000s | mape_pct | within25_pct | bias_000s |
| --- | --- | --- | --- | --- | --- |
| baseline | 1352 | 623.45 | 27.268 | 58.136 | -126.63 |
| elo_strength | 1352 | 613.252 | 26.753 | 58.358 | -129.835 |
| lean_pregame | 1352 | 609.336 | 26.774 | 58.284 | -130.312 |
| market_competitiveness | 1352 | 620.84 | 27.185 | 58.136 | -125.551 |
| nonlinear_blend25_predicted_1m_plus | 1352 | 590.69 | 26.431 | 61.021 | -169.874 |
| nonlinear_depth5 | 1352 | 739.659 | 37.478 | 48.077 | -250.918 |
| nonlinear_depth5_blend25 | 1352 | 591.957 | 27.45 | 60.207 | -163.881 |

## Consistency by season

| year | variant | mae_000s | mape_pct |
| --- | --- | --- | --- |
| 2021 | baseline | 610.042 | 30.246 |
| 2022 | baseline | 549.112 | 25.29 |
| 2023 | baseline | 650.888 | 27.589 |
| 2024 | baseline | 556.389 | 27.698 |
| 2025 | baseline | 776.293 | 25.318 |
| 2021 | nonlinear_blend25_predicted_1m_plus | 592.162 | 30.153 |
| 2022 | nonlinear_blend25_predicted_1m_plus | 521.608 | 24.507 |
| 2023 | nonlinear_blend25_predicted_1m_plus | 642.491 | 26.954 |
| 2024 | nonlinear_blend25_predicted_1m_plus | 515.525 | 26.614 |
| 2025 | nonlinear_blend25_predicted_1m_plus | 702.574 | 23.637 |

Across all 1,821 games, the eligible blend improves MAE from 499.49K to 474.09K
and MAPE from 37.16% to 36.54%. On dashboard-like games after Week 1, MAE falls
from 613.80K to 579.80K and MAPE from 26.84% to 25.91%.

## Important limitations

The 1M rule was introduced after the initial experiment revealed worse
percentage errors from applying the nonlinear blend universally. It was a
single follow-up threshold, not a threshold sweep, but this is still adaptive
model development on reused historical holdouts. It must not be represented
as an untouched or independently confirmed test.

The exploratory five-season block bootstrap estimates dashboard-like MAE
improvement at 18.10K–49.80K (90% interval). This interval is not adjusted for
candidate selection and has only five season blocks.

The candidate worsens mean signed bias from -126.63K to -169.87K on dashboard-
like games. Thus it improves per-game accuracy without resolving the specific
2026 underprediction concern. It also requires serving the extra pregame inputs
consistently and revalidating its calibration and missing-input behavior.

## Recommendation

Advance this fixed candidate to the existing full promotion checks and an
untouched prospective comparison; retain current production until then. Keep
the 25% weight and 1M threshold fixed during that validation. Audit remaining
NBC audience definitions independently. No forecast, production artifact,
Firestore record, or repository push was changed by this experiment.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 RatingsAndRegression/targeted_improvements.py
```

Outputs: targeted_improvements_predictions.csv, summary.csv, by_year.csv,
and uncertainty.csv (each filename uses the targeted_improvements_ prefix).
