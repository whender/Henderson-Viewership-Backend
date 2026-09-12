# Week 1 major-network day interactions — September 12, 2026

## Requested specification

Five indicators: Week 1 × major network × weekday (Tuesday–Thursday), Friday, Saturday, Sunday, and Monday. Major means exactly NBC, CBS, ABC, FOX and ESPN; ESPN2 is excluded. Week 1 uses the second FBS schedule block, following the initial Week 0 block and ending on Monday. No new Week 0 features, corrected power flags, team decay or additional opening main effects are added in this experiment.

The five indicators cover all requested day categories; there is no extra major-Week-1 intercept. Redundant columns are removed using training data only. The Monday interaction is exactly the existing Monday indicator in this historical sample, so it cannot supply a separate coefficient. Monday predictions can still move slightly when the remaining model parameters and calibration are refitted.

## Historical result

For 74 Week 1 games on the five major networks in the 2021–2025 season-ahead backtest, the unpenalized specification reduces average absolute error from 893K to 840K (5.94%). Average percentage error falls from 26.71% to 25.72%; within-30% improves from 67.57% to 71.62%; >50% misses fall from 13 to 10. Within-20% declines from 48.65% to 47.30%. MAE improves in all five seasons. The exploratory 90% season-block interval for the MAE change is approximately −84K to −23K; it does not account for the historical candidate-selection process.

A sensitivity variant penalizing the new coefficients with lambda=10 gives a smaller 893K to 886K improvement and wins three seasons. Both variants are supplied in the preview CSV. The main preview uses the stronger historical candidate, selected before considering any 2026 actuals.

| variant | n | mae_000s | mape_pct | within20_pct | within30_pct | over50_pct | year_wins | season_ci05_000s | season_ci95_000s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 74 | 893.2 | 26.7 | 48.6 | 67.6 | 17.6 | 0 | 0.0 | 0.0 |
| week1_major_days | 74 | 840.2 | 25.7 | 47.3 | 71.6 | 13.5 | 5 | -84.1 | -23.2 |
| week1_major_days_shrink10 | 74 | 886.4 | 26.6 | 45.9 | 68.9 | 14.9 | 3 | -16.1 | 2.9 |

| variant | year | n | mae_000s | mape_pct | over50_pct |
| --- | --- | --- | --- | --- | --- |
| current | 2021 | 17 | 850.8 | 32.3 | 23.5 |
| current | 2022 | 14 | 1,040.6 | 33.0 | 28.6 |
| current | 2023 | 10 | 990.5 | 25.8 | 20.0 |
| current | 2024 | 17 | 519.5 | 17.2 | 0.0 |
| current | 2025 | 16 | 1,145.5 | 25.9 | 18.8 |
| week1_major_days | 2021 | 17 | 731.0 | 28.8 | 17.6 |
| week1_major_days | 2022 | 14 | 1,027.9 | 32.9 | 14.3 |
| week1_major_days | 2023 | 10 | 967.5 | 26.7 | 20.0 |
| week1_major_days | 2024 | 17 | 488.8 | 16.8 | 0.0 |
| week1_major_days | 2025 | 16 | 1,085.7 | 25.1 | 18.8 |

## 2026 Week 1 forecast preview

All audiences are millions. Published forecasts are the saved rounded values; percentage changes use the exact current-model replay, which agrees within rounding. The existing 2026 Monday pooling adjustment is retained for both replay and candidate. These are point forecasts; old uncertainty intervals are not reused for the candidate.

| date | day | network | matchup | Published M | Candidate M | Change % |
| --- | --- | --- | --- | --- | --- | --- |
| 09/03/26 | Thu | ESPN | Colorado vs Georgia Tech | 1.69 | 1.96 | 16.0 |
| 09/04/26 | Fri | ESPN | #7 Miami vs Stanford | 1.93 | 1.67 | -13.7 |
| 09/04/26 | Fri | FOX | Fresno St. vs #14 USC | 2.39 | 2.18 | -8.8 |
| 09/05/26 | Sat | ABC | East Carolina vs #13 Alabama | 3.61 | 3.89 | 7.7 |
| 09/05/26 | Sat | FOX | North Texas vs #6 Indiana | 1.18 | 1.28 | 8.1 |
| 09/05/26 | Sat | ESPN | Oregon St. vs #23 Houston | 1.17 | 1.25 | 6.8 |
| 09/05/26 | Sat | ABC | Baylor vs Auburn | 2.93 | 2.84 | -3.2 |
| 09/05/26 | Sat | CBS | Boise St. vs #2 Oregon | 3.19 | 3.40 | 6.6 |
| 09/05/26 | Sat | FOX | Boston College vs Cincinnati | 1.38 | 1.32 | -4.3 |
| 09/05/26 | Sat | ESPN | Texas St. vs #5 Texas | 1.31 | 1.42 | 8.3 |
| 09/05/26 | Sat | ESPN | Missouri State vs #8 Texas A&M | 1.31 | 1.42 | 8.4 |
| 09/05/26 | Sat | ABC | Clemson vs #11 LSU | 6.44 | 6.15 | -4.4 |
| 09/05/26 | Sat | NBC | Western Michigan vs #16 Michigan | 2.55 | 2.74 | 7.1 |
| 09/05/26 | Sat | ESPN | UCLA vs California | 1.25 | 1.19 | -4.6 |
| 09/06/26 | Sun | NBC | Washington St. vs #17 Washington | 3.77 | 4.93 | 30.8 |
| 09/06/26 | Sun | ABC | #24 Louisville vs #9 Mississippi | 4.06 | 4.61 | 13.5 |
| 09/06/26 | Sun | NBC | Wisconsin vs #4 Notre Dame | 5.80 | 6.77 | 16.7 |
| 09/07/26 | Mon | ESPN | #19 SMU vs Florida St. | 4.11 | 4.06 | -1.1 |

These are model refits, not flat day multipliers. Adding an interaction changes the other fitted coefficients and candidate-specific rank calibration too, so games on the same day can move by different amounts or even in opposite directions. The full-model Sunday interaction coefficients should not be interpreted as stand-alone audience multipliers: the existing Sunday coefficient shifts simultaneously.

## Method and limitations

Historical training uses strictly earlier seasons. The 2019 fold seeds rank calibration for the 2021–2025 holdouts. The interaction is added to both primary and exact-network log regressions; competition inputs and the nonlinear component remain fixed to isolate the change. Candidate-specific rank corrections use earlier rolling predictions only. Predictions outside Week 1 on the five major networks are explicitly held at the current baseline; this restriction is part of the tested proposal. Assertions verify every held-out game, current-model reconstruction and unchanged out-of-scope predictions.

The 2026 preview refits through 2025 on the saved model's original design matrix, reuses the saved 2026 pregame competition inputs and reconstructs both existing component predictions before adding features. Full-history candidate rank corrections come only from 2019/2021–2025 rolling predictions. Existing nonlinear and opening/Monday serving adjustments are retained in the preview. The retained weekly rows lack neutral-site context, so their nonlinear branch remains bypassed, as in current serving. No 2026 actual audience is used in fitting, calibration, selection or preview generation.

The historical benchmark excludes the later serving-only opening/Monday overlays, matching earlier analyses; the 2026 preview includes them. Therefore the historical 5.94% improvement is not a measured incremental gain on top of those deployed overlays. The Sunday interaction has only nine historical major-network Week 1 games, and there is just one Sunday outside opening week in the full historical sample. Monday adds no independent information. These support limitations still require care before deployment, notwithstanding the consistent historical directional improvement.

Published predictions and their actuals were not changed. No production artifact, Firestore record or deployment was modified.

## Reproduce

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 RatingsAndRegression/week1_major_day_interactions.py`. Outputs use the `week1_major_day_` prefix: predictions.csv, summary.csv, by_year.csv, by_day.csv, history.csv, feature_audit.csv, full_fit_coefficients.csv, 2026_preview.csv and this report.
