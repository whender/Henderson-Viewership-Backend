# Early-season Monday feature audit

The strongest candidate for Monday games is 75% current model plus 25% of the
prior-season Labor Day Monday audience mean. Monday MAE falls from 1.018M to
0.749M (26.4%); MAPE falls from 20.34% to 13.65%. It improves four of five games.
The Florida State–SMU preview rises from 3.862M to 4.111M (+249K, 6.5%).
The seven historical Monday audiences average 4.859M.

This is a calibration, rather than an additional regression feature. All seven
historical Monday observations already occur on Labor Day at 7 p.m. or later.
The early-Monday flag duplicates Monday exactly and cannot identify a separate
opening-week effect.

## Method

- Train only on years before each held-out season, 2021–2025; 1,821 held-out
  games, including five Monday games. Data also contain 2018 and 2019, no 2020.
- Refit both components of the production 50/50 primary/exact-network ensemble.
  Keep leakage-safe competition inputs fixed across candidates within each fold.
  Rank calibration uses only earlier rolling predictions, seeded by 2019.
- Baseline reproduces the saved strict rolling-origin shared-competition,
  rank-calibrated benchmark within numerical tolerance. The later Week 0 event
  overlay is excluded; it does not affect Monday games.
- Historical-mean/median calibrations use only earlier seasons and modify
  early Monday games only. No 2026 audience results enter fitting or selection.
- The full-history preview uses the actual served feature matrix and stored
  SMU–Florida State inputs. Its baseline matches the current serving prediction.
  Preserve the existing tiny difference between benchmark and serving log
  back-transformation formulas in their respective paths.

## Candidate comparison

Errors are in thousands of viewers; preview is in millions. Wins compare
individual held-out Monday absolute errors against baseline.

| variant | games | mae_000s | mape_pct | game_wins | prediction_millions |
| --- | --- | --- | --- | --- | --- |
| baseline | 5 | 1018.329 | 20.341 | 0 | 3.862 |
| early_monday | 5 | 1018.329 | 20.341 | 0 | 3.862 |
| holiday_mean50 | 5 | 791.183 | 17.646 | 2 | 4.929 |
| holiday_sun_mon | 5 | 1012.227 | 20.22 | 2 | 3.856 |
| monday_mean25 | 5 | 749.225 | 13.647 | 4 | 4.111 |
| monday_mean50 | 5 | 776.887 | 16.427 | 3 | 4.36 |
| monday_median50 | 5 | 755.023 | 16.102 | 3 | 4.361 |
| monday_rank_interaction | 5 | 1023.206 | 20.962 | 3 | 4.087 |
| pooled_holiday_nights | 5 | 1580.059 | 32.493 | 0 | 3.079 |

Definitions: early_monday adds Monday in September's first 14 days at 19:00+;
holiday_sun_mon adds a 19:00+ flag for Labor Day and the preceding Sunday,
retaining existing day effects; pooled_holiday_nights forces those Sunday and
Monday nights to share a day effect; monday_rank_interaction adds an early
Monday × both-ranked term (only one historical positive example). The four
remaining variants blend only Monday predictions toward historical Monday
mean/median or combined holiday-night mean at the specified weight.

## Individual Monday holdouts

All audiences and predictions below are thousands of viewers.

| year | Team 1 | Team 2 | actual_viewers_000s | baseline | monday_mean25 |
| --- | --- | --- | --- | --- | --- |
| 2021 | Mississippi | Louisville | 3078.0 | 2379.0 | 3181.2 |
| 2022 | Georgia Tech | Clemson | 4860.0 | 3837.3 | 4065.8 |
| 2023 | Clemson | Duke | 4390.0 | 4087.3 | 4260.1 |
| 2024 | Boston College | Florida St. | 4440.0 | 4526.8 | 4570.3 |
| 2025 | North Carolina | TCU | 6070.0 | 3089.5 | 3481.5 |

## Overall result and uncertainty

The best all-game regression feature is holiday_sun_mon: MAE 499.489K →
498.644K, a very small 0.845K improvement; Monday MAE improves only 6K and
the FSU–SMU estimate stays approximately 3.86M. The targeted 25% Monday mean
blend lowers overall MAE to 498.750K and leaves other games unchanged.
Neither gain reaches the existing benchmark's 3K overall practical threshold.

The paired five-game bootstrap gives the 25% blend an exploratory 90% interval
of approximately 109K–430K less Monday MAE. This interval is conditional on
these five observations and is not adjusted for candidate selection. The
25% blend narrowly beats the 50% median blend (749K vs 755K MAE); do not treat
that ordering or the chosen weight as established by a large sample. There
is no independent final test set after selecting candidates on 2021–2025.

Conclusion: modest evidence supports a forecast near 4.1M. A larger 4.36M
forecast from 50% Monday-mean blending performs worse in this backtest, and
the 4.93M Sunday/Monday-average preview wins only two of five Monday games.
Production artifacts and published predictions have not been changed.

## Reproduce

From the workspace root, using Python with pandas 3.0.1 and the project dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 RatingsAndRegression/monday_feature_audit.py
```

The script writes only monday_feature_audit_* research outputs.
