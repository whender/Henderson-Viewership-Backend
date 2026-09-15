# Recent audience calibration — September 15, 2026

Decision: keep experimental. Calibration materially reduces aggregate low bias,
but the audience-band variants do not improve overall MAE, and the best global
candidate has an uncertain 0.19% overall improvement with inconsistent years.
No production artifacts or published forecasts were changed.

## Design

Frozen rolling-season predictions for the current audience-interest + FOX Friday
architecture, excluding the experimental GameDay/Big Noon and slate-position
features. Fit an additional correction on the previous two calendar years of
out-of-sample predictions. Do not fit to in-sample model residuals or 2026 actuals.
2021 has no prior stored calibration outcomes, so remains unchanged. Score
2021–2025 and separately report 2022–2025 where correction can actually apply.

Three candidates, applying only to ABC/CBS/FOX/NBC/ESPN:

- Global mean: ratio of total actual to total predicted audience.
- Band mean: the same ratio in fixed predicted-audience bands <1M, 1–3M,
  3–5M, 5–8M, and >=8M.
- Band median: median actual/predicted ratio within those bands.

All ratios shrink toward 1 by n/(n+30), with factors bounded to 0.75–1.25.
Missing history is a no-op. Boundaries depend on original prediction, never
actual audience. Bands are deliberately simple and have hard boundaries;
these experimental factors are not production-serving code.

Evaluate MAE, signed bias, aggregate underprediction, MAPE, median APE and annual
results. Big-game groups are determined from ORIGINAL forecasts, so calibration
cannot improve its reported score by moving games between groups. A past-only
selection procedure requires both lower prior MAE and lower absolute prior bias;
otherwise retain the original model. These historical seasons informed earlier
model development and are not pristine final validation.

## Results

987 major-network games; positive MAE improvement means better.

| Candidate | MAE (viewers) | MAE improvement | Aggregate underprediction |
|---|---:|---:|---:|
| Current | 740,469 | — | 7.17% |
| Global mean | 739,039 | 0.19% | 1.66% |
| Band mean | 743,264 | -0.38% | 3.03% |
| Band median | 740,718 | -0.03% | 5.87% |

For the 154 games originally forecast at >=5M, global mean improves MAE by 2.74%
(1.466M to 1.426M) and changes total audience bias from 4.97% under to 0.81% over.
Band mean improves this group's MAE by 1.81%; band median by 1.19%.

Global mean improves overall MAE in 2023 and 2025, but worsens 2022 and 2024
(2021 unchanged). Its 95% year-cluster bootstrap MAE-change interval is -20,344
to +15,634 viewers, including no benefit. Only five seasons / four adjusted
years are available. In 2024 it overshoots total actual audience by 6.06% and
raises MAE from 667K to 695K. In 2025 it reduces MAE from 780K to 745K.
Past-only variant selection worsens pooled MAE 0.23%; it does not establish a
reliable improvement from choosing calibrators based on earlier performance.

## 2026 illustrative correction

Using only 2024–2025 held-out results, global mean implies a +5.88% adjustment
for major-network games. It would move the published Michigan–Oklahoma 5.29M
point estimate to 5.60M. Band mean would give 5.45M and band median 5.49M.
These are simple correction illustrations, not regenerated full-slate forecasts,
re-estimated intervals, or an update to the site. The unconfirmed 8.2M report
was not used anywhere in fitting or evaluation. Calibration alone would not
explain a miss of that size.

## Reproduce

```
python3 research/recent_audience_calibration_experiment.py
python3 -m unittest test_recent_audience_calibration
```

Source hash, fold configurations, annual/subgroup metrics, per-game predictions,
selection decisions and the 2026 candidate factors are saved alongside this file.
Tests verify exclusion of future/old outcomes, no-history and other-network
no-ops, fixed prediction-band boundaries, shrinkage and bounds.
