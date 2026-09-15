# Marquee calibration check — September 15, 2026

Targeting the correction is more promising than blanket calibration. This is
research only: production artifacts and forecasts were not changed.

Primary definition: original forecast >=5M on ABC/CBS/FOX/NBC/ESPN. This captures
154 held-out games in 2021–2025 without selecting by realized viewership. Fit a
ratio of summed actual/predicted audiences among eligible prior OOS forecasts
from the previous two calendar years. Shrink toward 1 by n/(n+30); cap factors
at 0.75–1.25. 2021 has no earlier stored calibration data and is unchanged.
Other games receive no correction. All input forecasts come from the current
architecture's frozen rolling-season FOX Friday experiment.

Sensitivity definitions: both ranked; and a hybrid of >=5M forecast OR both
ranked with >=3M forecast. Additional controls are median rather than mean ratio,
and learning the ratio from all major games but applying it only to >=5M games.
Rank indicators are joined by unique source_index from the historical frame.
No 2026 reports or actuals are used to fit/select these corrections.

## Results

| Candidate | MAE improvement, >=5M games | Overall major MAE improvement | >=5M aggregate underprediction |
|---|---:|---:|---:|
| Current | — | — | 4.97% |
| Marquee mean (primary) | 2.24% | 0.69% | 2.95% |
| Marquee median | 1.98% | 0.61% | 3.63% |
| Broad ratio applied to marquee only | 2.74% | 0.84% | -0.81% |
| Both-ranked only | 2.53% | 0.90% | 2.29% |
| Hybrid mean | 2.98% | 1.05% | 1.57% |

Primary marquee MAE falls from 1.466M to 1.433M. Hybrid reduces it to 1.422M.
Both primary and hybrid improve marquee MAE in all four adjusted years,
2022–2025; 2021 is a no-op. On 2024–2025 alone, marquee MAE improvements are
3.84% primary and 4.89% hybrid. Primary leaves every sub-5M game unchanged.
Hybrid overall major MAE falls from 740,469 to 732,665 viewers.

Primary 95% year-cluster MAE-change interval among marquee games is -55,325 to
-8,468 viewers, but date-cluster interval is -65,409 to +382. Hybrid date-cluster
interval is -94,104 to +7,123. Thus consistency across a few seasons does not
establish robust significance across game dates. We have repeatedly used these
seasons for model development; results are exploratory, not pristine validation.
Prior-only variant selection (requiring prior MAE and absolute bias improvement)
improves major MAE 0.84%, but variant definitions themselves were explored on
this historical data. Full annual/subgroup results and selection are saved.

## 2026 illustrations, not published forecasts

Fitting on 2024–2025 yields +5.05% for primary (78 eligible games), or +6.09% for
hybrid (101 eligible games). Applied to the existing published point estimates:

| Game | Published | Primary correction | Hybrid correction |
|---|---:|---:|---:|
| Michigan–Oklahoma | 5.29M | 5.56M | 5.61M |
| Ohio State–Texas | 11.97M | 12.57M | 12.70M |
| Alabama–Kentucky | 5.64M | 5.92M | 5.98M |

These corrections address measurable low bias but are too small to explain the
reported 2026 misses. No full-slate regeneration, interval recalibration or
production deployment is part of this check. The >=5M threshold is a hard gate;
prospective validation and boundary behavior warrant review before deployment.

Run `python3 research/marquee_calibration_experiment.py` and
`python3 -m unittest test_marquee_calibration`. Tests enforce prior-only fitting,
prediction-based selection, unchanged non-target games and empty-history no-op.
