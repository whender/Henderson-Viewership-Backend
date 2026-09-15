# Onsite pregame shows: September 15, 2026 experiment

Decision: retain as an experimental feature; do not change production forecasts.
Big Noon is the better candidate, but the small pooled improvement is uncertain
and the combined 2024–2025 major-network result deteriorates.

## Method

Two separate date-and-unordered-team-pair flags, independent of the game's TV
network. Historical source revisions, team normalization and unmatched visits
are saved. Onsite neutral-game events are included; studio/non-game shows are
excluded. No scores, winners, picks or show ratings enter the new features.
College GameDay has 98 matched appearances and Big Noon has 76 in the 2,389-row
2018–2025 fitting dataset (which excludes 2020). Other archived visits mostly
belong to postseason/FCS/unrated games outside this modeling sample. No fuzzy
date matching or automatic reassignment to another game is performed.

Compare current architecture (audience interest plus FOX Friday) with GameDay,
Big Noon, and both. Four variants are fully refitted within each rolling-season
fold. All fitting years precede the held-out year. Score 2021–2025: 1,821 games,
987 on ABC/CBS/FOX/NBC/ESPN. Rank calibration uses only prior held-out seasons;
nonlinear eligibility is held fixed across variants. Baseline predictions are
asserted equal to the prior FOX Friday experiment, to 0.00001 thousand viewers.
Existing exact network × kickoff interactions remain in the challenger branch.
New flags enter both linear branches and the nonlinear component. Competition
and audience-interest inputs are unchanged across variants.

This is a retrospective appearance experiment, NOT a point-in-time weekly
publication backtest. Most historical announcement dates are unavailable.
Primary-source spot checks are saved, with conservative known-by dates when
available; they do not certify every archive row. Future unlisted games are
unknown. Preview-only unlisted flags are assumed zero, explicitly, without
publishing. Dates and matched teams, not match outcomes, define the indicators.

## Results

Positive improvement means lower mean absolute error (MAE).

| Variant | All games MAE improvement | Five major networks | FOX |
|---|---:|---:|---:|
| GameDay | 0.28% | 0.25% | 0.53% |
| Big Noon | 1.32% | 1.59% | 5.69% |
| Both | 0.66% | 0.79% | 3.76% |

Big Noon reduces five-network MAE from 740,469 to 728,693 viewers; median absolute
percentage error improves from 18.93% to 18.43%. Among 71 held-out Big Noon
appearances, MAE improves 9.68%. GameDay alone improves MAE on its 72 held-out
appearances 11.48%, but delivers little overall gain and worsens major-network
median percentage error. Refit effects on other games matter too.

Big Noon improves major-network MAE in four of five years. It worsens 2024 from
667,386 to 686,119 viewers, and combined 2024–2025 MAE worsens 0.56%. The 95%
date-cluster bootstrap interval for the major-network MAE change spans -24,622
to +870 viewers; the year-cluster interval spans -31,725 to +7,045. Neither rules
out no benefit. Only five held-out seasons are available, and these seasons have
already informed earlier model choices; they are not pristine final validation.
A past-only variant-selection procedure achieves only 0.41% major-network gain.

## Michigan–Oklahoma preview

Fit through 2025 only, no 2026 actual outcomes; identical week-2 feature inputs:

| Variant | Pregame preview |
|---|---:|
| Current architecture | 5.299M |
| GameDay | 5.268M |
| Big Noon | 5.658M |
| Both | 5.610M |

Big Noon adds 359,281 viewers (+6.78%) relative to the controlled baseline.
The game itself has Big Noon = 1 and GameDay = 0. Small changes in the GameDay
variant arise from refitting the rest of the model, not a GameDay flag here.
These are controlled research previews, NOT replacements for the published
5.29M forecast or the separately rerun production model fitted with 2026 actuals.
No website, Firestore records, or production model artifacts are changed.

## Reproduce and checks

- `PYTHONPATH=/tmp/hv-show-parsing python3 research/collect_pregame_shows.py`
  (temporary dependency location used here; requires beautifulsoup4/lxml/requests).
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 research/pregame_show_experiment.py`
  requires the existing RatingsAndRegression research inputs and frozen
  tmp/week01_retrain historical model bundle used by previous experiments.
- `python3 -m unittest test_pregame_shows`: date, alias/order, dual-show/cross-network,
  future unknowns, announcement cutoff and duplicate-key checks.
- Per-game predictions, annual metrics, uncertainty, coefficients, matching
  audit and preview are adjacent to this report.
