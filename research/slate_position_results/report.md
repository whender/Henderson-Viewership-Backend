# Competition pecking order experiment — September 15, 2026

Decision: no promotion. The tested hierarchy interactions did not improve
held-out prediction accuracy. Production model artifacts and forecasts unchanged.

## What was tested

Use the existing no-competition intrinsic audience model to rank games within
same-date local windows of kickoffs no more than 90 minutes apart. This is a
local neighborhood per game, not a single rank covering the whole day. Training
intrinsic predictions are outcome-disjoint cross-fits; outer held-out seasons
use intrinsic models fitted only on earlier seasons. No actual audience, final
score, realized winner, or final predicted viewership enters the hierarchy.

Candidates, retaining the existing total competition feature:

- Leader: top predicted draw indicator plus total competition × leader.
- Share: own intrinsic audience / (own + competing intrinsic audience), plus
  total competition × share.
- Stronger: sum of intrinsic audiences of competitors bigger than the focal
  game. Alongside total competition, this fits different slopes for stronger
  and weaker competitors rather than discarding weaker competition.
- Control: log(1 + own intrinsic audience); also combine this control with each
  of the three candidates to distinguish hierarchy from a game-size proxy.

Tied largest draws both qualify as leaders; tied competitors are not stronger.
No-competitor games have zero competition interactions. For all-zero audience
inputs, share is zero. All features are finite; invalid dates/times or missing
intrinsic predictions fail validation.

## Validation

Current reference is the audience-interest + FOX Friday architecture; the
experimental onsite show flags are not active. Refit all linear/forest branches
within each outer season, preserving current eligibility and prior-season rank
calibration. Score 2021–2025: 1,821 games including 987 on ABC/CBS/FOX/NBC/ESPN.
2019 is the initial held-out calibration season. Every outer test season is
strictly later than the fitting years. Earliest inner training cross-fit uses
outcome-disjoint later training seasons as a fallback; it never accesses the
outer test season. This is the existing competition pipeline's policy.

Assert every training/test competition total agrees with the original feature
to 1e-8. Assert the current reference predictions reproduce the previous FOX
Friday experiment to 1e-5 thousand viewers. Hold the 90-minute window, underlying
slate membership, audience-interest inputs and eligibility fixed. Historical
slates contain available modeling rows, not every unmeasured televised game;
this limits what these results establish about a genuinely complete slate.

These seasons have been used for earlier model development; they are not a
pristine final validation set. The no-size-control ablations were added after
observing degradation from the size control. No deployment is justified by
searching variants on these same folds.

## Results

Positive values below mean *more* error (worse). Metric is mean absolute error.

| Candidate | All games error change | Five major networks | Major 2024–25 |
|---|---:|---:|---:|
| Leader interaction | +0.33% | +0.49% | +1.29% |
| Audience share interaction | +4.30% | +5.12% | +6.10% |
| Stronger vs weaker competition | +0.61% | +0.73% | +0.94% |
| Own-size control only | +6.32% | +5.14% | +2.90% |

The own-size-plus-interaction variants also worsen major-network error by
5.09–6.11%. Current major-network MAE is 740,469 viewers; leader = 744,068;
stronger competition = 745,906; share = 778,415.

Leader-only improves major-network MAE in 2021, but worsens every subsequent
season. Stronger-only improves 2024 but worsens the other four years. A selection
rule choosing only from earlier seasons' results worsens major-network MAE by
0.30%. Among the 358 held-out local leaders, the leader interaction improves
MAE just 0.15%; that small subgroup change does not justify the aggregate loss.

Regular linear-branch leader × competition coefficients are negative throughout
2021–2025 in both components; this does not support the hypothesized smaller
penalty for leaders. Those are conditional predictive associations, not causal
effects. The existing linear competition term is on log audience, so a common
coefficient represents approximately the same percentage effect, not the same
absolute viewer loss. The forest component can already model nonlinear effects.

## Saved output and reproduction

`feature_audit.csv`: all inner training and outer test intrinsic audiences,
rank, share, stronger competition and totals by fold. `predictions.csv`:
per-game results. Annual, subgroup, bootstrap, coefficient and selection tables
are adjacent. No hypothetical 2026 forecasts are published from this experiment.

Run from the backend with existing research inputs available:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 research/slate_position_experiment.py
python3 -m unittest test_slate_position
```

Tests cover boundary inclusion, different-day exclusion, ordering invariance,
ties, zero audiences, missing inputs and invariance to actual audience/scores.
