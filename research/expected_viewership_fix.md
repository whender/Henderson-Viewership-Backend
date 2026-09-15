# Dashboard expected audiences — September 15, 2026

The team-profile/comparison baseline previously called only the primary OLS
model. It omitted the serving ensemble, nonlinear component, opening-event
calibrations, Week 1 day model and audience-interest revision (including FOX
Friday). Legacy dashboard calendar reconstruction also changed the audited
August 29, 2026 NC State–Virginia row from Week 0 Power to Week 1 Power.

`expected_viewership.compute_expectations` now runs the shared serving pipeline
on the audited cleaned feature frame, before display-bucket reconstruction.
Schedule, pregame attention, Elo, spread, rankings, competition, and other
non-brand context stay fixed. One (team pages/comparison) or both (neutral
matchup summaries) team-indicator vectors are replaced with the equal-weight
mean across FBS teams having at least five rated appearances. The replacement
is seen by every active model component, rather than adding an adjustment
based only on the primary model's coefficients. Focal Colorado Deion terms
are removed as before. This is a conditional average-brand benchmark, not a
causal attribution or held-out forecast. Forests evaluate the neutral brand
vector; this is not an average of separately simulated replacement schedules.

The sanitized context snapshot is keyed and validated by source index, teams,
and date; refresh it when appending games using `research/export_expected_context.py`.
It exports only pregame context from the existing audited CFBD research cache
and opening-slate inputs. Elo maximum/minimum preserve the symmetric forest
features; rank bucket representatives preserve the exact rank-scope features.
Serving eligibility and missing-data fallbacks remain unchanged: date-scoped
2026 overlays are not extrapolated into older seasons, and missing pregame
Elo/lines or audience history do not trigger fabricated inputs.

Virginia–NC State actual: 3.220M. Verified old live Virginia expected: 1.3753M
(+134.1% actual vs expected). Corrected Virginia expected: 3.162575M (+1.816%);
NC State expected: 3.172776M (+1.488%); both-brand-neutral: 3.100287M.
The raw audited primary matrix also matches the weekly feature builder exactly
for this game. Published forecasts and accuracy metrics are not edited.

Validation: 67 unit tests pass, production-version pandas/scikit-learn startup
smoke check succeeds, all 2,423 games have finite expected values. Two added
regression tests check neutralization/context preservation, all three calls to
the shared serving pipeline, and rejection of mismatched source context.
