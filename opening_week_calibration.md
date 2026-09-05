# Opening-week audience calibration

For explicitly assigned Week 0 P4-vs-P4 (including Notre Dame) games on ESPN
or FOX, blend the final model estimate equally with the historical opening-event
mean. Other weeks, networks, non-power matchups, and historical years are exact
no-ops. Do not infer the scope from a Labor Day or August 28 cutoff.

The 2026 event mean is 4,613.4 thousand viewers from five historical games through
2025. The fixed 50% weight retains matchup-specific variation and limits the
influence of the small event sample. No 2026 actual audience is used to fit this
calibration. It is stored separately from the model artifact, bound to the
artifact SHA-256, and valid only for the next prediction year. Revalidate the
configuration whenever the model or training cutoff changes.

Validation applies the same fixed rule to the exact promoted model's saved
season-ahead predictions. Each holdout uses only prior-season opening-game
actuals. Four historical opening P4 games across 2021, 2022, 2024 and 2025 improve;
the other 1,817 games do not change. Opening-game MAE declines from 1.398M to
1.082M (22.6%). Overall MAE declines from 499.489K to 498.794K.

This is limited evidence: only four holdout games and five final training games.
It does not establish improved Week 1 performance, support other networks, or
guarantee results for a second P4 game on an opening slate. Its scope is purposely
narrow, and newly observed results should be tracked for future revalidation.

Revisions made after observing actuals must be stored in `revised_predicted`,
with revision metadata. Keep `predicted`, `actual`, and original forecast accuracy
intact. The dashboard labels original and revised estimates separately.
