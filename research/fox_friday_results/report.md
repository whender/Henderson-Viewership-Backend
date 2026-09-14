# Separate FOX Friday-night interaction

Research only. Production models and published forecasts unchanged.

Definition: FOX, Friday, kickoff at or after 6:30 p.m., excluding Black Friday and conference championships. Keep all existing pregame features; add one binary interaction to both linear ensemble components, regular/Week 1 branches and nonlinear component. Competition features and nonlinear eligibility are fixed for an isolated comparison.

Evaluation: train on prior seasons only, 2019 initializes rank calibration, score 2021–2025 (1,821 games, 987 five-major-network games). Baseline exactly reproduces the saved audience-interest holdout. No 2026 games enter training or calibration. Target sample is 13 games (9 in 2024, 4 in 2025). Before 2025, no target training examples exist, so the interaction cannot be estimated; candidate predictions match baseline through 2024. Thus the direct learned-slot evidence consists of only four 2025 games.

| Scope | Baseline MAE | New MAE | Improvement |
|---|---:|---:|---:|
| All games |465.77K|464.98K|0.17%|
| Five major networks |741.89K|740.47K|0.19%|
| FOX |909.91K|905.43K|0.49%|
| All 13 target games |794.92K|729.32K|8.25%|
| Four 2025 target games |732.58K|519.37K|29.10%|

2025 target median percentage error:24.64% →17.89%. Three of four improve. Rutgers–Iowa worsens:2.653M →2.361M versus3.010M actual. Minnesota–Nebraska improves:4.168M →3.640M versus2.660M; USC–Northwestern:2.286M →2.028M versus1.911M; Oregon–Minnesota:3.017M →2.658M versus2.327M.

Date-cluster bootstrap 95% MAE-change interval: major networks -3.27K to+0.07K; 2025 target -460.27K to+128.90K (negative is improvement). Both include zero. A season-cluster interval for target_2025 is degenerate because only one season is present; it must not be treated as evidence of significance. The interval calculation also does not account for the earlier subgroup inspection that motivated the interaction.

2026 Week2 preview uses stored full-slate inputs and fits only through2025. Kansas–Missouri:2.997636M →2.592421M, down405K (13.52%). The baseline matches the previously audited published3.00M replay. No actual audience is used. The full-season fit includes13 target observations; primary and exact regular-component log coefficients are -0.361 and-0.173. These are conditional coefficients; the final ensemble reduction differs from either coefficient alone.

Recommendation: promising, materially better fit for this slot, but retain as experimental pending more FOX Friday coverage and unseen-game validation. Do not present29% as an overall model improvement. No deployment performed.

## Deployment follow-up

User explicitly approved deployment after reviewing the four-game holdout limitation. The production audience-interest pregame and paired postgame components were refit on all 2,423 current training games through September 7, 2026, including 14 slot examples. Pregame calibration uses the interaction’s pre-2026 rolling predictions; postgame retains its existing calibration and has not separately demonstrated an accuracy gain. The previous core model remains the fallback when audience-interest data is unavailable. Core/intrinsic models and competition eligibility are unchanged. The resulting Kansas–Missouri point forecast is 2.63M (rather than the historical-only experiment’s 2.59M). Only this game’s saved pregame/postgame forecasts are scheduled for revision; unrelated games are preserved.
