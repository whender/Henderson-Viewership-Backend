# Conditional nonlinear release

The deployed candidate applies a 25% nonlinear blend only when the existing
rank-calibrated forecast is at least 1M and both pregame Elo values, a numeric
spread_home, and boolean neutral_site are provided. Otherwise it returns the
existing forecast exactly. Do not replace missing values with guessed numbers.

The conditional policy was replayed on 1,821 season-ahead holdouts (2021-2025).
MAE improves 499.489K to 481.584K; all five seasons improve. It passes the existing
3K improvement, 0.25-point MAPE guard, three-season win, 15K worst-season,
75% bootstrap probability, and 5K upper-CI gates. See the saved gate file.
The unrestricted complete-data research result is not the conditional-policy
headline. Default median imputation failed the uncertainty gates and is not used.

This remains retrospective model development on reused historical holdouts;
no independent prospective results are claimed. The all-game average downward
bias worsens, so improved MAE should not be described as fixing underprediction.

## Serving and rollback

The 91KB nonlinear_pregame.joblib is bound by SHA-256 to the unchanged primary
artifact. Rank calibration uses earlier rolling nonlinear-blend residuals.
Existing Week 0 and Monday overlays run afterward. Existing published forecasts
are not rewritten. The API accepts the four additional optional fields. The
local weekly generator now preserves CFBD neutralSite alongside Elo and spread.
Old clients without these fields keep their current predictions.

To disable this release, remove nonlinear_pregame.joblib and restart the service.
The primary artifact and its previous ensemble/calibration files remain intact.

## Validation

40 backend unit/integration tests passed. A saved 25-game input replay confirmed
all missing-context predictions and all predictions under 1M remain unchanged;
19 complete-context forecasts changed. Numeric outputs were finite and positive.

Training/validation script copies here document the release; run their originals
from the sibling RatingsAndRegression directory, which supplies their imports
and datasets. No credentials or private service-account files are included.
