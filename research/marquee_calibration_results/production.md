# Approved production promotion

User explicitly approved hybrid_mean and in-place refresh of weeks 2 and 3.
Production factor 1.060880650203136, fitted on 101 2024–2025 held-out eligible
forecasts, is stored in marquee_calibration.json and bound to the current model
artifact hashes. Eligible 2026 games: ABC/CBS/FOX/NBC/ESPN with uncalibrated
pregame >=5M or both ranked and uncalibrated pregame >=3M.

Apply once at the end of pregame inference. Saved/previous forecast text never
enters eligibility. Postgame inference applies the identical multiplier using
uncalibrated pregame eligibility, retaining pre/post feature consistency;
postgame accuracy was not separately validated for this calibration. Prediction
intervals retain the existing primary-interval translation method and are not
newly coverage-calibrated. Intrinsic competition inference remains unchanged.

The publisher regenerates all 34 week-2 and 35 week-3 full-slate forecasts from
saved features, preserving 13 and 18 displayed games respectively. Competition,
Deion override, ranks, scores, actuals and histories are preserved. No additional
prediction history or revised_predicted field is added. Existing revised_predicted
fields are removed to ensure a single canonical current forecast. Whole-document
backups are kept privately under tmp/marquee_production before compare-and-set
publication of both documents in one transaction.

The latest model differs from the older saved week-2 forecasts independently of
this calibration. Regeneration therefore also refreshes non-marquee week-2
forecasts. Week-3 non-marquee predictions are unchanged. Forecast timing labels
remain truthful: existing week-2 retrospective labels are retained; no separate
revision label or column is added.

90 backend tests pass, including threshold/year/network scope, model-hash mismatch
rejection, paired pre/post multiplication and protection against compounding a
saved calibrated prediction. Per-game before/after and multiplier data are in
published_preview.json. Scripts and settings are committed before publication;
live /model-status and /weekly-predictions are checked after deployment.
