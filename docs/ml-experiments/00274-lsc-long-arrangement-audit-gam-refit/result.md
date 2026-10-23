---
id: '00274'
slug: lsc-long-arrangement-audit-gam-refit
title: >-
  How long can an arrangement last before a no-lag forecast learns it as normal?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: done
size: >-
  medium. Stored: private bias tables under
  data/ml-experiments/00274-lsc-long-arrangement-audit-gam-refit/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible for how a transfer is
  injected and for which training window the fit uses, so both are fixed below before any score is
  read. Callers: this folder's scripts, which call the reviewed `studies.dp_gam` fit and predict
  functions unchanged. Step 8 of the LSC decision program (private brief
  `briefs/sw_lsc_decision_program_brief.md`).
hypothesis: >-
  A forecast with no lagged power learns a long-lasting change of running arrangement as the
  normal pattern, because the monthly refit trains on whatever the meter recorded. The learned share
  grows with the duration of the arrangement, and the duration beyond which the forecast of the
  normal running arrangement (NRA) is wrong by more than a stated tolerance sets the limit that can
  be told to NGED.
prediction: >-
  Written before any score is read. A 30-day transfer of 25% of the receiver's load, injected into
  the training data of the lagless convex additive model (GAM_ONLY) and refitted monthly, biases
  the forecast of the true NRA by under 0.1 points of effective capacity in the 48 hours after the
  transfer ends (headline cell: mean over substations and onset dates, 95% interval upper bound
  below 0.1; the mean transfer is about 10 to 15 points). This is expected to fail: the 24-month
  window gives 30 days about 4% of the weight, so a bias of 0.5 to 1 points (about 5% of the
  transfer) is expected right after the end, and more where a month or holiday term can absorb it.
  Further predictions: the bias in the 48 hours after the
  end grows roughly in proportion to the duration up to 90 days; a permanent step is learned to
  half of its size within 12 months with the 24-month window and within 6 months with the 12-month
  window; and the rows before the onset have a bias of exactly zero (positive control).
decision_rule: >-
  Pre-registered before any score is read. Bias is the signed mean of (forecast trained on the
  injected series minus forecast trained on the clean series) over the rows of a group, multiplied
  by the sign of the transfer, so that a positive bias means the transfer was learned as normal.
  The two forecasts come from the same cut, weather, and settings, so the bias is exactly what the
  injection changed, and the bias against the true NRA series equals it. A second measure is the
  signed error of the injected-trained forecast against the true NRA series, beside the clean-
  trained forecast's own signed error, because the clean forecast is not exact and the second
  measure is what NGED would see. The tolerance for "learned as normal" is 0.1 points of effective
  capacity (the brief's figure) and, because the bias scales with the transfer, 5% of the mean
  transfer; 0.5 points is reported beside both. The limit for NGED is the longest tested duration
  whose bias in the 48 hours after the end has a 95% interval upper bound under the tolerance, per
  size and window, with the duration interpolated linearly in log duration between tested values.
  The threshold is a statement to NGED, so no porting cost applies. Only the 30-day, 25% headline
  cell sets the verdict; the other predictions are reported as findings. Verdict: `supported` when
  the 30-day, 25% bias after the end is under 0.1 with the interval upper bound under 0.1; `not
  supported` when the lower bound is above 0.1; `partly supported` otherwise.
benchmark: >-
  The experiment's own zero-size control and the clean-trained forecast of each scenario. Reference
  for the setting: the GAM_ONLY arm of experiments 208 and 258 (a fixed fit, no refit).
row_set_id: >-
  Own rows: monthly forecast periods at 10 Primary substations from onset dates in December 2022,
  June 2023, December 2023, and June 2024; not comparable with the leaderboard
metric: >-
  Signed mean of (forecast trained on injected series minus forecast trained on clean series), in
  points of effective capacity, by phase.
code: ml-experiments/00274-lsc-long-arrangement-audit-gam-refit/
verdict: not supported
best_arm: >-
  Not applicable (one forecast, GAM_ONLY, tested under many injections). Headline cell: a 30-day
  transfer of 25% of load leaves the forecast 2.49 points of effective capacity too high (95%
  interval 2.04 to 3.04) in the 48 hours after the transfer ends, 19% of the transfer, against a
  tolerance of 0.1 points.
headline: >-
  A monthly-refitted forecast with no lagged power learns 8% of a 7-day transfer, 19% of a 30-day
  transfer, and 32% of a 90-day transfer as normal in the 48 hours after the transfer ends (24-month
  window, transfer of 25% of load). A permanent step is half learned after about 12 months.
lessons: >-
  No tested duration, down to 7 days, keeps the bias in the 48 hours after the end under 0.1 points,
  so the limit for NGED is shorter than 7 days. The bias is the same for a transfer that adds load
  and one that removes it, and for a share of load and a fixed block of MW. A shorter training window
  learns faster and the whole history learns slower.
take_home_message: >-
  A forecast that uses no recent power still learns a load transfer as normal, and it does so
  quickly: a transfer lasting only 7 days leaves the forecast 1.2 points too high for the next 2
  days, and one lasting 30 days leaves it 2.5 points too high (about a fifth of the transfer). The
  limit on how long an arrangement can last before it counts as normal is therefore shorter than 7
  days, so NGED should not rely on this forecast as the normal-arrangement forecast for any
  arrangement longer than a few days unless the transfer's rows are masked out of training. The next
  experiment should test 1 to 6 days and a training mask.
advice:
  - >-
    Mask the rows of every known load transfer out of the training window of the GAM, whatever its
    length, because a 7-day transfer is already learned as 8% of its size (Results, figure 1).
  - >-
    Do not tell NGED that arrangements shorter than a week are safe: the 48-hour bias is above the
    0.1-point tolerance at 7 days, and the tolerance is not reached in the tested range (Results).
  - >-
    Use a 24-month or longer training window when load transfers cannot be masked, because a
    12-month window learns a 30-day transfer at 32% against 19% (24 months) and 15% (whole history)
    (figure 2).
  - >-
    Score a no-lag forecast's bias after a transfer from refits cut at the end date and at later
    months, not only at the first of a month, because the 48-hour figure changed from 2.1 to 0.9
    points when one fit was cut 12 hours later (Checks).
production_status: not-for-production
adopted_in: none
---

**A forecast that uses no recent power still learns a load transfer as normal, and the limit on how
long an arrangement can last before it counts as normal is shorter than the shortest transfer we
tested, 7 days.** We took 10 Primary substations, added a transfer of 25% of each substation's own
power to its recorded power for 7 to 90 days (or for good), refitted the lagless convex additive
forecast (GAM_ONLY) at the start of every month, and compared each forecast with the forecast
trained on the unaltered series, which is the normal running arrangement (NRA). Verdict:
`not supported`.

The prediction that a 30-day transfer of 25% of load biases the forecast by under 0.1 points did
not hold: the bias in the 48 hours after the end is 2.49 points (95% interval 2.04 to 3.04), 19% of
the transfer. The experiment narrows the search to the shortest durations (1 to 6 days) and to a
training mask, which are the next tests. Everything here is a lab condition: the weather is actual
ERA5 and CAMS, not ensemble weather, and the result does not apply to a forecast with lagged power.
The method of the injection is the share-of-load shape of [experiment
141](../00141-switching-detector-fed-into-gam-k/result.md), and the forecast is the GAM-K recipe of
[experiment 252](../00252-gam-stack-at-eight-more-primaries/result.md).

![After the transfer ends](after_end_bias.svg)

![Learned share by training window](learned_share_by_window.svg)

![Learned share by phase](learned_share_by_phase.svg)

![Permanent step](permanent_step.svg)

## Results

All figures are in points of effective capacity (1 point is 1% of the substation's capacity), means
over 10 substations and 4 onset dates, with 95% intervals from resampling substations and onset
dates independently. A positive bias means the transfer was learned as normal. The mean transfer
was 13.3 points for the 25% size. The numbers are printed by `e00274_report.py`
(`printed_report.txt`).

- **Controls passed.** The bias in the 3,720 groups of BEFORE rows and in the 1,920 groups of
  size-0 rows is exactly 0.
- **Headline (24-month window, 25% share, receiver):** bias in the 48 hours after the end is 1.21
  (0.73 to 1.75) at 7 days, 2.67 (1.14 to 5.54) at 14, 2.49 (2.04 to 3.04) at 30, 3.94 (3.29 to
  4.68) at 60, and 4.20 (3.33 to 5.29) at 90 days. As a share of the transfer: 8.5%, 18.0%, 19.1%,
  30.4%, and 31.9%. The bias grows less than in proportion to the duration (the prediction of
  proportional growth to 90 days did not hold).
- **The bias scales with the transfer:** the 30-day share is 19% at 10%, 25%, and 40% of load, and
  a transfer of 10% of load gives 1.0 points (0.82 to 1.21) at 30 days.
- **Training window (30 days, 25%):** 12-month window 32% (25% to 38%), 24-month window 19% (16% to
  22%), whole history 15% (11% to 20%).
- **Decay after the end (30 days, 25%):** 2.8% of the transfer in months 1 to 3 (bias 0.37, 0.31
  to 0.44) and 5.0% in months 4 to 12 (0.66, 0.48 to 0.86). For 90 days the shares are 5.3% and
  15.2%: the bias comes back in the same season of the following year.
- **During the transfer:** learned share 6.2% at 60 days and 11.4% at 90 days. The during rows are
  lagged by up to a month, because a fit at a cut has seen only the earlier part of the transfer,
  and for transfers of 30 days or shorter every during row precedes the second refit, so the
  during bias is 0 by construction.
- **Permanent step (25%):** the share learned is about 0.2 from month 2 to month 10, 0.48 at month
  12 (24-month window) and 0.61 at month 13, and 1.0 from month 24. The 12-month window reaches
  0.55 at month 11 and 0.97 at month 12. The whole history reaches 0.54 at month 24 and 0.7 at
  month 36. The predictions were half learned within 12 months (24-month window; reached at month
  13, so borderline) and within 6 months (12-month window; not met, reached at month 11).
- **Sign and shape (30 days, 25%, 24-month window):** a donor (load removed) 2.47 (2.05 to 2.96)
  points; a fixed block of MW (flat shape) 2.72 (2.19 to 3.24) points. The learned share does not
  depend on whether the series gains or loses load, or on the shape.
- **Season and substation:** winter onset 2.65 (1.88 to 3.48), summer onset 2.33 (1.96 to 2.69).
  The substations range from 1.47 (D12) to 3.04 (D2) points; the share ranges from 17% to 21%.
- **The limit for NGED:** no tested duration, including 7 days, has an interval upper bound below
  0.1 points or below 5% of the transfer, for any size or window. The limit is therefore below 7
  days and is not located.
- **Signed error against the NRA series (30 days, 25%):** the clean-trained forecast is 2.8 points
  too high in the 48 hours after the end and the injected-trained forecast 5.3 points, so the
  figure NGED would see is larger than the learned bias alone (the 48-hour rows are few, 48 to 96
  per cell).

## Checks

- **The 48-hour figure depends on where the fit is cut.** For D3 with a transfer ending on 1 July,
  the mean forecast difference over one fixed week was 2.13 points for a fit cut at the end of the
  transfer, 0.88 for a fit cut 12 hours later, 0.76 a day later, and about 1.0 for fits cut 3 and
  10 days later (`diagnose_report.txt`). Why one half-day of extra clean data lowers the bias by
  60% was not found (the season-year of the solar capacity runs September to August, so it is not
  that boundary). Treat the 48-hour figure as an upper bound that holds right at the cut and the
  months 1 to 3 figure (2.8% at 30 days) as the steadier measure.
- **Where the bias sits:** for D3 and D11 after a 30-day winter transfer, the forecast difference
  is about the same in every temperature band and at every time of day, so the transfer is not
  being learned as a temperature response. For D3 it stays at 1.2 to 1.5 points in weeks 1 to 4.
- The median forecast minus the mean forecast does not apply: GAM_ONLY is a least-squares mean
  forecast. The signed error of the forecast against the NRA series is in the Results.
- **Audit triggers.** Trigger 2 (a result contradicting its own prediction) fired in size: the
  prediction expected 0.5 to 1 points and the experiment measured 2.5. The explanation tested is
  the sensitivity to the cut above, which does not remove the result (the bias at 12 hours later
  is 0.9 and at 3 months is 0.4, both above 0.1). Trigger 7 holds, because the statement to NGED
  rests on this result and no other experiment has replicated it. No Opus audit was run, because
  the brief limits the session to Sonnet; the manager is told so.

## Caveats and what to try next

- The forecast has no drift term and no lagged power (GAM-K settings); the production GAM stack with
  the Kalman filter correction would behave differently.
- The weather is actual ERA5 and CAMS, not ensemble weather.
- The transfer is injected as a share of the receiver's own power or a fixed block, so it follows
  the receiver's daily shape or is flat. Real transfers have shapes of their own.
- The existing masks (lockdown and hand-filtered events) remove some rows from both the clean and
  the injected training sets; real arrangements in the unmasked history are in both.
- Only monthly refits were tested; weekly refits would learn a transfer sooner.
- Next: transfers of 1 to 6 days, a training mask that removes known transfers (the bias should
  then be 0 during the mask), and the cause of the cut sensitivity.
