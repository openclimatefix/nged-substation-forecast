---
id: "00098"
slug: det-bg-solar-and-heating-offsets-in-switching-detector-residual
status: done
size: medium
hypothesis: >-
  The refit-offset neighbour-count switching detector (RO-NC, experiment 82's XL) subtracts a
  lagless XGBoost demand forecast (experiment 17's H0.5, trained on ERA5) from each Primary
  substation's measured power. Giving that forecast the best guess's two explicit offsets, an
  embedded-solar term (capacity times a solar anomaly) and a heating degree-hour term, removes
  structured weather error from the residual, so the detector finds more of the moved MW at the
  same number of unexplained alarms.
prediction: >-
  The solar offset lowers the median absolute residual in summer midday hours (10:00 to 14:00 UTC,
  April to September) at the 5 Primaries that have a solar capacity by 5% to 15%, and RO-NC's
  MW-weighted recall (pooled over the 16 Primaries) rises by +0.01 (-0.01 to +0.03) for the better
  of the ERA5 and CAMS solar arms. The change is small because 11 of the 16 series have no solar
  term and the model already sees ERA5 shortwave as an input. The heating offset adds +0.00 to
  +0.01. CAMS and ERA5 solar arms differ by under 0.01 in recall.
decision_rule: >-
  Four planned contrasts, each the paired difference in MW-weighted recall of RO-NC, pooled with
  equal weight over injection shapes S2, S3, and S4, on the draws of experiment 82 (same seeds),
  for transfers that start in even months, intervals at 98.75% (a 95% interval split across four)
  from 20,000 resamples of whole calendar months. C1 is arm `era5_solar` minus the control `ctrl`.
  C2 is arm `cams_solar` minus `ctrl`. C3 is arm `heat` (the better solar arm plus the heating
  offset, where better means the smaller median absolute summer-midday residual on odd months at
  the 5 solar series) minus `ctrl`. C4 is `heat` minus its solar parent. Every arm is held to 80
  unexplained odd-month alarms. The verdict is `supported` when the lower end of the best of C1 to
  C3 is above zero and its point value is at least +0.02, `small gain` when that lower end is above
  zero but the point value is under +0.02, `refuted` when the upper ends of C1, C2, and C3 are all
  under +0.02, and `inconclusive` otherwise. The +0.02 threshold is experiment 70's: the offsets
  add a capacity table, an irradiance feed, and a coefficient fit to the detector's residual
  model. The residual noise, forecast MAE of H0.5, recall by donor step, and the same detectors
  PF14 and D0 are descriptive.
benchmark: >-
  `ctrl`: H0.5 refitted with a zero base margin under the refit schedule below. `h05_stored`
  (experiment 17 and 71's stored fits) is the replication check: RO-NC must reproduce experiment
  82's 0.770 (0.665 to 0.857) MW-weighted recall and PF14's 0.656.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 69 to 71 and 82 do, not forecasts)
metric: >-
  MW-weighted recall at 80 unexplained odd-month alarms (headline), pooled over S2, S3, and S4,
  even months; median absolute residual by hour band and season; forecast MAE of H0.5 itself;
  recall by donor step in noise units.
code: ml-experiments/00098-det-bg-solar-and-heating-offsets-in-switching-detector-residual/
verdict: inconclusive
best_arm: >-
  `heat` (CAMS solar offset plus heating offset): RO-NC finds 0.776 of the moved MW (98.75%
  interval 0.672 to 0.861) against 0.766 (0.654 to 0.855) for `ctrl`; C3 is +0.010 (-0.010 to
  +0.034). The two solar-only arms are refuted: C1 -0.002 (-0.019 to +0.012), C2 +0.002 (-0.013 to
  +0.009).
lessons: >-
  A cleaner summer midday residual did not make the detector find more moved MW. The CAMS solar
  offset lowered the median absolute residual of the 5 solar series by 12% in June to August
  midday hours and the detector's recall did not move, so the misses are not weather error in the
  residual: they are small donor steps with no metered receiver, which a quieter residual in 5 of
  16 series cannot reach. The ERA5 solar offset left the residual unchanged, because H0.5 already
  reads ERA5 shortwave as an input.
advice:
  - >-
    Do not add a solar or heating offset to the residual model of the switching detector to raise
    recall, because RO-NC's recall changed by -0.002 to +0.010 across the three offset arms with
    every 98.75% interval straddling zero (C1 to C3).
  - >-
    Look for the missing recall in the signal, not the residual's weather error, because the CAMS
    offset cut the summer midday residual by 12% and recall at the 5 solar series moved by +0.003
    (-0.028 to +0.022) (figure 3, donor split).
  - >-
    Use CAMS rather than ERA5 irradiance when a solar offset is wanted for a residual, because the
    CAMS offset cut the summer midday median residual from 5.26 to 4.63 points of capacity and the
    ERA5 offset left it at 5.27 (figure 3).
  - >-
    Expect transfers from Primaries with embedded solar to stay the hard case, because RO-NC finds
    0.51 of their moved MW against 0.94 for the other donors, in every arm (figure 4).
production_status: not-for-production
adopted_in: none
---

# Experiment 98: do the best guess's solar and heating offsets sharpen the switching detector?

**Giving the demand forecast behind the switching detector the best guess's solar and heating
offsets did not make the detector find more moved power, even though the CAMS solar offset made the
summer midday residual 12% smaller.** So the misses that remain are not caused by weather error in
the residual. They sit among small transfers from substations whose neighbours are unmetered, which
a quieter residual does not reach. The verdict is `inconclusive`: both solar-only arms are refuted
by the pre-registered rule, and the heating arm's upper end (+0.034) was too wide to rule out the
+0.02 threshold.

A load transfer is a switching operation that moves part of one Primary substation's demand onto
neighbouring substations for hours or days. The refit-offset neighbour-count detector (RO-NC,
called XL in [experiment 82](../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md))
subtracts an XGBoost forecast of demand from the measured power and raises an alarm when the
24-hour average of the difference steps. We tested whether telling that forecast about solar output
and heating explicitly, as the best guess does, shrinks the difference enough to find more
transfers. We re-ran the experiment 82 detector on the same synthetic transfers, held to 80 alarms
that no logged closure explains, and compared the share of the moved megawatts (MW) found.

![Share of moved MW found, by arm](figure_1_recall_by_arm.svg)

![Planned contrasts](figure_2_contrasts.svg)

![Summer residual by hour band](figure_3_summer_residual.svg)

![Recall by donor type](figure_4_recall_by_donor.svg)

## Results

- **Arms.** `h05_stored` is experiment 82's own result, reproduced (0.770). `ctrl` is H0.5 refitted
  here with a constant starting margin. `era5_solar` and `cams_solar` add a base margin of minus
  the capacity over effective capacity times the anomaly of a unit PV shape, at the 5 series that
  have a capacity (experiment 64's D9, D10, D11, D14, and D16), with weight fixed at -1. `heat`
  adds experiment 94's heating term at all 16 series to the CAMS solar arm, which had the smaller
  summer midday residual on odd months (5.29 against 5.73 for ERA5 and 5.41 for `ctrl`).
- **Row set.** The 611 headline-band transfers per shape of experiment 82, same seeds, shapes S2,
  S3, and S4 pooled, starting in even months, 16 Primary substations. No forecast rows, so no
  tail-calibration or curtailment-cost table applies.
- **RO-NC MW-weighted recall (98.75% interval):** `h05_stored` 0.770 (0.659 to 0.860); `ctrl`
  0.766 (0.654 to 0.855); `era5_solar` 0.764 (0.647 to 0.853); `cams_solar` 0.768 (0.650 to 0.856);
  `heat` 0.776 (0.672 to 0.861). PF14 stays at 0.656 to 0.662 in the first four arms and 0.686 in
  `heat`.
- **Planned contrasts:** C1 `era5_solar` minus `ctrl` -0.002 (-0.019 to +0.012); C2 `cams_solar`
  minus `ctrl` +0.002 (-0.013 to +0.009); C3 `heat` minus `ctrl` +0.010 (-0.010 to +0.034); C4
  `heat` minus `cams_solar` +0.009 (-0.008 to +0.037). `ctrl` minus `h05_stored` is -0.004 (-0.014
  to +0.003), so the cheaper refit schedule did not change the detector.
- **Residual noise.** Median absolute residual of the 5 solar series, June to August 2022 to 2026,
  midday (10:00 to 14:00 UTC): `ctrl` 5.26 points of capacity, `era5_solar` 5.27, `cams_solar` 4.63,
  `heat` 4.63. Afternoon: 4.63, 4.78, 4.00, 3.99. On odd months from April to September, the
  selection slice, the gain is only 2% (5.41 to 5.29). Mean absolute error of H0.5 over all 16
  series and all hours: `ctrl` 6.72, `era5_solar` 6.75, `cams_solar` 6.70, `heat` 6.68 points.
- **By donor (exploratory).** When the donor is one of the 5 solar series (325 transfers per
  shape) RO-NC finds 0.512 (`ctrl`) against 0.515 (`cams_solar`) and 0.537 (`heat`); when it is
  one of the other 11 (286 transfers) 0.940 in every arm.
- **By donor step.** The misses are the small steps, as in experiment 82: with no metered receiver
  and a step of 2 to 4 noise units, `ctrl` finds 140 of 550 transfers, `cams_solar` 145, and `heat`
  155.

## Caveats

- **Lab shortcuts.** ERA5 weather in the residual model (a live service would use ENS), the
  24-hour hindsight end rules of experiment 82, and solar capacities fitted partly on data after
  some scored dates (experiment 64's held value before July 2025, experiment 78's season-year value
  after).
- **Schedule.** Two H0.5 models refitted on the 1st and the 16th of each month, 150 rounds at eta
  0.15, against experiment 17's 500 rounds at eta 0.05. `ctrl` matches the stored fit's mean
  absolute error for series 7 to three decimals and the detector's recall to within 0.004, so
  the contrasts should not depend on the schedule.
- **Only 5 of 16 series have a solar term**, so the pooled recall cannot move much by
  construction; the donor split shows the 5 solar series alone also stay flat.
- **Fixed weight -1 on the solar margin**, set before the run. Experiment 64's fitted weights of
  -0.3 to -0.6 were for noisy forecast anomalies; a different weight was not tried.
- **CAMS has no value for about 10% of daylight half-hours** (dawn, dusk, and gaps), where the
  anomaly counts as zero. ERA5 shortwave is taken as the half-hour ending at its timestamp.
- **Not run.** The blend arm (4) was not run because ERA5 and CAMS recall did not differ, and the
  bank-holiday arm (6) because experiment 47 already found it leaves recall unchanged.

## What the next experiment should answer

The residual's weather error is not what limits recall. The 401 of 725 missed transfers with a
small donor step and no metered receiver need a different signal, such as the longer averaging
window or the receiver-balance check listed under experiment 82.
