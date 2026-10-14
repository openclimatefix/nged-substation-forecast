---
id: "00016"
slug: pooled-mixture-calibration-solar-wind
status: planned
size: complex
hypothesis: >-
  For the 6 solar farms and 3 wind farms, a correction fitted on earlier validation months (a scalar
  spread factor, a spread factor per lead band, a location-and-scale calibrator, or a free quantile
  calibrator) brings the 5% to 95% band of the pooled ERA5-trained mixture closer to 90% coverage in
  every lead band, without raising the continuous ranked probability score (CRPS) by more than 0.05
  points of capacity.
prediction: >-
  The lead bands are 0 to 24 h, 1 to 2 days, 3 to 6 days, 7 to 10 days, and 11 to 15 days. In
  experiment 00002's scores the uncorrected wind band covers 95.3%, 94.4%, 90.4%, 89.6%, and 90.1%
  of cases, so it over-covers by more than 2.5 points in the first two bands. The uncorrected solar
  band covers about 92.5% of the cases with observed power of at least 1% of capacity, so it
  over-covers by 2 to 3 points in at least one band. A scalar spread factor cannot fix wind, because
  the fitted factor sits near 1 and leaves the first two wind bands more than 2.5 points above 90.
  The spread factor per lead band and the location-and-scale calibrator bring every wind band within
  2.5 points of 90. No arm changes the CRPS of either technology by more than 0.1 points of
  capacity, because the bands start within about 5 points of nominal. A per-technology fit cannot
  remove the spread between farms of the same technology (wind farm coverage ranges from 88% to 95%
  in 00002's scores).
decision_rule: >-
  Coverage is the share of cases with the observed power between the mixture's 5% and 95% quantiles
  inclusive, on non-trivial cases, those whose uncorrected 95% quantile exceeds 1% of capacity (so
  that night-time solar zeros do not inflate it). The CRPS is scored on all cases. An arm reaches 90%
  for a technology when its coverage lies within 2.5 points of 90 in every lead band and is closer
  to 90 than `none` in every band where `none` lies more than 2.5 points away. The write-up reports
  `none` against the same tolerance, so that a reader can see whether any band needed a correction.
  The CRPS contrast is the arm minus `none`, with a month-resampled interval at 99.4% (5% with a
  Bonferroni correction across the 8 planned contrasts, conservative for a verdict that rests on 4).
  Supported for a technology if some arm reaches 90% and its CRPS interval has an upper bound of at
  most +0.05 points. Refuted if no arm reaches 90%, or if `none` already lies within 2.5 points of
  90 in every band (no correction was needed). Inconclusive if an arm reaches 90% but its CRPS
  interval has an upper bound above +0.05, which the write-up names as coverage bought at a CRPS
  loss.
benchmark: >-
  The uncorrected pooled mixture (`b_mixture_full` of experiment 00002), with the beta-transform
  blend (`b_mixture_full_blp`), the stage 2 calibrator (`c_stage2_full`), and a month-by-half-hour
  climatology as references.
row_set_id: fold_a_tuesdays_v1
metric: >-
  CRPS estimated from 9 quantile levels, points of capacity; coverage of the 5% to 95% band, percent
verdict: pending
production_status: not-for-production
adopted_in: none
code: ml-experiments/00016-pooled-mixture-calibration-solar-wind/
pull_request: pending
---

# Does a correction bring the pooled mixture's 5% to 95% band to 90% coverage for solar and wind?

## Size

**The size is complex, because the experiment adds new methods and admits more than one defensible design.**

- **What gets stored:** private predictions only. No schema in `packages/contracts/` changes.
- **Production serving path:** not touched. The calibrators run on stored quantiles after the fact.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. The correction can scale about the median or about
  each quantile, and can be fitted globally or per lead band.
- **Code whose callers cannot be named without searching:** no. The mixture arithmetic copies
  `studies/stage1_pilot/stage2_common.py`, and the stored member quantiles are experiment 00002's.

## Which earlier advice this plan follows

- **Followed:** pool the ERA5-trained quantiles over the ECMWF ensemble (ENS) members before any
  calibrator (00002); report the pooled mixture beside every calibrated arm (00002); fit the
  correction on earlier validation months only, as 00002's beta-transform blend does.
- **Departs from:** nothing. No arm uses lagged power, so the advice on lagged power does not
  apply.
- **Not tested:** the PV ratio target and the wind `base_margin` that the research plan lists for
  this proposal. Experiment 00012 found the two within 0.05 points of its benchmark (C3, C4), and both change
  stage 1a and not its calibration.

## Setup

**Rows and series.** The 6 solar farms (labels A to F) and 3 wind farms (W1 to W3) of
`fold_a_tuesdays_v1`: 53 runs on Tuesdays, valid times from 2025-07-01 to 2026-06-30, every
valid time with an observed power. Labels are anonymised in the write-up.

**Inputs.** Stage 1a is an XGBoost model per series, trained on ERA5 weather and fitted before
2025-06-17, so every validation case is out of sample. The stage 2 calibrator of 00002 is not used
here. The 9 quantile levels (5%, 10%, 25%, 40%, 50%, 60%, 75%, 90%, 95%) of each of the 51
ENS members, from the ERA5-trained stage 1a of experiment 00002 (`full` variant), pooled into a mixture as in 00002
separately for each fitting seed (0, 1, 2); each seed is corrected and scored on its own. ERA5 is used only in training, so no past weather enters at
forecast time, and no arm uses a lag or a calendar column.

**Fitting the corrections.** The cases of a scored month are the runs whose initialisation time
falls in that month. Each arm is fitted per technology on the cases with valid times before the
first run of the scored month, so that a fit sees no outcome later than the earliest forecast it
is applied to. A fit uses only non-trivial cases, those whose uncorrected 95% quantile exceeds 1%
of capacity, and trivial cases (night-time solar zeros) are left uncorrected. A fit needs at least
5,000 earlier non-trivial cases (per lead band for the band arms); with fewer the arm returns the
uncorrected mixture, so the first months of the year, which cannot be fitted, are scored
uncorrected. The write-up states which months and bands this affects, gives the number of earlier
runs beside each fit's case count (one run supplies a few hundred cases that share one weather
situation), and reports coverage over the corrected months alone as well as over all months.

## Arms

| Arm | Correction of the mixture's quantiles `q` and median `m` |
|---|---|
| `none` | None |
| `spread` | `m + k (q - m)`, one `k` per technology, the value on a grid of 0.60 to 1.60 in steps of 0.05 whose coverage is nearest 90% |
| `spread_band` | The same, with one `k` per technology and lead band |
| `locscale` | `m + a + k (q - m)`, one `(a, k)` per technology and lead band, minimising the mean CRPS on a grid (`a` from -0.05 to +0.05 in steps of 0.01, `k` as above) |
| `free_quantile` | `q_t + d_t`, where `d_t` is the t-quantile of the observed power minus `q_t` over the fitted cases, per technology, lead band, and level |

The `locscale` fit draws at most 30,000 of the fitting cases, at random with a fixed seed, to
limit the cost of its grid search. Each corrected quantile is clipped to 0 to 1 of capacity and the nine are then sorted.

## Planned contrasts

Each is the arm's CRPS minus that of `none`, in points of capacity, per technology, over all
leads (the lead-band tables are descriptive).

| Id | Technology | Arm |
|---|---|---|
| C1, C2 | wind, solar | `spread` |
| C3, C4 | wind, solar | `spread_band` |
| C5, C6 | wind, solar | `locscale` |
| C7, C8 | wind, solar | `free_quantile` |

## Checks before the contrasts are read

- **Reproduction.** `none`, averaged over the three seeds, must reproduce 00002's `b_mixture_full`
  mean CRPS to within 0.05 points per technology (11.11 for wind and 5.05 for solar), or the
  experiment stops.
- **Leakage review.** A reviewer reads the code to confirm that no fit reads a case whose valid
  time is not before the first run of the scored month.

## Resolution limit

The data are 9 series in two weather regions and one validation year. The write-up reads coverage
differences of a few percentage points per technology and descriptively, and shows coverage with a
month-resampled 95% interval.
