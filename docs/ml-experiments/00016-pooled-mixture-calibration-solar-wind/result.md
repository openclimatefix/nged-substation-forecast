---
id: "00016"
slug: pooled-mixture-calibration-solar-wind
status: planned
size: complex
hypothesis: >-
  For the 6 solar farms and 3 wind farms, a correction fitted on earlier validation months (a scalar
  spread factor, a spread factor per lead band, a location-and-scale calibrator, or a free quantile
  calibrator) brings the 5% to 95% band of the pooled ERA5-trained mixture to 90% coverage in every
  lead band without raising the continuous ranked probability score (CRPS).
prediction: >-
  The uncorrected mixture covers 94% to 95% of the observed power in the first two days and about
  90% from day 3 (experiment 00002's scores, wind), so it over-covers at short leads. A scalar
  spread factor cannot fix both ends, and so leaves at least one wind band more than 2.5 points
  from 90%. The spread factor per lead band and the location-and-scale calibrator bring every band
  within 2.5 points of 90% and change the CRPS by less than 0.1 points of capacity.
decision_rule: >-
  Coverage is the share of cases with the observed power between the mixture's 5% and 95%
  quantiles, on cases whose uncorrected 95% quantile exceeds 1% of capacity (so that night-time
  solar zeros do not inflate it). An arm reaches 90% for a technology when its coverage lies within
  2.5 points of 90 in every one of the five lead bands. The CRPS contrast is the arm minus no
  correction, with a month-resampled interval at 99.4% (5% with a Bonferroni correction across the
  8 planned contrasts). Supported for a technology if some arm reaches 90% and its CRPS interval has
  an upper bound of at most +0.05 points. Refuted for a technology if no arm reaches 90%.
  Inconclusive otherwise. A CRPS gain needs an interval wholly below zero and an estimate of at
  most -0.10.
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

- **Followed:** pool the ERA5-trained quantiles over the ENS members before any calibrator
  (00002); report the pooled mixture beside every calibrated arm (00002); fit the correction
  without any validation month later than the one scored, as 00002's blend does.
- **Departs from:** nothing. No arm uses lagged power, so the advice on lagged power does not
  apply.
- **Not tested:** the PV ratio target and the wind `base_margin` that the research plan lists for
  this proposal. Experiment 00012 found them within 0.05 points of its benchmark (C3, C4), and they change
  stage 1a and not its calibration.

## Setup

**Rows and series.** The 6 solar farms (labels A to F) and 3 wind farms (W1 to W3) of
`fold_a_tuesdays_v1`: 53 runs on Tuesdays, valid times from 2025-07-01 to 2026-06-30, every
valid time with an observed power. Labels are anonymised in the write-up.

**Inputs.** The 9 quantile levels (5%, 10%, 25%, 40%, 50%, 60%, 75%, 90%, 95%) of each of the 51
ENS members, from the ERA5-trained stage 1a of experiment 00002 (`full` variant, seeds 0, 1, 2),
pooled into a mixture as in 00002. ERA5 is used only in training, so no past weather enters at
forecast time, and no arm uses a lag or a calendar column.

**Fitting the corrections.** Each arm is fitted per technology on the cases whose valid times end
at least 14 days before the scored month starts, the rule 00002's blend uses. A fit needs at
least 5,000 earlier cases (per lead band for the band arms); with fewer the arm returns the
uncorrected mixture, so the first months of the year, which cannot be corrected, are scored
uncorrected. The write-up states which months and bands this affects.

## Arms

| Arm | Correction of the mixture's quantiles `q` and median `m` |
|---|---|
| `none` | None |
| `spread` | `m + k (q - m)`, one `k` per technology, the value on a grid of 0.60 to 1.60 in steps of 0.05 whose coverage is nearest 90% |
| `spread_band` | The same, with one `k` per technology and lead band |
| `locscale` | `m + a + k (q - m)`, one `(a, k)` per technology and lead band, minimising the mean CRPS on a grid (`a` from -0.05 to +0.05 in steps of 0.01, `k` as above) |
| `free_quantile` | `q_t + d_t`, where `d_t` is the t-quantile of the observed power minus `q_t` over the fitted cases, per technology, lead band, and level |

Each corrected quantile is clipped to 0 to 1 of capacity and the nine are then sorted.

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

- **Reproduction.** `none` must reproduce 00002's `b_mixture_full` mean CRPS to within 0.01
  points per technology, or the experiment stops.
- **Leakage review.** A reviewer reads the code to confirm that no fit reads a case whose valid
  time is later than 14 days before the scored month.

## Resolution limit

Nine series in two weather regions, one validation year. Coverage differences of a few percentage
points are read per technology and descriptively.
