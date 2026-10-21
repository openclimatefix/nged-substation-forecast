---
id: "00139"
slug: differentiable-physics-pv-as-of-capacity-in-gam-k
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - differentiable-physics-forecast-engine
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00139-differentiable-physics-pv-as-of-capacity-in-gam-k/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design,
  because the convex model already subtracts a capacity times a unit solar shape and only the source
  of the capacity changes. Callers: this folder's scripts, which copy experiments 111, 129, and 136
  and call the reviewed `studies.dp_*` modules without changing them.
hypothesis: >-
  Feeding the convex model with the Kalman filter (GAM-K, experiment 111) the as-of solar capacity
  of experiment 129 in place of experiment 45's pinned capacity lowers the quantile score on the
  sunniest third of half-hours, and does not raise it on the darkest third. The mechanism is that
  the pinned capacity is a hindsight growth prior, and the as-of capacity is read from each
  Primary's own power before the forecast origin.
prediction: >-
  Written before any fit. The as-of arm scores 0.03 to 0.10 points better than the pin on the
  sunniest third, and within 0.01 points on the darkest third. Over all half-hours the gain is under
  0.1 points, because experiment 129's ranges are 1.2 to 1.6 MW wide and the pin lies inside them at
  most Primaries. The hindsight arm beats the as-of arm by under 0.03 points. The no-PV arm is worse
  than the pin by 0.2 to 0.6 points on the sunniest third. The pin and as-of capacities differ most
  at D10 and D16, where capacity grew (held-forward error 0.30 and 0.70 MW).
decision_rule: >-
  Contrast C1 is the as-of arm minus the pin, read on the sunniest third of half-hours (by CAMS
  irradiance) and on the darkest third, at the 98.75% level of the month-block bootstrap on paired
  differences, as experiment 111. The hypothesis is `supported` when C1's interval on the sunniest
  third lies wholly below -0.05 points and the darkest-third upper bound is at most +0.02 points.
  The threshold is 0.05 points and not zero because the as-of capacity adds a fitting step (a
  capacity fit per origin) that the pin does not need. It is `partly supported` when the sunniest-
  third interval lies wholly below zero but not below -0.05, or the darkest-third bound is above
  +0.02. It is `refuted` when the sunniest-third interval lies wholly above zero. It is
  `inconclusive` otherwise. The hindsight arm is a labelled ceiling and never a result. The no-PV
  arm is a floor. The verdict is capped at `inconclusive` if the known-answer stage fails.
benchmark: >-
  Arm 1, the pin (experiment 45's `frozen` vintage, as experiment 111), on the same rows; absolute
  skill of climatology and manual_heuristic as stored by experiment 111
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored substation,
  at D5, D10, D11, D14, and D16 with D9 beside them (D11 also alone), over leads 0.5 to 168 hours
  from the 09:00 UTC `power_fcst_init_time`; sliced by the sunniest and darkest third of half-hours,
  by season, and by lead band; tail calibration at the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00139-differentiable-physics-pv-as-of-capacity-in-gam-k/
production_status: not-for-production
adopted_in: none
---

# Experiment 139: does an as-of solar capacity in GAM-K lower the forecast error on sunny half-hours?

Plan only; the write-up follows the runs.

## Plan

**Question.** The convex demand model already subtracts a solar capacity times a unit solar shape.
Today that capacity is experiment 45's pinned `frozen` vintage, which carries a hindsight growth
prior. [Experiment 129](../00129-differentiable-physics-pv-capacity-known-answer-test/result.md)
infers a capacity per season-year from the Primary's own power. Does feeding the as-of capacity of
experiment 129, with the solar shape driven by ensemble (ENS) irradiance, lower GAM-K's error on
sunny half-hours without raising it on dark ones? No new model: only the capacity source changes.

**Arms.** Every arm is GAM-K of [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md) driven by the ENS
ensemble mean, as in the best-guess assembly of [experiment
136](../00136-dp-bg-stage1-assembly-on-the-ensemble-mean/result.md), with monthly refits on a
24-month window and the shared filter noise setting. Only the capacity differs.

| Arm | Capacity | Role |
|---|---|---|
| 1. Pin | Experiment 45's `frozen` vintage | Control |
| 2. As-of | Experiment 129's arm 2 capacity, fitted on data before each 1 July origin and held for the following season-year | Scored arm; the only arm a live forecast could have had |
| 3. Hindsight | Experiment 129's capacity fitted on data to 2026-07-31 | Labelled ceiling, never a result |
| 4. No PV | Capacity zero | Floor: how much any explicit solar term is worth |

**The origin is 1 July and not 1 September.** The design note says 1 September, but the convex
model's season-years run from July to June (experiment 129 records the mismatch). The as-of table
therefore uses the origin 2025-07-01 for the scored year (the fit sees season-years up to 2024 and
holds the 2024 value for 2025) and the origin 2024-07-01 for the replicate year (up to season-year
2023). `e00129_asof.py` builds one origin; this experiment loops it over the two origins. Before the
first complete season-year at a Primary the adaptor falls back to the pin and flags the row.

**Adaptor.** A `PinnedCapacity`-shaped object (`studies.dp_pinned_capacity`) built from the as-of
table: the weekly path is each season-year's capacity from the origin's fit, and the capacity after
the origin is the latest season-year's value held forward. It lives in this experiment's folder.
Moving it into `packages/studies` would be a module change that needs the maintainer's review.

**Leakage check, first.** The as-of fit at an origin sees only rows before the origin. Experiment 129
divided its estimates by a divisor taken from recoveries fitted with data to 2026-06-30, which is
disclosed there and is hindsight at the origin. This experiment either rebuilds the divisor per
origin or states the size of the effect: it compares the divisor of each season-year with the
median across season-years, and reports the as-of arm with the divisor fixed at 1 as a sensitivity
arm. A Sonnet code reviewer checks the table for any use of data after the origin.

**Known-answer stage, before any forecast arm.** Add a real metered farm's output (anonymised) at
50% of a Primary's as-of capacity to its net power for the forecast year, with the capacity left
unchanged. Check that the GAM-K forecast under-predicts by about the injection on sunny half-hours.
Then raise the capacity by the injected amount and check that the error falls by 0.75 to 1.25 of the
injection, as in experiment 129. Also rerun experiment 111's gate: the calibrated fake ENS scores
within 0.05 points of ERA5 and CAMS.

**Stop rules.** Stop and report a wiring fault if the injection is not recovered within 0.75 to 1.25
at 4 of the 5 Primaries. After the first replicate, stop if arm 2 minus arm 1 has an interval lying
wholly above zero on the sunniest third, and diagnose it (an as-of capacity too low at the growing
Primaries D10 and D16 is the likely cause) instead of running more arms.

**Rows and slices.** `fold_a_daily_gap_v1` (the 292 daily ENS runs of experiment 87), with the
replicate on the even months of 2024-07 to 2025-06 as in experiment 111. Slices: sunniest and
darkest third of half-hours by CAMS irradiance, season, lead band, D11 alone, and per Primary.
Absolute skill per arm beside every contrast, and tail calibration at the 0.01, 0.05, 0.95, and 0.99
quantiles. Forecast rows stay before 2026-07-01.

**Records.** Ratio calibration of an ensemble mean damps cloud variability, so report the forecast
solar output minus the CAMS-driven output by CAMS clear-sky index band. The capacity unit is the
99th percentile of alternating-current output over daylight half-hours in MW, named wherever a
number appears. Report per Primary, since D10 and D16 grew.

**D9 and D18.** Experiment 129 covered D5, D10, D11, D14, and D16 only, so D9 has no as-of table and
is reported with the pin and no-PV arms only. D18 is not scored.

**Compute.** CPU only, at most 6 cores with one thread per worker, each job under `systemd-run
--user --scope -p MemoryMax=16G -p OOMPolicy=continue`. Time one Primary and one month before the
full grid, because experiment 111 underestimated its XGBoost compute 20 times. The as-of fits use
`e00129_asof.py`'s GPU path (one job at a time).

**Five triggers.** Stored: private files only. Serving path: not touched. Degradation rule: not
touched. More than one defensible design: no. Callers not nameable without searching: no.

**Reviews.** One Sonnet plan review, one Sonnet code review (leakage first), one Sonnet write-up
review, one Sonnet prose review. An Opus review only with the maintainer's approval.

**Advice followed.** From experiment 129, use an as-of capacity fitted before each forecast origin
and never the hindsight capacity in a forecast test, and count only season-years with at least
12,000 training half-hours when estimating growth. From experiment 125, do not rely on member
spread for tail skill.
