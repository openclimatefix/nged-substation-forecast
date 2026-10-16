---
id: "00051"
slug: pvdis-3b-midday-loss-of-the-solar-term
status: planned
size: simple
hypothesis: >-
  At June-to-August midday on day+2 and day+3, the solar term of experiment 00046 scores worse than
  the best guess `d0` (C6 +0.477 and +0.172 points of capacity) mainly because the fixed
  coefficient of 1 is too large for the sunniest rows, which a PV inverter clips, and not because
  the ENS members add noise.
prediction: >-
  The residual of the oracle-shifted median turns positive above a knee in the CAMS clear-sky
  index at 3 or more of the 5 verdict series, at a knee between 0.6 and 0.9, which implies a
  DC:AC ratio between 1.1 and 1.5. A clip fitted on one half of the weeks lowers the oracle's
  median error on the other half by 0.05 to 0.15 points of capacity. The ensemble-mean shift
  removes under a third of the ENS shift's excess over `d1l`.
decision_rule: >-
  Rows are the 5 verdict series, June to August, valid 09:00 to 15:00 UTC, day+1 to day+3, from
  experiment 00046's stored predictions (no new fit). Per-row values are averaged over the 3 seeds.
  Intervals are 95%, resampling whole ISO weeks of forecast start, 10,000 resamples (1,000 where a
  hinge is refitted in each resample). Let e be the actual minus the `d1l` median shifted by the
  oracle with coefficient 1, in points of capacity; it is zero if the oracle shift is exactly right,
  and positive where the shift over-corrects. Three repairs to that median are scored out of fold
  (fit on odd ISO weeks of the valid time and score on even, then the reverse) by the mean absolute
  error of the median. Repair A is a hinge, e = a + c * max(0, csi - k), fitted per series with
  the knee k on a grid from 0.4 to 0.95 of the CAMS clear-sky index. Repair L is a straight line,
  e = a + b * csi, the comparison for A. Repair B is a coefficient fitted per series and hour bin
  (09-12 and 12-15 UTC), median = `d1l` median minus coefficient times shift. Repair C is the shift
  built from the ensemble-mean anomaly (no member spread), scored on the quantile score and the
  sunny 0.05-quantile coverage against the member-based ENS shift. (a) clipping explains the loss
  when repair A beats repair L out of fold with an interval wholly below zero, the slope c above
  the knee is positive with an interval above zero at 3 or more series, and repair A lowers the
  out-of-fold error against the unrepaired oracle shift by at least 0.05 points with an interval
  wholly below zero. (b) a too-large coefficient explains it when the slope of the `d1l` residual
  on the oracle shift has an interval wholly above -0.85 in at least 4 of the 6 cells of three
  months by two hour bins, pooled across series, (a) fails, and repair B lowers the out-of-fold
  error by at least 0.05 points with an interval wholly below zero. (c) member noise explains it
  when the quantile score of the ENS shift minus that of the ensemble-mean shift has an interval
  wholly above zero and is at least half of the ENS shift minus `d1l`, at day+2 and day+3 pooled.
  More than one cause may hold; if none holds the verdict is inconclusive and the write-up says
  what the residuals show.
  A load that rises with sunshine mimics a clipping knee (visitor load at coastal resorts, cooling
  of crop stores, irrigation), so cause (a) also needs the splits below. The hinge is compared with
  a straight line out of fold, by the mean absolute error of the median on the held-out weeks, not
  by the in-sample sum of squares: (a) requires the hinge to beat the straight line out of fold. A
  knee that appears in the inland series, in term-time weekdays, and in school-holiday weekdays and
  weekends alike is clipping. A knee that is absent inland, or that concentrates in school-holiday
  and weekend rows, is load, and the write-up names it as load. The splits are coastal (within 10
  km of the open coast, by distance computed in the process and never published) against inland
  series, and school-holiday weeks (22 July to 31 August) plus weekends against term-time weekdays. The coastal split compares the series that
  qualify with the rest, and the write-up states how many qualify. Temperature at fixed
  clear-sky index is also read: the ERA5 2 m temperature is binned in three terciles within each
  clear-sky-index bin, and the residual is read by tercile. A residual that grows with temperature
  at fixed clear-sky index points to load, since clipping depends on irradiance and hot panels
  clip slightly less.
benchmark: >-
  The `d1l` median (experiment 00046), shifted by the oracle with coefficient 1, for the repairs;
  `d1l_ens_shift` for repair C.
row_set_id: fold_a_every5d_v1
metric: >-
  Mean absolute error of the median and quantile score, in points of capacity (1 point is 1% of
  the series' effective capacity), plus the fraction of actuals below the 0.05 quantile in sunny
  rows.
code: ml-experiments/00051-pvdis-3b-midday-loss-of-the-solar-term/
---

# Experiment 51: why does the solar term lose at sunny summer middays?

*Planned; the write-up replaces this page.* Experiment 46 added a solar correction to a demand
forecast and found that at June-to-August midday it scored worse than the same forecast without the
correction, 0.48 points at two days ahead. This experiment asks why, using only the numbers
experiment 46 already stored, with no new forecast model.

## Size

- **What is stored:** one results folder under `data/ml-experiments/00051-*`; no predictions.
- **Serving path:** untouched.
- **Degradation rule:** untouched.
- **Designs:** three causes are tested side by side, so no choice among designs is needed.
- **Callers:** the script imports experiment 46's helpers; nothing imports the script.

All five are "no", so the size is simple.

## Planned analysis

1. Plot the residual of the `d1l` median against the CAMS clear-sky index by hour, by lead day, and
   by Primary, before and after the oracle shift.
2. Cause (a): fit a hinge in the clear-sky index to the oracle-shifted residual per series, and
   convert the knee to a DC:AC ratio through the unit output `u` at the knee.
3. Cause (b): the slope of the `d1l` residual on the oracle shift by hour and month.
4. Cause (c): the ENS shift against the ensemble-mean shift.

## Non-inferiority margin for PVDIS-5

Set in the "next" paragraph after the results are read, and stated in advance here: the margin is
0.10 points of capacity (about 2% of `d0`'s 4.62), which is the amount by which `d0`'s radiation
inputs improved on `d1l` (0.104). A solar term that loses to `d0` by more than the gain the
radiation inputs brought has no claim to replace them.
