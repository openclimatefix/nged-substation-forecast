---
id: "00051"
slug: pvdis-3b-midday-loss-of-the-solar-term
status: done
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
  and weekend rows, is load, and the write-up names it as load. The splits are coastal (within 10 km
  of the open coast, by distance computed in the process and never published) against inland series,
  and school-holiday weeks (22 July to 31 August) plus weekends against term-time weekdays. The
  coastal split compares the series that qualify with the rest, and the write-up states how many
  qualify. Temperature at fixed clear-sky index is also read: the ERA5 2 m temperature is binned in
  three terciles within each clear-sky-index bin, and the residual is read by tercile. A residual
  that grows with temperature at fixed clear-sky index points to load, since clipping depends on
  irradiance and hot panels clip slightly less.
benchmark: >-
  The `d1l` median (experiment 00046), shifted by the oracle with coefficient 1, for the repairs;
  `d1l_ens_shift` for repair C.
row_set_id: fold_a_every5d_v1
metric: >-
  Mean absolute error of the median and quantile score, in points of capacity (1 point is 1% of
  the series' effective capacity), plus the fraction of actuals below the 0.05 quantile in sunny
  rows.
code: ml-experiments/00051-pvdis-3b-midday-loss-of-the-solar-term/
verdict: partly supported
best_arm: >-
  Not applicable: the experiment compares repairs of one stored forecast and fits no new arm. The
  repair that lowers the median's error most is a coefficient fitted per series and hour bin
  (-0.991 points, -1.473 to -0.495), which a straight line in the clear-sky index matches.
headline: >-
  At June-to-August midday the oracle shift's coefficient of 1 is too large: the residual of the
  `d1l` median has a slope of -0.42 to -0.69 on it in every month and hour bin, and a fitted
  coefficient lowers the median's error by 0.991 points (-1.473 to -0.495). A clipping knee is not
  distinguished from a straight line (hinge minus line -0.103, -0.233 to +0.030). Noisy ENS members
  add 0.129 points (0.057 to 0.212) to the ENS shift's score.
lessons: >-
  The summer-midday loss is mostly a coefficient that is too large, and clipping cannot be told
  apart from a straight line in this sample. The fitted coefficient of experiment 46 removes the
  loss at day+2 (+0.004 against `d0`, -0.255 to +0.241, post hoc).
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Fit the solar coefficient by hour of day and season in PVDIS-5, because the slope of the `d1l`
    residual on the oracle shift is -0.42 in June and -0.69 in July at 09-12 UTC, and a fitted
    coefficient lowers the median's out-of-fold error by 0.991 points (-1.473 to -0.495), (B,
    repairs figure).
  - >-
    Do not report a clipping knee or a DC:AC ratio from these rows, because a hinge beats a straight
    line out of fold by only 0.103 points (-0.233 to +0.030) and the implied ratios (1.8 to 3.2)
    are not credible for rooftop inverters (hinge table).
  - >-
    Keep the member spread of the ENS shift but check the sunny 0.05-quantile coverage, because the
    shift with member spread puts 0.010 to 0.027 of sunny actuals below the 0.05 quantile and the
    ensemble-mean shift puts 0.039 to 0.086 there, against a target of 0.05 (coverage figure).
  - >-
    Set PVDIS-5's non-inferiority margin to 0.10 points of capacity and judge it on the upper end
    of a 95% interval from at least 25 weekly blocks, because 13 blocks gave an upper end of +0.110
    for the fitted shift minus `d0` (post hoc, next section).
---

# Experiment 51: why does the solar term lose at sunny summer middays?

We asked why a correction for rooftop solar panels made a demand forecast worse around midday in
summer. [Experiment 46](../00046-embedded-solar-share-of-primary-forecast-error/result.md) lowered
each forecast of a Primary substation's power by the panels' capacity times how much sunnier the day
was than average. At 09:00 to 15:00 UTC in June to August, two days ahead, the correction scored
0.48 points worse than the same forecast without it, where a point is 1% of the substation's
capacity. We tested three explanations on the forecasts that experiment 46 had already stored,
without fitting a new forecast model: the panels' inverters cap their output on sunny days, the
correction is simply too large, or the 51 ensemble members of the weather forecast add noise. The
correction built from the true sunshine is too large: the actual power fell by only 0.4 to 0.7 of
the amount the correction removed, at every month and time of day. A cap on the panels' output could
not be told apart from that, and the noisy members explained a smaller part of the loss.

**The verdict is `partly supported`: a too-large coefficient explains the loss, clipping is not
distinguished from it, and member noise adds a smaller part.** The decision rule was fixed in the
plan before any result was read. The scripts were run once before the code review, with the same
rule, and again after it; the review changed which series count as coastal and added checks, not
the rule. All figures are for the 5 Primaries of experiment 46, June to August, valid 09:00 to
15:00 UTC, 1 to 4 days ahead (3,575 forecast cases, each averaged over 3 seeds, in 13 weeks of
forecast start). Intervals are 95%, from resampling whole weeks. The series are labelled D9, D10,
D11, D14, and D16, as in experiment 46.

## Raw data: the error left after the oracle shift

**The oracle shift lowers the forecast too far on sunny rows, and by about the same amount at every
lead day, hour, and Primary.** Let *e* be the actual power minus the forecast after the oracle shift
(the shift built from the true sunshine), in points of capacity. A value of 0 means the shift was
exactly right, and a positive value means the shift lowered the forecast too far. The mean of *e* is
negative on cloudy rows (about -3 to -4 points), crosses 0 near a clear-sky index of 0.7, and
reaches +5 to +8 points above 0.9. The grey dots are the cases, and the line is their mean.

![Error after the oracle shift against the clear-sky index, by lead day](figure_1_by_lead_day.svg)

![The same by hour of day](figure_2_by_hour.svg)

![The same by Primary](figure_3_by_primary.svg)

## Cause (b): the coefficient is too large

**The actual power falls by 0.4 to 0.7 of what the oracle shift removes, so a coefficient of 1
over-corrects.** A coefficient of 1 would give a slope of -1 of the `d1l` residual on the shift. The
slope is between -0.42 and -0.69 in each of the 6 cells of month by hour bin (June -0.42 and -0.51,
July -0.69 and -0.65, August -0.69 and -0.60), by lead day -0.61, -0.54, and -0.61, and by Primary
-0.52 to -0.81. In 4 of the 6 cells the interval lies wholly above -0.85, which meets the rule's
count; the 2 August cells have lower ends of -1.03 and -1.12. A coefficient fitted per series and
hour bin on half the weeks lowers the median's mean absolute error on the other half by 0.991
points (-1.473 to -0.495), against the oracle shift with coefficient 1.

![The residual against the oracle shift](figure_4_residual_against_shift.svg)

## Cause (a): a clipping knee is not distinguished from a straight line

**A hinge that is flat below a knee and rises above it fits no better than a straight line out of
fold.** The hinge lowers the median's error against the unrepaired oracle by 1.023 points (-1.526 to
-0.521), but a straight line in the clear-sky index lowers it by 0.920 (-1.382 to -0.437). The
hinge beats the line by 0.103 points (-0.233 to +0.030), and the rule needed an interval wholly
below 0. Both lower the error through a free intercept, so the first number alone is not evidence of
clipping. The hinge slope above the knee is positive at all 5 series, with intervals above 0, and
the knees lie between a clear-sky index of 0.40 and 0.65 with wide intervals (0.4 to 0.78). At the
knee, the unit output of the CAMS shape implies a DC:AC ratio of 1.8 to 3.2, which is not credible
for rooftop inverters, whose ratios are usually under 1.5. Not reporting a per-Primary knee for
PVDIS-6 is therefore the honest reading: these rows cannot place one.

| Series | Knee (clear-sky index) | Slope above the knee (points per unit index, 95% interval) | Implied DC:AC |
|---|---|---|---|
| D9 | 0.40 | 24.1 (19.7 to 32.6) | 3.2 |
| D10 | 0.575 | 32.2 (18.2 to 58.8) | 2.1 |
| D11 | 0.55 | 20.7 (15.5 to 33.5) | 2.2 |
| D14 | 0.60 | 24.1 (17.5 to 45.5) | 2.0 |
| D16 | 0.65 | 15.3 (9.7 to 25.5) | 1.8 |

![Out-of-fold repairs of the median](figure_5_repairs.svg)

**The knee does not concentrate in the rows where a load would show, with one exception.** The
sunshine-dependent loads at the coast (visitors, cooling, irrigation) would give a positive
residual that grows with the clear-sky index, as a clipping knee does. 2 of the 5 Primaries are
within 10 km of the open coast. The slope above each series' own knee is 24.1 at those 2 and 22.7 at
the other 3, and 25.2 on term-time weekdays against 17.3 on school-holiday weeks and weekends (no
intervals; point estimates only). The hinge beats the line by 0.161 points (-0.282 to -0.038, 10
blocks) in holiday-and-weekend rows alone, which is the load signature, and by -0.081 (-0.253 to
+0.097) on term-time weekdays. At fixed clear-sky index, the sunniest rows' error rises with 2 m
temperature by 0.67 points per kelvin (0.21 to 1.16). Clipping depends on irradiance, and hot panels
clip slightly less, so a rise with temperature points to a load or to a response the forecast does
not model. This is a second contribution to the over-correction that the data cannot separate from
the coefficient.

## Cause (c): noisy ENS members add a smaller part

**The member spread adds 0.129 points to the ENS shift's score, and about 0.19 more comes from the
shift's size.** The table gives the quantile score of the first arm minus the second, at day+2 and
day+3 pooled (2,340 cases), with its interval.

| Contrast (negative favours the first arm) | Difference (95% interval) |
|---|---|
| ENS shift minus `d0` (experiment 46's C6) | +0.324 (-0.144 to +0.761) |
| ENS shift minus `d1l` (C3) | +0.057 (-0.495 to +0.570) |
| ENS shift minus ensemble-mean shift (cause c) | +0.129 (+0.057 to +0.212) |
| Ensemble-mean shift minus `d1l` | -0.072 (-0.625 to +0.441) |
| Fitted-coefficient ENS shift minus ENS shift (post hoc) | -0.435 (-0.689 to -0.139) |
| Fitted-coefficient ENS shift minus `d0` (post hoc) | -0.111 (-0.348 to +0.110) |

The loss that prompted the experiment reproduces: C6 is +0.477 (+0.057 to +0.842) at day+2 and
+0.172 (-0.580 to +0.807) at day+3, the figures of experiment 46. Over the two days pooled, the
interval of C6 includes 0 when 13 weeks are resampled, so the loss is small and uncertain here. The
fitted-coefficient shift of experiment 46, fitted on the training window, removes it: it scores
+0.004 (-0.255 to +0.241) against `d0` at day+2 (post hoc). The oracle shift over-covers the sunny
lower tail too, so member noise cannot be its only source.

![Share of sunny actuals below the 0.05 quantile](figure_6_coverage.svg)

## What it could not show

- **The hinge test has little power.** Its interval has 13 blocks, a straight line is a good
  approximation over most of the range, and the knees are not pinned down.
- **The out-of-fold folds are weeks of the valid time.** One forecast run spans up to 4 valid days
  and can fall into both folds, which mildly favours every repair. The intervals resample fixed fits,
  so they are somewhat narrow.
- **The slope may be attenuated by error in the CAMS sunshine.** Error in the shift pulls the
  slope towards 0, so some of the "too-large coefficient" may be the difference between CAMS and
  the real sunshine. A coefficient fitted on CAMS has the same property, which is the one that
  matters for use.
- **The coastal flag rests on a coarse coastline** (7 points, a 10 km threshold), and 2 of 5
  Primaries qualify.
- **Both loads and clipping may be present.** The data support a coefficient below 1; they do not
  exclude a smaller clipping effect on top of it.

## What the next experiment should answer

**PVDIS-5 should fit the solar coefficient by hour and season, test non-inferiority against `d0` with
a margin of 0.10 points of capacity, and not report a clipping knee.** The margin is set in the
plan: 0.10 points is about 2% of `d0`'s 4.62 and about the amount (0.104) by which `d0`'s radiation
inputs improved on `d1l`, so a solar term worse than `d0` by more would give back the gain the
radiation inputs brought. Non-inferiority means the upper end of the 95% interval of (solar term
minus `d0`) lies below +0.10. The fitted shift's upper end here is +0.110 over 13 weeks at summer
midday day+2 and day+3 (post hoc), so the margin cannot yet be declared met; PVDIS-5 should resample
at least 25 weeks of forecast start (April to September) and repeat the test for 0 to 48 hours
separately. A per-Primary clipping knee needs a different design: rows at or near the inverter's
rated output, such as the clearest days at midday, with the load splits repeated on more than two
coastal series.
