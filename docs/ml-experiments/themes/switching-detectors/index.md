# Switching detectors

**The best detector we have built finds about three quarters of the megawatts (MW) moved by injected
load transfers but only about a third of the real ones, and nothing yet shows that using it
improves a forecast.** A load transfer is a switching operation that moves part of one Primary
substation's demand onto neighbouring substations for hours or days. Our best detector (the
refit-offset neighbour-count detector, RO-NC, called XL in experiments 70 to 82) found 0.770 of the
moved MW on injected transfers ([experiment 82](../../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md);
98.33% interval 0.665 to 0.857). On the 35 real transfers lasting over 1 day that a language model
labelled as visible in the power readings, it found 12 (0.34; 95% interval 0.21 to 0.51), and it
found none of the 31 shorter ones ([experiment 101](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)).
Every number here is a lab result: the detector reads an ERA5-based forecast residual and needs up
to 14 days of hindsight to place the end of a transfer.

## Does it work?

**Early detectors found under a fifth of injected transfers.** In
[experiment 43](../../00043-sw2-1-detector-scoreboard/result.md), a detector that reads one
substation's residual (D0) and a detector that checks that the donor's step is balanced by its
neighbours (D1) each found under a fifth of 600 injected transfers, and no more than 0.29 in any
size group. The neighbour check helped at a 4-hour averaging window and hurt at a 24-hour window
(0.21 against 0.29). [Experiment 37](../../00037-switching-first-look/result.md) was inconclusive
on 18 real transfers.

**Most gains came from placing the end of each transfer and from using the neighbours' steps.**
Moving only the end of D0's alarm interval to the strongest opposite step in the next 14 days found
+0.191 more of the moved MW (+0.148 to +0.237, [experiment 61](../../00061-sw2-6-paired-step-intervals/result.md),
a descriptive arm). A later rule that ends the interval at the first opposite step of at least half
the onset step (called PF14) did better still ([experiments 66 and 69](../../00066-sw2-16-end-rule-as-baseline/result.md);
post hoc rule, no pre-registered verdict), and PF14 became the benchmark. Two further changes each
added about 0.05, and together added +0.114 (+0.074 to +0.182) over PF14:

- **Lifting the blind spot at the monthly refit** of the forecasting model raised recall by +0.050
  (+0.012 to +0.107) on injected transfers
  ([experiment 71](../../00071-sw2-18-refit-offset-and-gap-bridging/result.md)).
- **Counting the neighbours that step the same way as the alarming substation** raised recall by
  +0.048 (+0.021 to +0.096), but found 7 of the 35 real transfers against 8 for PF14
  ([experiment 70](../../00070-sw2-19-same-sign-neighbour-count/result.md)).
- **Using both** found 0.770 against 0.656 for PF14, about the sum of the two gains
  ([experiment 82](../../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md)).
  Neither change helps transfers lasting 15.6 to 60 days.

![RO-NC finds the most moved power of the five detectors on real transfers over 1 day](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/figure_1_recall_by_detector.svg)

**Real transfers are found much less often than injected ones.** RO-NC found 0.357 of the moved MW
on the 35 real transfers over 1 day (0.169 to 0.553), against 0.770 on injected transfers. Of the
35, 6 sit where the detector's residual panel holds no readings and 12 have a step under 4 noise
units. RO-NC found 10 of the other 17, where injected transfers of the same step size predict about
14. Every detector needs the 24 hours after a step, so no detector found any of the 31 real
transfers shorter than 1 day. With 35 events, the intervals are wide, and they cannot rank RO-NC
against the refit-offset detector alone
([experiment 101](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)).

![RO-NC finds 6 of 14 large real transfers and 2 of 10 with no metered receiver](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/figure_2_recall_by_band.svg)

**Many changes to the detector did not help.** These changes found no more of the moved MW, at the
same number of unexplained alarms, than the single-series detector:

- A bank-holiday column in the forecast: +0.011 (-0.010 to +0.044), inconclusive
  ([experiment 47](../../00047-sw2-4-better-residual-switching-detector/result.md)).
- Explicit solar and heating corrections in the forecast: +0.010 (-0.012 to +0.034), inconclusive
  ([experiment 98](../../00098-det-bg-solar-and-heating-offsets-in-switching-detector-residual/result.md)).
- A neighbour veto, including stronger vetoes: +0.008 and +0.013, inconclusive
  ([experiment 55](../../00055-sw2-8-neighbour-veto-single-series-detector/result.md),
  [58](../../00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md)).
- Combining four averaging windows (-0.050, refuted,
  [experiment 59](../../00059-sw2-13-several-windows-in-one-detector/result.md)), weighting by the
  expected daily profile (-0.000, refuted,
  [experiment 60](../../00060-sw2-7-profile-weighted-step-detector/result.md)), and removing the
  residual's half-hour-to-half-hour memory (-0.097, refuted,
  [experiment 62](../../00062-sw2-9-whitened-residual-step-detector/result.md)).
- Scoring the onset and return steps as one pair (refuted, -0.133 and -0.214,
  [experiment 61](../../00061-sw2-6-paired-step-intervals/result.md)) and one joint fit of power
  flows across all substations (refuted, 0.108 against 0.646,
  [experiment 67](../../00067-sw2-15-joint-edge-flow-estimator/result.md)).
- Refitting the forecast without detected transfers, three times over: +0.011 (+0.000 to +0.034),
  inconclusive ([experiment 68](../../00068-sw2-5-train-detect-retrain-loop/result.md)).

**What the detectors miss is mostly small steps and unmetered receivers.** On injected transfers
the recall gap between transfers with and without a metered receiver (0.50 against 0.17) mostly
disappears at equal step size, because 211 of 274 transfers without a metered receiver step by less
than the alarm threshold
([experiment 58](../../00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md)). Among the
real transfers, the 12 with a step under 4 noise units hold 10.6 of the 42.7 MW.

**Using the detector has not helped a forecast yet.** Leaving detected switching out of the 14-day
error correction made the forecast worse (error up 57%, 0.4% to 134%,
[experiment 38](../../00038-switching-mask-value/result.md), refuted). In the year scored, a
perfect detector would lower the best guess's quantile score by 0.42% (-0.47% to 1.96%), because
switching in the reliable band touches 3 of 16 Primary series
([experiment 44](../../00044-sw2-2-switching-share-of-best-guess-score/result.md), refuted).
Giving the forecast corrected lags and MW columns from the detector's labels did not lower the
error on rows with a lag inside a logged event: +0.48 points of capacity
(-0.06 to +1.72, [experiment 86](../../00086-sw2-12-detector-labels-for-nra-forecast/result.md),
inconclusive). The labeller that supplies the real transfers is itself imperfect: it found 0.74 of
large, long injected transfers, under its 0.80 bar
([experiment 41](../../00041-sw2-0a-labeller-gate/result.md)).

## Best design so far

**RO-NC, the benchmark PF14 with two changes.** The recipe:

1. Subtract an XGBoost forecast that uses no past power from each Primary substation's power, and
   refit it monthly.
2. Take the 24-hour step of that residual from whichever of two models, refitted half a month
   apart, did not refit within 24 hours of the stamp.
3. Score each candidate alarm with a logistic regression on its own step, the largest opposite
   neighbour step, the number of neighbours, and the number of neighbours that step the same way.
4. Set the threshold to allow 80 unexplained alarms on odd months, and end each interval at the
   first opposite step of at least half the onset step.

Experiments 71, 70, and 82 tested the parts. Experiment 71 found that the second model adds nothing
over simply lifting the suppression on injected transfers (-0.005), though it found all 3 real
transfers that start at a refit against 2.

## What to implement in production

**Nothing is ready for the serving path.** The detector reads ERA5 weather and needs up to 14 days
of hindsight, so it can only run retrospectively. Experiments 70, 71, and 82 are marked
`candidate` as detectors. Experiments 38 and 86 found no forecast benefit from using one. The
evidence supports building RO-NC, if at all, as a training-time and event-table tool, in line with
the [switching events](../../../roadmap/switching-events.md) roadmap page.

## Gotchas and implementation details

Each gotcha names the experiments that taught it and what to check to see whether it still applies.

- **Count events, not half-hours, and show every event.** 99.9% of the confirmed share of the
  stack's threshold-weighted score lay on two series, almost all in one 370-hour closure
  ([experiment 37](../../00037-switching-first-look/result.md)), and January 2026 held 95% to 105%
  of every masking gain ([experiment 38](../../00038-switching-mask-value/result.md)). Without
  per-event results, a pooled half-hour total reads as evidence from many closures when it comes
  from one. Check again if the scored period is longer than the year used there, or the series are
  not the 16 Primary substations. - **Match the unexplained-alarm count on every month a rule
  reads.** A threshold matched on odd months left even months 15% under round 0's alarm count
  ([experiment 68](../../00068-sw2-5-train-detect-retrain-loop/result.md)), and a shared threshold
  grid left D0 at 0.14 alarms per series-month and D1 at 0.38 ([experiment
  37](../../00037-switching-first-look/result.md)). Without matching, a detector looks better by
  raising more alarms. Check again if the detector's alarm rate depends on a different residual
  model or period. - **Spread injected transfers across series and time before training on them.**
  Experiment 58's 620 even-month transfers shrank to 103 once no two on one series lay within 1 day
  ([experiment 68](../../00068-sw2-5-train-detect-retrain-loop/result.md)). Without spreading, the
  training set holds a few overlapping events counted many times. Check again if the injection
  sampler changes. - **Score on injection shapes the detector does not assume, on bins of step size,
  and on the real transfers.** Recall is a function of the donor step in noise units ([experiment
  58](../../00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md)), a detector built on the
  injection's own shape is flattered ([experiment
  60](../../00060-sw2-7-profile-weighted-step-detector/result.md)), and injected recall of 0.770
  became 0.34 on real transfers ([experiment
  101](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)). Without these,
  a gain on injected transfers is read as a gain on real ones. Check again if the real labelled set
  grows beyond the 35 transfers over 1 day. - **The end of an interval is the hard part, and it
  needs hindsight.** The strongest-opposite-step end rule needs up to 14 days of hindsight
  ([experiment 65](../../00065-sw2-14-combined-single-series-and-neighbour-score/result.md)), and an
  interval that ended on the first failing 24-hour block, or started at a return step, produced a
  60-day mask ([experiment 38](../../00038-switching-mask-value/result.md)). Without this, a
  detector scored with hindsight ends is mistaken for one that could run live. Check again if a
  live-capable end rule is tried. - **Recompute `moved_share_of_receiver` before scoring against the
  experiment 41 labels.** 62 of 81 stored values are NaN, partly because series ids were written as
  strings ([experiment 44](../../00044-sw2-2-switching-share-of-best-guess-score/result.md)).
  Without the recomputation, a filter on that column silently drops most events. Check again if the
  labelled set is rebuilt.

## What we have not settled

- Whether any detector finds the real transfers with a step under 4 noise units, or those shorter
  than 1 day.
- How RO-NC does on real transfers once the readings missing from two Primary substations are
  restored (6 of the 35 events).
- Whether the detector's labels improve a forecast of the normal running arrangement. The 82 logged
  events in experiment 86 are too few to show it.

The [Research plan](plan.md) lists what is planned next, and the [Experiments](experiments.md) page
lists every experiment.
