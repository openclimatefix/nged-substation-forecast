# Switching detectors

**The best detector so far, RO-NC, finds about three quarters of the moved power in large injected
load transfers and about a third of that in real ones, and no experiment yet shows that using a
detector improves a forecast.** A load transfer is a switching operation that moves part of one
Primary substation's demand (a Primary substation is an 11 kV or 6.6 kV substation that feeds local
circuits) onto neighbouring substations for hours or days. Injected transfers are synthetic
transfers added to the real power readings, so their start, end, and size are known. A transfer is
large when it moves at least 25% of the receiving substations' load.

RO-NC (the refit-offset neighbour-count detector, called XL in [experiment
82](../../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md)) found 0.770 of the
moved MW on large injected transfers lasting over 1 day On the 14 real transfers of that kind it
found 0.559, so that band alone shows no gap. Across all 35 real transfers lasting over 1 day it
found 0.357 of the moved MW, and it found none of the shorter ones ([experiment
101](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)).

Every number here comes from the lab. Recall is the share of the moved megawatts (MW) that sit in
found transfers. A detector finds a transfer when it raises an alarm starting within 2 hours of the
transfer's start and places the end within the larger of 6 hours and 10% of the duration. The
detector reads the residual (measured power minus the forecast of an XGBoost demand model that uses
no past power) of a model trained on ERA5 (the European Centre for Medium-Range Weather Forecasts'
fifth-generation reanalysis, an estimate of past weather). It uses hindsight, meaning data after the
event, to place the end of each transfer. Experiment 63 is tagged to this theme but not cited,
because it is a joint linear fit and not a detector.

## How well the detectors find transfers

**Early detectors found under a fifth of injected transfers.** In [experiment
43](../../00043-sw2-1-detector-scoreboard/result.md), a detector that reads one substation's
residual (D0) and a detector that checks that the donor's step is balanced by its neighbours' steps
(D1) each found under a fifth of 600 injected transfers, and no more than 0.29 in any size group.
The neighbour check helped at a 4-hour averaging window and hurt at a 24-hour window (0.21 against
0.29). In [experiment 37](../../00037-switching-first-look/result.md), D1 found 0.326 of the
switched MW on its 18 real validation transfers (95% interval 0.093 to 0.579), which is inconclusive
against the 0.35 bar.

**Placing the end of each transfer gave the first large gain.** Moving only the end of D0's alarm
interval to the strongest opposite step in the next 14 days found +0.191 more of the moved MW
(+0.148 to +0.237; [experiment 61](../../00061-sw2-6-paired-step-intervals/result.md), a descriptive
arm with no pre-registered verdict). A post hoc rule that ends the interval at the first opposite
step of at least half the onset step, called PF14, found 0.688 against 0.646 for the 14-day
strongest-step rule ([experiment 66](../../00066-sw2-16-end-rule-as-baseline/result.md)). PF14
became the benchmark ([experiment 69](../../00069-sw2-17-missed-onsets-of-the-end-rule/result.md)).

**Two further changes each added about 0.05, and together added +0.114 on experiment 82's draw of
injected transfers.** Each draw is a fresh set of injected transfers, so gains are comparable only
within a draw.

- **Adding a second XGBoost demand model, refitted half a month later,** raised recall by +0.050
  (+0.012 to +0.107). PF14 ignored every alarm within 24 hours of the monthly refit (the XGBoost
  demand model is refitted on the 1st of each month), a blind spot that the second model removes.
  Lifting the suppression alone gave +0.056 (+0.016 to +0.113; [experiment
  71](../../00071-sw2-18-refit-offset-and-gap-bridging/result.md); PF14 found 0.665; transfers of
  25% to 40% of the receiving load).
- **Counting the neighbours that step the same way as the alarming substation** raised recall by
  +0.048 (+0.021 to +0.096), but found 7 of the 35 real transfers against 8 for PF14 ([experiment
  70](../../00070-sw2-19-same-sign-neighbour-count/result.md); PF14 found 0.665; transfers of 25% or
  more).
- **Using both** found 0.770 against 0.656 for PF14, a gain of +0.114 (+0.074 to +0.182), where the
  parts alone gained +0.056 and +0.050 (transfers of 25% or more; the bands differ between the
  experiments) ([experiment
  82](../../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md)). None of the three
  helps transfers lasting 15.6 to 60 days.

Intervals above are the ones each experiment reports, from 95% to 99.17%.

![RO-NC and X find the most moved power of the five detectors on real transfers over 1 day](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/figure_1_recall_by_detector.svg)

**Real transfers are found less often than injected ones, partly for reasons outside the detector.**
RO-NC found 0.770 on injected transfers (98.33% interval 0.665 to 0.857), 0.559 (0.203 to 0.816) on
the 14 large real transfers, and 12 of the 35 real transfers over 1 day (0.34; 0.21 to 0.51) with
0.357 of their MW (0.169 to 0.553). Of the 35 real transfers over 1 day, which total 42.7 MW, 6 sit
where the detector's residual panel (the table of residuals for all 16 Primary series) holds no
readings, and 12 have a step under 4 noise units. A noise unit is the usual size of the detector's
24-hour step on odd-numbered months. RO-NC found 10 of the other 17, where injected transfers of the
same step size predict about 14. Every detector needs the 24 hours after a step, so none found any
of the 31 real transfers shorter than 1 day. With 35 events the intervals are wide, and the events
cannot rank RO-NC (0.357) against X (0.350), the refit-offset detector alone ([experiment
101](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)).

![RO-NC finds 6 of 14 large real transfers and 2 of 10 with no metered receiver](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/figure_2_recall_by_band.svg)

**Many changes to the detector found no clear gain.** Each change below was tested against a
detector without it, at the same number of unexplained alarms. An unexplained alarm is one more than
24 hours from any closure in the switching log of National Grid Electricity Distribution (NGED).
Refuted means the experiment's rule ruled the gain out.

- **Refuted:** four averaging windows in one detector (-0.050; [experiment
  59](../../00059-sw2-13-several-windows-in-one-detector/result.md)); weighting each half-hour by
  the expected daily profile (-0.000; [experiment
  60](../../00060-sw2-7-profile-weighted-step-detector/result.md)); removing the residual's
  half-hour-to-half-hour memory (-0.097; [experiment
  62](../../00062-sw2-9-whitened-residual-step-detector/result.md)); scoring the onset and return
  steps as one pair (-0.133 and -0.214; [experiment
  61](../../00061-sw2-6-paired-step-intervals/result.md)); one joint fit of power flows across all
  substations (0.108 against 0.646; [experiment
  67](../../00067-sw2-15-joint-edge-flow-estimator/result.md)).
- **Inconclusive or partly supported:** a bank-holiday column in the demand model (+0.011, -0.010 to
  +0.044; [experiment 47](../../00047-sw2-4-better-residual-switching-detector/result.md)); a
  neighbour veto (+0.008, and +0.013 for a stronger veto; [experiment
  55](../../00055-sw2-8-neighbour-veto-single-series-detector/result.md) and [experiment
  58](../../00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md)); refitting the demand model
  three times without detected transfers (+0.011, +0.000 to +0.034; [experiment
  68](../../00068-sw2-5-train-detect-retrain-loop/result.md)); explicit solar and heating
  corrections, measured against RO-NC (+0.010, -0.012 to +0.034; [experiment
  98](../../00098-det-bg-solar-and-heating-offsets-in-switching-detector-residual/result.md));
  accepting weaker onset alarms that a step back confirms (+0.007, -0.014 to +0.040; the gain of
  +0.146 was cancelled by the stricter threshold that held the alarm count, -0.140; [experiment
  69](../../00069-sw2-17-missed-onsets-of-the-end-rule/result.md)).
- **Also small or negative:** a 4-hour window for the end rule lost 0.147 against 24 hours (-0.248
  to -0.054; [experiment 66](../../00066-sw2-16-end-rule-as-baseline/result.md)), and combining the
  own step with only the neighbours' opposite-sign steps gained +0.019 (-0.011 to +0.062;
  [experiment 65](../../00065-sw2-14-combined-single-series-and-neighbour-score/result.md)).
  Counting the neighbours that step the same way, which experiment 65 tried post hoc, is the change
  in which experiment 70 then found a gain on injected transfers.

**What the detectors miss is mostly small steps with no metered receiver.** A metered receiver is a
receiving substation whose own power is measured. In experiment 58, 211 of 274 injected transfers
without a metered receiver step by less than the alarm threshold, and among transfers that step by
4.0 noise units or more, D0 still found only 0.56 with a metered receiver and 0.35 without
([experiment 58](../../00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md)). Before the PF14
end rule, a wrong interval end was the main miss of the single-series detector D0: among large
transfers with a metered receiver, most misses were an end outside tolerance, not a missed alarm
([experiment 59](../../00059-sw2-13-several-windows-in-one-detector/result.md), [experiment
60](../../00060-sw2-7-profile-weighted-step-detector/result.md)). For RO-NC, 0.195 of the moved MW
sits in transfers with no alarm at the start and 0.035 in transfers with a wrong end, and experiment
82 puts the remaining misses mostly among small steps with no metered receiver.

**Using a detector has not yet improved a forecast, on 16 Primary series, over one year for
experiments 44 and 86 and four years for experiment 38, with labels that use hindsight.** Positive
changes in error mean a worse forecast.

- **Experiment 38 masked D1's flags out of the 7-day error correction** of the stack (the stack is
  the project's forecast pipeline, which corrects the XGBoost demand model's error with the mean
  error of recent days). The error on the tail (the half-hours above each series' 99th percentile)
  outside logged switching rose 57% (0.4% to 134%). The rise came from an interval builder that ran
  one interval for 60 days. A post hoc builder lowered the tail error by 5.4% (-15.1% to +1.5%), and
  a mask built from NGED's logs, the oracle mask, lowered it by 8.9% (-21.8% to 0.0%), with 95% of
  that gain in one closure in January 2026 ([experiment
  38](../../00038-switching-mask-value/result.md)).
- **A perfect detector would lower the best-guess forecast's quantile score by 0.42% (-0.47% to
  1.96%)**, because switching in the reliable band touches 3 of 16 Primary series. The best guess is
  the project's current best combination of inputs for the Primary demand forecast ([experiment
  44](../../00044-sw2-2-switching-share-of-best-guess-score/result.md)).
- **Lags corrected with XL's labels did not help** the XGBoost demand model. On rows outside
  switching, the quantile score rose by 0.056 points of capacity (0.016 to 0.126). On rows with a
  lag inside a logged event, it rose by 0.48 against plain lags (-0.06 to +1.72), and oracle labels
  changed it by -0.29 (-2.06 to +0.68). The test could not show a gain from either label source.
  Experiment 86 belongs to the `switching-labels-in-the-forecast` theme ([experiment
  86](../../00086-sw2-12-detector-labels-for-nra-forecast/result.md)).

The labeller that supplies the real transfers is itself imperfect: a language model that found 0.74
of large, long injected transfers, under its 0.80 bar ([experiment
41](../../00041-sw2-0a-labeller-gate/result.md)).

## Best design so far

**RO-NC, which is PF14 with two changes.** The recipe:

1. Subtract the forecast of an XGBoost demand model that uses no past power from each Primary
   substation's power, and refit the model monthly.
2. Take the 24-hour step of that residual from whichever of two such models, refitted half a month
   apart, did not refit within 24 hours of the stamp. Simply lifting the suppression near the refit
   does as well on injected transfers (the second model changes recall by -0.005), though the second
   model found all 3 real transfers that start at a refit against 2 ([experiment
   71](../../00071-sw2-18-refit-offset-and-gap-bridging/result.md)).
3. Score each candidate alarm with a logistic regression on its own step, the largest opposite
   neighbour step, the number of neighbours, and the number of neighbours that step the same way.
4. Set the alarm threshold to allow 80 unexplained alarms over all 16 Primary series on odd months,
   and end each interval at the first opposite step of at least half the onset step.

Experiments [70](../../00070-sw2-19-same-sign-neighbour-count/result.md) and
[82](../../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md) tested the other
parts.

## What to implement in production

**Nothing is ready for the serving path.** Three experiments are marked `candidate`, and each is
training-time only for the same two reasons: the residual uses ERA5 weather, which arrives about 5
days after the event, and the end rule uses hindsight.

- **Experiment 71** (lift the suppression near the monthly refit): +0.050 on injected transfers, and
  the second model's extra effect is small.
- **Experiment 70** (same-sign neighbour count): +0.048 on injected transfers, but unconfirmed on
  real ones (7 against 8).
- **Experiment 82** (both together): +0.114 on injected transfers, and 12 against 8 on the 35 real
  transfers, a weak signal.

One option, which no experiment has tested, is to run RO-NC retrospectively to build a table of
switching events for NGED. The basis is that RO-NC is the best detector found. Against it,
experiment 44 found that a perfect detector would improve the best guess's quantile score by little,
so the table would serve NGED's own use of the events and not a forecast gain. The [switching
events](../../../roadmap/switching-events.md) roadmap page owns the decision.

## Gotchas and implementation details

Each gotcha names the experiments that taught it and what to check to see whether it still applies.

- **Count events, not half-hours, and show every event.** 99.9% of the confirmed share of the
  stack's tail score lay on two series, almost all in one 370-hour closure ([experiment
  37](../../00037-switching-first-look/result.md)), and January 2026 held 95% to 105% of every
  masking gain ([experiment 38](../../00038-switching-mask-value/result.md)). Without per-event
  results, a pooled half-hour total reads as evidence from many closures when it comes from one.
  Check again if the scored period is longer than the year used there, or the series are not the 16
  Primary substations.
- **Match the unexplained-alarm count on every month a rule reads.** A threshold matched on odd
  months left even months 15% under the first round's alarm count ([experiment
  68](../../00068-sw2-5-train-detect-retrain-loop/result.md)), and a shared threshold grid left D0
  at 0.14 alarms per series-month (alarms per series per calendar month) and D1 at 0.38 ([experiment
  37](../../00037-switching-first-look/result.md)). Without matching, a detector looks better by
  raising more alarms. Check again if the detector's alarm rate depends on a different demand model
  or period.
- **Spread injected transfers across series and time before training on them.** Experiment 58's 620
  even-month transfers shrank to 103 once no two on one series lay within 1 day ([experiment
  68](../../00068-sw2-5-train-detect-retrain-loop/result.md)). Without spreading, the training set
  holds a few overlapping events counted many times. Check again if the injection sampler changes.
- **Compare like with like when quoting injected and real recall.** 0.770 is MW-weighted recall on
  large injected transfers lasting over 1 day, and 0.34 is a count over all 35 real transfers over 1
  day, 21 of them under 25% of the receiving load. The matching real band gives 0.559 (0.203 to
  0.816) ([experiment
  101](../../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)). Without
  matching, a gap that the sample cannot show is reported as a finding. Check again if the labelled
  real set grows beyond 35 transfers over 1 day.
- **Score on several injection shapes and by donor step.** Recall is a function of the donor step in
  noise units ([experiment 58](../../00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md)),
  and experiment 60 scored five shapes of injected transfer so that a step statistic built for one
  shape could be tested on the others ([experiment
  60](../../00060-sw2-7-profile-weighted-step-detector/result.md)). Without this, a mix of small and
  large steps hides which transfers are missed. Check again if the injection builds transfers a new
  way.
- **Interval ends need hindsight, and a bad end rule dominates the misses.** The
  strongest-opposite-step end rule needs up to 14 days of hindsight ([experiment
  65](../../00065-sw2-14-combined-single-series-and-neighbour-score/result.md)), and an interval
  that ended on the first failing 24-hour block, or started at a return step, ran for 60 days
  ([experiment 38](../../00038-switching-mask-value/result.md)). In experiments 59 and 60 a wrong
  end was the main miss of the early detector, though not of RO-NC. Without allowing for this, a
  detector scored with hindsight ends is mistaken for one that could run live. Check again if an end
  rule that does not use hindsight is tried.
- **Recompute `moved_share_of_receiver` before scoring against the experiment 41 labels.** 62 of 81
  stored values are NaN, partly because series ids were written as strings ([experiment
  44](../../00044-sw2-2-switching-share-of-best-guess-score/result.md)). Without the recomputation,
  a filter on that column silently drops most events. Check again if the labelled set is rebuilt.

## What we have not settled

- Whether any detector finds the real transfers with a step under 4 noise units, or those shorter
  than 1 day.
- How RO-NC does on real transfers once the readings missing for two Primary substations are
  restored (6 of the 35 events).
- Whether the detector's labels improve a forecast of the normal running arrangement. The 82 logged
  events in experiment 86 are too few to show it.

The [Research plan](plan.md) lists what is planned next, and the [Experiments](experiments.md) page
lists every experiment.
