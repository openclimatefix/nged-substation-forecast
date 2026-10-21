---
id: "00156"
slug: switching-faster-onset-alarm-from-opposite-steps
title: "A faster onset alarm from simultaneous opposite steps at a metered neighbour, feeding a mask forecast"
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00156-switching-faster-onset-alarm-from-opposite-steps/
  only (alarm tables, interval tables, per-row score tables). Serving path: not touched. Degradation
  rule: not touched. Designs: more than one defensible alarm rule (the step windows, the pairing
  test, the veto of same-sign steps), so the windows and the threshold grid are pre-registered and
  all three windows are run. Callers: this folder's scripts only; they copy the modules of
  experiments 147 and 151 they need and read the stored residual panels, worlds, and intervals
  without changing them.
hypothesis: >-
  RO-NC (the refit-offset neighbour-count detector) learns of a load transfer about 48 hours after
  it starts, because its step statistic needs 24 hours of data after the step and its peak test
  needs 24 hours more. A transfer between two metered substations makes opposite-signed steps of
  similar size at the two, while weather, Economy 7, and holidays make same-signed steps. An alarm
  that fires when one node steps by k noise units and a connected metered node steps the opposite
  way by at least k/2, measured over a window of 2, 4, or 6 hours, therefore learns of the transfer
  within hours instead of days, at the same rate of real false alarms as RO-NC. A mask forecast
  (readings from the alarm onset set missing until the end rule fires) driven by this alarm then
  scores better than the same mask driven by RO-NC, because the mask starts days sooner.
prediction: >-
  Written before any result. At the same false-alarm rate as RO-NC (at most 80 unexplained alarms
  in odd months), the median onset-known lag of the new alarm is 3 to 8 hours (the window plus
  scheduling), against about 48 hours for RO-NC. The share of injected transfers with a metered
  partner (coverage) is 0.6 to 0.85. The MW-weighted recall of the new alarm on all injected
  transfers is lower than RO-NC's (0.15 to 0.35 against RO-NC's 0.36 to 0.45) because the rule
  needs a metered partner and a short window has a worse signal-to-noise ratio, and the union of
  the two alarms recalls more than RO-NC alone by 0.05 to 0.15. The mask forecast driven by the
  union scores 0.1 to 0.4 points of capacity better than the mask driven by RO-NC on all rows after
  the onset (experiment 151: MASK closed 21% of the gap on those rows), and is no worse on real rows
  outside events. D8 and D9 supply 20 to 40 of the false alarms at the matched rate, as they do for
  RO-NC.
decision_rule: >-
  Pre-registered before any fit. Alarm settings: windows W of 2, 4, and 6 hours (4, 8, and 12
  half-hours); k on the grid 2.5 to 8.0 in steps of 0.25 noise units; the headline setting for each
  window is the smallest k whose real alarms in odd months that no logged closure explains number
  at most 80 (RO-NC's count). The headline window is the one with the highest MW-weighted recall on
  injected transfers that start in odd months at its matched k (tuning months); it is chosen before
  any even-month score is read. Four planned contrasts, on injected transfers that start in even
  months (the headline-band draw behind experiments 138 and 147), with whole calendar months of
  onset resampled 20,000 times. C1, the decisive detection contrast: median onset-known lag in
  hours of the headline alarm minus RO-NC, on the transfers both find, 99% interval; `supported`
  when the upper end is at most -24 hours. C2: MW-weighted recall of the headline alarm minus
  RO-NC on all injected transfers, 95% interval, reported with the coverage; it decides nothing
  alone, because the rule cannot see transfers without a metered partner. C3: MW-weighted recall of
  the union (an interval from either alarm, a duplicate merged) minus RO-NC, 95% interval. C4, the
  decisive consumer contrast: the mean quantile score of the mask forecast driven by the union
  (MASK-UNION) minus the mask driven by RO-NC (MASK-RONC), in points of capacity, on all rows with a
  forecast origin after the onset of an injected transfer, counting missed transfers, with 99%
  interval from calendar months of target time; `supported` when the upper end is at most -0.10
  points (a pipeline step to run two alarms is worth porting only for a gain of that size; 0.10
  points is about 3% of the 2.9-point gap between no correction and the true series in experiment
  151). A no-harm check, C5: MASK-UNION minus no correction on real rows outside logged transfers,
  `supported` when the upper end of the 95% interval is at most +0.05 points. The verdict is
  `supported` when C1, C4, and C5 are all supported, `partly supported` when C1 is supported and
  C4 is not, or C4 is supported and C5 is not, `not supported` when C1 is not supported and the point
  value of C4 is not negative, and `inconclusive` when fewer than 8 calendar months have scored
  rows. If no k on the grid gives at most 80 false alarms for a window, that window is `not
  supported` and the next window is the headline.
benchmark: >-
  RO-NC with the 6-hour lookahead end rule of experiment 148 on arm B (the frozen GAM residual of
  experiment 130), and for the consumer MASK-RONC and no correction (K0) of experiment 151.
row_set_id: >-
  fold_a_daily_gap_v1 rows of experiment 141 (09:00 UTC initialisation, leads 0.5 to 168 h, six
  decision substations, the 44 injected worlds of experiment 141, rows outside the hand-filtered
  switching events), as in experiment 151; not comparable with the leaderboard
metric: >-
  Median onset-known lag (hours); MW-weighted recall; coverage; count of real unexplained alarms in
  odd months; mean quantile score and absolute error of the median forecast against the true
  normal-running-arrangement series, in points of effective capacity.
code: ml-experiments/00156-switching-faster-onset-alarm-from-opposite-steps/
---

**Plan only; the write-up follows the run.**

## Design

- **Residual.** Arm B of experiment 130, the frozen GAM residual of each metered substation, on the
  16-node graph `ADJACENCY` of experiment 147. Injected worlds use `residual + moved MW` (the fast
  path), as in experiments 141 and 147.
- **Statistic.** For each window W, `level_shift` with gap 0: the mean of the W half-hours from `t`
  minus the mean of the W half-hours before `t`. Its noise unit is 1.4826 times the median absolute
  deviation over odd months of each node's own statistic at that window, so each window length
  has its own units. The alarm at `t` is known at `t + W`, when the after-window is complete, and
  uses no later data (causal).
- **Alarm rule.** At `t`, node i alarms when `|step_i| >= k` noise units and some connected
  metered node j steps the opposite way by at least `k/2` noise units (its own unit) at `t-1`, `t`,
  or `t+1`, and the partner's megawatt step lies within 0.4 to 2.5 times node i's (the balance range
  of experiment 43, so that `r_i + r_j` shows no step). The alarm is rejected when two or more
  connected metered nodes (or the only neighbour) step the same way by `k/2` units. The first
  stamp at which the rule holds, with no alarm on the series in the previous W half-hours, is the
  onset.
- **Intervals.** Each alarm becomes an interval on the alarm series and each partner series. The
  end and the stamp from which it is known come from the same machinery as RO-NC's intervals
  (experiments 147 and 148, merged on one Primary substation within 48 half-hours), so the
  onset alarm is the only difference between the two. The mask forecast treats readings from
  the onset-known stamp until the end is known (the 6-hour-lookahead lag of 60 half-hours after the
  end) as missing, exactly as MASK in experiment 151.
- **Arms.** NEW (headline window and k), RO-NC, and UNION (an interval from either, merged).
  Consumer arms: K0 (no correction), MASK-RONC, MASK-NEW, MASK-UNION, CLEAN (true series).
- **Real checks.** The 35 real transfers over a day, the count of real alarms by series with D8 and
  D9 reported separately, and the share of unexplained alarms that have a balanced opposite
  neighbour (experiment `sw_false_alarms`).
- **Weighting.** Each transfer weighted equally as well as pooled by rows, and with and without D11.
- **Compute.** CPU only, 4 cores, thread caps 1, under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`.
  Retraining: none, since the GAM-K forecasts of experiment 141 are cached.

## Checks and caveats (to fill in after the run)

- Median minus mean forecast per site, the mean-forecast error beside each contrast, and which of
  the seven audit triggers fired.
- Lab shortcuts: the residual comes from ERA5 weather and the moved megawatts are injected, as in
  experiments 141 and 151.
