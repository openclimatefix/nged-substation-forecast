---
id: '00265'
slug: dp-gas-engine-bands-prior-and-visibility-switch
title: >-
  Can an outside prior and a causal "engine visible" switch make D8's widened forecast bands honest
  in every quarter, and do they help or harm the other Primaries?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
size: >-
  medium, answering the five triggers: stored, private prediction files under
  data/ml-experiments/00265-dp-gas-engine-bands-prior-and-visibility-switch/ only; serving path not
  touched; degradation rule not touched (the switch is scored, not installed); more than one design
  is defensible (the shape of the prior, its shrinkage, the switch thresholds), so the choices are
  fixed below before any score is read; callers are this folder's scripts, which read experiment
  191's stored forecasts. Sonnet only, as the maintainer's brief requires.
hypothesis: >-
  Widening a Primary's forecast bands from its own past residuals ([experiment
  264](../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md), arm 2b) is dishonest in the first
  quarter, when there is no history, and after a change in the site's behaviour. A prior taken from
  the other Primaries' residuals, shrunk towards the site's own trailing residuals as they accrue,
  fixes the first quarter. A causal switch on the share of large drops in the trailing 30 days
  returns the widening to normal when D8's engine becomes visible in the readings (around April
  2026). Together they hold D8's q05-to-q95 coverage within 5 points of 0.90 in every quarter, at no
  cost in quantile score, and without harming Primaries that have no engine.
prediction: >-
  Written before any new score is read. The descriptive table of 264 (reference coverage by month at
  D8) was read, so P1 and P3 are not blind. (P1) The control (264's arm 2b, own history only) holds
  0.80 coverage in the first quarter and fails the rule there. Arm 2 (prior) lifts the first-quarter
  coverage to at least 0.85, because D8's reference is too narrow by a factor that the other
  Primaries' residuals do not show, so the prior cannot be too narrow by more than the pooled
  difference; if the pooled prior is narrower than D8's need the first quarter stays under 0.85 and
  P1 fails. (P2) Arm 2 does not worsen the quantile score at D8 by more than 0.5% against the control.
  (P3) The switch at on 7% and off 4% turns off at D8 once, between 2026-04-01 and 2026-05-15, and
  never turns on at any other Primary. (P4) Arm 3 holds coverage in [0.85, 0.95] in 2026Q2 and lowers the
  D8 quantile score against arm 2 by under 1%. (P5) At least one Primary
  besides D8 has a heavy lower tail in the census. (P6) Arms 2 and 3 do not worsen the quantile
  score of any Primary without a heavy tail by more than +0.5% (upper end of the 95% interval).
decision_rule: >-
  Exploratory: 264's scored year chose the switch thresholds and the 12 months are the only year of
  forecasts, so every verdict is a fit-to-the-year check. Reference forecast: experiment 191's arm
  a0 on `fold_a_daily_gap_v1`, seed 0, scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours,
  outside the hand-filtered events, the same rows as 264. Control: 264's arm 2b, rebuilt here by
  the same method. Pass rule (the maintainer's brief), judged for a candidate arm at D8: (a) q05-to-q95
  coverage within 5 points of nominal, so in [0.85, 0.95], in every quarter of the scored year,
  counting the first quarter in full (July included) and the quarters after April 2026; (b) quantile
  score at D8 at most 0.5% above the control's, on the point estimate; (c) no placebo worsens: at
  each Primary that the census does not call heavy, the quantile score against the control rises by
  at most +0.1% on the point estimate and at most +0.5% at the upper end of the 95% month-block
  interval. Verdicts: `supported` when arm 2 or arm 3 passes (a), (b) and (c); `partly supported`
  when a candidate passes (b) and (c) and improves on the control's coverage in the first quarter
  but misses (a) elsewhere; `not supported` otherwise. The threshold in (b) is zero-cost in
  principle: the prior is a stored table of ratios and the switch is a 30-day share, so both add
  one pipeline step, and a score loss above 0.5% would not repay it. The 95% interval gates (c).
benchmark: >-
  The control (264's arm 2b) at the same sites, rows and leads; the unwidened reference (arm 1) is
  reported for context. Experiment 251's best-guess stack does not cover D8.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  q05-to-q95 coverage by quarter, tail coverage and pinball loss at q01, q05, q95 and q99, mean
  absolute error of the mean forecast, and the curtailment stand-in described under Arm 5.
code: ml-experiments/00265-dp-gas-engine-bands-prior-and-visibility-switch/
date: 2026-10-23
---

**Plan only: this page is the pre-registered plan, and results replace it when the experiment is
finished.**

We ask whether the widening of [experiment
264](../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md) can be made honest in every quarter.
D8 is a primary substation whose readings appear to include a 6 MW gas engine that cannot be
detected. In 264, widening the forecast band from the residuals already seen (a residual is the
reading minus the forecast median) lifted the share of readings inside the q05-to-q95 band from
0.42 to between 0.80 and 0.87, but the first quarter had only weeks of history and no arm held the
share within 5 points of 0.90 in every quarter.

## Arms

All arms use the same rows as 264 and the same as-of rule: a residual enters a fit only if its
valid time is at least 6 hours before the fit, fits repeat every 7 daily origins, and the window is
the trailing 365 days.

1. **Control.** 264's arm 2b: widen both tails by cell quantiles (six-hour block by season) of the
   ratio of the residual to the band's half-width, from the site's own window, applied from 28 days
   of history. Reproduced here, and checked against 264's printed D8 quantile score of 9.14.
2. **Arm 2, prior.** The same cells, but each cell's quantile is the weighted mean `w * own +
   (1 - w) * prior` with `w = n / (n + K)`, where `n` is the cell's count of own resolved rows and
   `K = 2,000` rows (about 6 weeks). With no own rows the table is the prior, so the widening works
   from the first origin. The prior is the pooled cell quantiles of the ratios of the other five Primaries
   (from D5, D8, D10, D11, D14, and D16, leaving the site itself out), over the whole scored year. This is a
   lab shortcut: a live service would take the prior from earlier years of the other Primaries.
   Arm 2c is the same with the pool restricted as-of to the fit window (falling back to the
   whole-year prior until 28 days exist), which shows how much the hindsight in the prior matters.
3. **Arm 3, visibility switch.** Arm 2, but the own rows in the fit window are only those taken while
   the switch had the same state as at the origin. The switch is on when 7% or more of the last 30
   days' resolved half-hours had a residual below -25 points (-2.5 MW), and off when the share falls
   below 4% (hysteresis). When the switch is off and the site has few off-state rows, the widening is
   the prior, which is the "normal" widening. The 7% and 4% settings come from 264's sweep, which
   found that 5% and 3% never turn off, so they are post hoc. The 5% and 3% settings are run as a
   sensitivity check. Flip dates are reported at both.
4. **Census, then application.** For all six Primaries with stored forecasts the census prints, by
   quarter, the share of readings below the reference's q05, the share above q95, and the share of
   residuals below -25 points. A site has a **heavy lower tail** when the share below q05 is 0.15 or
   more in at least two of the four quarters (nominal 0.05). Arms 2 and 3 are run at every site,
   and a site that is not heavy is a placebo, which must not worsen under rule (c). The other
   heavy sites, if any, are scored by rules (a) and (b).
5. **Curtailment cost at D8.** D8 never exports (its readings run from 0 to 99 points, never
   negative), so the export breach that `studies.node_curtailment` scores cannot occur at D8. The
   stand-in treats D8's demand as the flow that must stay under a limit. The limit is `f` times the
   maximum reading in the first quarter (July to September 2025), for `f` of 0.9, 0.75 and 0.5, the
   score runs on October 2025 to June 2026 in node-only mode, with 10 MW effective capacity (0.1 MW
   per point), at the breach ratio 20, and with the manual-heuristic arm not computed. The stand-in
   exercises the upper band, not the lower band that the engine threatens, so it cannot show the
   engine's effect on a generator export. It reports the energy proxy per arm, and nothing else is
   claimed from it. The Brier stand-in of 264 (reading below the 2nd percentile) is reported beside it.
6. **Neural network for the widening factor.** Not run. The brief makes it conditional on beating
   arm 2 by a pre-registered margin, and the shrinkage already has two parameters. If arm 2 fails (a)
   in the first quarter for lack of a good prior, a small network is the next experiment.

## Resources and reviews

CPU only, 4 cores, thread caps of 1, memory capped at 16 GB. One Sonnet reviewer reads the code for
leakage and one checks the write-up against the printed report, then a prose review. An audit by a
fresh Sonnet agent runs if a trigger in the `ml-experiment` skill fires.
