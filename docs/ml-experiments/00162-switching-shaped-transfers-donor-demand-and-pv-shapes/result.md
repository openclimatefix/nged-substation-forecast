---
id: "00162"
slug: switching-shaped-transfers-donor-demand-and-pv-shapes
title: >-
  Do load transfers follow the donor's demand and solar shapes, and does a constant-MW injection
  flatter the switching detectors?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private tables and arrays under
  data/ml-experiments/00162-switching-shaped-transfers-donor-demand-and-pv-shapes/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible design (how
  the demand and solar terms are split, how a shaped transfer is matched to a constant one), so the
  matching rule and the arms are pre-registered. Callers: this folder's scripts only; they copy the
  modules of experiments 130, 147, and 156 they need and read their stored panels, worlds, and
  intervals without changing them.
hypothesis: >-
  A load transfer moves houses, factories, and solar sites, not a constant number of megawatts, so
  the power it moves follows the donor's demand and solar shapes. Part 1: on the logged real
  transfers, a donor residual during the transfer is fitted better by a daily shape, -(a x the
  donor's demand term + b x its solar term) from the convex model (GAM), than by a constant offset,
  and the fit separates a from b only for transfers that last several days and contain both sunny
  and dull days. Part 2: an injection whose moved power follows the donor's solar term vanishes at
  night and is small at midday in winter, so the half-size-reversal end rule and the step threshold
  of the detectors misfire on it, and a constant-MW injection (and, less so, today's injection,
  which moves a share of the donor's net power) flatters the detectors.
prediction: >-
  Written before any result. Part 1: for the logged transfers over 1 day, the shape model
  (constant plus a x demand term plus b x solar term, three parameters) cuts the residual sum of
  squares inside the interval by a median of 10% to 30% relative to a constant, measured on held-out
  alternate days, and a is positive (interval excluding zero) at more than half of the donors; b is
  identified (95% interval excluding zero) only at transfers of 3 days or more with both bright and
  dull days. A pure solar transfer and a pure demand transfer injected into quiet stretches can be
  told apart (the fit puts more than 70% of the moved energy in the correct term) from 3 days of
  both sunny and dull weather onward, and not below 1 day. Part 2: relative to a constant-MW
  injection of the same mean megawatts, a solar-shaped injection lowers the pair alarm's
  MW-weighted recall by 0.05 to 0.20 and lengthens the median lag from step to alarm by 2 to 12
  hours; a demand-shaped injection changes recall by less than 0.05; the end rule (the first
  half-size reversal) ends solar-shaped intervals early on 20% to 50% of the transfers; and the
  mask forecast scores 0.05 to 0.30 points of capacity worse on solar-shaped transfers than on
  constant ones.
decision_rule: >-
  Pre-registered before any fit. Models and tests are given in the plan section "Design"; the
  contrasts are 95% intervals with whole calendar months (part 2) or whole transfers (part 1)
  resampled 20,000 times, and the nine contrasts below share a Bonferroni level of 0.05/9 (99.4%).
  P1: median over the logged real transfers over 1 day of the held-out reduction in residual sum of
  squares of the shape model against the constant, 99.4% interval; `supported` when the lower end is
  above 0. P2: share of donors whose a has a 95% interval above zero, tested against one half.
  P3: on 200 positive-control injections into quiet real stretches, the share of moved energy the
  fit places in the right term, by duration (under 1 day, 1 to 3 days, over 3 days) and sunny share;
  reported, not tested. P4 to P6, on the same injected transfers in the 44 worlds of experiment 141
  (even months), difference in MW-weighted recall of the pair alarm (4-hour window, k of
  experiment 156), RO-NC's recall, and the median onset-known lag: solar-shaped minus constant.
  P7: the share of true interval length covered by the detected interval after the end rule:
  solar-shaped minus constant. P8: mask forecast mean quantile score on rows after onset: MASK-UNION
  on solar-shaped minus constant. P9: the same for the demand-shaped arm. The verdict is `supported`
  when P1 and at least two of P4, P5, P7, and P8 show the solar-shaped arm worse than constant with
  the interval excluding zero; `partly supported` when P1 is supported and fewer than two of those
  four are; `not supported` when P1 is not supported and no part-2 contrast excludes zero;
  `inconclusive` when fewer than 20 logged transfers over 1 day have usable terms. No threshold for
  porting is set, because the experiment changes no production step: its use is deciding whether
  the earlier constant-MW and share-of-net-power results can be trusted.
benchmark: >-
  The constant-MW arm (the same events, with the mean megawatts of the share-of-net-power
  injection of experiment 156 spread evenly over the interval) and, for the forecast, the
  weather-model-only arm and the true normal-running-arrangement series of experiment 151.
row_set_id: >-
  fold_a_daily_gap_v1 rows of experiment 141 (09:00 UTC initialisation, leads 0.5 to 168 h, six
  decision substations, the 44 injected worlds of experiment 141, rows outside the hand-filtered
  switching events), as in experiments 151 and 156; not comparable with the leaderboard
metric: >-
  Held-out reduction in residual sum of squares; fitted a and b with intervals; share of moved
  energy placed in the right term; MW-weighted recall; median onset-known lag (hours); share of the
  true interval covered; mean quantile score in points of effective capacity.
code: ml-experiments/00162-switching-shaped-transfers-donor-demand-and-pv-shapes/
verdict: null
production_status: not-for-production
adopted_in: none
---

**Plan.** The question comes from the switching roadmap's alpha idea
([Approach 4](../../roadmap/switching-events.md)): a switch moves houses, factories, and rooftop
solar sites, so the power exchanged between two substations follows the donor's demand shape and
solar shape, not a constant. The experiment has two parts.

**Part 1 fits the logged real transfers.** For each of the 35 logged transfers longer than 1 day,
and for each donor and receiver, the residual of the convex model
([experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md) arm B) inside the
interval is fitted by three models: a constant offset; a pure daily shape (a x the donor's demand
term plus b x its solar term); and a constant plus the shape. The convex model's terms come from
`studies.dp_gam.predict_gam` with the `columns` mask, which returns the sum of chosen terms; the
demand term is every column except the capacity columns, and the solar term is the capacity
columns. Known-answer checks come first: the two terms must sum to the stored forecast, and 200
shaped transfers injected into quiet stretches of the real residual (known a and b) must be
recovered, which also maps which durations and sunny/dull mixes identify a and b. A placebo fits
the same model at random quiet intervals to give the null distribution of the fit improvement.
The experiment 147 detected intervals are fitted the same way. A second check fits each receiver's
opposite step with the donor's terms and with the receiver's own terms.

**Part 2 re-scores the detectors on shaped injections.** The events of the 44 worlds of
[experiment 141](../00141-switching-detector-fed-into-gam-k/result.md) are re-injected with
four moved-power rules that share each event's mean megawatts: a constant; today's rule (a drifting
share of the donor's net power, which follows demand and net solar already); the donor's demand
term only; and the donor's solar term only (a 50/50 mix is added if time allows). The pair alarm
of [experiment 156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md), RO-NC,
the end rule, and the mask forecast are re-run on each. The onset alarm and RO-NC run on the fast
path (residual plus moved megawatts), as in experiment 156.

**Existing injections are already partly shaped.** Experiments 43 to 156 moved a drifting share of
the donor's net power, which follows the donor's daily shape (demand minus solar) but not the
solar term alone. The "constant-MW injector" in the brief is therefore a new reference arm, not the
current injector.

**Past-weather caveat.** The convex-model terms use ERA5 weather (past weather), as in experiments
130 and 156, and ERA5 arrives about 5 days late, so the shape terms in part 1 are a lab ceiling for
a live service. Part 2 uses the same ERA5-based residual panel as experiment 156.

**Size.** What gets stored: private files only. Serving path: not touched. Degradation rule: not
touched. Designs: several (matching rule, term split), pre-registered above. Callers: the folder's
scripts only. Medium: one Sonnet code review, one Sonnet write-up review, one Sonnet prose review.

**Limits.** Four cores, one thread each, under `systemd-run` with `MemoryMax=16G`. Series, worlds,
and arms run as independent jobs.
