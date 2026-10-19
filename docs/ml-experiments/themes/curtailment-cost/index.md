# Curtailment cost

**We can now score a demand forecast by the curtailment cost of its errors at a bulk supply point,
but the two supply points of the pilot hold too few events to rank forecasts, so no forecast has yet
been shown to curtail less.**

Curtailment is the capping of generators beneath a substation when its export would pass a limit.
The scoring code plans the caps from a forecast of the substation's export, then charges the energy
lost to the caps plus a penalty for export that stays over the limit. Both experiments of the theme
are worked examples of that code, and neither had a registered prediction. Experiment 84 ran it on
2 [bulk supply points](../../glossary.md#bulk-supply-point-and-grid-supply-point), Boston and
Skegness, and experiment 83 scored the tails of the forecasts of 20 demand series.

![Half-hours beyond each limit, by node and season](../../00084-substation-limit-curtailment-cost-worked-example/figure_1_event_census.svg)

**Boston and Skegness together have 19 half-hours beyond 0.9 times their earlier maximum export in
the scored year, and more than 100 per node only at half that maximum.** The scored year runs from
2025-07-02 to 2026-06-30, and the figure counts export and import in separate panels. Source:
[experiment 84](../../00084-substation-limit-curtailment-cost-worked-example/result.md).

## Does it work?

**The code runs and reproduces a hand-worked answer exactly.** All eight quantities of a made-up
half-hour (a 45.0 MW limit and two generators) agree with the hand calculation ([experiment
84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). The verdict of both
experiments is `inconclusive`, because neither registered a contrast.

**The pilot cannot rank forecasts at the headline limit.** At 0.9 times each node's earlier maximum
export, the cost rests on 19 half-hours, and no difference between arms is significant at the 5%
level. At 0.75 times, with 46 half-hours, the [stack](../../glossary.md#stack) costs GBP 2,440,720
more than the [best guess](../../glossary.md#best-guess) (95% interval 96,371 to 6,686,043). At 0.5
times, with 323 half-hours, the best guess costs GBP 10,802,520 less than
[climatology](../../glossary.md#climatology) (-20,880,964 to -3,342,553), and its gap to the NGED
baseline includes zero. We have no evidence that NGED operates at 0.5 times, so that limit is a
rehearsal ([experiment 84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)).
The costs are retrospective, from an energy proxy at GBP 2,000 per MWh, and are a calibration to
history, not a saving. The stack is the demand model fitted once and corrected from its own recent
errors, and it has no half-life.

![Avoidable cost per arm at three scaled limits](../../00084-substation-limit-curtailment-cost-worked-example/figure_4_avoidable_cost.svg)

**The best guess (weekly retrained) has the lowest avoidable cost of the arms at all three scaled
limits, but only some of its gaps are significant.** The figure has one panel per scaled limit at
GBP 2,000 per MWh, and shorter bars are better. Source: [experiment
84](../../00084-substation-limit-curtailment-cost-worked-example/result.md).

**Most of the best guess's cost at 0.9 times is breach, not unneeded cuts.** Of its GBP 571,823
avoidable cost, 81% is export that stayed over the limit: the forecast ordered 13.6 MWh of cuts on
event half-hours against 35.1 MWh needed. Another 20% is output cut when metered export was within
the limit. Avoidable
cost is an arm's cost minus the perfect forecast's cost ([experiment
84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)).

**A forecast that fits a whole distribution can plan far more curtailment.** Student-t and
skew-normal fits planned 16 to 24 times more curtailed energy than the stored forecast at 0.9
times, on 17 half-hours at one node, so the costs of the arms could not be separated ([experiment
97](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)).

**The stored demand forecasts of experiment 48 are too narrow in both tails.** Over 20 demand
series, every lead, and runs from 2025-07-02, coverage of the 0.05 quantile is 0.066 to 0.085 and of
the 0.95 quantile is 0.891 to 0.927, and weekly retraining has the lowest pinball loss at both
levels. The experiment gives no interval, and the 0.01 and 0.99 levels were not stored ([experiment
83](../../00083-curtailment-cost-and-tail-calibration-worked-example/result.md)). The finding does
not carry over to the later [BG2](../../glossary.md#bg2) forecast, whose q05 covers 0.035, so its
lower tail is too wide ([experiment
97](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md), Primary substations
at 12 to 36 hours).

## Best design so far

**The best of the metric designs tested here plans caps from the 0.95 quantile of the substation's
export forecast and scores at a limit scaled from the node's earlier maximum.** The generators
beneath the node cut the excess in proportion to their capacity, and the cost is the energy proxy
with a breach ratio of 20 ([experiment
84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). This is a metric
design, not a forecast design: no forecast design curtailed less with significance at the headline
limit.

## Candidates for production

**Nothing is ready.** Both experiments are `not-for-production`. The cost-savings roadmap describes
the metric and the larger delivery that would allow a ranking ([cost-savings
metrics](../../../roadmap/cost-savings-metrics.md)).

## Gotchas and implementation details

- **Score curtailment at the substation, not at a single meter.** Experiment 83 first built a
  single-series metric and withdrew it, because the limit binds at a substation ([experiment
  83](../../00083-curtailment-cost-and-tail-calibration-worked-example/result.md)). Otherwise the
  metric scores each meter against its own history, which does not measure curtailment. Re-check if
  a per-meter limit becomes known.
- **Count the events before ranking.** At 0.9 times the maximum only 19 half-hours qualify, and no
  difference is significant ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). Otherwise a ranking
  rests on a few half-hours. Re-check when the delivery of about 2,500 series is scored.
- **Score at a scaled limit, and show the published thermal limit as a reference.** Boston's export
  reached its published limit once, and the maintainer's unconfirmed understanding is that NGED
  curtails early to keep headroom ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). Look for NGED's
  answer in [issue #1318](https://github.com/openclimatefix/nged-substation-forecast/issues/1318).
- **Label every cost "calibration to history, not a saving".** The perfect forecast still costs GBP
  87,460 at 0.9 times, because the cap also removes output above a generator's estimated capacity
  ([experiment 84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)).
  Otherwise a cost reads as money NGED would have saved.
- **The NGED baseline is an assumption.** It is the [manual
  heuristic's](../../glossary.md#manual_heuristic) substation forecast with every generator at full
  capacity, from the maintainer's unconfirmed understanding of NGED's practice ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). Re-check once NGED
  describes its practice.
- **Rank under the energy proxy and under the DCUSA form separately.** Changing the price does not
  reorder the arms, but the DCUSA form charges every instructed generator its whole headroom, and
  climatology's avoidable cost becomes negative under it ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). Otherwise the
  choice of cost form decides the ranking unnoticed.
- **Report how many generator-half-hours had no forecast.** Across the six arms, 13% had none, so
  those generators were not capped and added to breach ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). Otherwise a
  forecast set with more gaps looks worse for that reason alone. Re-check with other forecast sets.
- **Report the unscored half-hours.** Scoring needs a stored forecast at 12 to 36 hours lead, which
  drops 5 of 46 events at 0.75 times, and the dropped half-hours hold 14% of the excess energy at 0.5
  times ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)).
- **Check the sign of each node.** Boston and Skegness are inferred as import-positive, Bicker Fen's
  sign is unconfirmed, and Walpole's is unknown, so the metric excludes them ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). A wrong sign swaps
  export and import. Re-check when NGED supplies the signs.
- **Store the 0.01 and 0.99 quantiles, and score tails with `studies.tail_calibration`.** The stored
  sets of experiment 48 stop at 0.95, so the tail table could not be read at 0.99 ([experiment
  83](../../00083-curtailment-cost-and-tail-calibration-worked-example/result.md)). Later sets store
  11 levels ([experiment
  97](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)).
- **Publish generator figures only as sums over at least 4 generators.** Experiment 84 reports no
  generator singly, and the table joining generators to substations stays in the private data
  folder ([experiment
  84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)).

## What we have not settled

- Whether any forecast curtails less than another with significance, once about 2,500 series are
  scored (the January 2027 delivery).
- How NGED chooses its operating limit, and how it allocates caps among generators.
- Whether the tails of the forecasts are too narrow in general: experiments 83 and 97 disagree on
  the lower tail, on different forecasts and periods.

The [Research plan](plan.md) lists what is planned next, and the [Experiments](experiments.md) page
lists every experiment.
