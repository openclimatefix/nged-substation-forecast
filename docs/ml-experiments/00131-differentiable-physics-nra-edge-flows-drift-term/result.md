---
id: "00131"
slug: differentiable-physics-nra-edge-flows-drift-term
title: "Differentiable physics, NRA: edge flows with a per-node drift term"
part: NRA-A2
theme: differentiable-physics-network
also_relevant_to: [switching-detectors]
status: planned
size: >-
  medium. Stored: private files only. Serving path: untouched. Degradation rules: untouched. One
  defensible design (116's flow objective plus a per-node random-walk offset). Callers: this
  experiment's scripts only; the drift term is a module change in its own reviewed pull request.
hypothesis: >-
  Adding a slowly varying offset to each substation, fitted jointly with the flows on the
  interconnection edges, lets the flow fit find injected transfers that last hours to days without
  putting flow on pairs of substations that are not connected, which experiment 116's fit could not
  do.
prediction: >-
  At the setting the tuning year chooses, the fit finds at least 0.77 of the megawatt-hours of
  injected transfers of 24 to 72 hours and at least 0.5 of those of 3 to 24 hours, with under 0.05
  MW-days a year on each unconnected pair. We expect the placebo reading to be the one that is
  hardest to pass (drift in the residuals is 23 to 38 MW-days a year per pair, step 0), and put the
  chance that all four readings pass in both years at about one in three. Transfers of 7 to 30 days
  are expected to be partly absorbed by the drift term (recall under 0.5).
decision_rule: >-
  See "Decision rule". The gate passes only if all four readings pass in the tuning year and in the
  replicate year. Nothing is loosened after reading.
benchmark: >-
  Experiment 116's registered fit (no drift term) re-tuned on the same grid and the same transfers;
  experiment 116's pairs without an interconnection as placebos.
row_set_id: dp_mw_v1 (tuning 2023-24, replicate 2024-25, scored 2025-26, each frozen at 1 July)
metric: >-
  Megawatt-weighted recall of injected transfers, median moved-MW ratio, flow on unconnected pairs
  in MW-days a year, release share; then (only if the gate passes) quantile score outside events
  and on injected rows, as in experiment 116.
---

## Plan

This page is the plan, committed before any fit. It follows
[experiment 116](../00116-differentiable-physics-nra-edge-flows-between-primaries/result.md), whose
flow fit found injected transfers only at settings that also put flow on pairs of substations with
no interconnection. A drift audit on 116's residuals (step 0, exploratory) found that slow
drift alone gives 23 and 38 MW-days a year on the two tuning placebo pairs, and that the drift is
mostly shared by both substations of a pair rather than opposite-signed, so the fix tested here is
a per-node offset, not a pair term.

### Question and method

**Question.** With a per-node drift term, can a flow fit recover injected transfers that last
hours to days without putting flow on unconnected pairs?

**Model.** Experiment 116's objective (Huber loss on the residuals of the frozen convex model, a
step penalty and a level penalty on each flow) plus a per-node offset `d_i(t)`. The offset is a
random walk: its change between half-hours has variance `q_d`, given as a daily variance in points
of effective capacity squared (a point is 1% of the substation's effective capacity). The fit
alternates between flows and offsets until both settle (`studies.dp_edge_flow.fit_flows`, drift
term in pull request 1431). The residuals are experiment 116's arm-A0 residuals, so the drift term
is the only change. Only half-hourly flows are used; 116's 3-hour blocks are dropped.

**Big events.** A big event moves at least 10% of the receiving substation's 99th-percentile
cleaned demand (rows at or before the cut with a reading and outside hand-filtered events) for at
least 3 hours. The thresholds are 0.56 to 1.1 MW at the four gate receivers, set before any gate
reading (`e00131_threshold.py`). Transfers of 25% and 40% of the receiving load clear them in 80% and
96% of cases, and transfers of 0.5 MW and 10% of the load rarely do. 88% of the 66 hand-filtered
events have a labelled size above the smallest threshold. The gate and the tuning read only the
injected transfers that clear the threshold. Events below it are not looked for, and the page says
so.

**Injected transfers.** 116's injection code with new seeds. Sets `T` (tuning) and `G` (gate) hold
400 transfers each, balanced over 4 sizes (0.5 MW, and 10%, 25% and 40% of the receiving load) and 2
durations (3 to 24 hours and 24 to 72 hours, log-uniform within each band). Set `L` holds 200
transfers of 25% and 40% that last 7 to 30 days. The two gate edges are D5–D16 and D10–D11.
Each transfer is fitted on a window that starts 7 days before it and ends 7 days after its
48-hour post-event rows.

**Placebos.** 116's pairs without an interconnection (tuning D5–D10 and D16–D14, replicate D5–D11
and D16–D10), fitted in 72-day tiles over the year, with the placebo path listed first and last;
the larger flow counts, because 116's zero placebo was an artefact of the path order.

### Tuning (tuning year only)

**Stage 1, screen.** 60 settings: step constant {3, 10, 30, 100, 300} × level constant {0.3, 1.0}
× drift variance `q_d` in {none, 0.0001, 0.0004, 0.0016, 0.0064, 0.0256} points² per day (a
standard deviation per day of 0.01 to 0.16 points), Huber threshold 1.5 noise scales. A level
constant of 0 is left out because it makes the minimiser non-unique. Each setting is scored on the
big transfers of set `T` and on 4 evenly spaced 72-day placebo tiles scaled to a year. The setting
"none" is the benchmark: the same grid without a drift term.

**Stage 2, finalists.** The settings under 0.05 MW-days a year on both placebo pairs in both orders
at stage 1 are ranked by their margin, the smaller of (recall for 3 to 24 hours ÷ 0.5) and (recall
for 24 to 72 hours ÷ 0.77). The top 3 are re-read on the whole year of placebo tiles; among those
still under 0.05 the one with the largest margin is chosen. The chosen setting's Huber threshold is
then compared at 1.0 and 3.0 noise scales by the same rule. If no setting is under 0.05 at stage 1,
the one with the smallest placebo flow is chosen, and the gate is then read anyway. The same rule
runs separately for "none", to give the benchmark's own best setting. The chosen settings are
written to `decisions.json` before the gate is read.

### Decision rule

**The gate** is read on set `G` in the tuning year and again in the replicate year (a 2024-25 draw
of the same set), with the settings frozen. It passes only if all four readings pass in both
years:

1. **Moved-MW ratio:** the median over big transfers of the fitted mean flow over the true mean
   flow lies between 0.8 and 1.25.
2. **Recall:** the pooled megawatt-weighted recall is at least 0.77 for transfers of 24 to 72 hours
   and at least 0.5 for transfers of 3 to 24 hours.
3. **Placebo:** under 0.05 MW-days a year on each of 116's placebo pairs for that year, with the
   placebo path first and last.
4. **Release:** among big transfers that the hindsight fit found (recall of at least 0.5), at least
   half have a causal flow under 20% of the moved MW at the larger of 6 hours and 10% of the
   duration after the end, counting the 3.5-hour telemetry delay. The number of transfers read is
   reported.

**Not read as a verdict:** the 7 to 30 day arm (set `L`, 200 transfers, read in the tuning and
replicate years; it reports whether the drift term absorbs long transfers), the real-event
comparison, the benchmark, and any variant chosen after reading a result (labelled post hoc).

**If the gate passes,** forecast contrasts C2 (training without the found flows changes the day-ahead
quantile score outside events by under 0.05 points, non-inferiority) and C3 (causal flows improve
the score on injected rows and the 48 hours after them) run as in experiment 116, with the winter
and cold-night breakdowns. Verdicts: gate passes and C2 and C3 read, then as 116's decision rule
states; gate fails: `inconclusive`, and the take-home says whether the drift is separable from
transfers at this residual quality. If the gate fails every reading again, the session stops and
asks the maintainer before any audit. No threshold changes after reading.

**Descriptive, no verdict.** The hindsight flows of the whole decision graph are compared with the
31 logged pairs and the hand-filtered events (recall and precision, pairs named by D-label only). If
experiment 130 has reported by then, the injected transfers and real events that its detector
misses are listed with this fit's recall on them.

### Cost, compute, and reviews

CPU only, at most 4 workers while the machine is busy, memory capped at 12 GB. A Sonnet code review
covers the module change (pull request 1431) and the scripts; a Sonnet numbers review and a
Sonnet prose review cover the page. No Opus.
