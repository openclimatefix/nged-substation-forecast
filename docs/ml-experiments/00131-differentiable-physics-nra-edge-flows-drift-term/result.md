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
  MW-days a year on each unconnected pair. We expect the placebo reading to be the hardest reading
  to pass (drift in the residuals is 23 to 38 MW-days a year per pair, step 0), and put the
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

**The plan tests one change to experiment 116: a drift offset for each substation.** This page
is the plan, committed before any fit. [Experiment
116](../00116-differentiable-physics-nra-edge-flows-between-primaries/result.md) fitted flows on the
interconnections between Primary substations (the normal running arrangement, NRA, is the layout
without those transfers). Its flow fit found injected transfers only at settings that also put flow
on pairs of substations with no interconnection. A drift audit on 116's residuals (exploratory, no
fit) found that slow drift alone gives 23 and 38 MW-days a year (megawatts of flow summed over the
days of a year) on the two tuning placebo pairs. The drift is mostly shared by both substations of a
pair, not opposite-signed, so this experiment adds a per-substation offset, not a pair term.

### Question and method

**Question.** With a per-substation drift term, can a flow fit recover injected transfers that last
hours to days without putting flow on unconnected pairs?

**Model.** The objective is 116's: the Huber loss (quadratic near zero, linear beyond 1.5 noise
scales) on the residuals of the frozen convex model, plus a step penalty and a level penalty on each
flow. This experiment adds an offset `d_i(t)` for each substation. The offset is a random walk. Its
change over one day has variance `q_d`, in points of effective capacity squared (a point is 1% of
the substation's effective capacity), so the variance per half-hour row is `q_d` divided by 48. The
fit alternates between flows and offsets until both settle (`studies.dp_edge_flow.fit_flows`, drift
term in pull request 1431). The residuals are the arm-A0 residuals of experiment 116, so the drift
term is the only change. Only half-hourly flows are used.

**A big event moves a substation by at least a set number of megawatts.** The threshold is 10% of
the receiving substation's 99th-percentile cleaned demand, taken from rows of the tuning year's cut
that hold a reading and fall outside hand-filtered events. A transfer is big when its mean moved
megawatts reach the threshold; every injected transfer lasts at least 3 hours. The threshold is 0.56
to 1.1 MW at the four gate receivers, is frozen at the tuning cut for every year, and is set before
any gate reading (`e00131_threshold.py`). In set `T`, injected transfers of 25% and 40% of the
receiving load clear the threshold in most cases, and transfers of 0.5 MW and 10% rarely do. The
script also prints 5 hand-filtered events, picked by a fixed random seed, beside the threshold, and
the page reports that comparison. Events below the threshold are not looked for.

**Injected transfers.** The code is 116's, with new seeds. Sets `T` (tuning) and `G` (gate) hold 400
transfers each, balanced over 4 sizes (0.5 MW; 10%, 25%, and 40% of the receiving load) and 2
durations. The durations are 3 to 24 hours and 24 to 72 hours, log-uniform within each band. The
split into two equal-count bands departs from a single log-uniform draw over 3 to 72 hours, because
the gate reads the two bands separately. Set `L` holds 200 transfers of 25% and 40% that last 7 to
30 days. The two gate edges are D5–D16 and D10–D11. Each transfer is fitted on a window that starts
7 days before it and ends 7 days after its 48-hour post-event rows, so the drift term has a week of
context on each side. Windows are shorter than the 72-day placebo tiles below, and a year-long fit
may behave differently from either; the page says so.

**Placebos.** The placebo pairs are 116's pairs without an interconnection (tuning D5–D10 and
D16–D14, replicate D5–D11 and D16–D10). Each is fitted in 72-day tiles over the year (the last tile
is shorter, and the total is scaled by rows covered to a year), with the placebo path listed first
and last. The larger flow counts, because the zero placebo reading of 116 came from the path order.
Placebo tiles contain whatever real events fall in them, and injected windows exclude events within
48 hours.

### Tuning (tuning year only)

**Stage 1 screens 60 settings.** The grid is step constant {3, 10, 30, 100, 300} × level constant
{0.3, 1.0} × drift variance `q_d` in {none, 0.0001, 0.0004, 0.0016, 0.0064, 0.0256} points² per day,
with a Huber threshold of 1.5 noise scales. The five positive values of `q_d` give a standard
deviation per day of 0.01 to 0.16 points. The value "none" is the benchmark, the same grid without a
drift term, and is not one of the five tuned values. A level constant of 0 is left out because the
minimiser is then not unique. Each setting is scored on the big transfers of set `T` and on 4 evenly
spaced 72-day placebo tiles, scaled to a year.

**Stage 2 reads the finalists on the whole year.** Settings under 0.05 MW-days a year on both
placebo pairs in both path orders at stage 1 are ranked by their margin, the smaller of (recall for
3 to 24 hours ÷ 0.5) and (recall for 24 to 72 hours ÷ 0.77). Ties go to the larger step constant.
The top 3 are re-read on every placebo tile of the year. Among those still under 0.05, the setting
with the largest margin is chosen. The Huber threshold of the chosen setting is then compared at
1.0, 1.5, and 3.0 noise scales by the same rule. If no setting is under 0.05 at stage 1, the setting
with the smallest placebo flow is chosen, the gate is read anyway, and the verdict is `inconclusive`.
The same rule runs separately for "none", which gives the benchmark's best setting. The chosen
settings are written to `decisions.json` before the gate is read.

### Decision rule

**The gate passes only if all four readings pass in the tuning year and in the replicate year.** It
is read on the big transfers of set `G`, drawn in each year, with the settings frozen. Only the
replicate year is independent of tuning: the tuning-year placebo reading repeats the stage-2
computation on the same tiles, and the tuning-year recall readings use a different seed on the same
residuals. The scored year (2025-26) is read only for C2 and C3, once, and only if the gate passes.
The thresholds 0.77, 0.5, 0.05, and 0.8 to 1.25 are experiment 116's, kept by the brief for this
experiment so that the two experiments' gates are comparable.

1. **Moved-MW ratio.** The median over big transfers of the fitted mean flow over the true mean flow
   on the transfer's edge, over the transfer's rows and signed towards the receiver, lies between
   0.8 and 1.25.
2. **Recall.** Recall is the sum over rows of the smaller of the fitted and true flow (positive in
   the true direction), over the sum of the true flow, pooled over transfers. It is at least 0.77
   for transfers of 24 to 72 hours and at least 0.5 for transfers of 3 to 24 hours.
3. **Placebo.** Under 0.05 MW-days a year on each placebo pair of the year, with the placebo path
   first and last.
4. **Release.** A big transfer is found when its hindsight recall is at least 0.5. At least half of
   the found transfers must have a causal flow under 20% of the moved MW at the larger of 6 hours
   and 10% of the duration after the end. The causal flow comes from refitting, with the drift term,
   the trailing 30 days of residuals up to 3.5 hours before the deadline. The reading needs at
   least 30 found transfers; fewer counts as a fail. The count is reported.

**Not read as a verdict:** the 7 to 30 day arm (set `L`, read in the tuning and replicate years; it
reports whether the drift term absorbs long transfers), the real-event comparison, the benchmark,
and any variant chosen after reading a result (labelled post hoc). Flow on unconnected paths inside
the injected windows is not read.

**If the gate fails on any reading,** the verdict is `inconclusive`, and the take-home says whether
the drift is separable from the transfers at this residual quality. Post hoc variants may be
explored, labelled as such, and play no part in the verdict. If every reading fails, the session
stops and asks the maintainer before any audit.

**If the gate passes,** C2 and C3 run as in experiment 116, with the winter and cold-night
breakdowns. C2 is non-inferior when the upper end of the 98.33% interval for the change in
day-ahead quantile score outside events (training without the found flows, minus training on every
row) is under 0.05 points. C3 improves when the causal flows lower the quantile score on injected
rows and the 48 hours after them by at least 0.5 points. Both met: `supported`. One met: `partly
supported`. Neither: `not supported`. No threshold changes after reading.

**Descriptive, no verdict.** The hindsight flows of the whole decision graph are compared with the
31 logged pairs and the hand-filtered events (recall and precision, pairs named by D-label only). If
experiment 130 has reported by then, the injected transfers and real events that its detector misses
are listed with this fit's recall on them. The residuals' noise scales come from rows before the
cut, where the frozen model was trained, so they understate the out-of-sample scale; 116 shares
this property.

### Reviews, compute, and earlier advice

A Sonnet reviewer read this plan before any fit. A Sonnet code review covers the module change
(pull request 1431) and the scripts, and a Sonnet numbers review and a Sonnet prose review cover the
finished page. There is no Opus review. The fits use CPU only, at most 4 workers while the machine
is busy, with memory capped at 12 GB. The plan follows the advice of experiment 116: read the
placebo with the path first and last, and fit a drift term before tuning the penalties. Any new
data-processing finding goes into `docs/ml-experiments/data-processing.md` with the write-up.
