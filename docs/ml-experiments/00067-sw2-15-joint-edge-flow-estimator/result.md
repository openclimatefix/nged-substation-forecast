---
id: "00067"
slug: sw2-15-joint-edge-flow-estimator
status: planned
size: complex
hypothesis: >-
  A single convex fit of block-shaped load flows across every boundary between neighbouring Primary
  substations, fitted to all the residuals at once, finds more of the MW moved by injected load
  transfers than the best sequential detector (D0M, the 24-hour single-series detector with the
  strongest-opposite-step end rule), at the same count of unexplained alarms on odd months.
prediction: >-
  MW-weighted recall, even months, pooled over the injection shapes S2, S3, and S4 of experiment
  60, with the alarm budget matched at 80 unexplained odd-month alarms: D0M with the PE end rule
  reaches 0.63 to 0.65 (0.646 in experiment 65). The joint estimator (EF) reaches 0.68 to 0.74, a
  gain of +0.03 to +0.09, because conservation of power across boundaries removes common-mode
  alarms and places both ends of a transfer from the same block. The gain sits in transfers with a
  metered receiver and a donor step of 2.5 to 6 noise units. Transfers with no metered receiver
  gain nothing, because their flow lands in the slack term and the slack has no partner to check.
decision_rule: >-
  Two planned contrasts, read at 97.5% (Bonferroni across two) from 20,000 resamples of whole
  calendar months: EF minus D0M (C1) and EF minus LOGIT_SAME (C2), in MW-weighted recall over
  experiment 58's 620 headline-band transfers that start in even months, equal weight over S2, S3,
  and S4. LOGIT_SAME is experiment 65's post hoc logistic score with the count of same-sign
  neighbour steps (PE end rule), a benchmark chosen after seeing its result and labelled so. EF is
  held to at most 80 unexplained odd-month alarms, counted per series (an edge run counts once for
  each of its two series), and to no more than 10 above D0M's unexplained alarms on even months.
  EF is `supported` when C1's interval lies wholly above zero and its point value is at least
  +0.05. EF is `small gain` when C1's interval lies wholly above zero and its point value is under
  +0.05. EF is `refuted` when C1's upper end lies below +0.02. Otherwise it is `inconclusive`. C2
  is read without a verdict and says whether EF also beats the best neighbour score. The threshold
  of +0.05 is set by the cost of porting: the estimator adds a graph, a CVXPY solve per window, and
  three tuned penalty weights to a detector that has one threshold, and the estimator serves only
  training-time masks and the NGED event table, so a gain under +0.05 does not repay that.
benchmark: >-
  D0M with the PE end rule (experiment 61), 80 unexplained odd-month alarms, as in experiment 65.
  Experiment 66 (SW2-16) is still running; its winner replaces D0M if it merges before this
  experiment's write-up, and both are reported.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55,
  58, 59, 61, and 65 do, not forecasts)
metric: >-
  MW-weighted recall (headline): the share of the MW moved by injected transfers that sits in
  transfers the detector finds, where a transfer is found when an alarm interval on one of its
  series starts within 2 hours and ends within the larger of 6 hours and 10% of the duration.
  Unweighted recall is secondary, and so is the share of moved MW missed.
code: ml-experiments/00067-sw2-15-joint-edge-flow-estimator/
production_status: not-for-production
adopted_in: none
---

# Experiment 67: does one convex fit of load flows across all boundaries find more of the moved load?

*Status: planned. This page is the plan, committed before any code runs. The write-up replaces it.*

## What we ask, and how

**We ask whether fitting every boundary between neighbouring substations at once finds more
switched load than looking at each substation alone.** A load transfer moves power off a donor
Primary substation and onto its neighbours. The detectors in experiments 43 to 65 look for a step
in each substation's residual (measured minus predicted power) and then check the neighbours. The
estimator here instead assigns the residuals of all 16 substations to a flow of MW across each of
the 30 boundaries, with a slack term per substation for load moved to a neighbour that is not
metered. A flow is zero almost all the time and flat when non-zero, and the fit penalises both
non-zero flows and changes in flow. Every run of non-zero flow is one detected transfer, with its
own onset and end.

## Method

- **Network.** The 16 trial Primaries and the 30 boundaries of `ADJACENCY`. Eight of the 16 are
  metered only in MVA, a magnitude with no direction. They enter as slack-only nodes: their
  residual can be explained only by their slack, and a flow into one is seen only at its metered
  end.
- **Data.** The stored residual panel of experiment 37 (`res_mw`), averaged to hourly bins and
  divided by each series' robust standard deviation on odd months. No whitening (experiment 62).
- **Fit, per window of 70 days.** Minimise a Huber loss (threshold 3 noise units) of the residual
  minus the flows into and out of each metered node minus the slack, plus `lambda` times the
  absolute value of the flows and slack (the sparsity penalty), plus `rho * lambda` times a group
  fused penalty (the sum over time of the Euclidean norm of the one-hour changes in the flows and
  slack at each node), with the slack's level penalty multiplied by `m`. The fit is a CVXPY convex
  problem solved with Clarabel.
- **Read-out.** A run is a stretch of one flow or slack that stays above 0.25 noise units, with
  gaps of up to 3 hours bridged. Runs under 24 hours are dropped, the same duration the 24-hour
  detectors cannot see below. A run on a boundary gives an interval on both of its series; a slack
  run gives one on its own series. An interval's onset and end are the run's.
- **Injected transfers** are experiment 58's 1,900, in its batches and with its seeds, injected
  with experiment 60's shapes. For each transfer the fit uses the window from 21 days before its
  onset to 49 days after, cut from the batch's injected residual. The real-residual fit uses
  windows of 70 days at a stride of 28 days and keeps runs whose onset lies in the middle 28 days.
- **Tuning, on odd months only.** `lambda` is set so that the fit on the real residual has at most
  80 unexplained odd-month alarms, the count of D0M, where an alarm is explained when a logged
  closure touches its series within 24 hours. Then `rho` (1, 3, 10) and `m` (1.5, 3) are chosen to
  maximise MW-weighted recall on 300 odd-month headline-band transfers, one shape per transfer
  (S2, S3, or S4 by event number), at the matched `lambda`. The grid is fixed here, before any
  fit.
- **Scored on even months.** The 620 headline-band, even-month transfers under S2, S3, and S4, with
  the benchmark arms from experiment 65's code: D0M with the PE end rule and LOGIT_SAME. The
  labelled real transfers over 1 day are scored too (35 of them), and recall by donor step in
  noise units is split by metered receiver.

## Planned contrasts

- **C1:** EF minus D0M (PE), MW-weighted recall, pooled over S2, S3, S4.
- **C2:** EF minus LOGIT_SAME (PE), same pooling, read without a verdict.

## Sizing

- **What gets stored:** nothing outside the private `data/ml-experiments/00067-*` folder. No Delta
  table, no contract.
- **Production serving path:** not touched. The estimator needs up to 49 days of hindsight, so it
  serves training masks and the event table, not a live input.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. The flow shape (flat or a fraction of the donor's
  profile), the loss, the bin width, and the group structure are all open; the plan fixes one
  choice of each and names the others as follow-ups.
- **Spans code whose callers cannot be named without searching:** no. The experiment copies the
  harness of experiment 65 into its own folder, as experiments 60 to 65 each did.
- **Size:** complex, because a new method has more than one defensible design. CVXPY is not a
  locked dependency, so the scripts run with `uv run --with cvxpy`; adding it to the lockfile is a
  separate decision.

## Advice followed

- Experiment 65: use D0M with the PE end rule as the benchmark, not D1. Followed.
- Experiment 65: pre-register any score that uses same-sign neighbour counts. Followed as C2 with
  LOGIT_SAME as a labelled benchmark.
- Experiment 62: no whitening. Followed.
- Experiment 60: score on shapes the detector does not assume (S2, S3, S4). Followed; the flat
  flow assumes none of them.
