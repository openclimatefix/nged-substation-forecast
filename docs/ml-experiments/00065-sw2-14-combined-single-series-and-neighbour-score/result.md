---
id: "00065"
slug: sw2-14-combined-single-series-and-neighbour-score
status: planned
size: medium
hypothesis: >-
  One alarm score that adds a substation's own 24-hour step evidence to the opposite-step evidence
  from its neighbouring substations, with neither acting as a gate, finds more of the MW moved by
  injected load transfers than the 24-hour single-series detector (D0M), the neighbour-confirmation
  gate (D1), and the neighbour veto (D2, experiment 55), at the same count of unexplained alarms
  on odd months.
prediction: >-
  MW-weighted recall (the share of the moved MW that sits in found transfers), even months, pooled
  over the injection shapes S2, S3, and S4 of experiment 60, with interval ends taken from the
  strongest opposite step in the next 14 days (PE, experiment 61): D0M reaches 0.63 to 0.65. The
  likelihood-ratio score (LR) gains +0.00 to +0.04 over D0M and the logistic-regression score
  (LOGIT) gains +0.01 to +0.05. The gain sits in transfers with a metered receiver whose donor step
  is under 4.0 noise units, which D0M cannot alarm on, and no group of transfers loses more than
  0.02. D1 and D2 are expected at or below D0M, as in experiments 43, 55, and 58.
decision_rule: >-
  Six planned contrasts, read at 99.17% (Bonferroni across six) from 20,000 resamples of whole
  calendar months, each version (LR, LOGIT) minus each benchmark (D0M, D1, D2) in MW-weighted recall,
  PE end rule, equal weight over S2, S3, and S4, over experiment 58's 620 headline-band transfers
  that start in even months. Every arm is held to at most 80 unexplained alarms on odd months, the
  count of D0M. A version is `supported` when its interval against each of D0M, D1, and D2 lies
  wholly above zero, its point value against D0M is at least +0.02, and its even-month
  unexplained-alarm count is no more than 10 above D0M's (a count near 80 varies by about 9 from
  chance alone). A version is `partly supported` when its intervals against D1 and D2 lie wholly
  above zero but its interval against D0M does not. The experiment is `refuted` when the upper end
  of both versions' intervals against D0M lies below +0.02, `inconclusive` otherwise, and takes the
  better version's verdict if the two differ. The threshold of +0.02 is set by the work of porting:
  the combined score adds a neighbour-step computation and a calibration fitted on injected
  transfers to a detector that needs neither. The same contrasts under the tracked end rule are
  reported and do not change the verdict.
benchmark: >-
  D0M (the 24-hour step detector at 4.0 noise units, 80 unexplained odd-month alarms), D1 (the
  neighbour-confirmation gate), and D2 at experiment 55's veto strength of 2.0, each with the PE end
  rule, and each with its alarm threshold matched to 80 unexplained odd-month alarms.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55,
  58, 59, and 61 do, not forecasts)
metric: >-
  MW-weighted recall (headline): the share of the MW moved by injected transfers that sits in
  transfers the detector finds, where a transfer is found when an alarm interval starts within 2
  hours and ends within the larger of 6 hours and 10% of the duration. Unweighted recall is
  secondary.
code: ml-experiments/00065-sw2-14-combined-single-series-and-neighbour-score/
---

# Experiment 65: does one score from a substation's own step and its neighbours' opposite steps find more of the moved load?

*Status: planned. This page is the plan and is committed before any code runs.*

## Plan

**Question.** Experiment 55 used neighbours to veto alarms, and experiment 43's D1 used them to
confirm alarms. In both, a weak own step cannot raise an alarm however clearly the neighbours
stepped the other way. A load transfer with a metered receiver produces two steps of opposite sign
at the same time, so strong evidence on either side should be enough. Experiment 58 found that a
metered receiver lifts recall by +0.161 at an equal donor step, which is the evidence this
experiment tries to use.

**Candidates.** A candidate is a local peak, within 24 hours either side, of one series' 24-hour
step statistic of at least 2.5 noise units, excluding the stamps near a monthly refit, as in
experiment 61. Every alarm of D0M (4.0 noise units) is a candidate. For each candidate we compute
two statistics, both in noise units of the neighbour's or the series' own odd-month scale:
`x1`, the size of the candidate's own step; and `x2`, the largest opposite-sign step among its
neighbours within one half-hour of the stamp (0 when no neighbour has data), using the neighbour
list and the step definition of D1 and D2. Two topology numbers also describe each candidate: the
number of neighbours, and whether any neighbour is a Primary metered only in MVA (a magnitude with
no direction).

**Two scores, fixed before any fit.**

- **LR, the likelihood-ratio sum.** For each statistic, a table of log-likelihood ratios over fixed
  bins (`x1`: 2.5, 3, 3.5, 4, 5, 6, 8, 12; `x2`: 0, 1, 2, 3, 4, 6, 8), made non-decreasing by a
  running maximum, from the share of positive candidates in the bin against the share of negative
  candidates (each count plus 0.5). The score is the sum of the two log-likelihood ratios.
- **LOGIT, the logistic regression.** An L2-penalised logistic regression (penalty 1.0, standardised
  inputs, equal total weight on the two classes) on `x1`, `x2`, the number of neighbours, and the
  MVA-neighbour flag.

**Training.** Positive candidates lie within 2 hours of the onset of an injected headline-band
transfer (25% to 40% of the receiving load, over 1 day), on the donor or on a metered receiver, in
the residual with that transfer injected. Negative candidates are those of the real residual in odd
months that no logged closure explains. Only transfers that start in odd months train either score,
all five injection shapes of experiment 60 are pooled, and the holiday set is left out. Even months
are touched only by the final scoring. The alarm threshold of each score is the lowest that keeps the
real residual's odd-month unexplained alarms at 80 or fewer.

**Benchmarks and end rule.** D0M, D1, and D2 (veto strength 2.0) each have their threshold matched
to 80 unexplained odd-month alarms by the scan of experiments 55 and 61. Every arm is scored with
two end rules: the tracked end of experiment 43, and PE, which replaces the end by the strongest
opposite step in the next 14 days. The headline uses PE, because experiment 61 found that interval
ends were the main limit on recall and advised making PE the baseline. D0M with PE is the
24-hour benchmark.

**Reports.** MW-weighted recall and unweighted recall for each arm and shape (S1 to S5, with S2,
S3, and S4 as the headline because the combined score assumes none of them); the share of the moved
MW still missed; recall by donor step in noise units, split by metered receiver; the labelled real
transfers over 1 day (35), counted found per arm; even-month unexplained-alarm counts; the fitted
log-likelihood ratios and logistic coefficients.

**Post hoc, labelled and outside the verdict.** LOGIT with the count of same-sign neighbour steps
added (the veto's own evidence, which the pre-registered score omits), and D2 at veto strength 1.0.

## Sizing

- **Stored:** nothing outside the experiment's private folder.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, the two scores and the candidate floor, so the plan is
  reviewed and the two versions are fixed here.
- **Code whose callers cannot be named:** no; the code is a copy of experiment 61's harness plus one
  module.

The size is medium: a Sonnet plan and plan review, a Sonnet implementation and code review, a
Sonnet run and write-up, and a Sonnet prose review.

## Earlier advice

Followed: score the 24-hour window as the single-window benchmark (experiment 59); make PE the
baseline (experiment 61); report recall by donor step split by metered receiver, shapes S2 to S4,
and the labelled real transfers (experiments 58 and 60). Departed from: none.
