---
id: "00253"
slug: lsc-end-time-pair-rule
theme: lsc-detectors
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private per-transfer outcome tables under
  data/ml-experiments/00253-lsc-end-time-pair-rule/ only, no prediction set. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible pair rule (the balance
  tolerance, the joint score), so each is a separate arm. Callers: this folder's scripts only, which
  copy experiment 148's code and read experiment 138's stored arm B score and threshold unchanged.
hypothesis: >-
  When a load shift moves demand between two metered Primary substations, the end of the shift is
  estimated better from the pair than from either substation alone, because the true end restores
  the pair's summed demand and an unrelated step on one substation does not. Using the pair removes
  most of the disagreement between the end times that the two substations' cards show (a real
  shift between D5 and D16 on 14 January 2026 is shown as ending after 9.2 days on one card and
  2.0 days on the other).
prediction: >-
  Written before any result. On experiment 148's injected draw, restricted to transfers with at
  least one metered receiver, P1 and P2 raise the MW-weighted share found by 0.03 to 0.10 over R0
  and cut the median disagreement between the two sides' end times by more than half. On the real
  labelled transfers with two metered Primaries (about 10 of the 35 over one day), P2's end
  disagreement is under 2 days at 7 of 10, and the 14 January 2026 shift between D5 and D16 lands at
  2 to 4 days on both cards. The 2.0-day card is nearer the truth, because the 9.2-day card
  contains a later unrelated step. P3 (a small neural scorer) beats P2 by under 0.02 in share
  found, and is not adopted unless it gains at least 0.03. With 10% unequal step sizes the P1 and P2
  gains shrink by under a third.
decision_rule: >-
  Contrasts (3 arms minus R0; the 98.75% level of the brief's 4-arm Bonferroni is kept) in the
  MW-weighted share found, the false-end share, and the end disagreement, for the lookahead of 24
  hours. R0 is experiment 148's L24 rule applied to each side alone. The population is transfers
  with at least one metered receiver, pooled over shapes S2, S3, and S4, starting in even months, no
  alarm merging, on experiment 148's draw (same seeds). Intervals resample whole calendar months
  20,000 times. The hypothesis is `supported` when, for P2, the lower end of the share-found
  contrast is above -0.02, the false-end share is at most 0.10, and the median disagreement is
  lower than R0's with an interval that excludes zero on the injected transfers, and the number of
  real pairs with disagreement under 2 days is higher than R0's. It is `partly supported` when the
  injected result holds but the real pairs do not move, or when the result holds only without the
  10% unequal step sizes. It is `refuted` when P2's disagreement is not lower on the injected
  transfers. The smallest gain worth porting is zero: the rule costs nothing in production beyond
  the existing 24-hour wait, so the margin of -0.02 on recall is that of experiment 148. P3 is
  adopted only if P3 minus P2 is at least +0.03 in share found with an interval above zero.
benchmark: >-
  R0, experiment 148's rule at a lookahead of 24 hours, applied to each side alone (share found
  0.794 on the same draw).
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, not
  forecasts)
metric: >-
  MW-weighted share found, false-end share, late-end share, end error in hours, and the end
  disagreement between the two sides in hours.
code: ml-experiments/00253-lsc-end-time-pair-rule/
---

# Experiment 253: does the pair of substations give a better end time for a load shift?

**This experiment asks whether the end of a load shift is found more reliably when the two
substations involved are read together.** When demand moves from one Primary substation to
another, the donor's demand drops and the receiver's rises by the same amount, so the end of the
shift is the step that restores the pair's summed demand. Today each substation's card reads its
own end alone, and the two cards can disagree by days. We take the load-shift screen (LSC) of
[experiment 148](../00148-switching-end-rule-lookahead-ro-nc/result.md) and add three rules that use
the pair, and score them on the injected transfers of experiments 147 and 148, with a version in
which the two steps differ by 10%, and on the real labelled transfers.

## Arms

- **R0:** experiment 148's rule at a lookahead of 24 hours, each side alone (the card's rule).
- **P1:** the same rule, but a candidate end must be balanced. The partner steps are the opposite
  steps of neighbouring metered substations at the onset (at least 1.5 noise units, within one
  stamp). A candidate is balanced when the sum of the side's step and the partners' steps at that
  stamp is under half the side's own step. A side with no partner keeps R0.
- **P2:** P1 with a joint score. At each candidate stamp the score is the smallest, over the side
  and its partners, of the opposite step divided by that side's onset step. The lookahead rule of
  24 hours then picks the stamp that maximises the score, which is the stamp at which every side
  returns strongly.
- **P3:** P2 with the score replaced by a small neural network (scikit-learn multi-layer
  perceptron, 8 hidden units) over the opposite-step features of both sides, trained on a separate
  draw that starts in odd months.

## Checks to be run

- R0 reproduces experiment 148's L24 share found.
- The U10 version scales each transfer's receiver shares by a factor drawn as 0.9 or 1.1.
