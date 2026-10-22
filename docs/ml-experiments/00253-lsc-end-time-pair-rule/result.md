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
  Written before any result, and revised once after a plan review and before any code ran. On
  experiment 148's draw, transfers with at least one metered receiver (930 of 1,806 in shapes S2 to
  S4 starting in even months) are already found at 0.962 by the best-of-sides score, against a
  ceiling of 0.980 (the share whose onset is found), so no rule can gain more than 0.018 there. We
  predict that scored side by side, P1 and P2 change the share of sides whose end is within the
  tolerance by -0.01 to +0.02 on the clean draw (R0 is near its ceiling), and by +0.05 to +0.20 on
  the draw with an injected unrelated step, where R0 takes the unrelated step as the end. We
  predict that with 10% unequal step sizes P1 and P2 keep at least two thirds of the gain they have
  on the clean draw. On the real labelled transfers the pair rules end within the tolerance of the
  logged close at no fewer of the transfers with two metered substations than R0 does. A hypothesis
  about the 14 January 2026 shift between D5 and D16, not a prediction we can score: the 9.2-day
  card ends at a later unrelated step, so P2 puts both cards at 2 to 4 days. P3 beats P2 by under
  0.02 in share found, and is not adopted unless it gains at least 0.03.
decision_rule: >-
  The primary contrast is P2 minus R0; P1 minus R0, P3 minus P2, and P3 minus R0 are secondary.
  All four use the 98.75% level (Bonferroni for four contrasts). The unit is a side: a donor, or a
  metered receiver, of an injected transfer with at least one metered receiver (pair transfers).
  A side is found when an alarm interval on it starts within 2 hours of the true start and ends
  within the end tolerance (the larger of 6 hours and 10% of the duration) of the true end. The
  share found is MW-weighted by the MW moved, pooled over shapes S2, S3, and S4, for transfers that
  start in even months, on experiment 148's draw, with no alarm merging. The false-end share is the
  share of onset-found sides whose end is earlier than the true end by more than the tolerance. Intervals
  resample whole calendar months 20,000 times. P2 is `supported` when, on the clean draw, the lower
  end of its share-found contrast is above -0.02 and its false-end share is at most 0.10, and on
  the draw with the injected unrelated step its share-found contrast has an interval above zero,
  and on the draw with 10% unequal step sizes the first condition holds again. It is `partly
  supported` when the clean-draw and unequal-step conditions hold but the unrelated-step contrast
  interval includes zero. It is `refuted` when the upper end of the clean-draw contrast is under
  -0.02 or P2 scores below R0 on the unrelated-step draw. Otherwise it is `inconclusive`. The
  smallest gain worth porting is zero: the rules cost nothing in production beyond the 24-hour wait
  of experiment 148, so the margin is that experiment's -0.02. End disagreement (the gap in hours
  between the ends the two sides call, for pair transfers where both sides alarm) is reported as
  secondary and is zero by construction for no arm; the real labelled transfers are descriptive and
  scored by the end error against the logged close. P3 is adopted only if P3 minus P2 is at least
  +0.03 with an interval above zero.
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

## Definitions

- **Half-hour stamp:** the unit of time of every table here; 48 stamps make a day.
- **Alarm, onset, noise unit:** the load-shift screen (LSC) raises an alarm on a substation at the
  half-hour where its demand steps most, and the alarm opens an interval. The step is measured as
  the mean demand over the next 24 hours minus the mean over the 24 hours before. A noise unit is
  the usual size of that step in odd-numbered months, per substation, when nothing is injected.
- **Injected draw:** the 1,806 transfers that experiment 148 added to real demand, drawn with
  seeds 8200 and 8300 and not changed. Each moves a share of the donor substation's demand to one
  to three receivers, some metered and some not. The shapes S2, S3, and S4 give the transfer a
  flat, a receiver-shaped, and a mixed daily profile.
- **Detector:** arm B of experiment 138, with that experiment's stored scorer and threshold.
- **Share found:** the MW-weighted share of sides whose interval starts within 2 hours of the true
  start and ends within the tolerance of the true end; experiment 148 scored a transfer as found if
  any side was, and R0 reproduces that number as a check.

## Arms

- **R0:** experiment 148's rule at a lookahead of 24 hours, each side alone. The end is the stamp
  of the largest opposite step seen so far, fixed once no larger step follows for 24 hours.
- **P1:** R0, but only balanced stamps are candidates. A partner is a neighbouring metered
  substation (the experiment 138 neighbour table) whose own largest opposite-signed step within 2
  hours of the alarm is at least 1.5 noise units. A stamp is balanced when the side's step plus the
  partners' steps at that stamp is under half the side's step. Unbalanced stamps are skipped. A
  side with no partner, or with no balanced stamp that meets R0's size test, keeps R0's end.
- **P2:** P1 with a joint score. At each balanced stamp the score is the smallest, over the side
  and its partners, of that substation's opposite step divided by its own onset step. A stamp needs
  a score of 0.5 or more, and the end is the stamp with the largest score, fixed by the same
  24-hour lookahead.
- **P3:** P2 with the score replaced by the output of a small neural network (scikit-learn
  multi-layer perceptron, 8 hidden units, seed 253) over seven features of both sides: own and
  partner returns, balance, time since onset, return in noise units, and share of the running
  peak. The target is that the stamp is within the tolerance of the true end. Training uses 600
  injected transfers (shapes S2 to S4) that start in odd months, drawn with seed 8400.
- **Causality:** every arm uses only the steps up to 24 hours (plus the step window) after the
  candidate end. The true end is used only to pick populations, to train P3 on its own draw, and to
  score.

## Variants of the injected draw

- **Clean:** experiment 148's draw.
- **Unrelated step (C):** on one side of each pair transfer, chosen by seed, an extra step lasting
  3 days starts 2 to 20 hours after the true end, in the direction that restores the side's level,
  at 1.0 to 1.5 times the side's own return step. The other side shows nothing. This is the
  situation of the 14 January 2026 shift, and the clean draw never contains it.
- **Unequal steps (U10):** each transfer's receiver shares are multiplied by 0.9 or 1.1, chosen
  by seed, so the two sides' steps differ by 10%.

## Checks to be run

- R0 reproduces experiment 148's best-of-sides share found of 0.794 on the full draw.
- The partner topology is the same table that the injection draws receivers from, so partner
  finding is favourable on the injected draw; the 1.5-noise-unit test is the only detection step.
- Changes from the brief: P1 is a balance test on the candidate step instead of an end taken from the
  summed series, and the population is transfers with at least one metered receiver instead of
  transfers with both sides metered, because a transfer has one donor and the receivers vary.
