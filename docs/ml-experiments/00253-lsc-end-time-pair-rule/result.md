---
id: "00253"
slug: lsc-end-time-pair-rule
theme: lsc-detectors
also_relevant_to: []
status: done
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
verdict: inconclusive
best_arm: >-
  P2, the joint pair score: with an unrelated step added after the true end it finds 0.633 of the
  moved MW against 0.593 for R0 (+0.041, 98.75% interval +0.007 to +0.074), and it cuts the median
  gap between the two sides' ends from 8.5 to 2.5 hours. On the clean draw it finds 0.770 against
  0.788 (-0.017, -0.034 to +0.007).
headline: >-
  Clean draw: P2 minus R0 -0.017 (-0.034 to +0.007), P1 minus R0 -0.028 (-0.041 to +0.002), P3
  minus P2 -0.011 (-0.023 to +0.016). Unrelated step added: P2 minus R0 +0.041 (+0.007 to +0.074),
  false-end share 0.093 against 0.041. Step sizes 10% unequal: P2 minus R0 -0.012 (-0.029 to
  +0.011).
lessons: >-
  The pair rule trades a small loss on ordinary transfers for a gain when an unrelated step follows
  the end. The loss is concentrated in transfers with two or more metered receivers. The real
  14 January 2026 shift is not changed by any rule, because neither card's end comes from the step
  rules.
take_home_message: >-
  Reading the two substations together makes their end times agree when an unrelated step follows
  the true end: the median gap falls from 8.5 to 2.5 hours and the share of moved MW with a correct
  end rises by 0.04. On ordinary transfers the same rule loses about 0.02, so the experiment cannot
  say the pair rule is safe to port. It does not change the 14 January 2026 shift between D5 and
  D16, because one side has no step to read and the other side's candidate steps are all
  unbalanced.
advice:
  - >-
    Do not port the pair rule P2 yet: it loses 0.017 (-0.034 to +0.007) of the share found on
    ordinary transfers and gains 0.041 only when an unrelated step follows the end (contrasts P2
    minus R0, clean and C).
  - >-
    Score end rules side by side, not as the best of the sides: the best-of-sides share found is
    already 0.962 on transfers with a metered receiver, so it cannot show a gain of more than 0.018
    (ceiling check).
  - >-
    Do not add a neural scorer to the end rule: P3 minus P2 is -0.011 (-0.023 to +0.016) on the
    clean draw (secondary contrast).
  - >-
    When one side of a pair has no qualifying return step, tie its end to the other side's
    balanced end instead of keeping its level-tracked end; this is the case on 14 January 2026
    (case report).
production_status: not-for-production
adopted_in: none
---

# Experiment 253: does the pair of substations give a better end time for a load shift?

**Reading the two substations of a load shift together makes their end times agree and finds the
right end more often when an unrelated step follows the end, but it costs a little on ordinary
transfers, and it leaves the real 14 January 2026 shift between D5 and D16 unchanged.** When demand
moves from one Primary substation to another, the donor's demand drops and the receiver's rises by
the same amount, so the end of the shift restores the pair's summed demand. Each substation's card
now reads its own end alone. We took the load-shift screen (LSC) of [experiment
148](../00148-switching-end-rule-lookahead-ro-nc/result.md) and added three rules that use the
neighbouring substations. We scored all of them side by side on the 1,160 transfers that experiment
148 injected into real demand, on a copy where an unrelated step follows the end of one side, on a
copy where the two steps differ by 10%, and on the labelled real transfers.

On the clean draw the best rule, P2, finds 0.770 of the moved MW against 0.788 for the lookahead
rule R0 (-0.017, 98.75% interval -0.034 to +0.007). With an unrelated step added it finds 0.633
against 0.593 (+0.041, +0.007 to +0.074). The pre-registered rule needs the clean-draw lower end
above -0.02, so the verdict is `inconclusive`. The result narrows the search: a balance test on the
pair is worth keeping for the case it was built for, and the next change should make it fall back
to the other side's end where a side has no step of its own.

![Contrasts of the share found](contrasts.svg)

![Disagreement between the two sides' ends](disagreement.svg)

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

## Results

- **Check:** R0 on the full draw, scored as experiment 148 scored a transfer, finds 0.794, as in
  experiment 148.
- **Ceiling:** on transfers with a metered receiver, experiment 148's best-of-sides score is
  already 0.962 against a ceiling of 0.980 (the share whose start is found), so the brief's
  predicted gain of 0.03 to 0.10 in that score was out of reach. The plan was revised to score each
  side before any code ran.
- **Share found, side by side (95% interval), pair transfers, shapes S2 to S4, even months:** clean
  R0 0.788 (0.743 to 0.850), P1 0.760, P2 0.770 (0.718 to 0.848), P3 0.759; unrelated step added R0
  0.593 (0.549 to 0.640), P1 0.606, P2 0.633 (0.588 to 0.683), P3 0.628; unequal steps R0 0.781, P1
  0.754, P2 0.769, P3 0.754. The offline rule OFF finds 0.785, 0.621, and 0.779.
- **Contrasts (98.75%):** clean P2 minus R0 -0.017 (-0.034 to +0.007), P1 minus R0 -0.028 (-0.041
  to +0.002), P3 minus P2 -0.011 (-0.023 to +0.016); unrelated step P2 minus R0 +0.041 (+0.007 to
  +0.074), P1 minus R0 +0.013 (-0.010 to +0.068), P3 minus P2 -0.006 (-0.027 to +0.026); unequal
  steps P2 minus R0 -0.012 (-0.029 to +0.011), P1 minus R0 -0.027 (-0.042 to +0.002).
- **False-end share:** clean 0.047 (R0), 0.052 (P1), 0.053 (P2), 0.052 (P3); unrelated step added
  0.041 (R0), 0.104 (P1), 0.093 (P2), 0.112 (P3). P2's false-end share is under the 0.10 limit.
- **Disagreement between the two sides' ends (transfers where both alarmed):** clean, the median is
  0 hours for R0 and P2, and the share over 6 hours is 0.145 (R0) and 0.153 (P2). With the
  unrelated step the median falls from 8.5 hours (R0) to 2.5 (P2), and the share over 6 hours from
  0.579 to 0.339. P3's median is 5.0 hours.
- **Predictions:** the share-found prediction of -0.01 to +0.02 on the clean draw held for P2
  (-0.017) and failed for P1 (-0.028). The predicted gain of +0.05 to +0.20 with the unrelated step
  was not reached (P2 +0.041, P1 +0.013). The prediction that P3 beats P2 by under 0.02 held, and P3
  is not adopted.
- **Real labelled transfers (descriptive):** 55 transfers have two or more metered sides (111
  sides), and the start is found on 26 sides. The end is within the tolerance of the logged close
  on 20 sides for R0, 21 for P1 and P2, and 22 for P3. For the 6 transfers where both sides
  alarmed, the median gap between the ends is 6.8 hours for R0 and 0.0 for P2.
- **The 14 January 2026 shift:** every rule gives the same ends as the card: D5 ends after 9.2
  days and D16 after 2.0 days. D16 has no stamp where the opposite step reaches the 5.0 noise units
  the rule needs (its largest is 2.1), so its end comes from the level tracker. D5 has 13 qualifying
  stamps and none is balanced, so P1 and P2 fall back to R0's choice. The pair rule therefore
  cannot say which card is right.
- **Post hoc, one metered receiver only:** P2 minus R0 is -0.009 (-0.038 to +0.012) on the clean
  draw and +0.055 (-0.002 to +0.105) with the unrelated step; the share of transfers over 6 hours
  apart falls from 0.604 to 0.321 with the step. The loss on the clean draw is mostly in transfers
  with two or more metered receivers.
- **Post hoc, group balance (P1G, P2G):** summing every neighbour that stepped at the onset, of
  either sign, gives P2G minus R0 of -0.018 (-0.037 to +0.005) clean and +0.031 (-0.008 to +0.075)
  with the unrelated step. Group balance does not remove the loss.

## Checks

- **Forecast checks:** no forecast is scored, so the median-minus-mean and mean-error checks do not
  apply.
- **Audit triggers:** trigger 5 (a nested arm beats the arm that contains it) fires, because R0
  beats P1 and P2 on the clean draw. The explanation tested here is that the balance test rejects
  the true end when the group is not fully balanced; the single-receiver and group-balance runs
  above show the loss shrinks to -0.009 for one receiver and does not vanish with group balance.
  Trigger 2 (a prediction contradicted) fires for the unrelated-step gain, which came out at +0.041
  against +0.05 to +0.20. A Sonnet code and numbers audit replaced the Opus audit, because the
  brief allows Sonnet only and the manager session was not running; its findings are triaged in
  the pull request. The other five triggers did not fire. Leaving one series out was not run, and
  no claim here is per series.
- **Not run on all transfers:** the long transfers of experiment 148 and alarm merging were left
  out.

## Caveats

- **Partner finding is favourable on the injected draw,** because the neighbour table that finds
  partners is the table the injection draws receivers from. The test of 1.5 noise units is the only
  detection step.
- **The unrelated step is injected by us.** Its size (1.0 to 1.5 times the side's own step) and
  timing (2 to 20 hours after the end) were chosen to resemble the 14 January 2026 shift, so the
  gain is conditional on such steps being common. The labelled real transfers cannot tell how
  common.
- **Real pairs are few:** the start is found on 26 of 111 sides, and 6 transfers have both sides
  alarmed.

## What to try next

**The next experiment should make a side with no return step use the other side's balanced end,
and should count how often unrelated steps follow real ends.** Tying the ends together removes the
case the 14 January 2026 shift shows, where one side has nothing to read. Counting unrelated steps
in the real telemetry decides whether the gain of +0.04 is worth the loss of 0.017.

Code: `ml-experiments/00253-lsc-end-time-pair-rule/`; every number above is printed by
`e00253_contrasts.py` (saved at `data/ml-experiments/00253-lsc-end-time-pair-rule/printed_report.txt`)
and `e00253_case.py` (`case_report.txt` in the same folder).
