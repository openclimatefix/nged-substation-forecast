# The research default and the rule for adopting a change

**Each target family has one fixed research default forecast, one fixed test for adding a change to
it, and a [log of versions](research-default-log.md).** An experiment of kind `change` asks whether
the research default plus change x beats the research default. A win produces the next version of
the research default. The rule exists so that the experiments build one forecast in order, and so
that a gain is always measured against the current forecast and never against a stale or partial
one.

**Adopting a change into the research default does not put the change into production.** The
`release-model` skill governs production changes. This page says "research default" and "adopt"
because [production deployment](../architecture/production-deployment.md) already uses "champion"
and "promote" for the deployed XGBoost model. Every version has a name made of the family, the words
"research default", and a number N that counts up within the family from 1, for example "Primary
demand research default 1" for the GAM stack A+H+S. "C1" and "C2" already name the planned contrasts
on every result page, so a bare "default 3" or "v3" is never used. Every `result.md` and
pull-request body cites the version by name, for example "default plus x versus Primary demand
research default 1". The two arms of a `change` experiment are called "default" and "default plus
x".

**The research default is not what runs in production.** Production serves a much weaker XGBoost
model, stored as `xgboost_baseline`. [Experiment
87](00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md) scored it at 10.536 points on
the Primary quantile score, against 5.260 for [BG2](glossary.md#bg2). A page that names BG2 as a
benchmark says that production does not run BG2.

## Families

**Each family has its own research default, row set, headline metric, protected groups, benchmarks,
retraining cadence, and version line.** The family definition below fixes these values once, and an
experiment's plan never sets or changes them. A per-plan choice would let an experiment pick the
metric on which its change wins. The metrics used across the experiments so far include the quantile
score, the mean absolute error of the median, tail-weighted continuous ranked probability score
(twCRPS), a single quantile's pinball loss, and avoidable curtailment cost.

| Family | Rows and period | Lead range | First version of the research default | Retraining cadence |
|---|---|---|---|---|
| Primary demand (six Primaries) | `fold_a_daily_gap_v1`, scored year 2025-07-02 to 2026-06-30, replicate year 2024-25, rows outside hand-filtered events | 0.5 to 168 hours (7 days) | Primary demand research default 1: the GAM stack A+H+S of experiments [220](00220-dp-bg-stage4-assembly-with-drop-one/result.md) and [251](00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md) | Monthly for the GAM stack; weekly for its XGBoost benchmark BG2 |
| Metered solar farms A to F | Daylight half-hours, scored year 2025-09 to 2026-08, 6-month replicate | 12 to 36 hours | Metered solar farms research default 1: the best guess assembled by experiment 300, once it reports. Until then the family accepts no adoption | A single fit or monthly, as the family's first version states |
| Wind farms W1 to W3 and the public wind farms of experiment 288 | `fold_a_tuesdays_v1`, 53 Tuesday runs | To be named with the first default | Wind farms research default 1: to be named when the first default is chosen. No adoption until then | To be named with the first default |
| Bulk and grid supply points | `fold_a_daily_gap_v1` | To be named with the first default | Bulk and grid supply points research default 1: to be named when the first default is chosen. No adoption until then | Weekly |

**NGED care about the 1 day to 10 day horizon, so each family states its lead range, and a lead range shorter than that is a limit on every result.** The Primary-demand range stops at 7 days and the metered-farm range stops at 36 hours. A `change` result for such a family repeats its lead range in its headline caveat, for example "scored at 12 to 36 hour leads only".

**The headline metric for all four families is the quantile score over 11 levels, in points of
effective capacity.** A point is 1% of a series' capacity.

**A family with few series and few runs will rarely pass the gain test, and the rule does not lower
the bar for it.** Three wind farms on 53 runs give wide intervals, so a wind change will seldom
satisfy condition 1. A change that cannot reach the bar stays a candidate, and the maintainer may
judge it by a descriptive report instead.

**The Primary-demand family fixes the following tolerances, in points unless stated.** A tolerance
is the harm a protected group may suffer before an adoption fails.

| Protected group | Tolerance |
|---|---|
| Pooled six-Primary headline | 0.02 |
| 0.01 and 0.99 pinball tails (the worse end is scored, with coverage reported beside it) | 0.05 |
| Cold winter nights | 0.05 |
| Sunny daylight half-hours | 0.05 |
| twCRPS | 2% relative |

**The other three families use a headline tolerance of 0.02 and the same tail tolerance of 0.05.**
Their other protected groups are set when each family's first version is written.

**The minimum gain depends on the complexity of the change.** The point estimate of the gain must
reach the value for the change's tier.

| Tier | Minimum gain |
|---|---|
| A changed setting, an added column, or a changed loss | 0 |
| A new fitted step, a new input, or a new data source | 0.02 points |
| A new head | 0.03 points |

[Experiment 278](00278-stack-retraining-cadence/result.md) anchors the values: it used 0.02 points
for the convex model's cadence and 0.03 for the head.

**Two further lists are part of each family's definition.** The reported groups have no pass or
fail: each Primary on its own, and the rows near a load shift (seven logged transfers are too few
for an interval, per [experiment
277](00277-lsc-curtailment-metric-on-existing-forecasts/result.md)). The admissible subgroups for a
targeted gain are a short list set in advance (cold nights, the Economy 7 step, sunny daylight), and
a targeted adoption on a subgroup also needs no harm on the family headline.

**The cumulative harm budget is 2 times the group's tolerance, measured against the family's first
version.** The no-harm condition compares each adoption only with the previous version, so n
adoptions that each cost up to the tolerance could cost n times the tolerance in total. At each
adoption, the current research default must also stay within the budget against the first version,
using the first version's stored forecasts.

## The three fixed benchmarks

**Each family keeps three benchmarks whose recipes never change: `manual_heuristic`, `climatology`,
and one frozen XGBoost-based reference.** The rows grow as months arrive, and the benchmarks are
extended with the same recipe and never retuned. Every `change` page reports the research default's
absolute score against the three benchmarks, with intervals, because a chain of small contrasts can
hide a drift in absolute skill. Diagnostic pages need not report them.

**For Primary demand the frozen reference is BG2.** BG2 is the stored arm `p_s6_bg2` of experiment
87. It is a quantile blend of five forecasts, with an explicit solar offset at the 5 of 16 Primaries
that have a clean solar capacity estimate. The five forecasts are a weekly-refitted XGBoost model, a
monthly-refitted XGBoost model, an analogue model (the experiment 87 page does not state its type),
an XGBoost model pooled by substation type, and the frozen stack of experiment 48. For the bulk and
grid supply points, BG2 is the weekly-refitted XGBoost model alone (`b_weekly`). The metered-farm
pages compare with the XGBoost forecast of experiment 243 and the floors of experiment 224, and the
frozen reference for the farm and wind families is named when each family's first version is
written.

**The GAM stack trails BG2 on sunny April-to-June half-hours.** [Experiment
251](00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md) found the stack worse
than BG2 in April, May, and June (+0.162, +0.220, and +0.054 points), and at D5 on sunny half-hours.
Every Primary-demand `change` page reports this finding until a version of the research default
closes the gap.

## Reusing the research default's code and forecasts

**A `change` experiment starts from the stored forecasts of the research default and, when the
change acts inside a fitted stage, from the default's code.** The change is one of two kinds, and
the experiment's page says which.

- **Tier 1 changes act on stored forecasts.** A clip, a band-widening rule, a blend, a
  recalibration, or a quantile interpolation needs only the research default's stored forecasts. The
  D8 as-of widening, the lead-band interpolation, the D5 clip, and the blend of experiment 287 are
  all of this kind.
- **Tier 2 changes act inside a fitted stage.** A change to the convex model, the head, the solar
  summary, or the features needs the research default's code. The experiment imports that code
  directly from the folder of the experiment that holds it, with that folder on `PYTHONPATH`, as the
  `ml-experiment` skill describes. The code moves into `packages/studies` in a reviewed pull request
  only when several experiment folders need it or when production might one day need it.

**The research default's definition holds four items.** These are the code (a script in an
experiment folder and its commit, a module and commit in `packages/studies`, or "stored forecasts
only"), a configuration file naming every setting and the seed policy, the stored forecasts for
every scored, replicate, and holdout period, and the scores. The seed policy matters: averaging
three seeds of the correction network scored 0.123 points below arm D, against 0.104 for the mean
single seed ([experiments 201 and 220](00220-dp-bg-stage4-assembly-with-drop-one/result.md)), so
"more seeds" would otherwise adopt as a change.

**Each version's stored forecasts are written once and never overwritten.** Several worktrees share
one `data/` folder, and an earlier re-run has already clobbered another session's evidence.
Experiments in flight read Primary demand research default 3 while research default 4 is being
built. Each version is written once with `experiment_store.write_predictions` under the adopting
experiment's id and a fixed name. Where a refit is needed to reproduce the default, the reproduction
uses the default's cadence exactly, because refit cadence is itself a contrast (experiment 278).

## What an experiment must contain to be eligible

**An experiment is eligible for adoption only when its front matter says `kind: change` and its
plan, committed before any score is read, states five items.** Diagnostics, audits, data checks, and
detection or decision studies are `kind: other` and never adopt. The five items are the following.

- The family and the research default version the experiment starts from.
- The arms: the default, and the default plus x. The default's arm reads the stored forecasts of
  that version, with a reproduction gate when the experiment refits the default, as experiment 278
  did.
- The tier of the change (tier 1 or tier 2) and its complexity tier from the minimum-gain table.
- The pre-registered default-plus-x arm or arms.
- The controls that apply: a known-answer test, a placebo, a leakage check.

**An experiment that evaluates a blend reports each constituent model family alone beside the blend.** The single-family table comes first, and the page states the blend's added value over the best single family, with an interval. The first production update is a relatively simple model implemented as one model family, for example XGBoost only, while blends stay of interest. A blend is adopted into the research default only with that table present.

**The default plus x is produced on the research default's full row set.** Its stored forecasts then
serve as the next version without a separate regeneration job.

## The adoption rule

**A change is adopted only when all five conditions hold.**

1. **Gain.** The 99% month-block interval of (default plus x, minus default) on the family headline
   lies wholly on the better side of zero, and the point estimate is at least the minimum gain for
   the change's tier. With k pre-registered arms the interval is read at the level 1 − 0.01/k.
2. **No harm.** Each protected group is gated by a one-sided 95% non-inferiority bound: the upper
   bound of the harm must not exceed the group's tolerance. The six Primaries are pooled for this
   gate, and the per-Primary results are reported. A Primary whose point estimate is worse than the
   tolerance triggers a look, not a failure. Because every group must pass, the gate is an
   intersection test and needs no multiplicity correction. A neutral change passes a group with
   probability Φ(τ/SE − 1.645), where τ is the group's tolerance and SE is its standard error, so a
   tolerance of 2 standard errors passes a neutral change 64% of the time and a tolerance of 3
   standard errors passes it 91% of the time. A change that truly harms a group by exactly the
   tolerance passes that group 5% of the time.
3. **Replication.** The gain has the same sign in a second period, and the replicate estimate is at
   least half of the smaller of the main estimate and the minimum gain. Where the minimum gain is
   zero, the condition reduces to the same sign. Leave-one-series-out is a check for changes tuned
   on one series and does not stand in for a second period: a change with no second period stays a
   candidate. A change that fails condition 3 is not retried on the same replicate period with a
   tweaked x unless the tweak is pre-registered as a new experiment and the failed attempt is
   counted.
4. **Controls.** The experiment's own controls passed, and none of the audit triggers in the
   `ml-experiment` skill is unresolved. The audit's findings are triaged in the pull request before
   the adoption. A change that meets condition 3 satisfies audit trigger 7, and a candidate that
   fails condition 3 never becomes a base for later work.
5. **Serving.** The change uses only inputs available at forecast time. If the change would need a
   serving-path change to port, the page says so.

**A change that wins on condition 1 but fails conditions 2 to 5 is recorded as a candidate, not
adopted.** Candidates are retested when the missing evidence arrives, or when a rebase run is
available.

**The 99% interval is closer to a 96% interval.** In a check run for this rule, a percentile
bootstrap that resampled independent normal months with a nominal 99% interval covered 96.5% of the
time at 12 months, 92.0% at 6 months, and 83.7% at 4 months (1,500 trials of 2,000 resamples each).
The real procedure also resamples seeds and has autocorrelated months, so the figures are
indicative. For periods under 12 months, add a t test on the monthly paired differences or use a
studentised bootstrap. Adoption contrasts use 99% whatever the number of arms, as a stated exception
to the [interval-level convention](glossary.md#interval-level-convention), and a `change` page says
"the 99% interval lies wholly on the better side of zero" in place of the `study` skill's
"statistically significant at the 5% level".

**The recorded gain of each version is an upper bound.** Winners are selected on the scored year, so
the recorded estimate overstates the true gain. In a simulation, a true gain of 2 standard errors
was recorded on average as 2.7. The log therefore also reports the cumulative gain on the replicate
period, which carries no selection bias, beside the scored-year sum.

## The holdout

**The frozen period is the existing holdout: valid times on or after 2026-07-01.** No experiment
scores those valid times, which are kept for a finalist. The `release-model` skill scores the
holdout once per release, on the sign of the effect only, and a month may confirm one release only.

**The holdout holds under four months and no winter on 2026-10-23.** It covers July to October 2026,
and the months from November to February arrive by about March 2027. Until then, a change that acts
in winter (the heating curve, cold nights, the Economy 7 step) is adopted on the replicate year
alone, and its log entry reads "winter-unconfirmed". The holdout confirms its sign in March 2027 or
later.

**The metered-farm family has already scored part of the holdout.** Experiments 224, 243, 257, 270,
272, 275, 284, 287, and 293 state a scored year of 2025-09 to 2026-08, so July and August 2026 are
no longer untouched for metered solar farms. That family's holdout starts at 2026-09-01 at the
earliest.

**The same-months shrinkage check pauses a family's adoptions only on evidence of shrinkage.** The
check compares the holdout months with the same calendar months of the scored year (July to October
2025 for the current holdout). Adoptions in the family pause when the one-sided 90% upper bound of
the holdout months' cumulative gain lies below half of the cumulative gain on those same scored-year
months. A month read for a decision, or for a pause investigation, joins the scored months, and the
holdout rolls forward as new months arrive. In a simulation of a 12-month scored year and a 4-month
holdout, the check paused the family after 29% of false adoptions, 7% of adoptions with a true gain
of 2 standard errors, and 3% of adoptions with a true gain of 3 standard errors. It understates the
chance of a pause for a change whose effect is seasonal.

## Several arms, post hoc arms, and experiments planned before the rule

**An experiment may hold several default-plus-x arms, and at most one of them is adopted.** Each arm
is read at the level 1 − 0.01/k, and the others become candidates that rebase on the new version. A
bundle of two components is adopted as one change only when it was pre-registered as one change and
its leave-one-out arms show neither component harmful.

**Only a pre-registered arm can be adopted.** A post hoc arm becomes a candidate that needs a fresh
experiment. When an experiment tests k variants of one change (tilt 20 and tilt 27, for example),
the best variant needs replication on its own after the Bonferroni read.

**Experiments planned before the rule are grandfathered, and their winners reach the research
default only through a rebase run.** Their front matter has no `kind`, and `experiment_tables.py
--check` accepts its absence for an experiment whose id is below the cut-off. A proposal with no
plan commit yet is re-planned as `kind: change` where it tests a change.

## The rebase rule for stale results

**A result measured against an older version stays valid as an estimate of x on that version, but it
answers a stale question.** The estimate (research default 3 plus x, minus research default 3) is
still correct for research default 3. It is not an estimate of x on research default 4. Adoption one
change at a time is path-dependent: a change that failed against one version may pass against a
later one, and two changes that act on the same error can both win alone while only the first to
merge adopts. The two gains of experiment 220 added to a stacking ratio of 0.99, while the drift
term improved the bare GAM-K by 0.40 points and worsened arm D by 0.03.

1. A result that passes against a version that is no longer current becomes a candidate with the
   status "candidate (stale)".
2. A short rebase run scores (current default plus x, minus current default) against the stored
   forecasts of the current version, under the same five conditions and with no new plan.
3. A rebase is needed unless every adoption since the experiment's version belongs to a different
   family.
4. Where two candidates change the same component (two widening rules, for example), the rebase
   scores them head to head.
5. Adoptions go in merge order. The second of two overlapping changes gets a rebase run against the
   new version.
6. Changes rejected against the last few versions stay on a retest list that the periodic ablation
   revisits.

**After every fifth adoption in a family, an ablation experiment removes each adopted component in
turn.** The ablation is an experiment with its own id and a planned `result.md`. A component is
dropped only when the interval of its removal lies wholly on the better side, or when its removal is
non-inferior within the tolerance and the component sits in a higher complexity tier.

## Who adopts, and what the log holds

**The experiment's own pull request carries the adoption.** Its front matter holds `promoted_to`
(the name of the new version, for example "Primary demand research default 2") and the scores
against the three benchmarks, and the same pull request adds a row to the
[log](research-default-log.md). The reviewing session checks the five conditions before merging and
assigns the version number. Versions are serial within a family.

**Each log row holds eight items.** These are the change, the adopting experiment, the scored-year
and replicate gains with intervals, the absolute headline score, the three benchmarks' scores on the
family's rows, the stored-forecast name, the commit of the code, and the status "scored-year gain is
an upper bound". A result against an old version keeps its verdict with the status "candidate
(stale)" until a rebase runs. A component that a later ablation removes sets `production_status:
superseded` on its experiment.

**Adoption needs the maintainer's approval in four cases.** These are a new data source, a cost
above 30 pounds sterling or a recurring cost in production, a change that would need a serving-path
change to port, and a change whose code must enter `packages/studies`. Other adoptions run
autonomously under the merge rules in the `ml-experiment` skill.

**A model release is a separate step.** The Primary-demand research default is a GAM stack, and a
model release is a numbered, trained XGBoost model, so most demand adoptions are not release
candidates until the GAM stack has a production design. The `release-model` skill starts from the
log of versions since the last release.

## Retraining cadence

**The cadence is part of the research default, so each family's definition states one, and a change
of cadence is a `change` experiment.** The evidence differs by family.

| Family | Evidence on cadence | Cadence |
|---|---|---|
| Primary demand, XGBoost benchmark BG2 | Weekly beats monthly by 0.123 (experiment 87) to 0.13 (experiment 48) points | Weekly, with a 12-month-window monthly refit as a research proxy ([experiment 85](00085-cheaper-retraining-schedules-rank-arms-like-weekly-demand/result.md)) |
| Primary demand, GAM stack | Weekly minus monthly +0.001, quarterly minus monthly +0.010 (experiment 278) | Monthly |
| Bulk and grid supply points | Weekly beats monthly by 0.160 points (0.066 to 0.247, experiment 87) | To be named with the first default | Weekly |
| Metered solar and wind farms | Monthly minus frozen -0.044 for solar and +0.007 for wind (experiment 48), inside a ±0.05 band | A single fit, or monthly, as the family's first version states |

**Quarterly refits are not used for the XGBoost-based families.** Experiment 85 found that quarterly
refits scored 0.38 points above weekly refits on the same 12-month window, and advised against using
them to compare arms that differ in cadence or in how fast they adapt. For the GAM stack, quarterly
refits save about 19 seconds per arm-year and add a loss of 0.010 points (99% interval -0.014 to
+0.032), concentrated in winter, so they are not shown adequate. Their effect on the power of a
default-plus-x contrast has not been measured.
