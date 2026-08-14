# Intervention log

An append-only record of every occasion on which a human had to intervene in the running service.
This is the artefact that
[T1.1](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself) is
scored from.

It exists because T1.1 is the only test on the
[engineering hypotheses](../design-philosophy/engineering-hypotheses.md) page that **cannot be
measured retrospectively**. Every other test can be reconstructed later — from the scenario suite,
from MLflow timestamps, from the runbooks, or from billing history. "How many times did a human
have to intervene, and why?" is unrecoverable unless it is written down as it happens.

## How to add an entry

Append a row to [the log](#the-log) below. One row per intervention, newest last. The columns are
deliberately few, so that logging an intervention is never the reason an intervention goes
unlogged:

| Column | What goes in it |
|---|---|
| **Date** | `YYYY-MM-DD HH:MM` UTC of the intervention, not of the underlying fault. The time of day matters: T1.1 scores out-of-hours interventions separately, and a date alone cannot be scored for that |
| **Trigger** | What alerted us — a Sentry alarm, a missed check-in, an NGED email, a routine glance at the Dagster UI |
| **Cause** | One of the [cause categories](#cause-taxonomy) below |
| **Minutes** | Human-minutes spent, start to finish, rounded to the nearest 5 |
| **Runbook?** | Did [Operating the live service](operations.md) already cover it? `yes` / `partial` / `no` |
| **Notes** | One sentence. Link the issue or PR if there is one |

An intervention is **any occasion a human had to do something to the running service**, including
the ones that turn out to be trivial. Feature work and deliberate upgrades are not interventions;
unglamorous keep-it-running chores — a credential rotation, a certificate, a dependency bump forced
by an upstream deprecation — are, and belong in `routine-ops`.

A run that failed and then recovered on its own retry is *not* an intervention, but log it anyway,
with `Minutes = 0` and `Cause = self-recovered`. Self-recovery is evidence for the design rather
than against it, and it is only evidence if somebody wrote it down.

"Runbook?" is not bookkeeping. A gap in [operations.md](operations.md) is itself a finding: T1.1
claims a non-expert can run this service from the runbooks alone, and every `no` is a point against
that claim.

## Cause taxonomy

The taxonomy is the substance of the test.
[T1.1](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself) predicts
that **at least 90% of entries fall into `upstream-contract`** — that essentially the only thing
that should ever need a human is an upstream provider changing the shape of what they publish.
Every entry in another category counts against that 90%, which is exactly why the categories are
recorded separately rather than lumped together. The threshold is deliberately not 100%, so a
single fluke cannot falsify the claim on its own.

| Category | Meaning |
|---|---|
| `upstream-contract` | An upstream provider changed a format, schema, column, unit or file layout. **The one category T1.1 predicts.** |
| `upstream-outage` | Upstream data absent or stuck for long enough that a human had to act, without the contract itself changing |
| `infrastructure` | The host, container, scheduler, network or cloud account — anything below our own code |
| `our-bug` | A defect in this codebase |
| `model` | Forecast quality required a human decision — a promotion, a rollback, a retrain |
| `routine-ops` | Keep-it-running work that needed a human without anything having failed: credential rotation, a certificate, a dependency bump forced by an upstream deprecation |
| `self-recovered` | Not an intervention at all — a run that failed and recovered on its own retry, logged with `Minutes = 0` because self-recovery is evidence for the design |

Categories are append-only, like the hypothesis labels themselves. If something genuinely does not
fit, add a category rather than stretching an existing one — a taxonomy bent to fit the data
measures nothing.

## The scoring window opens at v1.0

Log everything from day one, but **score T1.1 only from v1.0 onward**.

While the system is being actively rebuilt through v0.2–v0.9, a good deal of what looks like an
intervention is really development churn, and counting it would spuriously falsify the ≥90%
threshold. The pre-v1.0 entries are still worth having: they are what the cause taxonomy is built
from, and they are the honest record of what the service actually demanded of us on the way up.

The reverse also holds, and matters more. A *quiet* pre-v1.0 stretch does not score in favour of
H1 either. Counting the good weeks of an excluded window while discounting the bad ones would be
the plainest possible case of the selective reading these hypotheses exist to prevent.

## The log

*No interventions recorded yet.*

| Date | Trigger | Cause | Minutes | Runbook? | Notes |
|---|---|---|---|---|---|

## Periods covered

Recording the periods, and not only the entries, is what makes an empty table mean something. An
empty log with no stated period is indistinguishable from a log nobody kept.

| Period | Version | Scope | Interventions | Scores T1.1? |
|---|---|---|---|---|
| 2026-07-15 18:00 UTC → 2026-08-13 19:00 UTC | v0.1 | 32 time series, 6-hourly `live_forecasts` on AWS | 0 | No — pre-v1.0 |
| 2026-08-14 00:00 UTC → ongoing | v0.2 | 6-hourly `live_forecasts` on AWS, every slot checked by `live_forecasts_are_healthy` | 0 | No — pre-v1.0 |

Figures below are stated as of **06:00 UTC on 14 August 2026**. Every count in this section moves
within the day, so the as-of instant is part of the measurement rather than a formality.

### v0.1 on AWS, 2026-07-15 to 2026-08-13

The first `live_forecasts` run on AWS was the 18:00 UTC slot on 15 July 2026, and the last was the
18:00 UTC slot on 13 August 2026, an hour before the v0.1 stack was retired at roughly 19:00 UTC
that evening to make way for v0.2. That is a window of 29 days and 1 hour, over which the schedule
called for 117 consecutive 6-hourly forecast slots and a daily `ecmwf_ens` run, with **zero
interventions and zero observed failures**.

The VM was deployed once, on 15 July, and was not touched again until it was retired: no code was
pushed to AWS during the period. So the window is genuinely unattended rather than quietly
maintained.

*Verified by* counting distinct `power_fcst_init_time` values with `fold_id = "live"` in the
`power_forecasts` Delta table across the period, cross-checked against the Dagster run history and
the Sentry missed-check-in monitor, which never alarmed. That count was last run on 7 August, when
it matched the 91 slots the schedule called for by then; the final 26 slots of the window are
recorded here on the strength of the Dagster run history and the silent monitor alone, and the
distinct-`power_fcst_init_time` count still needs re-running over the closed window.

Three caveats, without which the number would be worth less than it looks:

- **"Zero observed failures" is a weaker claim than "zero failures".** v0.1 implemented very little
  failure detection: `live_forecasts_are_healthy` — the check that reads each slot's rows back and
  reports missed NWP runs — arrives with v0.2, after this window closed. What is
  genuinely verified is that every scheduled slot produced output, not that nothing degraded
  quietly on the way. A silently-stale input is precisely the failure mode this stack is built to
  make visible, and at v0.1 it would not yet be visible.
- **A month is short, and this is the easy case.** v0.1 is 32 time series and one ECMWF run
  per day. The dominant cause T1.1 predicts — an upstream contract change — may simply not have
  happened yet in a window this short. A quiet four weeks is consistent both with "the design
  works" and with "nothing has been thrown at it".
- **It does not score.** The window opens at v1.0, [as above](#the-scoring-window-opens-at-v10).

So what this is, stated plainly: **weak, non-scoring evidence for H1, drawn from a window the
scoring rule excludes and gathered with detection too thin to see quiet degradation.** The deployed
stack served every scheduled slot for 29 days without a human touching it. That is worth recording,
and it is not worth more than that.

### v0.2 on AWS, from 2026-08-14

v0.2 was deployed on the evening of 13 August 2026, replacing the v0.1 stack at roughly 19:00 UTC.
Its first `live_forecasts` run was the 00:00 UTC slot on 14 August 2026, which is where this period
starts. The deployment itself is a deliberate upgrade, so it is not an intervention and has no row
in [the log](#the-log).

The window is too young to say anything about yet. What is different from v0.1, and what makes the
next stretch worth more than the last one, is that `live_forecasts_are_healthy` now reads each
slot's rows back and reports missed NWP runs — so a slot that produces output from stale inputs is
visible rather than silent, and the first caveat above no longer applies from here on.

## See also

- [Engineering Hypotheses → H1](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself)
  — the hypothesis this log scores, and the other three tests that sit alongside T1.1.
- [Operating the live service](operations.md) — the runbooks whose coverage the `Runbook?` column
  measures.
- [Inherent Stability](../design-philosophy/inherent-stability.md) — the design that is meant to
  keep this log short.
