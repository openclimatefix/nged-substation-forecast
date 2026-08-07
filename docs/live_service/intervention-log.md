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
| **Date** | `YYYY-MM-DD` of the intervention, not of the underlying fault |
| **Trigger** | What alerted us — a Sentry alarm, a missed check-in, an NGED email, a routine glance at the Dagster UI |
| **Cause** | One of the [cause categories](#cause-taxonomy) below |
| **Minutes** | Human-minutes spent, start to finish, rounded to the nearest 5 |
| **Runbook?** | Did [Operating the live service](operations.md) already cover it? `yes` / `partial` / `no` |
| **Notes** | One sentence. Link the issue or PR if there is one |

An intervention is **any occasion a human had to do something to the running service that was not
planned work** — including the ones that turn out to be trivial. A run that failed and then
recovered on its own retry is *not* an intervention, but it is worth a note in the same row as the
next real entry, because self-recovery is evidence for the design rather than against it.

"Runbook?" is not bookkeeping. A gap in [operations.md](operations.md) is itself a finding: T1.1
claims a non-expert can run this service from the runbooks alone, and every `no` is a point against
that claim.

## Cause taxonomy

The taxonomy is the substance of the test.
[T1.1](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself) predicts
that **at least 90% of entries fall into `upstream-contract`** — that essentially the only thing
which should ever need a human is an upstream provider changing the shape of what they publish.
Every entry in another category is evidence against the hypothesis, which is exactly why the
categories are recorded separately rather than lumped together.

| Category | Meaning |
|---|---|
| `upstream-contract` | An upstream provider changed a format, schema, column, unit or file layout. **The one category T1.1 predicts.** |
| `upstream-outage` | Upstream data absent or stuck for long enough that a human had to act, without the contract itself changing |
| `infrastructure` | The host, container, scheduler, network or cloud account — anything below our own code |
| `our-bug` | A defect in this codebase |
| `model` | Forecast quality required a human decision — a promotion, a rollback, a retrain |
| `routine-ops` | Planned-ish work that still needed a human: credential rotation, a certificate, a dependency bump forced by an upstream deprecation |

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
| — | — | — | — | — | — |

## Periods covered

Recording the periods, and not only the entries, is what makes an empty table mean something. An
empty log with no stated period is indistinguishable from a log nobody kept.

| Period | Version | Scope | Interventions | Scores T1.1? |
|---|---|---|---|---|
| 2026-07-15 18:00 UTC → ongoing | v0.1 | 32 time series, 6-hourly `live_forecasts` on AWS | 0 | No — pre-v1.0 |

### v0.1 on AWS, from 2026-07-15

The first `live_forecasts` run on AWS was the 18:00 UTC slot on 15 July 2026. As of 7 August 2026
that is 22 days and 12 hours — 91 consecutive 6-hourly forecast slots, and 24 daily `ecmwf_ens`
partitions — with **zero interventions and zero observed failures**. Every expected forecast
exists.

Three caveats, without which the number would be worth less than it looks:

- **"Zero observed failures" is a weaker claim than "zero failures."** v0.1 implements very little
  failure detection — the asset check on `live_forecasts` that reports missed NWP runs is still
  open as [#424](https://github.com/openclimatefix/nged-substation-forecast/issues/424). What is
  genuinely verified is that every scheduled slot produced output, not that nothing degraded
  quietly on the way. A silently-stale input is precisely the failure mode this stack is built to
  make visible, and at v0.1 it would not yet be visible.
- **Three weeks is short, and this is the easy case.** v0.1 is 32 time series and one ECMWF run per
  day. The dominant cause T1.1 predicts — an upstream contract change — may simply not have
  happened yet in a 22-day window. A quiet three weeks is consistent both with "the design works"
  and with "nothing has been thrown at it".
- **It does not score.** The window opens at v1.0, [as above](#the-scoring-window-opens-at-v10).

What it does establish is narrower, and still worth writing down: the deployed stack ran unattended
for three weeks without anyone touching it, which is the precondition for H1 rather than evidence
for it.

## See also

- [Engineering Hypotheses → H1](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself)
  — the hypothesis this log scores, and the other three tests that sit alongside T1.1.
- [Operating the live service](operations.md) — the runbooks whose coverage the `Runbook?` column
  measures.
- [Inherent Stability](../design-philosophy/inherent-stability.md) — the design that is meant to
  keep this log short.
