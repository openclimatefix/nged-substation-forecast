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

| Date | Trigger | Cause | Minutes | Runbook? | Notes |
|---|---|---|---|---|---|
| 2026-08-13 19:56 | Sentry alarm — but from the pre-v0.2 code on a laptop, not from AWS | `upstream-outage` | <5 | yes | Dynamical.org first published the 2026-08-09 00Z ECMWF run with a variable wholly missing, and v0.1 treated that as fatal, so the partition was re-materialised by hand. Four forecast slots ran on the previous day's run in the meantime. [#493](https://github.com/openclimatefix/nged-substation-forecast/pull/493) added the retry that covers it |

## Periods covered

Recording the periods, and not only the entries, is what makes an empty table mean something. An
empty log with no stated period is indistinguishable from a log nobody kept.

| Period | Version | Scope | Interventions | Scores T1.1? |
|---|---|---|---|---|
| 2026-07-15 18:00 UTC → 2026-08-14 00:00 UTC | v0.1 | 28 time series, 6-hourly `live_forecasts` on AWS | 1 | No — pre-v1.0 |
| 2026-08-14 00:00 UTC → ongoing | v0.2 | 31 time series, 6-hourly `live_forecasts` on AWS, with `live_forecasts_are_healthy` reporting on each slot | 0 | No — pre-v1.0 |

Figures below are stated as of **07:00 UTC on 14 August 2026**. Every count in this section moves
within the day, so the as-of instant is part of the measurement rather than a formality.

### v0.1 on AWS, 2026-07-15 to 2026-08-13

The first `live_forecasts` run on AWS was the 18:00 UTC slot on 15 July 2026, and the last was the
18:00 UTC slot on 13 August 2026, an hour before the v0.1 stack was retired at roughly 19:00 UTC
that evening to make way for v0.2. Over that window the schedule called for 117 consecutive
6-hourly forecast slots, and **every one of them produced a forecast for all 28 time series**. One
ECMWF run was lost and one human intervention was needed, both described below.

The VM was deployed once, on 15 July, and no code was pushed to AWS until it was retired. The one
operator action in the window was the NWP backfill logged above, so the period is close to
unattended but not entirely so.

*Verified by* counting distinct `power_fcst_init_time` values with `fold_id = "live"` and
`experiment_name = "xgboost_cv_0001"` — v0.1's promoted model — in the `power_forecasts` Delta
table on S3. All 117 scheduled slots are present, every consecutive pair is exactly six hours
apart, and all 28 time series appear in every one of the 117.

### The 9 August ECMWF run, and what it shows

Dynamical.org publishes each ECMWF run as roughly 40 separate Icechunk commits, so a run can be
readable and incomplete at the same time. The 2026-08-09 00Z run was first published with a weather
variable wholly missing, and repaired 3 hours 25 minutes later. v0.1 treated a wholly-missing
variable as fatal, so its `ecmwf_ens` run failed and no partition was written for 9 August. The
partition was re-materialised by hand on 13 August at 19:56 UTC.

The forecast did not stop. Four slots — 2026-08-09 12:00 UTC through 2026-08-10 06:00 UTC — ran on
the 8 August 00Z run instead, at 36, 42, 48 and 54 hours old against the 12–30 hours a healthy slot
sees. Every other slot in the window used NWP no older than 30 hours.

That is [Principle 1](../design-philosophy/design-principles.md) — the power forecast never stops —
working in production rather than on paper, and it is the more interesting result on this page. A
missing input degraded the forecast instead of stopping it, and the degradation was bounded and
visible after the fact. v0.2 closes the gap that made the intervention necessary at all: a
wholly-missing variable is now a retryable "not ready yet" rather than a fatal error, with a
four-hour retry budget that covers the 3h25m this republication took.

Three caveats, without which the window would be worth more than it is:

- **The deployment had no telemetry at all, so nothing on AWS *could* have alerted.** The earliest
  Sentry commit of any kind — the failure hook, the check-in and the freshness warning all arrived
  in the same week — lands on `main` on 21 July, six days after the box was deployed and never
  updated. Whatever was running there therefore predates Sentry entirely: the missed-check-in
  monitor never existed on that box, and the alarm that did surface the missed run came from newer
  code running on a laptop.
  `live_forecasts_are_healthy`, the check that reads each slot's rows back and counts missed NWP
  runs, landed later still. The four degraded slots were reconstructable only because
  `nwp_init_time` travels on every forecast row: the degradation was recoverable from the data, but
  nothing in the deployment announced it.
- **A month is short, and this is the easy case.** v0.1 is 28 time series and one ECMWF run
  per day. The dominant cause T1.1 predicts — an upstream contract change — did not happen in a
  window this short; a partial publication is a milder fault than a changed schema.
- **It does not score.** The window opens at v1.0, [as above](#the-scoring-window-opens-at-v10).

So what this is, stated plainly: **weak, non-scoring evidence for H1, drawn from a window the
scoring rule excludes.** The deployed stack served all 117 scheduled slots over 29 days, absorbed
one lost NWP run by degrading rather than stopping, and cost a human about a minute. That is worth
recording, and it is not worth more than that.

### v0.2 on AWS, from 2026-08-14

v0.2 was deployed on the evening of 13 August 2026, replacing the v0.1 stack at roughly 19:00 UTC.
Its first `live_forecasts` run was the 00:00 UTC slot on 14 August 2026, which is where this period
starts. The deployment itself is a deliberate upgrade, so it is not an intervention and has no row
in [the log](#the-log).

v0.2 forecasts 31 time series, three more than v0.1, under the promoted model
`xgboost_cv_0003`. The 00:00 and 06:00 UTC slots on 14 August both carry all 31.

Three things make the next stretch better evidence than the last. `live_forecasts_are_healthy`
reads each succeeding slot's rows back and reports missed NWP runs, so a slot forecasting from
stale inputs is recorded as degraded rather than passing unremarked. A wholly-missing NWP variable
is now retried for four hours instead of failing, which is what would have made the 9 August
intervention unnecessary. And the deployment carries Sentry, which v0.1's never did, so an
`ecmwf_ens` run that *does* exhaust its retries reports itself from AWS rather than waiting for
somebody to run the code on a laptop.

What still would not reach us is the degraded *slot*, where nothing failed. Dagster runs no check
for an asset that raised, so this is the succeeding-run case: `live_forecasts_are_healthy` returns
its warning to the Checks view and sends nothing to Sentry, and the slot's check-in reports the
service healthy regardless, so a degraded run looks like a good one from outside — the gap at
[#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501).

## See also

- [Engineering Hypotheses → H1](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself)
  — the hypothesis this log scores, and the other three tests that sit alongside T1.1.
- [Operating the live service](operations.md) — the runbooks whose coverage the `Runbook?` column
  measures.
- [Inherent Stability](../design-philosophy/inherent-stability.md) — the design that is meant to
  keep this log short.
