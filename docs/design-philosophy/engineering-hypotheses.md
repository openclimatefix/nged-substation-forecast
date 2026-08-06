# Engineering Hypotheses

This page states what we claim Flexpectation's engineering will achieve, how each claim is tested,
and what would falsify it. It is deliberately written as a set of **hypotheses with numbers**
rather than a set of aims.

Its counterpart is the [design principles](design-principles.md) list — the
bets we are making in order to achieve the hypotheses below. The relationship between the two, and
the admission test it implies, are stated on that page.

## Why hypotheses rather than aims

Three reasons.

**NIA funding is for transferable learning, including negative results.** A pre-registered
threshold that we then miss is a publishable finding; an aim that we quietly fall short of is not.
Six report issues ([#128](https://github.com/openclimatefix/nged-substation-forecast/issues/128),
[#130](https://github.com/openclimatefix/nged-substation-forecast/issues/130),
[#131](https://github.com/openclimatefix/nged-substation-forecast/issues/131),
[#132](https://github.com/openclimatefix/nged-substation-forecast/issues/132),
[#135](https://github.com/openclimatefix/nged-substation-forecast/issues/135),
[#156](https://github.com/openclimatefix/nged-substation-forecast/issues/156)) are the natural
consumers.

**It converts arguments into measurements.** Several of the claims below — particularly the ones
about operational burden — are genuinely disputed, and a document that argues one side persuades
nobody. A number that resolves itself is a better outcome, and pre-registering one signals
confidence in a way that prose cannot.

**It forces the measurement artifacts to exist in advance.** Most of these tests need something
built before they can be scored: a baseline forecaster, a scenario suite, an intervention log.
Writing the hypothesis first is what puts those on the roadmap early enough to be useful.

The commitment this entails is real: a hypothesis without a number is an aim wearing a lab coat.
Each one below has a threshold and a window, and we must be willing to record a falsification.

## The claims

**H*n*** is a hypothesis; **T*n.m*** is its *m*th **test**. These labels are cited from issues and
from NIA reports, so **append, never renumber**.

| | Test | Threshold | Resolvable |
|---|---|---|---|
| T1.1 | Operability | ≥90% of interventions caused by an upstream format change; zero out-of-hours | ~2 quarters of v1.0 |
| T1.2 | Graceful degradation | Every series emits a forecast; beats `nged_incumbent` at rungs 0–2 | v0.3, after [#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) |
| T1.3 | Faithful uncertainty | PICP within ±5 percentage points of nominal in every degradation regime (tolerance provisional until [#443](https://github.com/openclimatefix/nged-substation-forecast/issues/443)) | v0.5 |
| T2.1 | Experiment throughput | ≥100 registered experiments per person, in a peak month | v0.5 |
| T3.1 | Promotion effort | ≤1 command | v0.3 |
| T3.2 | Rollback effort | ≤1 command | v0.3 |
| T4.1 | Cost at v1 scale | ≤£50/month for the whole running service | v1.0, from the AWS bill |
| T4.2 | Cost at v2 scale | ≤£200/month for the whole running service | v2 |
| T5.1 | Operability by a non-expert | Every game-day exercise recovered from the runbooks alone | Handover phase |
| T6.1 | Scale without redesign | No structural change forced by going from 32 to ~2,500 series | v2 |

## H1 — a service that mostly runs itself

> Manual attention is needed only when an upstream input format changes. The service degrades
> gracefully and legibly, and it propagates uncertainty faithfully.

The design that is meant to deliver this is [Inherent Stability](inherent-stability.md).
The background argument for *why* leniency is affordable is
[Uptime: lenient by design](../background/requirements.md#uptime-lenient-by-design) — but note that
this is a defensive argument (an outage does not cost much) whereas H1 is a stronger positive claim
(interventions will be rare). The positive claim is the one actually in dispute, and a sceptic is
not moved by "it's fine when it breaks".

**T1.1 — Operability.** Interventions per quarter, classified by cause, counted from the
[intervention log](#the-intervention-log). The headline quote says "only"; the test operationalises
that at ≥90% so a single fluke cannot falsify the claim on its own. "Zero out-of-hours" means no
intervention is ever needed outside normal working hours — the posture that makes
next-business-day recovery honest. This is the only test on this page that cannot be measured
retrospectively.

**T1.2 — Graceful degradation.** Run the failure-scenario suite across every time series and check
two things: that a forecast is emitted at all, and that it still beats `nged_incumbent`. Blocked on
[#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147), which builds the
baseline to compare against.

**T1.3 — Faithful uncertainty.** PICP and pinball loss computed per degradation regime, from the
leaderboard's scenario dimension. This is the sharpest test we have, and it needs no new metric —
only the scenario dimension, which is the same machinery T1.2 uses. The tolerance is pre-registered
at ±5 percentage points (a nominal 90% interval must achieve 85–95% coverage in every regime);
[#443](https://github.com/openclimatefix/nged-substation-forecast/issues/443) may tighten it before
the first scoring, but any change must be recorded here.

## H2 — a hundred experiments per person in a peak month

> When experimentation is the active workstream, one person can register at least 100 leaderboard
> experiments in a month.

The requirement this comes from is
[ML experimentation at scale](../background/requirements.md#ml-experimentation-at-scale).

**T2.1 — Experiment throughput.** Registered leaderboard experiments per person per month, taken
from MLflow run timestamps. Two framing choices are deliberate.

It is a **peak** claim. There will be months spent hardening the production service or writing
documentation, and a quiet month is not a falsification — the claim is about what the machinery
allows when we lean on it.

And it is **count-only**. A single N-configuration sweep can inflate the number, and we accept
that, because a simple count that MLflow already records beats a "decision-grade experiments"
qualifier that would need a human-effort log to measure. If the threshold is ever met only by
config sweeps, that will be obvious from reading the runs, and the fix is to *append a T2.2* then —
never to redefine T2.1.

## H3 — one-click promotion, and one-click rollback

> Moving the leaderboard winner into production, or reverting it, is a single command each way.

**T3.1 — Promotion effort.** Commands required to get from "the leaderboard says X won" to "X is
serving", following the runbook. The mechanism already exists (the `promoted_model` asset — see
[Production Deployment](../architecture/production-deployment.md#promote-the-champion-via-a-dagster-asset-not-a-script));
what is missing is a runbook that pins down what counts as one command.

**T3.2 — Rollback effort.** Commands required to get from "X is serving" back to the previous
champion. Promotion without rollback is not safe at any speed, and rollback is the damping half of
[inherent stability](inherent-stability.md#the-rules) — so this is not the optional
half of H3.

## H4 — it runs for pocket money

> The whole running service costs under £50/month at v1 scale and under £200/month at v2 scale.

This is probably the most transferable finding of the set, and it is a second, independent answer to
the worry that a service like this must carry heavy operational overhead. The estimates it is pinned to are in
[AWS Running Costs](../architecture/aws-costs.md): ~£25–35/month at v1 and a projected ~£70–140/month
at v2. The thresholds sit above those estimates deliberately, so that the hypothesis is a claim
about the architecture rather than a restatement of the spreadsheet.

**T4.1 and T4.2 — Cost.** Read the monthly AWS bill. No per-experiment instrumentation is needed:
training today runs on laptops, the planned weekly AWS retrain is bounded at ~£1/month, and a
backtest on AWS is bounded at well under £1 per run at v1 scale — all inside the bill either way.

## H5 — operable by a non-expert

> An NGED operator can run the service day to day from the runbooks alone.

This is already designed as the **operator contract** in
[Handover to NGED](../roadmap/handover.md#1-the-operator-contract) — an enumeration of roughly ten or
fewer actions the operator is ever expected to take. Framing it as a hypothesis turns the
[game days](../roadmap/handover.md#6-game-days-and-in-person-training) into a measurement rather than
purely a training exercise.

**T5.1 — Operability by a non-expert.** During the game days, the NGED operator recovers from each
scripted failure — NWP feed broken, disk full, daemon killed, credential expired, forecast slot
missed — unaided, using only the runbooks. An exercise that needs OCF intervention is a
falsification of T5.1 for that exercise, and a documentation bug to fix.

## H6 — scale without redesign

> The architecture goes from 32 to ~2,500 time series without structural change.

This is the central engineering bet of the project. It is what justifies building the v2
architecture during v1 rather than prototyping first and rewriting later, and it is only truly
resolvable at v2 — which is an argument for writing it down now, while the prediction still costs
something to make.

**T6.1 — Scale without redesign.** At v2, no change was forced by scale alone to the data
contracts, the asset graph, or the storage layout. Changes to configuration, partition counts and
machine sizes do not count against it; a new table, a changed schema, or a restructured asset graph
does.

## The intervention log

**T1.1 is the only test on this page that cannot be measured retrospectively.** Every other test can
be reconstructed later: T1.2 and T1.3 are computed by re-running the scenario suite and the
leaderboard, T2.1 comes from MLflow timestamps that live forever, T3.1 and T3.2 can be counted
whenever the runbooks exist, T4.1 and T4.2 come from billing history, and T5.1 and T6.1 are
measured at events that have not happened yet. But "how many times did a human have to intervene,
and why?" is unrecoverable unless it is recorded as it happens.

Two clauses keep that measurement honest.

The artifact is deliberately cheap — an append-only log with the date, the trigger, a cause
category, the human-minutes spent, and whether a runbook already existed — so there is no excuse for
not keeping it.

But its **measurement window opens at v1.0**. Interventions during v0.2–v0.9, while the system is
being actively rebuilt, are development churn, and counting them would spuriously falsify the
≥90%-upstream threshold. Log everything from day one — the pre-v1.0 entries still feed the cause
taxonomy — but score T1.1 only from v1.0 onward.

The cause taxonomy is the substance of the test, since T1.1 predicts that essentially every entry
falls into "upstream format or contract change".

## Recording a falsification

If a test fails, the result is recorded here rather than quietly dropped: the threshold stays, and
the hypothesis gains a short note saying what was measured, when, and what we think the cause was.
A falsified engineering hypothesis is one of the more useful things this project can hand to the
next DNO that tries it.

Six hypotheses is the sensible ceiling. Each one carries a measurement cost, and a page of thirty
claims nobody scores is worse than a page of six that are actually resolved.
