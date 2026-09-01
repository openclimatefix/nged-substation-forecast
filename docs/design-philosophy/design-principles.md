# Design Principles

The principles below govern this project's engineering decisions — what gets built, where code
lives, and which technologies are adopted. Each principle is stated compactly here, with the full
argument in the pages it links to — the list below is an index, not the argument. Each entry says
concretely what the principle buys, in several sections:

- The "**Without it**" section describes the everyday failure the principle exists to prevent,
- "**Decided**" names a real decision it made (informed by this principle),
- "**Serves**" names the [engineering hypothesis](engineering-hypotheses.md) it is a bet on, and
- "**Detail**" links the full argument.

Some of these principles are, frankly, obvious — but a principle does not have to be surprising to
be useful. Writing them all down is what makes them explicit and checkable: a design proposal can
then be tested against the *whole* list, rather than against whichever two or three principles
happen to be at the front of the reviewer's mind that day.

Checking proposals against the whole list this way pays off twice over now that much of this repo
is written with an AI coding agent, because an agent can check every new addition against the whole
list on every change — precisely the job a human reviewer's attention is too scarce to do reliably.
Principles that live only in an experienced engineer's head cannot be applied that way; written-down
principles can.

The principles are **bets, not truths**: each one is scored by the [engineering
hypotheses](engineering-hypotheses.md). A principle that fails its test will be reported as a
negative result, not rewritten out of the record. The defence against merely collecting fashionable
engineering trends is the admission test described below.

## Where these principles come from

The framing — a greenfield chance to test-drive other industries' best practice and produce a field
report rather than a rulebook — is stated on the [section overview](index.md).

We deliberately researched best practices across multiple industries. Several of the ideas on these
pages are borrowed from disciplines that have been solving the same shape of problem for longer than
data engineering has existed: *inherent stability* comes from vehicle dynamics, *fail-operational*
from avionics autoland (the flight-deck technology that fully automates the landing phase) and
ISO 26262, *blast radius* from site reliability engineering, and mobile telecoms supplies the case
that a heavily-corrupted channel can still run genuinely unattended — see
[H1, a service that mostly runs
itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself). Borrowing across disciplines
is the point, not a flourish: some of them have been shipping safety-critical systems for fifty
years.

Not every borrowed idea survives contact with reality, and we record those outcomes too — that is
what makes this a field report rather than a manifesto. *Error budgets* were examined and declined
([Deliberately absent](#deliberately-absent) below); [five practices we respect are not yet
absorbed](#industry-best-practices-we-have-not-yet-absorbed); and Postel's law is named on the
[Inherent
Stability](inherent-stability.md#not-postels-law) page precisely so that nobody mistakes it for what
we do.

## How principles relate to hypotheses

Principles are constraints on *decisions*; in contrast, hypotheses are claims about *outcomes*. A
hypothesis can be falsified by a measurement. A principle cannot — it is a decision rule, not a
factual claim — so it is retired by different routes: the hypothesis it serves is falsified, which
discredits the bet behind it; or measurements override it in specific decisions often enough that
the exceptions become the rule; or it turns out never to actually decide anything, and is deleted
as decoration. The two are connected in one direction: the principles
are the bets we are making in order to test the hypotheses. That gives us a test for admission to
this list — **name the hypothesis it serves and a decision it actually decided**. A principle
serving no hypothesis is either decoration or a sign of a missing hypothesis; a hypothesis with no
principle behind it is a claim we are merely hoping comes true.

## The principles

### 1 — The power forecast never stops

If data inputs are disrupted, the forecast gets less certain instead of stopping — the full
argument, the degradation ladder and the failure-modes table are in [Inherent
Stability](inherent-stability.md). An error in **our own code** — an empty promoted model, a
contract violation, a join that has fanned out — gets the opposite posture: fail as early as
possible, because degrading around our own bug would deliver a wrong forecast and bury its cause.
How far such a failure then spreads is a different axis, and the business of
[principle 10 ("*every write is atomic and idempotent, and every failure is confined to one
partition*")](#10-every-write-is-atomic-and-idempotent-and-every-failure-is-confined-to-one-partition)
rather than of this one. The plan is to deliver graceful degradation through the **model itself** —
a machine-learning (ML) model that can, at least partially, handle missing inputs — rather than
through fallback logic wrapped around a model that assumes complete data. (Note that this decision
to "never stop" will not be appropriate for energy-forecasting systems where an uncertain forecast
might be more harmful than *no* forecast. But, in Flexpectation, there are strong arguments that
our forecast will *always* be better than NGED's incumbent baseline, even when we have no live
data.)

Degrading is only half the principle: **we must be notified that the forecast degraded.** Three
channels carry that — a widened uncertainty band on the forecast row (designed, not yet built 🚧 —
see [Widening bands](inherent-stability.md#widening-bands-the-in-band-signal)), a
`power_forecast_warnings` row naming which feed degraded and since when (not yet built 🚧), and a
Sentry event for the data failures a human can act on, such as a missed daily ECMWF run that says
Dynamical.org is having problems. Each answers a different question and none substitutes for
another; all three are required, and two of the three are still to be built. What each is for, and
who reads it, is set out in
[Three audiences, three channels](inherent-stability.md#three-audiences-three-channels). A Sentry
event says whether we broke or an input degraded, and both kinds have to be delivered — see
[Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence),
which defines the two kinds.

*Without it:* every wobble in an upstream feed becomes an outage — the service raises at 06:00
because one meter went quiet, NGED open their dashboard to a gap instead of a forecast, and a
developer spends the morning re-running a pipeline whose only real problem was a missing input.

*Decided:* every asset check in the repo is non-blocking `WARN`; there is deliberately no
`ERROR`-severity check anywhere. Non-blocking never means non-notifying: a check that detects
degradation must still send a Sentry event, without failing the run it is warning about. On the
other side of the line, `PowerForecast.validate` hard-fails a whole slot on a duplicated primary
key, because only a bug of ours can duplicate a forecast row: every numerical weather prediction
(NWP) write replaces its partition, so the duplication cannot come from the data at rest. One
production raise is not yet on the right side of that line: a sustained NWP outage still fails
`live_forecasts` rather than degrading to a weather-blind forecast
([#446](https://github.com/openclimatefix/nged-substation-forecast/issues/446)).

*Serves:* [Hypothesis 1: a service that mostly runs
itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself).

*Detail:* [Inherent Stability](inherent-stability.md), whose
[rules](inherent-stability.md#the-rules) are the fine-grained form of this principle together with
the complexity-offline and strict-contracts principles.

### 2 — Complexity belongs offline, not in the serving path

When a capability could be built into the training loop *or* into the production service, build it
into the training loop — training runs unattended, in front of nobody; see [Where complexity should
live](inherent-stability.md#where-complexity-should-live) for why that asymmetry decides it.
Production forecasting systems commonly solve real problems *in* the serving path — a post-processing
step that corrects for recent forecast errors, a switch to a separately-trained fallback model when
an input feed is down, a blend of models specialised per horizon — and each of those is a reasonable
answer to a real need. The bet this principle makes is that the same needs can be met by training a
single model to handle whatever gets thrown at it, for example: recent-error correction learned from
lagged-power features, missing-input tolerance trained in rather than switched to, one model spanning
the whole horizon. Whether that bet actually improves the forecast, and where it trades skill away,
is exactly what the planned failure-scenario suite is meant to measure.

*Without it:* the serving path grows `if-then-else` branches — corrections, fallbacks, blends —
that are easy to leave under-exercised: unless a team deliberately tests all these execution paths,
their first real execution happens during an incident, unattended.

*Decided:* `promoted_model` copies the champion to local disk, so production inference makes no
MLflow call at all.

*Serves:* [Hypothesis 1: a service that mostly runs
itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
[Hypothesis 3: one-click promotion, and one-click
rollback](engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback).

*Detail:* [Where complexity should live](inherent-stability.md#where-complexity-should-live) —
including the two qualifiers that keep this principle honest — and [Bake the model into the
image](../architecture/production-deployment.md#bake-the-model-into-the-image-at-build-time).

### 3 — One execution path from research to production

The code that implements the best-performing model in R&D *is* the code we deploy. Note the
direction of travel: this is emphatically **not** "push the research notebook to production" — it
is the opposite. Research runs on the production pipeline, so research code is held to production
standards from the first experiment onwards. Exploratory notebooks are still allowed, but an idea
becomes an *experiment* — a candidate that can enter the leaderboard, and therefore be promoted to
production — only once it is implemented in the pipeline's own code, behind the same data contracts
and tests as everything else. There is then no "now rewrite the research code for production" step,
because there was never a second, scruffier implementation to rewrite (and this should *accelerate*
research, not slow it down). This is also what makes a one-command promotion *safe* rather than
merely fast: the model that won the leaderboard is, bit for bit, the model that serves, so there is
no re-implementation whose divergence from the measured version can only be discovered in
production.

*Without it:* research code is rewritten for production, the two implementations drift apart, and
the deployed model no longer does what the winning experiment measured. And it takes *longer* to
get the best model into production, which is bad for users, and bad for developers too because the
*true* test of forecasting skill is to test the model in production. The price of avoiding that is
real — every experiment has to run on production-quality pipeline code from day one — and we pay it
deliberately.

*Decided:* this was the argument that settled Dagster over Airflow, and it is why splitting the
live service onto a second orchestrator remains rejected.

*Serves:* [Hypothesis 2: a hundred experiments per person in a peak
month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
[Hypothesis 3: one-click promotion, and one-click
rollback](engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback).

*Detail:* [Nothing gets rewritten on the way to
production](../ml_experimentation/mlops-approach.md#nothing-gets-rewritten-on-the-way-to-production),
[Why Dagster, not Airflow?](../architecture/why-dagster-not-airflow.md).

### 4 — An experiment must be cheap to try, and cheap to abandon

Most research ideas fail — that is the normal condition of research, not a symptom of a bad team —
so the number of good ideas a project finds is set by how many it can attempt, which is in turn set
by what a single attempt costs: wall-clock time, money, and how much shared code has to be touched.
We treat that marginal cost as a design constraint in its own right, rather than as a pleasant
side-effect of tidy architecture. It is also the deliberate counterweight to [principle 3
("*one execution path from research to
production*")](#3-one-execution-path-from-research-to-production): insisting that every experiment
runs on production-quality pipeline code is only affordable while an experiment stays cheap, so
when that price starts to climb, the answer is to make the shared pipeline easier to extend — never
to quietly reopen a research-only shortcut.

*Without it:* trying an unusual idea means editing code that every other experiment depends on, so
a speculative idea has to clear a far higher bar than a safe idea before it is worth attempting —
and since the wilder ideas are also the likeliest to fail, the cost of failure is exactly what
decides whether they are ever tried at all.

*Decided:* a new model family is a
[`BaseForecaster`](../api/ml_core/index.md#ml_core.base_forecaster.BaseForecaster) subclass, and a
model that wants a different view of the data — a CNN wanting a spatial NWP crop rather than a
table — swaps its
[`feature_engineer`](../api/ml_core/index.md#ml_core.base_forecaster.BaseForecaster.feature_engineer)
for a different
[`FeatureEngineer`](../api/ml_core/index.md#ml_core.features.FeatureEngineer) implementation,
instead of editing shared feature-engineering code; most experiments are
a config change rather than a code change, and can be tried without provisioning anything; and an
abandoned experiment leaves nothing behind but its own partition, which nothing else depends on.

*Serves:*
[Hypothesis 2: a hundred experiments per person in a peak month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
[Hypothesis 4: it runs for pocket money](engineering-hypotheses.md#h4-it-runs-for-pocket-money).

*Detail:*
[The Universal Model Interface](../architecture/overview.md#the-universal-model-interface),
[Tweaking a config for an experiment](../ml_experimentation/model-configuration.md#tweaking-a-config-for-an-experiment),
[Getting started on your laptop](../getting-started.md).

### 5 — Everything around the model is general-purpose

The pipeline is agnostic to the ML model, the geography, and the energy resource being forecast.
The code that runs *before* the ML model — ingest, feature engineering, cross-validation fold
construction — and the code that runs *after* it — metrics, the leaderboard, the production
inference loop, delivery — is written to serve any model and any set of sites. Knowledge of the *ML
model* is confined behind the `BaseForecaster` interface. Knowledge of the *geographical place*
enters as data and configuration — coordinates, capacities, a boundary polygon, a timezone — and is
confined to a thin, named layer rather than sprinkled through the pipeline. This is not portability
for its own sake, and we are not paying today to make some hypothetical other country easier
tomorrow: we want it because it is the same property that lets a new model be tried without editing
shared code ([principle 4, "*an experiment must be cheap to try, and cheap to
abandon*"](#4-an-experiment-must-be-cheap-to-try-and-cheap-to-abandon)), and that lets one pipeline
serve 32 time series or 2,500 without being rewritten. What makes this affordable rather than
merely tidy is lazy evaluation: because each layer only ever describes work, the engine can fuse
the layers' constraints at the end and pay nothing for their mutual ignorance — see [principle 11
("*push the work down to the query engine; materialise once, as late as
possible*")](#11-push-the-work-down-to-the-query-engine-materialise-once-as-late-as-possible).

*Without it:* the pipeline accumulates `if model_type == …` branches and local assumptions — a
hard-coded timezone, an assumed half-hourly settlement period, a national voltage taxonomy — and
each one becomes a place where a new model, a new data source or a new region has to fork the
shared code instead of plugging into it.

*Decided:* lags and rolling windows are expressed as **durations** rather than as counts of
half-hour periods, so the reporting interval is not baked into the feature grammar; the H3 gridding
helper accepts any polygon, and the NWP download bounds are derived at runtime from whatever grid
it is handed rather than from a hard-coded bounding box; `BaseForecaster` deliberately permits one
sub-model per time series, a single global model, or anything in between.

*Serves:* [Hypothesis 5: scale without
redesign](engineering-hypotheses.md#h5-scale-without-redesign), [Hypothesis 2: a hundred
experiments per person in a peak
month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month).

*Detail:* [The Universal Model Interface](../architecture/overview.md#the-universal-model-interface).
The principle is not yet fully honoured: the place-specific assumptions do sit in one thin layer
(mostly the `contracts` package), but two hard-coded `"Europe/London"` defaults sit outside it —
the dashboard's `DISPLAY_TIME_ZONE` and `ml_core`'s `DEFAULT_LOCAL_TIMEZONE` — which is precisely the
leak the principle exists to prevent. `ml_core`'s default reaches `FeatureEngineer.engineer()` as
an overridable parameter, but the value itself still lives in general-purpose `ml_core` code
rather than in `contracts`, so it remains an instance of the leak.

### 6 — The whole system must be exercisable on one laptop

The laptop runs the same code that runs in the cloud. Downloading the weather archive, training, a
full 51-member backtest, running the live service, opening the dashboards — all of it has to be
runnable end-to-end on a single ordinary machine, with no cloud account, nothing to provision, and
no containers to orchestrate. What differs between a laptop and the production deployment is
*configuration* — where the storage roots point, which MLflow tracking server is used, which
environment tag telemetry carries — never the code path. In contrast, a "laptop mode" implemented
as a separate, simplified code path would be a second system to maintain, and debugging it would
teach you about the toy rather than about production. (This is the instinct of [principle 3 ("*one
execution path from research to production*")](#3-one-execution-path-from-research-to-production)
applied to a different axis: principle 3 forbids a second implementation for research versus
production; this one forbids a second implementation for laptop versus cloud.) This is a constraint
on design rather than a happy accident: it rules out architectures that can only be exercised on a
cluster, and it is a large part of why the storage layout was optimised for size. A year of the
full 51-member ECMWF ENS archive over Great Britain is ~40 GB on disk and takes roughly a minute
per day of data to download and convert — an evening's work for a year of weather, on a domestic
Internet connection.

This principle's limit at v2 scale is worth stating plainly, and it is narrower than it might
sound. Nothing here casts doubt on whether the architecture *serves* ~2,500 time series — that is
[Hypothesis 5](engineering-hypotheses.md#h5-scale-without-redesign), and the production service
runs on a rented machine sized for the job. What v2 scale tests is the stronger, additional claim
that the *whole* system still fits on one developer's laptop. Even there the likely pinch is
wall-clock rather than feasibility — a full v2-scale 51-member backtest may simply take too long to
sit through — and the ordinary answer is to develop against a subset of time series, which costs
nothing this principle cares about, since the loop still closes locally. If even that turns out not
to hold, it is a result to report, not a rule to quietly drop.

*Without it:* the debugging loop runs through somebody else's computer. Reproducing a failure needs
credentials, a provisioned cluster, and a wait, so the loop closes a handful of times a day instead
of continuously, and a newcomer's first day goes on access requests rather than on running the
system. That cost has grown over the last few years rather than shrunk: a coding agent can only
close a loop it is able to run unaided. A system that is fully exercisable on one machine is
therefore one an agent can be pointed at, while a system whose failures only reproduce in the cloud
is not.

*Decided:* the project consistently defaults to what a laptop can run, and lets configuration —
not a second code path — point at the cloud instead:

- the data-table roots default to **local paths** and take an `s3://` URI only by configuration;
- the MLflow tracking URI defaults to a local SQLite file and names a server only by configuration;
- Sentry separates a laptop from the production box by an `environment` tag rather than by a
  different code path — *how* those values arrive differs by compute (a `.env` file on a laptop,
  container environment variables on AWS) but `Settings` reads them identically, and a laptop can
  even rehearse the full object-store path against a local MinIO server;
- a fresh clone reaches a trained model without any cloud resource;
- input pruning and `init_time` chunking were sized by what a laptop has (a full 51-member
  validation prediction, ~321M rows, peaks at ~9 GB) rather than by what an instance could be
  rented with; and
- the 13-bit significand rounding that makes the archive locally storable was adopted on measured,
  not assumed, compression.

*Serves:* [Hypothesis 2: a hundred experiments per person in a peak
month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
[Hypothesis 4: it runs for pocket money](engineering-hypotheses.md#h4-it-runs-for-pocket-money).

*Detail:* [Which settings for which
environment](../live_service/setup.md#at-a-glance-which-settings-for-which-environment),
[Performance and Scale](../architecture/performance.md), [Getting started on your
laptop](../getting-started.md).

### 7 — Strict contracts at every boundary

We are liberal about missing inputs and strict about malformed inputs. Every tabular boundary is
a Patito schema, validated rather than assumed. This is the deliberate *opposite* of Postel's
law, and it is what stops [principle 1 ("*the power forecast never
stops*")](#1-the-power-forecast-never-stops) from decaying into "accept anything and hope".
Strictness also has a granularity: reject structurally-broken data outright, but tolerate
locally-corrupt values a model can absorb — throwing away an otherwise-good NWP run because a few
percent of its pixels are null would convert a tolerable problem into an outage.

*Without it:* a malformed file does not crash anything — instead it quietly (and insidiously)
shifts forecasts until someone notices the units are wrong. The failure mode is silent corruption
rather than a loud rejection at the boundary.

*Decided:* `AllFeatures`, `PowerForecast`, and the rest are validated schemas rather than
conventions, and every `PowerForecast` row is self-describing. `Nwp.validate` rejects a
de-accumulated variable that is null in *every* slice beyond lead-0 — a column carrying no weather
at all — but tolerates every smaller null pattern, from scattered per-pixel corruption up to a
whole `(ensemble_member, valid_time)` slice, which the `nwp_has_no_unexpected_nulls` check reports
as a `WARN`. That line is this principle's granularity clause doing real work: a run can arrive
with 2 of one variable's 4284 `(member, step)` slices empty, and rejecting it over 0.05% of a
single already-nullable variable is exactly the tolerable-problem-into-an-outage trade the clause
exists to forbid.

*Serves:*
[Hypothesis 1: a service that mostly runs itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
[Hypothesis 5: scale without redesign](engineering-hypotheses.md#h5-scale-without-redesign).

*Detail:*
[Strict data contracts](../architecture/forecast-delivery.md#strict-data-contracts-machine-verifiable),
[Not Postel's law](inherent-stability.md#not-postels-law),
[The guiding principle](../architecture/ecmwf-ens-known-issues.md#the-guiding-principle).

### 8 — Every experiment is scored identically

What varies is the model, never the measurement. A leaderboard is only worth having if two numbers
on it are genuinely comparable, which requires the population, the folds, the metric definitions,
and the pipeline to be held constant *by construction rather than by discipline*. Note what this
does **not** say: it does not say that an experiment may change only one variable at a time. The
space of ideas worth trying is far larger than the compute budget, so a single experiment will
often bundle several changes — two new features at once, or a feature together with a
hyperparameter — and spending a scarce budget that way is usually the right call. The price of
bundling is *attribution*: you learn that the bundle helped, not which part of it did, and buying
that attribution back is what a controlled ablation is for. The price we never pay is
comparability, because a score that cannot be set against the scores already on the board tells you
nothing at all, however cheaply it was obtained.

*Without it:* the leaderboard fills with numbers that cannot be compared — one model scored on easy
folds, another on hard folds — and every "which idea won?" decision is built on sand.

*Decided:* fold eligibility is derived from data coverage alone and **never** from the model or
config; a fold enters the leaderboard only once its validation window is complete; a new data
source is assessed by a controlled ablation before it may enter the leaderboard at all.

*Serves:*
[Hypothesis 2: a hundred experiments per person in a peak month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month).

*Detail:*
[Eligibility](../ml_experimentation/cross-validation-folds.md#eligibility),
[Complete validation windows only](../architecture/ml-orchestration.md#complete-validation-windows-only),
[Evaluating new data sources](../ml_experimentation/evaluating-new-data-sources.md).

### 9 — Provenance travels with the forecast data

Every forecast row carries enough to say where it came from, so a forecast can be explained,
reproduced, or invalidated without an external lookup.

*Without it:* "why was Tuesday's forecast odd?" becomes an afternoon of cross-referencing deploy
logs and run timestamps — answerable, but slowly, and only by whoever still remembers how the
pieces fit.

*Decided:* every
[`PowerForecast`](../api/contracts/index.md#contracts.power_schemas.PowerForecast) row carries
[`power_fcst_init_time`](../api/contracts/index.md#contracts.power_schemas.PowerForecast.power_fcst_init_time)
(when we made the forecast),
[`nwp_init_time`](../api/contracts/index.md#contracts.power_schemas.PowerForecast.nwp_init_time)
(when the weather forecast behind it was initialised),
[`power_fcst_model_name`](../api/contracts/index.md#contracts.power_schemas.PowerForecast.power_fcst_model_name)
and
[`power_fcst_model_version`](../api/contracts/index.md#contracts.power_schemas.PowerForecast.power_fcst_model_version)
— all four of which survive through to the table delivered to NGED — plus
[`experiment_name`](../api/contracts/index.md#contracts.power_schemas.PowerForecast.experiment_name),
[`fold_id`](../api/contracts/index.md#contracts.power_schemas.PowerForecast.fold_id) and
[`ml_flow_experiment_id`](../api/contracts/index.md#contracts.power_schemas.PowerForecast.ml_flow_experiment_id),
which are internal-only and projected out at the delivery boundary. Every MLflow run is stamped
with the git SHA and the Delta table versions it read.

*Serves:* [Hypothesis 2: a hundred experiments per
person in a peak
month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
[Hypothesis 5: scale without redesign](engineering-hypotheses.md#h5-scale-without-redesign).

*Detail:* [Two metric
stores](../architecture/ml-orchestration.md#two-metric-stores-one-division-of-labour), [The
Universal Model Interface](../architecture/overview.md#the-universal-model-interface).

### 10 — Every write is atomic and idempotent, and every failure is confined to one partition

Three properties of the same boundary. *Atomic*: a write becomes visible all at once, so a reader
sees the state before it or the state after it, never a half-finished middle. *Idempotent*:
re-running it is safe, so a retry cannot double-count. *Confined*: a failure cannot spread beyond
the partition it happened in — a *blast-radius* property, how much fails, which is a different axis
from [principle 1 ("*the power forecast never stops*")](#1-the-power-forecast-never-stops), which
is about which *way* a failure occurs.

Atomicity is the one that is easiest to lose without noticing, because several widely-used array
and table formats are plain directories of files with no commit boundary. A reader of such a
directory has no way to distinguish "the write finished" from "the write is halfway through". A
producer and a consumer running as independent processes will overlap sooner or later. It is also
the property that keeps principle 1 honest: with an atomic store, an interrupted write leaves the
previous version in place, so a consumer sees *stale* data — a state the [degradation
ladder](inherent-stability.md) already knows how to handle — rather than *torn* data, which nothing
downstream can handle.

*Without it:* a retry double-counts the rows it appended before failing; one experiment's crash
corrupts a neighbour's results; a dashboard or a downstream asset reads a table mid-write and
quietly forecasts from half a weather run; and re-running anything first requires working out what
state the last run left behind.

*Decided:* Delta Lake was adopted partly for this — every write lands as a single transaction-log
commit, and an in-progress write is never visible to a reader, on a laptop's local filesystem
exactly as on object storage, so the development loop exercises the same semantics production
depends on; re-materialising a fold overwrites its `(experiment_name, fold_id)` partition rather
than appending, so a retry cannot silently double-count; parallel experiments write to disjoint
partition directories and never touch each other.

*Serves:*
[Hypothesis 1: a service that mostly runs itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
[Hypothesis 5: scale without redesign](engineering-hypotheses.md#h5-scale-without-redesign).

*Detail:*
[ACID on object storage](../architecture/forecast-delivery.md#and-it-is-a-database-acid-on-object-storage),
[Idempotent writes and concurrency](../architecture/ml-orchestration.md#idempotent-writes-and-concurrency),
[Serve only the trained population](../architecture/production-deployment.md#serve-only-the-trained-population).

### 11 — Push the work down to the query engine; materialise once, as late as possible

No code between storage and the model boundary may force the data into memory (in Polars, by
calling `.collect()`), so the query engine sees the whole plan and prunes the scan before any data
crosses the wire. At this data scale the alternative is not slow, it is impossible.

The saving in memory is the obvious benefit, but the more valuable benefit is that laziness is what
makes [principle 5 ("*everything around the model is
general-purpose*")](#5-everything-around-the-model-is-general-purpose) affordable. Each layer
expresses only the constraint it actually knows about — the cross-validation code filters by date
and population, knowing nothing about which features were selected; the feature engineer adds
columns, knowing nothing about which fold it is serving; the model materialises at its own boundary
— and none of them has to be told what the others did. At the point of materialisation the engine
fuses the lot, pushing every filter and column selection from every layer down into the scan, so
the result costs no more than if one omniscient function had written the whole query by hand.
Without that, generality would be unaffordable: to avoid materialising a vast intermediate, the
cross-validation code would have to know which columns the model wanted, which is exactly the
model-awareness principle 5 exists to forbid.

*Without it:* a full 51-member backtest needs a cluster instead of a laptop — an unpruned NWP
materialisation is hundreds of gigabytes — and the pocket-money cost claim goes with it. The
quieter cost is architectural: every layer starts reaching into the next one's business to keep the
intermediate results small.

*Decided:* input pruning plus chunking by weather-run date is what holds the full 51-member
validation prediction to the laptop-sized peak quoted under [principle
6](#6-the-whole-system-must-be-exercisable-on-one-laptop).

*Serves:*
[Hypothesis 4: it runs for pocket money](engineering-hypotheses.md#h4-it-runs-for-pocket-money),
[Hypothesis 5: scale without redesign](engineering-hypotheses.md#h5-scale-without-redesign).

*Detail:*
[Lazy evaluation strategy](../architecture/performance.md#lazy-evaluation-strategy).

### 12 — Measure; do not assume

Performance, size and cost claims are benchmarked on real data, through the real code path, before
they are believed — and the measurement is written down next to the decision it justified, so a
later reader can tell which numbers are still true.

*Without it:* plausible-sounding defaults ship unexamined — `BYTE_STREAM_SPLIT` on NWP *sounds*
right and measures worse — and the docs fill with numbers nobody can reproduce or trust.

*Decided:* `BYTE_STREAM_SPLIT` is used for `power_fcst` but deliberately *not* for NWP, because it
measured *worse* there; the NWP scan-pruning rules were each verified with `LazyFrame.explain()`
rather than reasoned about.

*Serves:*
[Hypothesis 4: it runs for pocket money](engineering-hypotheses.md#h4-it-runs-for-pocket-money),
[Hypothesis 5: scale without redesign](engineering-hypotheses.md#h5-scale-without-redesign).

*Detail:*
[Storage formats](../architecture/performance.md#storage-formats-measured-not-assumed),
[Bounding feature-engineering memory](../architecture/performance.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output).

### 13 — A new technology must earn its place against one we already operate

The requirement is a burden of proof, not a ban: where a technology we already run does the job,
use it; where it genuinely does not, adopt the new technology deliberately and write down what it
bought. The reason for the asymmetry is that every additional service is one more service to
deploy, monitor, secure, upgrade, document and — if the service is one day handed over to NGED —
teach to a new operator: a cost that is paid forever and is easy to overlook at the moment of
adoption.

*Without it:* the stack accretes one "obviously useful" service per quarter, each cheap to adopt
and expensive forever after, until a very small team spends its time feeding infrastructure rather
than improving forecasts.

*Decided:* delivery to NGED reuses the Delta-on-S3 stack we already operate rather than adding a
REST (Representational State Transfer) API. The REST API is not rejected forever; it has a
documented set of conditions under which it would earn its keep. We *did* adopt Delta Lake,
Dagster, MLflow, Marimo, and Sentry, each for a stated reason recorded at the time.

*Serves:*
[Hypothesis 4: it runs for pocket money](engineering-hypotheses.md#h4-it-runs-for-pocket-money),
[Hypothesis 1: a service that mostly runs itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself)
— specifically
[T1.4, operability by a non-expert](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself).

*Detail:*
[An established industry pattern](../architecture/forecast-delivery.md#an-established-industry-pattern),
[When would a REST API earn its keep?](../architecture/forecast-delivery.md#when-would-a-rest-api-earn-its-keep),
[Considered but rejected designs](../architecture/production-deployment.md#considered-but-rejected-designs).

### 14 — Production jobs are coupled through data at rest, never through run status

Each scheduled production job reads whatever is on disk at the moment it runs. No production job
asks whether the job that produces its input succeeded, or ran at all. This is principle 10's
blast-radius property one layer up: [principle 10 ("*every write is atomic and idempotent, and
every failure is confined to one
partition*")](#10-every-write-is-atomic-and-idempotent-and-every-failure-is-confined-to-one-partition)
confines a failure inside the write it happened in, and this principle confines it inside the job
it happened in.

The common alternative is a chain of scheduled jobs — A at 06:00, B at 06:15, C at 06:30 — in
which B's real input is *the event of A having run*. That design has no way to distinguish "A
failed" from "A is still running" from "A had nothing to do", so one bad morning upstream takes
out every job downstream of it for the rest of the day, and recovery means replaying the chain in
order.

Ours cannot propagate a failure that way, because there is no channel for it to propagate down.
`live_forecasts` does not care whether `ecmwf_ens` succeeded in the last 24 hours, or whether
this hour's telemetry pull ran: it selects the freshest NWP run genuinely present as of its own
init time, and stamps that run's `nwp_init_time` on every row it writes. A failed telemetry pull
makes the 06:00 forecast slightly staler; it cannot make it late, and it cannot make it absent.
One case still falls short of that today: a *sustained* NWP outage, where no run on disk still
covers the horizon, makes `live_forecasts` raise rather than degrade — the one hard failure left
in the [failure-modes table](inherent-stability.md#failure-modes), which
[#446](https://github.com/openclimatefix/nged-substation-forecast/issues/446) will convert into a
weather-blind forecast.

Note the deliberate distinction between the *lineage* graph and a runtime precondition. Dagster
still knows that `live_forecasts` depends on `ecmwf_ens` and `power_time_series_and_metadata`;
that edge is what lets a developer ask Dagster to materialise an asset *together with its
upstreams* on a laptop, and what makes the graph legible in the UI. What we decline is letting
that edge become a gate in production.

*Without it:* in a chained design, one failed 06:00 ingest suppresses the 06:00 forecast and the
slots behind it, and someone has to replay the chain in order, out of hours, to catch up — the
precise out-of-hours intervention that the
[T1.1 operability test](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself) predicts
will never be needed.

*Decided:* the three production jobs run on three independent schedules with no run-status
coupling between them, and the `:55` offset on the telemetry pull is an optimisation for
freshness, not a precondition — if it is missed, the forecast still runs. A related decision
shows how far the preference for filesystem coupling goes: `promoted_model` was removed from
`live_forecasts`' deps altogether once it became clear the model arrives by filesystem path
rather than by data flow, so even the lineage edge was not worth the permanently un-materialised
parent it rendered on the production box, which has no MLflow and never runs promotion.

*Serves:*
[Hypothesis 1: a service that mostly runs itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself).

*Detail:* [Inherent Stability](inherent-stability.md#the-rules) — "*never make one production
job's run status a precondition for another's*".

### 15 — Transform data in feature engineering, not in the ingest, unless it saves a lot of storage

Changing a transform that runs at ingest means re-downloading and re-writing the whole archive.
Changing a transform that runs in feature engineering means editing a function and re-running an
experiment. Feature engineering is therefore the default home for a transform, and one earns a place
in the ingest only when a measurement says it should be there — almost always a storage measurement,
written down next to the decision.

The test is what the transform destroys. A transform that throws information away has to show what
it buys: the H3 spatial aggregation and the 13-bit significand rounding both did, and both are a
large part of why the archive fits on a laptop. A transform that merely rewrites the same
information in a different form has nothing to show, because feature engineering can produce that
form on demand, differently for each experiment. Converting wind's `u` and `v` components into speed
and direction is the second kind: it destroys nothing, and it hands every later stage an angle that
wraps at 360°, which ordinary interpolation, averaging, quantiles, and z-scores all get wrong.

The saving has to be large, not merely positive. Converting wind to speed and direction does save
storage — about 6% of the `nwp` table, measured, or roughly 2.4 GB of the ~39 GB a year of ECMWF ENS
takes — and that is not enough to justify freezing a wrapped angle into the archive. Compare the H3
aggregation, which is what makes a year of 51-member weather fit on a laptop at all. That contrast
is the principle: it is the *size* of the saving that decides, and the only way to know the size is
to measure it.

*Without it:* a transform nobody remembers choosing becomes impossible to revisit. Wind arrives as
`u` and `v`, the ingest converts it to speed and direction and drops the components, and every later
stage that interpolates, takes an ensemble quantile of, or standardises a direction is quietly wrong
— with the fix costing an overnight re-download rather than a config change.

*Decided:* the `nwp` table's H3 spatial aggregation and 13-bit significand rounding both stay in the
ingest, because each was measured and each buys a large saving. This principle is also what puts
wind's polar conversion up for review in v0.5
([#525](https://github.com/openclimatefix/nged-substation-forecast/issues/525)): the conversion
saves ~6%, which the principle says is too little to freeze a wrapped angle into the archive for,
and the storage experiment there is sized to confirm the number before the change is made.

*Serves:* [Hypothesis 2: a hundred experiments per person in a peak
month](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month) — a transform
frozen into the archive cannot be varied by an experiment.

*Detail:* [NWP variable conventions](../architecture/nwp-variable-conventions.md),
[Storage formats](../architecture/performance.md#storage-formats-measured-not-assumed).

### 16 — A failure names its own cause in the telemetry

When a failure occurs, the cause should be legible from the Sentry event and from what Dagster
shows, rather than reconstructed by trawling logs. The standard to aim at is `rustc`'s diagnostics:
name what broke, and say where. That makes each event a design surface rather than a by-product —
its tags decide whether an alert rule can route it, its fingerprint decides whether a stall
recurring hourly is one issue or 24, and its message decides whether the operator can act without
opening a shell on the box. This is load-bearing rather than good manners for two reasons.
[Principle 1 ("*the power forecast never stops*")](#1-the-power-forecast-never-stops) makes failure
quiet by design, so the telemetry is often the only channel that speaks at all; and under the
operating model preferred once this project's funding ends, the reader of the alert is a non-expert
at NGED holding the runbooks and nothing else.

*Without it:* the alert says a run failed. The operator opens Dagster, finds the run, reads the
step's logs, and works out from a stack trace which time series, which weather run or which of our
own assumptions actually broke — an hour per incident, and a cause recorded in the [intervention
log](../live_service/intervention-log.md) that is a guess.

*Decided:* every Sentry event sender carries a tag an alert rule can route on, and the stale-power
event names the worst-affected series — see [Send telemetry to Sentry, and alarm on
absence](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence) for
the routing rules and the mechanisms that keep those tags trustworthy.

*Serves:* [Hypothesis 1: a service that mostly runs
itself](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself) — specifically
[T1.4, operability by a non-expert](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
and [T1.1](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself)'s cause taxonomy, which is
only as good as what the telemetry says caused the failure.

*Detail:* [Send telemetry to Sentry, and alarm on
absence](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence),
which also defines the two kinds of event, and
[Three audiences, three channels](inherent-stability.md#three-audiences-three-channels).

## Deliberately absent

We have **no availability service-level objective (SLO) and no error
budget** — the pair of SRE practices that would normally quantify "how much downtime is acceptable"
and then spend that allowance deliberately. We have consciously declined both: the requirement is
recovery "next business day, via runbook", and the reasoning is in
[Uptime: lenient by design](../background/requirements.md#uptime-lenient-by-design). Recording the
rejection is the point — a practice we considered and declined is more useful to a reader than one we
never considered.

## Industry best practices we have not yet absorbed

The list above is not finished, and pretending otherwise would undermine it. A review of what
comparable systems practise surfaced five practices we respect but have **not yet absorbed into
the code or the plan**. They are parked on
[#449](https://github.com/openclimatefix/nged-substation-forecast/issues/449), to be considered
once the live service has run for long enough to inform them; we expect to adopt whichever of them
earn their keep. Recording them here keeps the gap honest — a reader comparing this project
against industry practice should find these named by us, not discovered as omissions.

### Input drift detection

Our checks detect input data that is *absent* or *malformed*; standard
MLOps practice also watches for data that is present, well-formed, and *different* — an upstream
model upgrade, a re-gridding, a re-metered substation. The case for it is strong here: on a 14-day
horizon a forecast is not fully scoreable for a fortnight, so forecast *error* is a badly lagging
indicator, while a shift in the *inputs* is visible the same day. But this is also the item with a
genuinely open design question, because our inputs are **expected to drift**: climate change moves
the weather distributions, and the grid beneath the power data is changing fast (solar, EVs, heat
pumps).

Great Britain's summer of 2026 is the concrete case, and it arrived while this page was being
written. All the figures here are the Met Office's provisional figures, current as of early
August
2026. July 2026 was the driest July in the England and Wales series that begins in 1836: England
recorded 6.5 mm of rain, 10% of its long-term average, and southern England recorded 1.9 mm — 3% of
average, and the driest month it has ever observed. The same month was the UK's sunniest July in a
series running from 1910, and for England and Wales the sunniest *calendar month* ever observed; it
was also the second-warmest July for the UK as a whole. Nor was this confined to one month: spring
2026 was already the warmest on record for England and Wales, and at the summer's halfway point the
UK was running 1.8 °C above the seasonal norm — warmer than 2025 at the same stage, and 2025 went
on to be the UK's warmest summer on record. Every one of those is a real shift in the distribution
of an input we feed the model, and not one of them is a fault. A naive "distribution changed" alarm
would have fired continuously from spring 2026 onwards while telling an operator nothing they could
act on, which is exactly how a naive alarm gets tuned into silence.

Note also how much the *choice of statistic* does here, which is part of why the design question is
open. Summer rainfall to mid-July stood at 42% of the full season's long-term average against the
roughly 50% normal for that date, which the Met Office fairly describes as just below average —
even as the same season was producing individual months that broke 190-year records. A detector
watching seasonal aggregates and one watching monthly or daily distributions would have told an
operator entirely different stories about the same summer.

The question to answer before adopting the practice is therefore whether the useful event is "the
distribution changed" or "the model is extrapolating — being asked about conditions unlike anything
it trained on". The two suggest different responses, and a summer like 2026's probably *should* be
flagged — not as a fault, but as a legitimate reason for wider uncertainty bands, routing the
signal into the in-band channel of [principle 1 ("*the power forecast never
stops*")](#1-the-power-forecast-never-stops) rather than to
a pager. There is also a model-side answer that is not monitoring at all: give the model features
that let it *represent* the regime rather than merely detecting that it is unusual — the planned
[weather-abnormality features](../roadmap/xgboost-improvements.md#weather-abnormality-climatology-z-score-features)
are that answer, and the two are complementary rather than alternatives. We do not yet know the
right design; working it out is the task.

### Shadow (champion–challenger) deployment

Shadow deployment means running a candidate model against live inputs in parallel with the
champion, recorded but not delivered, so that a promotion decision can rest on live behaviour
rather than on backtests alone. It catches the class of bug backtests cannot, such as
training/serving skew and availability differences that only exist live. It should be unusually
cheap here, because `power_forecasts` already partitions by `experiment_name`, so a shadow run is
just another partition scored by the existing production-monitoring metrics. Two open questions to
resolve first: a shadow lane is exactly the kind of serving-path complexity that [principle 2
("*complexity belongs offline, not in the serving
path*")](#2-complexity-belongs-offline-not-in-the-serving-path) pushes against, so it must earn its
place; and it interacts with [Hypothesis 3: one-click promotion, and one-click
rollback](engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback), since
one-command rollback is less frightening when the champion was already shadowed.

### A schema-evolution policy for the delivery contract

The [strict-contracts principle](#7-strict-contracts-at-every-boundary) freezes a
boundary; this is the missing account of how a frozen boundary is allowed to *change*
once NGED consume it — additive-only changes, a deprecation window, a version field a consumer can
branch on, and how a breaking change is announced. Of the five, this is the one with a deadline
attached: NGED first consume forecasts around v0.6, and the cost of having no policy is paid the
first time a delivered table needs to change.

### Statistical process control on forecast error

Control charts and CUSUM (cumulative sum) charts are manufacturing's answer to "detect a shift
without picking an arbitrary fixed threshold", with decades of theory behind them. Fixed thresholds
on a seasonal signal are either too loose in winter or too noisy in summer; SPC (statistical
process control) is designed for exactly that problem, and it answers "is the champion quietly
degrading?" cheaply. It shares the open question raised under [input drift
detection](#input-drift-detection) above: a summer like 2026's will legitimately worsen forecast
error, and the chart must not read that as a model regression.

### Naming poka-yoke (mistake-proofing)

Mistake-proofing, from manufacturing: design names and interfaces so the
wrong usage fails to parse rather than being merely discouraged. The codebase already practises
this in places — closed string vocabularies are `Literal` types, tabular shapes are validated
schemas, and `delta_store`'s write helpers make the storage format the path of least resistance
rather than a rule to remember — but it is a habit applied opportunistically, not yet a stated
rule.

## How to use this list

A proposed design change should be checkable against this list: if it violates a
principle, that is not a veto, but it does require saying which principle is being traded away and
what is bought in return. And a principle that stops deciding anything should be deleted rather than
left as decoration — the same discipline the hypotheses page applies to its thresholds.

**Before copying a principle into another system, check the assumption it is a bet on.** Most of the
list is portable as-is — strict contracts, one execution path, identically-scored experiments,
provenance, atomic and idempotent writes, telemetry that names its own cause, and
measure-don't-assume are
close to unconditional good practice, and
cheap experiments are too for any project that is doing research rather than only operating a fixed
model.
[Principle 1 ("*the power forecast never stops*")](#1-the-power-forecast-never-stops) is the
contingent one: it is downstream of the fact
that [an outage is cheap here](../background/requirements.md#uptime-lenient-by-design) and that
[the incumbent forecast is a floor beneath us](inherent-stability.md#ngeds-incumbent-forecast-is-the-floor);
a
system where a wrong-but-confident forecast costs real money — a trading desk, a control-room feed —
should invert it and fail closed. And the push-work-to-the-engine and new-technology principles are
general.

[Principle 15 ("*transform data in feature engineering, not in the ingest, unless it saves a lot of
storage*")](#15-transform-data-in-feature-engineering-not-in-the-ingest-unless-it-saves-a-lot-of-storage)
is general in its reasoning but contingent in its arithmetic: it assumes re-downloading the archive
is merely inconvenient rather than impossible, and that the storage a faithful representation costs
is affordable. A project ingesting a feed it cannot replay, or one whose storage bill dominates,
should expect the balance to come out differently.

For the finer-grained rules that sit underneath these — how to write the code rather than how to
shape the system — see [Code Style](../architecture/code-style.md) and
[Testing](../architecture/testing.md).
