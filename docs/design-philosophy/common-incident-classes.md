# Common incident classes in production forecasting services

Production services that ingest third-party weather, satellite or telemetry feeds tend to fail in
a small number of recurring shapes, largely independent of the specific stack behind them. This
page catalogues those shapes and states, for each, which mechanism in this project's design
targets it — honestly marking what already exists against what is designed but not yet built, and
naming the gaps outright. It is the same field-report standard the rest of this section holds
itself to: a practice that is not yet covered is more useful named than left implicit.

This page is broader than [Inherent Stability](inherent-stability.md), which argues the largest of
these shapes — an input going missing or stale — in full. Several of the shapes below sit outside
inherent stability's remit entirely: shared-compute contention, deploy drift, and fault
propagation across independent units of work are blast-radius and process concerns, not
input-degradation ones.

## Upstream data outage

A third-party feed — weather, satellite imagery, telemetry — stops arriving, for anywhere from
minutes to days, for reasons entirely outside the consuming service's control.

This is the shape [Inherent Stability](inherent-stability.md) is built around: the
[degradation ladder](inherent-stability.md#the-degradation-ladder) states what the forecast should
still produce at each stage of loss, from one missed weather-model run through to no fresh data at
all. Telemetry already degrades this way today — a stalled meter widens nothing yet, but it does
not stop the forecast either. Weather data is the sharper case: a sustained NWP outage currently
makes the live-forecast asset **raise** rather than degrade — one of three hard-failure rows in the
[failure-modes table](inherent-stability.md#failure-modes), and the only one tracked to change
([#446](https://github.com/openclimatefix/nged-substation-forecast/issues/446)). The other two stay
deliberate raises: an empty or unloadable promoted model is a promotion bug rather than a data
outage, and a duplicated forecast primary key means one of our own joins has fanned out, which
`PowerForecast.validate` rejects at the contract boundary rather than delivering. The NWP gap is tracked, and the fix is deliberately sequenced behind training
the model against realistic outages, so that a degraded forecast is measured rather than merely
produced.

## Upstream data malformed or silently reshaped

A feed keeps arriving on schedule, but the payload is wrong: null-filled, wrongly shaped, or
carrying a provider-side schema or unit change nobody announced.

The [strict-contracts principle](design-principles.md#7-strict-contracts-at-every-boundary) is the
direct answer — every tabular boundary is a validated Patito schema rather than an assumed shape,
with the deliberate asymmetry that structurally broken data is rejected outright while a tolerable
amount of local corruption is absorbed rather than discarded wholesale. [Known ECMWF ENS
data-quality issues](../architecture/ecmwf-ens-known-issues.md) is what that policy looks like
applied in full to one real, messy upstream feed. This is one of the better-covered shapes here,
precisely because it was designed for from the start rather than retrofitted after a malformed
payload reached a model.

## Silent output corruption

The forecast pipeline runs cleanly end to end and delivers a value — but the value itself is
wrong: an implausible magnitude, an internally inconsistent set of quantiles, or a forecast that
quietly diverges from reality — and nothing rejects or flags it before a consumer notices.

Two mechanisms already point the right way, both 🚧 planned rather than shipped. Normalising
`power_fcst` to **[−1, +1]**
([#246](https://github.com/openclimatefix/nged-substation-forecast/issues/246), the code today
still forecasts raw MW/MVA) will make an order-of-magnitude output error structurally harder,
since the value becomes capacity-bounded by construction rather than by a downstream check someone
has to remember to run. And quantile crossing — one quantile level coming out below a lower one —
is a *named* failure mode with a designed fix already: sorting each member's quantiles at predict
time
([Probabilistic Forecasting](../techniques/probabilistic-forecasting.md#turning-51-quantile-sets-into-one-the-pooling-recipe),
[#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263)).

The gap: `live_forecasts_are_healthy` already warns on a null, NaN or infinite `power_fcst`, but
nothing checks a *finite, plausible-looking* output for implausible magnitude or, once quantiles
ship, crossed levels — the same "validate, and warn on what validation can't reject outright"
pattern every input boundary in this project follows has not yet been applied to the output
itself ([#560](https://github.com/openclimatefix/nged-substation-forecast/issues/560)).

## Shared-compute blast radius

One heavy or misbehaving component — an expensive dashboard query, a bulk API request, an
ad-hoc analytical query — exhausts a resource (memory, database connections, disk) shared with
unrelated services, and takes all of them down together rather than just itself.

[Every write is atomic, idempotent and confined to one partition](design-principles.md#10-every-write-is-atomic-and-idempotent-and-every-failure-is-confined-to-one-partition)
is the data-layer answer, and the [accepted AWS architecture](../roadmap/live-service.md#aws-architecture)
extends the same instinct to compute: every live and backtest run dispatches to its own ephemeral
Fargate task, so a bad run cannot starve another one. The gap sits on the always-on control-plane
box itself, where the Dagster daemon, webserver and dashboard run as separate processes sharing
one machine's memory and disk with no per-process limit — today's mitigation is restart-on-failure
plus the [missed-check-in alarm](../roadmap/live-service.md#alert-on-absence-the-missed-check-in-alarm),
both reactive rather than preventive
([#561](https://github.com/openclimatefix/nged-substation-forecast/issues/561)).

## Orchestrator or scheduler silently stops

The process responsible for triggering scheduled work dies, hangs, or stops firing — and because
nothing failed loudly, the absence produces no alert of its own. It surfaces only when someone
notices stale output, which can be hours or days later.

This is the shape the
[missed-check-in alarm](../roadmap/live-service.md#alert-on-absence-the-missed-check-in-alarm)
exists to close, and closes about as directly as a single mechanism can: it is evaluated from
**outside** the deployment, on the logic that a dead process cannot be relied on to report its own
death, and it fires on the absence of a successful run rather than on any particular failure
signal. Of every shape on this page, this is the one with the most direct, already-shipped answer.

## Deploy or configuration drift between environments

A change — a schema migration, a container image, a configuration value — is verified in one
environment and assumed, never actually confirmed, to be live in another. The two silently diverge
until a run fails on the difference, or worse, produces a wrong answer that passes.

[One execution path from research to production](design-principles.md#3-one-execution-path-from-research-to-production)
removes an entire class of this: there is no second, separately deployed implementation for
research versus production to drift apart from. Atomic Delta commits and `promoted_model`'s
all-or-nothing swap — it refuses to replace the model on disk at all if the incoming config can't
be rebuilt — close the model-promotion instance of this shape specifically. What is not yet
addressed is a staged or canary rollout for a **code** deploy itself, as opposed to a model
promotion; today a new image goes live everywhere the next run picks it up. Left as an open
question rather than a proposed change — the single-image, single-environment deployment this
project runs today is a narrower target than a multi-service system, so the shape may not earn a
dedicated mechanism until the architecture grows past that.

## Fault propagation across independent units of work

A single malformed or missing record inside a larger batch — one site, one time series, one
partition — takes down the entire job, rather than being isolated to the piece it actually
affects.

Per-`(experiment, fold)` partitioning and Patito's own null-tolerance policy — reject a
structurally empty column outright, but absorb a locally corrupt value the model can tolerate —
both push in the right direction by construction. One case is now settled rather than assumed: NWP
rows duplicated for a single H3 cell fan the feature join out, `PowerForecast.validate` rejects the
duplicated primary key, and the whole slot is lost — including every series the duplicated cell
never touched. That is the propagation this section describes, accepted for now because delivering
a silently duplicated forecast is worse, and because the trigger is a bug in our own ingest rather
than a routine outage.

## How to use this page

None of the above is a request to build every mitigation at once — several of these shapes are
already fully addressed, one is an open question rather than a proposed change, and the rest are
tracked as the linked issues.
