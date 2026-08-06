# Design Principles

The principles below govern this project's engineering decisions — what gets built, where code
lives, and which technologies are adopted. They are deliberately short here and link out to the page
that argues each one in full — this is an index, not the argument. Each entry says concretely what
the principle buys: *Without it* paints the everyday failure the principle exists to prevent,
*Decided* names a real decision it made, *Serves* names the
[engineering hypothesis](engineering-hypotheses.md) it is a bet on, and *Detail* links the full
argument.

**Where they come from.** An aim of this project is to gather **industry best practice into a
single codebase for energy forecasting** — and not only best practice from the energy-forecasting
industry. Several of the ideas on these pages are borrowed from disciplines that have been solving
the same shape of problem for longer than data engineering has existed: *inherent stability* comes
from vehicle dynamics, *fail-operational* from avionics autoland and ISO 26262, *blast radius* from
site reliability engineering. Borrowing across disciplines is the point, not a flourish: a
discipline that has been shipping safety-critical systems for fifty years has usually already made
the mistake we are about to make. Not every borrowed idea survives contact, and we record those
outcomes too: *error budgets* were examined and declined (see "Deliberately absent" below),
*poka-yoke* (mistake-proofing, from manufacturing) remains a candidate
([#449](https://github.com/openclimatefix/nged-substation-forecast/issues/449)), and Postel's law is
named on the [Inherent Stability](inherent-stability.md#not-postels-law) page precisely so that
nobody mistakes it for what we do.

**How these relate to the [engineering hypotheses](engineering-hypotheses.md).** Principles are
constraints on *decisions*; hypotheses are claims about *outcomes*. A hypothesis can be falsified by
measurement, whereas a principle can only be overridden by a measurement or found not to be
load-bearing. The two are connected in one direction: the principles are the bets we are making in
order to achieve the hypotheses. That gives a test for admission to this list — **name the hypothesis
it serves and a decision it actually decided**. A principle serving no hypothesis is either
decoration or a sign of a missing hypothesis; a hypothesis with no principle behind it is a claim we
are merely hoping comes true.

1. **The power forecast never stops.** If data inputs are disrupted, the forecast gets less certain
   instead of stopping — and says so in the answer itself, through wider uncertainty bands. The
   forecast always does the best it can with whatever data it has, rather than blowing up; raising
   is reserved for states that are our own bug.
   *Without it:* every wobble in an upstream feed becomes an outage — the service raises at 06:00
   because one meter went quiet, NGED open their dashboard to a gap instead of a forecast, and a
   developer spends the morning re-running a pipeline whose only real problem was a missing input.
   *Decided:* every asset check in the repo is non-blocking `WARN`; there is deliberately no
   `ERROR`-severity check anywhere. *Serves:*
   [H1](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself). *Detail:*
   [Inherent Stability](inherent-stability.md), whose [ten rules](inherent-stability.md#the-rules)
   are the fine-grained form of this principle and principle 2.

2. **Complexity belongs offline, not in the serving path.** When a capability could be built into
   the training loop or into the production service, build it into the training loop: training runs
   in front of a human who can read the traceback, whereas the production service runs unattended.
   Capabilities that forecasting services commonly bolt onto the serving path go inside the model
   instead: correcting for recent forecast errors is not a post-processing step but a lagged-power
   feature the model learns from; a missing input triggers no switch to a separately-trained
   fallback model, because the one model is trained to tolerate missing inputs; and there is no
   blending of a short-horizon model with a long-horizon one, because the one model spans the
   horizon.
   *Without it:* the serving path accumulates special cases — a correction here, a fallback switch
   there, a blend per horizon — and each one is a branch that fires for the first time in
   production, during a real outage, with nobody watching.
   *Decided:* `promoted_model` copies the champion to local disk, so production inference makes no
   MLflow call at all. *Serves:*
   [H1](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
   [H3](engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback). *Detail:*
   [Where complexity should live](inherent-stability.md#where-complexity-should-live) — including
   the two qualifiers that keep this principle honest —
   and [Bake the model into the image](../architecture/production-deployment.md#bake-the-model-into-the-image-at-build-time).

3. **One execution path from research to production.** The artifact that won the experiment *is* the
   artifact we deploy — not a re-implementation of it. There is no "now rewrite the research code for
   production" step, because every experiment already runs on the production pipeline.
   *Without it:* the classic two-codebase failure — research code is rewritten "properly" for
   production, the rewrite quietly diverges, and the deployed model no longer does what the winning
   experiment measured.
   *Decided:* this was the deciding argument for Dagster over Airflow, and it is why splitting the
   live service onto a second orchestrator remains rejected. *Serves:*
   [H2](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
   [H3](engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback). *Detail:*
   [Nothing gets rewritten on the way to production](../ml_experimentation/mlops-approach.md#nothing-gets-rewritten-on-the-way-to-production),
   [Why Dagster, not Airflow?](../architecture/why-dagster-not-airflow.md).

4. **Strict contracts at every boundary — liberal about missing inputs, strict about malformed
   ones.** Every tabular boundary is a Patito schema, validated rather than assumed. This is the
   deliberate *opposite* of Postel's law, and it is what stops principle 1 from decaying into
   "accept anything and hope". Strictness also has a granularity: reject structurally-broken data
   outright, but tolerate locally-corrupt values a model can absorb — throwing away an
   otherwise-good NWP run because a few percent of its pixels are null would convert a tolerable
   problem into an outage.
   *Without it:* a malformed file does not crash anything — it lands, joins, and quietly shifts
   forecasts until someone notices the units are wrong; silent corruption instead of a loud
   rejection at the boundary.
   *Decided:* `AllFeatures`, `PowerForecast` and the rest are validated schemas rather than
   conventions, and every `PowerForecast` row is self-describing. `Nwp.validate` rejects a
   *whole-slice* null in a de-accumulated variable but tolerates scattered per-pixel nulls (the
   `nwp_has_no_unexpected_nulls` check reports them as a `WARN`) — usefully converting fine-grained
   catastrophe into a coarse-grained missed run, the form principle 1 already handles. *Serves:*
   [H1](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
   [H6](engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Strict data contracts](../architecture/forecast-delivery.md#strict-data-contracts-machine-verifiable),
   [Not Postel's law](inherent-stability.md#not-postels-law),
   [The guiding principle](../architecture/ecmwf-ens-known-issues.md#the-guiding-principle).

5. **Every comparison must differ in exactly one thing.** A leaderboard is only worth having if two
   numbers on it are genuinely comparable, which requires the population, the folds, the metric
   definitions and the pipeline to be held constant by construction rather than by discipline.
   *Without it:* the leaderboard fills with numbers that cannot be compared — one model scored on
   easy folds, another on hard ones — and every "which idea won?" decision is quietly built on sand.
   *Decided:* fold eligibility is derived from data coverage alone and **never** from the model or
   config; a fold enters the leaderboard only once its validation window is complete; a new data
   source is assessed by a controlled ablation before it may enter the leaderboard at all. *Serves:*
   [H2](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month). *Detail:*
   [Eligibility](../ml_experimentation/cross-validation-folds.md#eligibility),
   [Complete validation windows only](../architecture/ml-orchestration.md#complete-validation-windows-only),
   [Evaluating new data sources](../ml_experimentation/evaluating-new-data-sources.md).

6. **Provenance travels with the data.** Every row carries enough to say where it came from, so a
   forecast can be explained, reproduced or invalidated without an external lookup.
   *Without it:* "why was Tuesday's forecast odd?" becomes an archaeology project — nobody can say
   which NWP run or which model produced the row, so the honest answer is a shrug.
   *Decided:* `PowerForecast` carries `nwp_init_time`, model name and version, experiment name and
   MLflow experiment id; every MLflow run is stamped with the git SHA and the Delta table versions
   it read. *Serves:*
   [H2](engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month),
   [H6](engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Two metric stores](../architecture/ml-orchestration.md#two-metric-stores-one-division-of-labour),
   [The Universal Model Interface](../architecture/overview.md#the-universal-model-interface).

7. **Every write is idempotent, and every failure is confined to one partition.** Re-running
   anything must be safe, and a failure must not be able to spread beyond the partition it happened
   in. Note that this is a *blast-radius* property — how much fails — which is a different axis from
   principle 1, which is about which *way* a thing fails.
   *Without it:* a retry double-counts the rows it appended before failing, one experiment's crash
   corrupts a neighbour's results, and re-running anything first requires working out what state the
   last run left behind.
   *Decided:* re-materialising a fold overwrites its `(experiment_name, fold_id)` partition rather
   than appending, so a retry cannot silently double-count; parallel experiments write to disjoint
   partition directories and never touch each other. *Serves:*
   [H1](engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself),
   [H6](engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Idempotent writes and concurrency](../architecture/ml-orchestration.md#idempotent-writes-and-concurrency),
   [Serve only the trained population](../architecture/production-deployment.md#serve-only-the-trained-population).

8. **Push the work down to the engine; materialise once, as late as possible.** No code between
   storage and the model boundary may call `.collect()`, so the query engine sees the whole plan and
   prunes the scan before any data crosses the wire. At this data scale the alternative is not slow,
   it is impossible.
   *Without it:* a full 51-member backtest needs a cluster instead of a laptop — an unpruned NWP
   materialisation is hundreds of gigabytes — and the pocket-money cost claim goes with it.
   *Decided:* input pruning plus `init_time` chunking keeps a full 51-member validation prediction
   (~321M rows) under 9 GB on a laptop. *Serves:*
   [H4](engineering-hypotheses.md#h4-it-runs-for-pocket-money),
   [H6](engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Lazy evaluation strategy](../architecture/performance.md#lazy-evaluation-strategy).

9. **Measure; do not assume.** Performance, size and cost claims are benchmarked on real data,
   through the real code path, before they are believed — and the measurement is written down next
   to the decision it justified, so a later reader can tell which numbers are still true.
   *Without it:* plausible-sounding defaults ship unexamined — `BYTE_STREAM_SPLIT` on NWP *sounds*
   right and measures worse — and the docs fill with numbers nobody can reproduce or trust.
   *Decided:* `BYTE_STREAM_SPLIT` is used for `power_fcst` but deliberately *not* for NWP, because
   it measured *worse* there; the NWP scan-pruning rules were each verified with
   `LazyFrame.explain()` rather than reasoned about. *Serves:*
   [H4](engineering-hypotheses.md#h4-it-runs-for-pocket-money),
   [H6](engineering-hypotheses.md#h6-scale-without-redesign). *Detail:*
   [Storage formats](../architecture/performance.md#storage-formats-measured-not-assumed),
   [Bounding feature-engineering memory](../architecture/performance.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output).

10. **A new technology must earn its place against one we already operate.** This is a burden of
    proof, not a ban: where something we already run does the job, use it; where it genuinely does
    not, adopt the new thing deliberately and write down what it bought. The reason for the asymmetry
    is that every additional service is one more thing to deploy, monitor, secure, upgrade, document
    and — if the service is one day handed over to NGED — teach to a new operator: a cost that is
    paid forever and is easy to overlook at the moment of adoption.
    *Without it:* the stack accretes one "obviously useful" service per quarter, each cheap to adopt
    and expensive forever after, until a very small team spends its time feeding infrastructure
    rather than improving forecasts.
    *Decided:* delivery to NGED reuses the Delta-on-S3 stack we already operate rather than adding a
    REST API — and the REST API is not rejected forever, it has a documented set of conditions under
    which it would earn its keep. We *did* adopt Delta Lake, Dagster, MLflow, Marimo and Sentry, each
    for a stated reason recorded at the time. *Serves:*
    [H4](engineering-hypotheses.md#h4-it-runs-for-pocket-money),
    [H5](engineering-hypotheses.md#h5-operable-by-a-non-expert). *Detail:*
    [An established industry pattern](../architecture/forecast-delivery.md#an-established-industry-pattern),
    [When would a REST API earn its keep?](../architecture/forecast-delivery.md#when-would-a-rest-api-earn-its-keep),
    [Considered but rejected designs](../architecture/production-deployment.md#considered-but-rejected-designs).

**Deliberately absent.** We have **no availability service-level objective (SLO) and no error
budget** — the pair of SRE practices that would normally quantify "how much downtime is acceptable"
and then spend that allowance deliberately. We have consciously declined both: the requirement is
recovery "next business day, via runbook", and the reasoning is in
[Uptime: lenient by design](../background/requirements.md#uptime-lenient-by-design). Recording the
rejection is the point — a practice we considered and declined is more useful to a reader than one we
never considered.

**How to use this list.** A proposed change should be checkable against this list: if it violates a
principle, that is not a veto, but it does require saying which principle is being traded away and
what is bought in return. And a principle that stops deciding anything should be deleted rather than
left as decoration — the same discipline the hypotheses page applies to its thresholds.

**Before copying a principle into another system, check the assumption it is a bet on.** Most of the
list is portable as-is — strict contracts, one execution path, single-variable comparisons,
provenance, idempotent writes and measure-don't-assume are close to unconditional good practice.
Principle 1 is the contingent one: it is downstream of the fact that
[an outage is cheap here](../background/requirements.md#uptime-lenient-by-design) and that
[the incumbent forecast is a floor beneath us](inherent-stability.md#the-incumbent-is-the-floor); a
system where a wrong-but-confident forecast costs real money — a trading desk, a control-room feed —
should invert it and fail closed. And the push-work-to-the-engine and new-technology principles are
general, but their weighting here is turned up by the pocket-money budget and the very small team; a
funded platform team can rationally afford more services and more materialisation.

For the finer-grained rules that sit underneath these — how to write the code rather than how to
shape the system — see [Code Style](../architecture/code-style.md) and
[Testing](../architecture/testing.md).
