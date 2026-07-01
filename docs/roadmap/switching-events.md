# Estimating Latent Demand Under Switching Events — Approach & Implementation Roadmap

**Scope.** How to estimate, for each NGED primary substation, its *latent demand under the normal running arrangement* — the demand that would be metered if the network were never reconfigured — given that the network is in fact reconfigured roughly 10% of the time by switching events. Background on what switching events are and why they are hard is at [**Switching Events**](../background/switching-events.md). This document defines the staged modelling plan (v0.6 → v2.5 → v2.6).

> **Status: 🔬 Research / 🚧 Planned.** None of this is implemented yet. The v0.6 unsupervised
> statistical detector is the nearest-term piece (it feeds the
> [`substation_switching`](delivery-tables.md#table-5-substation_switching) table and the
> training-data mask); the v2.5 / v2.6 mixture models are later research. The post-v2.0 roadmap is
> not yet fully specified, so read "v2.5 / v2.6" as "some time after v2.0". See the
> [roadmap index](index.md) for status conventions and where this fits the overall plan. This is the
> **canonical** treatment of switching events — it supersedes the earlier "switching state-space
> model" sketch in [differentiable physics](differentiable-physics.md).

---

## Part 1 — The graph structure (and why it is not a GNN)

A **graph** is just **nodes** connected by **edges**. Here the nodes are substations and the edges connect substations that can exchange load (i.e. that can be electrically joined by some switching operation). The graph is the natural way to encode the one fact a per-substation model is blind to: that a switching event is a *cross-substation* phenomenon — load leaving one node reappears at others, so the drop at A and the rises at B and C are the *same power moving*.

It is worth being explicit, because the term invites confusion: **none of the approaches in this document require a graph neural network (GNN).** A GNN is a *trained* model that passes learned "messages" along edges; it is a heavyweight tool we do not need. We use the graph purely as a **data structure** — a fixed map of "who can exchange load with whom" — and run simple, mostly closed-form operations over it. In every stage the graph is used the same humble way: to look up which substations are even *eligible* to exchange load, pruning an otherwise `N × N` problem down to each node's handful of real neighbours. The per-stage specifics — the cheap neighbour-subset search in v0.6, the sparsity pattern on the mixing weights in v2.5, and the typed nodes in v2.6 — are described with each version in Part 2 below.

In short: the graph tells us *who can connect to whom*; the simple statistics and the differentiable forward model do the rest. If a future stage ever genuinely benefited from a trained GNN, that would be a deliberate escalation — but nothing here needs it.

> **A note on differentiable physics.** Only the v2.6 stage uses *differentiable physics*: physics-based forward models (e.g. irradiance → PV power) implemented so their latent parameters — capacity, panel orientation, and so on — can be recovered by gradient-based **inversion** (running the forward model backwards to fit observed power). The v0.6 detector and the v2.5 mixture model do not use it. The full treatment lives in [Differentiable Physics](differentiable-physics.md), which is the single source of truth for that machinery; we do not re-derive it here.

---

## Part 2 — The staged roadmap

Version numbers are aligned with the main codebase: **v0.6**, then **v2.5**, then **v2.6**. Each stage states its motivation, what it adds, what it misses, and its trade-offs. (Although, note that we haven't fully specified the roadmap _after_ v2, so read "v2.x" as just meaning "some time after v2 is operational"). Escalation between stages must be justified by *measured residual structure at the previous stage*, not by anticipation — this keeps effort matched to demonstrated need.

### Overview of the ladder

```text
v0.6  ── Unsupervised statistical detector on power data.
  │       Flags/masks switching events; VALIDATED against the 32-series logs.
  │       Produces the detection sensitivity floor (a quantified, honest limit).
  │
v2.5  ── Magnitude-only mixture model. The workhorse.
  │       Reconstructs latent NRA demand per substation. Fully unsupervised.
  │       Faithful to "arbitrary continuous slice, multi-donor" transfer.
  │
v2.6  ── Type-resolved mixture with differentiable PV/wind/demand modules.
          Lets a switching event move proportionally more PV than load.
```

---

### v0.6 — Unsupervised statistical switching-event detector

**Goal.** Flag periods of abnormal running arrangement using simple statistics on the power time series. **No GNN, no differentiable physics, no latent-variable inference, no switching-log inputs.**

**Motivation.** Before any reconstruction model, we need to (a) flag/mask switching-affected periods so they stop poisoning forecasting training data; (b) produce an evaluation set to validate heavier models later; and (c) — critically — *quantify how well switching events can be detected from power data at all*, since at scale that is the only signal available.

**Method — three stages, in order:**

#### Stage 1. **Per-series weather/calendar baseline, then detect changepoints on the residual.**

For each substation, form an expected-power baseline that is a function of **exogenous, switching-independent covariates only** — temperature, solar irradiance, recent weather, time-of-day, day-of-week, holidays — fitted across a long history (e.g. an XGBoost or GAM regression). Take the residual (observed − expected). A switching event then shows up as a *sustained level shift* in this residual — not a spike, not a slope. Detect level shifts with a standard mean-shift changepoint method (PELT or binary segmentation with an L2 cost, or CUSUM). **Output:** candidate step times and magnitudes per substation.

**Why the baseline must be weather/calendar-based and *not* a lagged-power baseline.** A tempting cheap baseline is "same half-hour last week." It is unsound here, and disqualified. If last week sat in a switching event and this week is normal (or vice versa), the residual shows a step of the same magnitude and shape as a real event — but with the **sign reversed**, because the contamination is in the *reference*, not the observation. Stage 2's balance attribution would then hunt for donor rises coincident with a source drop that is a baseline artifact, manufacturing phantom events and mis-attributing them. Worse, because switching events can persist for days to months, a lag can land *inside the same ongoing event*, so there is no step at all and a real event is masked entirely. Weather and clock time are unaffected by network topology, so a baseline built only from them cannot be contaminated by switching state — the residual then isolates "power the weather and clock don't explain," which is exactly where a topology change appears, with no comparison period to poison.

**Two residual-contamination routes to handle:**

- *Training-set contamination.* If the baseline is fitted on history that itself contains switching events, the fit is biased toward those contaminated periods. Fit **robustly** (quantile or Huber loss), or iteratively: fit → flag large residuals as candidate events → refit excluding them. Because events occupy only ~10% of the time, a robust fit recovers the NRA relationship and the events fall out as residuals. This closes a virtuous loop with the detector itself — detected events feed back to clean the baseline's training data.
- *Persistent events.* A months-long ARA appears as a residual level shift that *stays* shifted, not a transient. The changepoint detector handles this (it catches the onset step), but the baseline must **not** be allowed to slowly adapt and treat the new level as normal. Keep the baseline static (weather/calendar-driven only) over the detection window so a sustained ARA remains visible as a sustained residual offset.

#### Stage 2. **Node-level coincidence / balance attribution.**

A level shift at one substation could be many things (fault, new connection, meter error). What makes it a *switching event* is the conservation fingerprint: coincident, opposite-sign shifts at neighbours that *collectively balance*. Because transfers fan out to 2–3 neighbours, **do not match pairwise.** Instead, for each candidate drop of magnitude `Δ` at substation `i` at time `t`, solve a small constrained attribution: *which subset of `i`'s neighbours show coincident rises (≈ `t`) that sum to ≈ `Δ`?* With a handful of neighbours per primary this is cheap — enumerate subsets, or run a small non-negative least-squares of neighbour rises against the source drop. That candidate neighbour set is a fixed lookup from the network graph (the adjacency of who-can-exchange-load), with no learning over the graph — it is what keeps the search to a handful of substations rather than all `N`. Score by timing coincidence × magnitude-balance agreement. High score → switching event with an identified donor set; low score → "anomaly, unknown cause."

```text
residuals around time t (observed - expected), one row per substation:

 source  i :   ‾‾‾‾‾‾|____________   drop of Δ = 9 MW at t
                     t

 nbr     j :   ______|‾‾‾‾‾‾‾‾‾‾‾‾   rise +5
 nbr     k :   ______|‾‾‾‾‾‾‾‾‾‾‾‾   rise +4
 nbr     m :   ____________________  no change (0)   <- correctly excluded
                     t

 balance test:  +5 (j) + 4 (k) = 9  ==  Δ at i      -> switching event, donors {j,k}
                (pairwise matching would FAIL: neither j nor k alone equals 9)
```

#### Stage 3. **Composition corroboration (post-attribution, per recipient).**

*Aim.* Stages 1–2 tell us *that* a switching event happened, *when*, and *how much net power* moved to each donor. They do **not** tell us *what kind* of power moved. A slice of the network carries a mix of underlying demand and embedded generation (rooftop PV, small wind), and the meter only ever sees the *net* (demand minus generation). Two transferred slices with the same net magnitude can have completely different make-ups — one might be 8 MW of demand with negligible generation, another might be 11 MW of demand offset by 3 MW of PV, both netting to +8 MW at the donor. The aim of stage 3 is to get a cheap, qualitative read on that make-up: *was the moved slice demand-dominated, PV-dominated, or wind-dominated?* This is corroboration and enrichment, not detection — it does not change whether we flagged the event, but it characterises it.

*Why we want it.* Three uses. (a) **Sanity-checking the attribution:** a leg whose inferred composition is physically implausible (e.g. "pure PV moved at 2 a.m.") is a signal the attribution in stage 2 mis-assigned that donor. (b) **A free preview of v2.6:** the later type-resolved model estimates per-type transfer properly; having a rough independent read here lets us check the heavy model agrees with the cheap one. (c) **Richer event labels:** the delivered event list becomes "source → donors, magnitude *and* rough composition per leg," which is more useful to NGED and to downstream stages.

*Mechanism.* The make-up of a slice is exposed by *when, within the day,* its power moved — because demand, PV, and wind each have a distinct, well-known diurnal signature. After stage 2 has told us donor `j` picked up some load at event onset, look at the **shape of `j`'s residual step across the hours of the day** (e.g. average the step magnitude by half-hour-of-day over the event's duration):

- a step that appears mainly around **midday and vanishes overnight** → the moved slice was **PV-heavy** (PV only generates in daylight, so a slice rich in PV changes `j`'s net power most when the sun is up);
- a step that is **roughly flat, or tracks the evening demand peak** → **demand-heavy**;
- a step that is **large but uncorrelated with daylight, gusty/variable** → **wind-heavy**.

This is a histogram (step magnitude vs. hour-of-day), not a fitted model — deliberately cheap, matching the spirit of v0.5.

*Order matters — read composition off the recipient, never the source.* The diurnal shape must be measured on **each recipient's individual step**, *after* attribution has identified which donor took which leg. It must **not** be read off the source substation's lumped drop. The reason: a source commonly sheds *different* slices to *different* donors at once — say a PV-heavy slice to donor `j` and a demand-heavy slice to donor `k`. The source's own residual shows only the *sum* of everything it lost, which blends the two into a meaningless average that matches neither leg. Only the per-recipient steps separate cleanly into "what `j` got" vs. "what `k` got." (This is the same source-blends-everything pitfall noted for stage 1, applied to composition rather than magnitude.)

#### Other notes

**Validation against the 32-series logs (this is the point).** Score the unsupervised detector against the known switching events: detection precision/recall, accuracy of the recovered donor set, error in transferred magnitude, and — most importantly — the **detection sensitivity floor**: the transferred-magnitude threshold above which we reliably detect events and below which we cannot. Pair this with the forecast impact of missed small events. *The detector must not consume the logs as input — only as a scoring oracle.*

**Diagnostic precursor.** Before formal changepoints, a near-trivial check: first-difference each residual (steps → spikes) and, over rolling windows, confirm that a source's negative spike coincides with the *summed* rises of a neighbour subset. If known-neighbour groups show no such coincidence, the neighbour list or baseline is wrong — fix that first.

**What it delivers.** A labelled list of detected events (time, source, donor set, per-leg magnitude, rough per-leg composition, score); an ARA mask for the forecasting data; a validation set for later stages; and the quantified sensitivity floor.

**What this approach misses / cons.**

- Misses slow/gradual reconfigurations (because changepoint detection algos assume abrupt shift).
- Misses small partial transfers near the noise floor — but these are both the *hardest* and (per the tolerance principle) the *least important* for the forecast, so the failure mode is aligned with priorities. The sensitivity floor makes this limit explicit rather than hidden.
- **Blind to events that straddle the start of the record.** A switching state already in force before the data begins has no observable onset — it is only detectable if and when it *ends*. This is a structural blind spot, not a fixable one, and belongs in the sensitivity-floor reporting alongside the magnitude threshold.
- Struggles with temporally overlapping events on one neighbourhood.
- Composition read-off is qualitative only.

**Pros.**

- Trivial to build and debug; transparent; every flag is human-interpretable.
- Runs unsupervised, exactly as the scale regime demands.
- Produces the masking artifact and the honest sensitivity floor that everything downstream relies on.
- De-risks the programme at minimal cost.

---

### v2.5 — Magnitude-only mixture model (the workhorse)

**Goal.** Reconstruct a latent NRA demand `d_i(t)` per substation by modelling observed power as a time-varying mixture of each substation's own normal demand and its neighbours'.

**Motivation.** v0.6 only flags events; it does not reconstruct the clean NRA signal. v2.5 is the simplest model that actually *produces* latent demand — and it does so **without modelling anything below the primary**, sidestepping the (non-existent) feeder-discovery problem entirely. The latent objects are the substations themselves, which we observe. This is the workhorse: faithful to how the network actually behaves, and fully unsupervised.

**Method.** Each substation `i` has a latent normal-demand signal `d_i(t)`. Observed power is a time-varying mixture over the neighbourhood:

```text
observed_i(t) = α_ii(t)·d_i(t) + Σ_{j ∈ neighbours(i)} α_ij(t)·d_j(t)
```

- **The neighbourhood is the graph:** `neighbours(i)` is exactly `i`'s neighbour set in the network graph, so the edges act as a *sparsity pattern* on the mixing matrix — most `α_ij` are structurally fixed at zero, and only the handful corresponding to real edges are free parameters. This is what makes the model identifiable and cheap rather than an `N × N` free-for-all.
- Under NRA: `α_ii ≈ 1`, `α_ij ≈ 0`.
- During an ARA: weight shifts from a source onto **one or more** neighbours. Multiple `α_ij(t)` may be active at once for a single source.
- **Conservation = node-level flow balance:** weight leaving `i` is distributed across a subset of neighbours and must sum to the weight lost at `i` (approximately mass-preserving over the affected neighbourhood). **Do not** implement this as independent pairwise equal-and-opposite constraints — that is wrong given confirmed 2–3-way fan-out.
- **Priors / regularisation:** `α(t)` strongly regularised toward the identity (NRA) and **piecewise-constant in time**, because switching events are rare (~10%) and abrupt. A useful by-product: jumps in `α` are directly interpretable as detected switching events.

**Why "arbitrary continuous slice" is handled natively.** `α_ij(t)` is a continuous fraction, so "some load, cut anywhere, moved to several donors" is exactly representable. The continuous-fraction form — which earlier looked like a limitation — is in fact *fidelity* to a network where the transferred amount is genuinely continuous and the cut point is free.

**What it adds over v0.6.**

- Produces the actual latent NRA demand signal, not just event flags.
- Models routing continuously rather than detecting it after the fact.
- Accommodates multi-donor partial transfer through the summed mixture + node-level balance.

**What it misses / cons.**

- A fractional mixture `α_ij·d_j` moves a *scaled copy of neighbour j's whole aggregate demand*. The slice that really moved may have a *different shape* (e.g. unusually PV-heavy). v2.5 can match the step magnitude but carries the wrong shape with it. **Important nuance:** because there is *no stable sub-unit* with a "true" recoverable shape (movable cut points), this is largely **not a fixable limitation** — v2.5's approximation is about as good as the data structurally permits for the *demand* total. v2.6 only partially improves it, and only for the DER component.
- DERs are folded implicitly into `d_i` (no explicit PV/wind separation yet).

**Pros.**

- Well-posed, identifiable, far easier to fit than any sub-primary model.
- No sub-primary modelling whatsoever; fully unsupervised; scales.
- Degrades gracefully; faithful to continuous, multi-donor, partial transfer.
- The right workhorse: build it, measure residual structure around detected events, escalate only if residuals show systematic shape error.

---

### v2.6 — Type-resolved mixture with differentiable physics modules

**Goal.** Decompose each substation into physically-typed components (demand, PV, wind), each from its own differentiable module, and let each *type* transfer with its own routing weights — so a switching event can move proportionally more PV than load.

**Motivation.** v2.5's residual error is wrong-*shape* transfer. Part of that shape error is *type mix*: the moved slice may carry disproportionate PV or wind relative to the parent's aggregate. Separating the physical types lets the model represent that. The differentiable modules also clean the meter of DERs generally (PV/wind net off demand whether or not switching occurs), which is independently valuable.

**Method.** Decompose each substation into typed components, each from its own differentiable forward module:

```text
d_i(t) = gross_demand_i(t)
       − pv_metered_i(t)   − pv_unmetered_i(t)
       − wind_metered_i(t) − wind_unmetered_i(t)
```

- **Demand:** temperature- and time-of-week-shaped.
- **PV:** irradiance-driven (NWP/satellite) via differentiable panel physics (temperature/spectral correction, inverter clipping); capacity is a latent parameter.
- **Wind:** wind speed via a differentiable power curve.
- **Metered vs unmetered split:** metered modules are tightly constrained by registered capacity/generation and mainly *remove* known generation cleanly; unmetered modules carry the genuine latent inference (latent capacity, latent siting), confined to the smaller residual — a meaningful conditioning win.

Mixing operates **per component-type**, each with its own routing weights:

```text
observed_i(t) = Σ_j [ α^dem_ij(t)·gross_demand_j(t)
                    − α^pv_ij(t)·pv_j(t)
                    − α^wind_ij(t)·wind_j(t) ]
```

Each substation is now a small *bundle* of typed nodes rather than one node, and routing happens per type. But the graph stays a plain **data structure**, exactly as in the earlier stages: when the i→j boundary is active, each *type* moves with its own weight — structure-plus-arithmetic, with no message-passing network trained.

**Prior structure.**

1. **Coupled switching, separate composition.** A reconfiguration is one electrical action — the types do not switch at unrelated times. Introduce one latent **switching indicator per ordered pair**, `s_ij(t) ∈ {0,1}`, piecewise-constant and sparse, governing *whether* the i→j boundary is active; the **type composition** (what fraction of demand/PV/wind rides along) applies only when `s_ij(t)=1`. Types switch *together*; the moved slice can still be disproportionately one type.
2. **Per-pair composition prior — DOWNGRADED.** An earlier design proposed a *learnable per-boundary* prior `θ_ij` ("the i→j boundary is usually 90% PV"), treating it as a stable feeder fingerprint. **This is downgraded to at most a weak, shared empirical prior, or dropped entirely.** Reason: movable cut points mean a boundary has *no stable composition* — what crosses depends on where the switch was opened this time — and `θ_ij` could not be fitted at scale anyway, since that requires labels we won't have. Do **not** rely on per-boundary learned composition.
3. **Multi-donor (one-to-many).** As in v2.5, several `s_ij(t)` may be active for one source at once; conservation is node-level flow balance across the donor set, not pairwise.

**What it adds over v2.5.**

- Captures disproportionate-*type* transfer (more PV than load can move).
- Differentiable modules give separately-shaped PV/wind/demand signals, so a step can be attributed to the right type by its temporal signature (midday irradiance-shaped → PV; evening temperature-tracking → load).
- Cleans DERs from the demand estimate generally, not just during switching.

**What it misses / cons.**

- Still moves a *scaled fraction of neighbour j's total PV/wind/demand*, not the specific moved slice's profile. Fixes wrong *type-mix*; does **not** fix wrong *within-type shape*. Given movable cut points, the within-type shape is not stably recoverable anyway, so this residual is largely irreducible and almost certainly second-order for forecasting.
- Larger latent space (three routing matrices): risk the optimiser trades PV- vs wind- vs demand-transfer to fit one step. **Mitigation:** the three types have distinct observable temporal signatures and cannot masquerade as each other if priors lean on irradiance/wind/temperature covariates; regularise each routing weight toward identity and piecewise-constant. When genuinely confounded (a windless, overcast event), accept non-identifiability and let uncertainty reflect it.

**Pros.**

- Physically grounded; the extra freedom is tied to physics, not invented structure.
- Still only primary-level objects; no sub-primary modelling; unsupervised; scales.
- Produces interpretable DER estimates as a by-product.

---

## Part 3 — Considered but rejected: the feeder-block model

An earlier plan included a further stage that modelled the **actual switchable physical units (feeders / load blocks)** explicitly — decomposing each substation into discrete blocks, each routed as a unit. **This stage is retired**, for two independent and decisive reasons:

1. **The unit does not exist.** NGED have been explicit that the network is meshed and run radially with movable cut points; load is a near-continuous distribution splittable almost anywhere. There is no stable, re-identifiable feeder with a persistent identity or composition to discover and route. The model would be trying to recover units that do not persist.
2. **It lives in the unlabelled regime.** This stage is precisely what would run at full scale (~1,161 primary substations), where **no switching labels exist.** Any block model that needed supervision to identify blocks is doomed there twice over.

**Possible deferred successor (a research bet, not a planned step).** If within-type shape error in v2.6 ever proves to materially hurt the forecast, the conceptually correct (but much harder) direction is to model **load as distributed along network arcs with movable cut points**, inferring the cut location rather than assuming discrete blocks. This is a genuine spatial-inference research problem and should be explicitly *deferred*, not chased — and only entertained if v2.6 residuals demonstrate the need.

---

## Part 4 — Cross-cutting implementation requirements

These apply at every stage and are the things most easily got wrong:

- **Fully unsupervised at runtime.** The production model uses **power time series only**. Switching logs are never a runtime input. They exist for the 32-series trial only, and serve exclusively as a **gold-standard test set**.
- **Do not fit pilot-only parameters and rely on them at scale.** Anything learned only on the 16 labelled primaries that cannot be set for the other ~1,145 is forbidden as a production dependency. The *method* generalises; a pilot lookup does not.
- **Conservation is node-level flow balance, everywhere.** One source's lost power is absorbed by a *subset* of neighbours whose pickups sum to it. Never implement conservation as independent pairwise equal-and-opposite matches — confirmed 2–3-way fan-out makes that wrong.
- **Composition is read from recipients, never the source.** Any per-leg composition estimate (v0.6 stage 3) must come from each recipient's individual step *after* attribution; the source's step blends all simultaneous outgoing legs.
- **Partial transfer is the common case.** Expect arbitrary continuous transferred magnitudes down to the noise floor. Quantify the detection sensitivity floor; do not assume a clean event/no-event separation.
- **Routing/switching priors:** regularise toward the identity (NRA) and piecewise-constant in time; switching is rare and abrupt.
- **Metered vs unmetered DER** kept as separate modules from v2.6 onward; metered tightly constrained, unmetered carrying the latent inference.
- **Interpretability artifacts** (detected events, inferred `s_ij`, DER estimates) are deliverables in their own right for NGED validation, not just internal state.

---

## Part 5 — Open items / dependencies

- **Switching logs for the 32-series trial** (held; e.g. the DINDER example). Used as the gold-standard validation set for v0.6 and beyond. **Not** available at full scale — this asymmetry drives the whole design.
- **Neighbour/adjacency structure** for the trial substations — which substations can exchange load (needed to define graph edges and the attribution search). Even approximate adjacency helps; note that because cut points move, "adjacency" means "can be electrically connected by some switching," not a fixed feeder map.
- **Confirmed by NGED:** (a) multi-recipient transfer (2–3 donors) is the norm; (b) partial transfers (some, not all, of a substation's load) are the common and harder case; (c) no stable "feeder" unit exists; (d) switching labels exist only for the trial area, not at scale.
