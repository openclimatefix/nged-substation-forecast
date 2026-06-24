# Estimating Latent Demand Under Switching Events — Approach & Implementation Roadmap

**Scope.** How to estimate, for each NGED primary substation, its *latent demand under the normal running arrangement* — the demand that would be metered if the network were never reconfigured — given that the network is in fact reconfigured roughly 10% of the time by switching events. This document defines what switching events are, why they are hard, and a staged modelling plan for different versions of the NGED forecasting system (v0.6 → v2.5 → v2.6).

---

## Part 1 — What is a switching event?

### 1.1 The HV network is meshed but run radially

NGED's 11 kV / 6.6 kV high-voltage (HV) distribution network is **physically meshed**: there are many parallel electrical paths between substations. However, it is **operated radially** — at any given moment, a set of switches is held *open* to break the parallel paths, so that power flows in a tree (each load fed from exactly one source, no loops). The radial tree you observe at any instant is just *one configuration* of an underlying mesh.

This matters enormously, and it is the crux of everything below:

- **Almost any switch can be opened or closed.** The network can be reconfigured into an enormous number of valid radial trees by choosing which switches are open. The configuration is not fixed.
- **There is no stable, re-identifiable "feeder."** Intuitively one might imagine a substation's load is divided into a handful of fixed "feeders," each a chunk that moves as a unit. NGED have been explicit that this is *not* how it works: because a switch can be opened essentially anywhere along a meshed path, the cut points themselves move. There is no persistent sub-unit with a stable identity or a stable composition.
- **Load is a near-continuous distribution along the network.** Demand is spread along the HV circuits and can be split at (almost) any point. When operators reconfigure, the amount of load that moves is "whatever happened to sit between the old cut point and the new one" — an unknown, continuously variable quantity.

The picture below contrasts the *physical* mesh (all paths exist) with the *operated* radial state (some switches held open, marked `/`, so power flows in a tree). `[A]`–`[D]` are primary substations; `===` is an energised circuit; `/` is a normally-open switch; `·` marks the load tapped along each circuit.

```
   PHYSICAL MESH (all paths exist)          OPERATED RADIALLY (open switches break loops)

        [A]=====·=====[B]                        [A]=====·=====[B]
         ||           ||                          ||           /        <- open: B's tail
         ·            ·                           ·                         now fed from D
         ||           ||                          ||
        [C]=====·=====[D]                        [C]=====·=====[D]
         (a loop exists A-B-D-C-A)                (no loop: radial tree rooted at A)
```

Both pictures are the *same wires*. Operators choose which switches are open, so the radial tree on the right is just one of many valid configurations of the mesh on the left. A switching event moves the `/` marks — and because the tapped load `·` is spread continuously along every circuit, moving a switch transfers *whatever load sat between the old and new cut points*.

Switching events can last from minutes to months. Each substation is in a switching event for
roughly 10% of the time. One "saving grace" is that, the smaller a switching event, the less we care
about it (because a small switching event won't harm our forecast much).

### 1.2 What actually happens during a switching event

A **switching event** is a reconfiguration: one or more open switches are closed and/or one or more closed switches are opened, changing which source feeds which load. It can be:

- **Planned** (maintenance, load balancing), or
- **Unplanned** (automatic protection response to a fault).

The **observable effect at the substation meters** is a sustained, roughly step-change in net power: load that used to be metered at substation A is, after the event, metered at substations B, C, … instead. The total power across the affected neighbourhood is conserved — power is neither created nor destroyed by reconfiguring — but *where it is metered* changes.

### 1.3 Worked example (from NGED's control-room log)

NGED supplied a real fault example. At 12:01:38 on 25/05/2026 the **DINDER** primary tripped on a fault (an 11 kV breaker tripped). Over the following ~90 seconds, the control system restored supply by closing switches that picked DINDER's load up from **multiple primaries**:

- **Cathedral Park** (= **Wells** primary)
- **16TD11**, **Cowl Street**, **16MW2** (all = **Shepton Mallet** primary)

So a single source substation's load fanned out to **at least two distinct primaries** (Wells and Shepton Mallet), via several separate switching operations. This is a *whole-primary* transfer triggered by a fault.

### 1.4 The two facts that make this hard

NGED have confirmed two things that shape the entire problem:

1. **Multi-recipient is the norm.** When load is diverted, it typically fans out to **2–3 neighbouring substations**, not one. Conservation must be reasoned about as a **node-level flow balance**: one source's lost power is absorbed by a *subset* of neighbours whose individual pickups sum to the source's loss.

2. **Partial transfer is the common case, and it is harder.** The clean whole-primary transfer in the worked example (a big, obvious step) is the *easy, rarer* case. The common case is that **only some of a substation's load is diverted** — an arbitrary continuous slice, cut at a movable point. The transferred magnitude is a free continuous variable with no minimum size, so small partial transfers shade continuously down into the measurement noise. **Detection difficulty scales inversely with how much load moved.**

The diagram shows why "how much moved?" has no clean answer. The load `·` is spread along the circuit between `[A]` and `[B]`; the cut point (open switch `/`) can sit anywhere, and wherever it sits determines how much load each end keeps.

```
   Circuit between two substations, load tapped continuously along it:

   [A]==·==·==·==·==·==·==·== / ==·==·==[B]
        \__________________/     \____/
             fed from A         fed from B    <- cut point here: A keeps most


   [A]==·==·==·==·==·== / ==·==·==·==·==[B]
        \____________/    \____________/
          fed from A        fed from B        <- cut point moved left: B keeps more

   The amount transferred = whatever load lies between the OLD and NEW cut points.
   It is continuous, unknown, and usually only a SLICE (not the whole substation).
```

### 1.5 The labels asymmetry (the dominant design constraint)

We have **switching labels (control-room logs) only for the 32-series trial area** (16 primaries plus associated generation/GSP/BSP series). When the system expands to the full ~1,500 substations, **we will have no switching labels.**

This is the single most important architectural fact in this document. It means:

- **The production model must run fully unsupervised, on power time series alone.** Switching logs cannot be a runtime input, because at scale they do not exist.
- **The 32-series logs are a one-time gold-standard *test set*, not a crutch.** We use them to *validate* the unsupervised method (measure detection precision/recall, recipient-set accuracy, magnitude error), then throw the method — not the labels — at the full network.
- **Nothing may be learned only where labels exist and relied upon at scale.** Any per-substation or per-boundary parameter fitted only on the labelled 16 primaries is a parameter we cannot set for the other ~1,484. The 16 labelled primaries are a *yardstick*, not a representative seed. The thing that must generalise is the *method*, not a lookup table fitted to the pilot.

### 1.6 The target variable

NGED's stated requirement is to forecast each substation **as if it were always in its normal running arrangement (NRA)** — its design topology. So the quantity we want is not the raw meter reading (which is contaminated by whatever switching state happened to hold), but the **latent demand under NRA**: what the substation would have metered had no reconfiguration occurred. Network planners reason about headroom and capacity under the design topology, so this is the operationally useful signal.

(Separately, the meter also nets demand against **distributed energy resources** — behind-the-meter solar PV, small wind, batteries. Recovering true demand also means accounting for these. Part 3 introduces explicit DER modelling; the earlier stages fold DERs implicitly into the latent signal.)

---

## Part 2 — Concepts from first principles

Two ideas recur below and are worth defining plainly for readers new to them.

### 2.1 Differentiable physics

A **forward model** is a physics-based equation that predicts an observation from underlying causes. For example, solar PV power can be predicted from irradiance, panel capacity, temperature, and inverter limits using well-understood physics. "**Differentiable**" means the model is implemented in a framework (PyTorch) that can compute the *gradient* of its output with respect to its inputs and parameters — i.e. "if I nudge the panel capacity up a little, how much does predicted power change?"

Why this is useful: if you have the *observed* power and a differentiable forward model, you can run the model **backwards**. You start with guessed values for the unknowns (e.g. latent demand, panel capacity), compare the model's prediction to the observation, and use the gradient to iteratively adjust the unknowns until the prediction matches. This is called **inversion**, and it is just gradient-based optimisation — conceptually the same machinery as fitting any model by minimising a loss, except the model in the middle is a physics equation rather than a generic regressor. The payoff is that the recovered parameters are physically meaningful and the model needs far less data than a black box, because physics constrains the solution.

### 2.2 Graphs (the structure — note: we do *not* need a full GNN)

A **graph** is just **nodes** connected by **edges**. Here the nodes are substations and the edges connect substations that can exchange load (i.e. that can be electrically joined by some switching operation). The graph is the natural way to encode the one fact a per-substation model is blind to: that a switching event is a *cross-substation* phenomenon — load leaving one node reappears at others, so the drop at A and the rises at B and C are the *same power moving*.

It is worth being explicit, because the term invites confusion: **none of the approaches in this document require a graph neural network (GNN).** A GNN is a *trained* model that passes learned "messages" along edges; it is a heavyweight tool we do not need. We use the graph purely as a **data structure** — a fixed map of "who can exchange load with whom" — and run simple, mostly closed-form operations over it. Concretely, the graph is used as follows in each stage:

- **v0.6 (detector):** the edges define, for each substation `i`, the small candidate set of neighbours to search when attributing a detected drop. The node-level balance test ("which subset of `i`'s neighbours show coincident rises summing to `i`'s drop?") is run *only over `i`'s graph neighbours*, which is what keeps the subset search cheap. No learning over the graph — just a lookup of which substations to consider.

- **v2.5 (mixture model):** the edges define which mixing weights `α_ij(t)` are even allowed to be non-zero — a substation can only receive load from its graph neighbours. The graph is therefore a *sparsity pattern* on the mixing matrix: most `α_ij` are structurally fixed at zero, and only the handful corresponding to real edges are free parameters. This is what makes the model identifiable and cheap rather than an `N × N` free-for-all.

- **v2.6 (type-resolved model):** the graph gains **typed nodes**. Each substation is no longer a single node but a small bundle of nodes — `gross_demand`, `metered_PV`, `unmetered_PV`, `metered_wind`, `unmetered_wind` — each carrying its own signal. Edges still connect substations that can exchange load, but routing now happens per node-type (`α^demand_ij`, `α^pv_ij`, `α^wind_ij`), so the graph expresses "when the i→j boundary is active, each *type* of node can move with its own weight." This is how the model represents a switching event that carries proportionally more PV than load. It is still just structure-plus-arithmetic — no message-passing network is trained.

In short: the graph tells us *who can connect to whom* (and, in v2.6, *which typed components sit at each site*); the simple statistics and the differentiable forward model do the rest. If a future stage ever genuinely benefited from a trained GNN, that would be a deliberate escalation — but nothing here needs it.


---

## Part 3 — The staged roadmap

Version numbers are aligned with the main codebase: **v0.6**, then **v2.5**, then **v2.6**. Each stage states its motivation, what it adds, what it misses, and its trade-offs. (Although, note that we haven't fully specified the roadmap _after_ v2, so read "v2.x" as just meaning "some time after v2 is operational"). Escalation between stages must be justified by *measured residual structure at the previous stage*, not by anticipation — this keeps effort matched to demonstrated need.

### Overview of the ladder

```
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

1. **Per-series changepoint on the residual.** For each substation, form a cheap expected-power baseline (e.g. existing XGBoost forecast _without_ any lagged features) and take the residual (observed − expected). A switching event is a *sustained level shift* in this residual — not a spike, not a slope. Detect level shifts with a standard mean-shift changepoint method (PELT or binary segmentation with an L2 cost, or CUSUM). Detrending first ensures we detect shifts in *behaviour-relative-to-normal*, so weather and time-of-day don't masquerade as events. **Output:** candidate step times and magnitudes per substation.

2. **Node-level coincidence / balance attribution.** A level shift at one substation could be many things (fault, new connection, meter error). What makes it a *switching event* is the conservation fingerprint: coincident, opposite-sign shifts at neighbours that *collectively balance*. Because transfers fan out to 2–3 neighbours, **do not match pairwise.** Instead, for each candidate drop of magnitude `Δ` at substation `i` at time `t`, solve a small constrained attribution: *which subset of `i`'s neighbours show coincident rises (≈ `t`) that sum to ≈ `Δ`?* With a handful of neighbours per primary this is cheap — enumerate subsets, or run a small non-negative least-squares of neighbour rises against the source drop. Score by timing coincidence × magnitude-balance agreement. High score → switching event with an identified donor set; low score → "anomaly, unknown cause."

   ```
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

3. **Composition corroboration (post-attribution, per recipient).** Once the donor set is identified, characterise *what* moved on each leg by the diurnal shape of that recipient's step: midday-peaking → PV-heavy; flat/evening-peaking → load-heavy. This is a histogram, not a model. **Order matters:** read composition off each *recipient's* individual step *after* attribution, never off the *source's* lumped step — a source shedding PV-heavy load to one donor and load-heavy load to another shows only a meaningless blend in its own residual.

**Validation against the 32-series logs (this is the point).** Score the unsupervised detector against the known switching events: detection precision/recall, accuracy of the recovered donor set, error in transferred magnitude, and — most importantly — the **detection sensitivity floor**: the transferred-magnitude threshold above which we reliably detect events and below which we cannot. Pair this with the forecast impact of missed small events. *The detector must not consume the logs as input — only as a scoring oracle.*

**Diagnostic precursor.** Before formal changepoints, a near-trivial check: first-difference each residual (steps → spikes) and, over rolling windows, confirm that a source's negative spike coincides with the *summed* rises of a neighbour subset. If known-neighbour groups show no such coincidence, the neighbour list or baseline is wrong — fix that first.

**What it delivers.** A labelled list of detected events (time, source, donor set, per-leg magnitude, rough per-leg composition, score); an ARA mask for the forecasting data; a validation set for later stages; and the quantified sensitivity floor.

**What it misses / cons.**
- Misses slow/gradual reconfigurations (changepoint assumes abrupt shift).
- Misses small partial transfers near the noise floor — but these are both the *hardest* and (per the tolerance principle) the *least important* for the forecast, so the failure mode is aligned with priorities. The sensitivity floor makes this limit explicit rather than hidden.
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

```
observed_i(t) = α_ii(t)·d_i(t) + Σ_{j ∈ neighbours(i)} α_ij(t)·d_j(t)
```

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

```
d_i(t) = gross_demand_i(t)
       − pv_metered_i(t)   − pv_unmetered_i(t)
       − wind_metered_i(t) − wind_unmetered_i(t)
```

- **Demand:** temperature- and time-of-week-shaped.
- **PV:** irradiance-driven (NWP/satellite) via differentiable panel physics (temperature/spectral correction, inverter clipping); capacity is a latent parameter.
- **Wind:** wind speed via a differentiable power curve.
- **Metered vs unmetered split:** metered modules are tightly constrained by registered capacity/generation and mainly *remove* known generation cleanly; unmetered modules carry the genuine latent inference (latent capacity, latent siting), confined to the smaller residual — a meaningful conditioning win.

Mixing operates **per component-type**, each with its own routing weights:

```
observed_i(t) = Σ_j [ α^dem_ij(t)·gross_demand_j(t)
                    − α^pv_ij(t)·pv_j(t)
                    − α^wind_ij(t)·wind_j(t) ]
```

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

## Part 4 — Retired stage: the feeder-block model

An earlier plan included a further stage that modelled the **actual switchable physical units (feeders / load blocks)** explicitly — decomposing each substation into discrete blocks, each routed as a unit. **This stage is retired**, for two independent and decisive reasons:

1. **The unit does not exist.** NGED have been explicit that the network is meshed and run radially with movable cut points; load is a near-continuous distribution splittable almost anywhere. There is no stable, re-identifiable feeder with a persistent identity or composition to discover and route. The model would be trying to recover units that do not persist.
2. **It lives in the unlabelled regime.** This stage is precisely what would run at full scale (~1,500 substations), where **no switching labels exist.** Any block model that needed supervision to identify blocks is doomed there twice over.

**Possible deferred successor (a research bet, not a planned step).** If within-type shape error in v2.6 ever proves to materially hurt the forecast, the conceptually correct (but much harder) direction is to model **load as distributed along network arcs with movable cut points**, inferring the cut location rather than assuming discrete blocks. This is a genuine spatial-inference research problem and should be explicitly *deferred*, not chased — and only entertained if v2.6 residuals demonstrate the need.

---

## Part 5 — Cross-cutting implementation requirements

These apply at every stage and are the things most easily got wrong:

- **Fully unsupervised at runtime.** The production model uses **power time series only**. Switching logs are never a runtime input. They exist for the 32-series trial only, and serve exclusively as a **gold-standard test set**.
- **Do not fit pilot-only parameters and rely on them at scale.** Anything learned only on the 16 labelled primaries that cannot be set for the other ~1,484 is forbidden as a production dependency. The *method* generalises; a pilot lookup does not.
- **Conservation is node-level flow balance, everywhere.** One source's lost power is absorbed by a *subset* of neighbours whose pickups sum to it. Never implement conservation as independent pairwise equal-and-opposite matches — confirmed 2–3-way fan-out makes that wrong.
- **Composition is read from recipients, never the source.** Any per-leg composition estimate (v0.6 stage 3) must come from each recipient's individual step *after* attribution; the source's step blends all simultaneous outgoing legs.
- **Partial transfer is the common case.** Expect arbitrary continuous transferred magnitudes down to the noise floor. Quantify the detection sensitivity floor; do not assume a clean event/no-event separation.
- **Routing/switching priors:** regularise toward the identity (NRA) and piecewise-constant in time; switching is rare and abrupt.
- **Metered vs unmetered DER** kept as separate modules from v2.6 onward; metered tightly constrained, unmetered carrying the latent inference.
- **Interpretability artifacts** (detected events, inferred `s_ij`, DER estimates) are deliverables in their own right for NGED validation, not just internal state.

---

## Part 6 — Open items / dependencies

- **Switching logs for the 32-series trial** (held; e.g. the DINDER example). Used as the gold-standard validation set for v0.6 and beyond. **Not** available at full scale — this asymmetry drives the whole design.
- **Neighbour/adjacency structure** for the trial substations — which substations can exchange load (needed to define graph edges and the attribution search). Even approximate adjacency helps; note that because cut points move, "adjacency" means "can be electrically connected by some switching," not a fixed feeder map.
- **Confirmed by NGED:** (a) multi-recipient transfer (2–3 donors) is the norm; (b) partial transfers (some, not all, of a substation's load) are the common and harder case; (c) no stable "feeder" unit exists; (d) switching labels exist only for the trial area, not at scale.
