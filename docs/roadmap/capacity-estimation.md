# Effective-capacity estimation

> **Status: 🚧 Planned (Phase 1, v0.6 / v0.7) · 🔬 Research (Phase 2, v2).** This page is the
> *plan* for applying [differentiable physics](../techniques/differentiable-physics.md) (DP) to
> capacity estimation; the method itself — the forward models, variational machinery, and the
> graph-structured engine — is explained on that techniques page. See the
> [roadmap index](index.md) for status conventions and where this fits the overall plan.

To minimise engineering risk, DP is introduced in two sequential phases that line up with the
project [roadmap](index.md): basic capacity estimation for metered generators in v1, full
disaggregation in v2.

## Phase 1: Dynamic capacity estimation for metered generators (v1)

Targeted at [roadmap v0.6 / v0.7](index.md#v06-v07-switching-events-dynamic-generator-capacity).
This is deliberately the **simplest** application of DP, and **none of the "clever" latent-demand or
ARA inversion happens here.** We deploy a basic DP model of the **metered PV and wind sites** purely
to estimate their physical parameters — most importantly the effective capacity, which bumps up and
down over time as a result of maintenance, faults, and build-out — feeding the
[`effective_capacity`](delivery-tables.md#table-4-effective_capacity) delivery table. Because this
turns the MVP's single scalar-per-series capacity into a time-varying series, the metrics pipeline
must also swap its `time_series_id`-only NMAE-denominator join for a temporal as-of join — see
[Normalising NMAE by `effective_capacity`](metrics-and-leaderboard.md#normalising-nmae-by-effective_capacity).

- **The problem:** A generator's effective capacity drifts over time — turbines fail, inverters drop
  out, panels soil and degrade (or are cleaned and replaced). A static nameplate value introduces
  large downstream errors.
- **The solution:** With the site's coordinates locked and live weather passed through the DP
  module, any residual between expected and actual power is backpropagated to update the
  **effective-capacity parameter** ($\mu_{\text{capacity}}$) on a rolling basis. This doubles as a
  real-time health and availability monitor. The per-site forward model is
  [`DifferentiableSolarPlant`](../techniques/differentiable-physics.md#3-the-core-building-block-differentiablesolarplant)
  (and its wind analogue).
- **What effective capacity must exclude:** ANM curtailment is a deliberate, network-driven
  reduction, not a loss of physical capability. We identify curtailed periods from NGED's
  curtailment/ANM feed and model them as a separate
  [curtailment gate](../techniques/differentiable-physics.md#6-scaling-to-the-full-grid-a-graph-structured-dp-engine),
  so the capacity estimate reflects only the asset's true availability. Folding curtailment into
  capacity would corrupt exactly the signal NGED needs — and
  [the regularisation section below](#regularising-the-capacity-parameter) covers how we further
  stop the capacity parameter from absorbing unexplained noise.
- **How capacity feeds the forecast:** as described in the roadmap, this is a **two-pass** approach —
  the first pass estimates effective capacity (here), and the second normalises each generator's time
  series by its effective capacity before the power-forecast model trains on it. XGBoost continues to
  do the actual power forecasting in v1; DP is responsible only for *estimating capacity* of
  *metered* generators.
- **Causal vs smoothed capacity — a lookahead trap in the two-pass scheme.** A capacity series
  regularised over the whole record
  ([below](#regularising-the-capacity-parameter)) is a *smoother*: it uses
  future observations, so the estimate for a given day changes once a later fault is seen. That is
  correct for the historical [`effective_capacity`](delivery-tables.md#table-4-effective_capacity)
  table and the NMAE denominator — but the capacity used to normalise at forecast init time, in live
  running *and in backtests*, must be the **causal (filtered) estimate available at that init
  time**, or backtest skill is quietly inflated by lookahead. This is the same no-lookahead
  invariant the feature pipeline enforces for power lags.
- **Irradiance inputs:** the beam/diffuse decomposition the
  [solar model](../techniques/differentiable-physics.md#3-the-core-building-block-differentiablesolarplant)
  needs is covered by the v0.6 weather ingests. **CM SAF SARAH-3** provides global (SIS), direct
  (SID) and direct-normal (DNI) irradiance at 0.05° / 30-minute resolution from 1983 (diffuse =
  SIS − SID) — the primary Phase 1 input, matching the half-hourly metering. **CERRA** provides
  global plus time-integrated direct short-wave (diffuse by subtraction; accumulated fluxes from
  3-hourly forecast cycles, so temporally coarser). The live **ECMWF ENS** feed carries only GHI —
  fine for Phase 1, but Phase 2 DP *forecasting* of PV needs a differentiable GHI → DNI/DHI
  decomposition model (or `fdir` added to the upstream dataset). See
  [data sources](data-sources.md#weather-data).

## Regularising the capacity parameter

The capacity parameter must not be free to bounce around at the data's sampling rate, or it will simply soak up whatever noise the rest of the model cannot explain. The right regulariser depends on whether the capacity can physically *fall* as well as rise.

### Metered effective capacity (can go up *or* down)

The effective capacity of a metered generator changes in both directions: it drops when turbines fail or inverters trip, and recovers when they are repaired. We therefore represent it as a smoothly-varying latent series and penalise high-frequency movement — for example a total-variation (or random-walk) penalty on the step-to-step change. This lets capacity track genuine, persistent changes (a turbine offline for a fortnight) while refusing to chase half-hourly noise. Sudden, sustained drops are exactly the generator-fault signal we want to surface.

### Unmetered installed capacity (grows monotonically)

The *installed* capacity of an unmetered fleet behaves differently: it essentially only ever grows, as more households and businesses fit panels. Here a monotonic representation is the right prior. We model capacity as a cumulative sum of per-week increments, each constrained to be **non-negative**, with an L1 (sparsity) penalty pushing most weekly increments to exactly zero — because installs happen in occasional bursts, not every week. The running total is then non-decreasing by construction. (This is the representation
[`UniversalSolarFleetNode`](../techniques/differentiable-physics.md#4-scaling-to-aggregate-fleets-universalsolarfleetnode)
implements.)

### Keeping weather bias out of capacity

The irradiance driving the forward model is itself biased: NWP and satellite products carry regional, *seasonal* error (satellite retrievals degrade at low UK winter sun angles, for example). Per site, that bias is indistinguishable from slow capacity drift — an unconstrained fit will alias it into exactly the signal we deliver to NGED. The mitigation exploits structure: every site in a region sees the **same** weather bias, while genuine capacity changes are site-specific. Fitting a **shared regional irradiance-bias term jointly across the metered fleet** separates the two. This is the Phase 1-sized version of what
[the weather encoder](../techniques/differentiable-physics.md#5-combining-differentiable-physics-with-the-weather-encoder)
does in v2.

### Economic curtailment (not in the ANM feed)

[Phase 1](#phase-1-dynamic-capacity-estimation-for-metered-generators-v1) keeps *network-driven* (ANM) curtailment out of capacity via the curtailment gate — but generators also **self-curtail economically**, increasingly commonly during negative-price periods, and that never appears in NGED's ANM feed. Unmodelled, it reads as capacity loss. Mitigations: treat capacity as an **upper envelope** (an asymmetric, quantile-flavoured loss that penalises under-predicting the best observed output far more than over-predicting typical output), and/or mask periods flagged as negative-price from public market data.

### A cheap baseline to beat

The DP estimator should be benchmarked against a deliberately simple baseline on the metered ground truth: a rolling quantile of clear-sky-normalised output (observed power ÷ physics-predicted clear-sky power), and/or the convex capacity-change detection in SLAC's [solar-data-tools](https://github.com/slacgismo/solar-data-tools). If DP cannot beat the cheap estimator, that is a finding worth surfacing early — and the baseline slots into the same leaderboard discipline used for the forecasting models.

## Phase 2: Full graph-structured disaggregation (v2)

Part of [roadmap v2.0](index.md#v20-scale-up-future-research). Once the metered assets are accurately
tracked, we unlock
[the full graph-structured architecture](../techniques/differentiable-physics.md#6-scaling-to-the-full-grid-a-graph-structured-dp-engine).
The DP modules for the *unmetered* fleets
([`UniversalSolarFleetNode`](../techniques/differentiable-physics.md#4-scaling-to-aggregate-fleets-universalsolarfleetnode))
are bound to the graph's nodes. The network uses the verified metered assets and spatial weather
cues to disaggregate the mixed substation signals, cleanly separating true gross demand from hidden
behind-the-meter renewable generation. This is also where DP graduates from *estimating capacity* to
directly *forecasting power* (including for MVA-metered sites), and where the **latent-demand
inversion** of
[the forward model](../techniques/differentiable-physics.md#2-the-core-idea-inversion-through-a-differentiable-forward-model)
is realised in full. Abnormal running arrangements — recovering the demand that would have been
metered under the normal running arrangement — are handled in their own canonical doc,
[Switching events & latent demand](switching-events.md).
