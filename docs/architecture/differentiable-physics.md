# Differentiable Physics (DP) for NGED

This framework bridges pure machine learning with mechanistic domain knowledge to solve the net-demand disaggregation problem across the National Grid Electricity Distribution (NGED) network.

---

## 1. Why Differentiable Physics?

Pure black-box models (like standard neural networks or XGBoost) require vast amounts of data to learn basic physical invariants and are prone to overfitting or hallucinating non-physical behaviour. We inject a **Differentiable Physics (DP)** layer directly into our computational graph for three core reasons:

* **Data and sample efficiency:** The model does not have to spend capacity re-learning solar geometry (where the sun is) or the shape of a turbine power curve (roughly cubic in wind speed between cut-in and rated, then flat to rated power, then zero beyond cut-out). The physics engine supplies these constraints for free, letting the ML components focus entirely on atmospheric and behavioural nuances.
* **True invertibility (inputs as parameters):** Because every operation in our physics engine preserves gradients, we can backpropagate errors all the way to the *inputs*. This lets us treat unobserved variables (like local irradiance) or system configurations as parameters that can be solved via gradient descent.
* **Interpretability:** Instead of inspecting uninterpretable latent layers, our system updates explicit physical parameters like tilt, azimuth, or capacity. This lets engineers immediately audit the model's assumptions.

---

## 2. Python Sketch: `DifferentiableSolarPlant`

We model each physical parameter as a learnable Normal distribution $\mathcal{N}(\mu, \sigma^2)$ — a mean-field variational posterior — and train with the reparameterisation trick (`rsample()`) so gradients flow through the sampling step.

Crucially, the training objective is an **ELBO**, not a bare reconstruction loss: a power-reconstruction term *plus* a KL term that pulls each posterior toward a fixed physical prior. The KL term is not optional. Minimising power error alone always rewards shrinking $\sigma \to 0$, so the parameter "uncertainty" we are trying to capture would simply collapse. The prior does double duty: it keeps the posterior spreads honest, and it injects weak domain knowledge (e.g. "panels point roughly south at a typical UK roof pitch") that regularises sites with little data.

Two physics details the sketch gets right:

- **Irradiance transposition.** Plane-of-array (POA) irradiance is *not* GHI scaled by the angle of incidence — GHI already bakes in a cosine-of-zenith projection. We decompose the resource into beam, sky-diffuse and ground-reflected components (an isotropic sky model) and transpose each correctly. The beam term uses DNI (not GHI) projected by the angle of incidence.
- **Capacity is a power.** The learnable `dc_capacity` is the DC nameplate in power units; POA is normalised by the reference irradiance (1000 W/m²) so that capacity falls out in MW at standard test conditions. The AC inverter limit is a separate clip (see §8).

Angle convention: azimuth is measured from due south, with east negative and west positive (east = −90°, south = 0°, west = +90°).

Here's a quick Python code sketch of the rough idea for applying differentiable physics to solar:

```python
import torch
import torch.nn as nn
import torch.distributions as dist

STC_IRRADIANCE = 1000.0  # reference plane-of-array irradiance at standard test conditions (W/m^2)
GROUND_ALBEDO = 0.2      # typical broadband ground reflectance


class DifferentiableSolarPlant(nn.Module):
    """Differentiable physical model of a single metered solar site.

    Each physical parameter is a mean-field variational posterior N(mu, sigma^2).
    Training maximises an ELBO: a power-reconstruction term plus a KL term against a
    fixed prior (see `kl_divergence`). The KL term is what stops the posterior spreads
    from collapsing to zero under a pure reconstruction loss.

    Azimuth convention: measured from due south, east negative, west positive.
    """

    def __init__(self, prior_tilt: float, prior_azimuth: float, prior_dc_capacity: float) -> None:
        super().__init__()
        # Variational posteriors. tilt/azimuth are in radians; capacity lives in
        # log-space so it stays strictly positive without a gradient-breaking clamp.
        self.tilt_mu = nn.Parameter(torch.tensor(prior_tilt))
        self.tilt_log_std = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ~ 0.13 rad initially

        self.azimuth_mu = nn.Parameter(torch.tensor(prior_azimuth))
        self.azimuth_log_std = nn.Parameter(torch.tensor(-2.0))

        self.log_capacity_mu = nn.Parameter(torch.log(torch.tensor(prior_dc_capacity)))
        self.log_capacity_log_std = nn.Parameter(torch.tensor(-2.0))

        # Fixed priors. Weakly-informative: "panels point roughly south at a UK roof pitch,
        # with a capacity near nameplate". These also keep the posterior spreads from collapsing.
        self.priors = {
            "tilt": dist.Normal(torch.tensor(prior_tilt), torch.tensor(0.3)),
            "azimuth": dist.Normal(torch.tensor(prior_azimuth), torch.tensor(0.5)),
            "log_capacity": dist.Normal(torch.log(torch.tensor(prior_dc_capacity)), torch.tensor(0.5)),
        }

    def posteriors(self) -> dict[str, dist.Normal]:
        """The current variational posterior for each physical parameter."""
        return {
            "tilt": dist.Normal(self.tilt_mu, self.tilt_log_std.exp()),
            "azimuth": dist.Normal(self.azimuth_mu, self.azimuth_log_std.exp()),
            "log_capacity": dist.Normal(self.log_capacity_mu, self.log_capacity_log_std.exp()),
        }

    def kl_divergence(self) -> torch.Tensor:
        """Sum of KL(posterior || prior) over all parameters — the regulariser in the ELBO."""
        q, p = self.posteriors(), self.priors
        return sum(dist.kl_divergence(q[key], p[key]) for key in q)

    def forward(
        self,
        dni: torch.Tensor,             # direct normal irradiance (W/m^2)
        dhi: torch.Tensor,             # diffuse horizontal irradiance (W/m^2)
        ghi: torch.Tensor,             # global horizontal irradiance (W/m^2)
        sun_zenith_rad: torch.Tensor,
        sun_azimuth_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Predict DC power. All inputs and operations preserve gradients."""
        q = self.posteriors()

        # Reparameterised samples (one per forward pass).
        tilt = q["tilt"].rsample().clamp(0.0, torch.pi / 2)
        azimuth = q["azimuth"].rsample()
        dc_capacity = q["log_capacity"].rsample().exp()  # strictly positive by construction

        # Angle of incidence on the tilted plane.
        cos_aoi = (
            torch.cos(sun_zenith_rad) * torch.cos(tilt)
            + torch.sin(sun_zenith_rad) * torch.sin(tilt) * torch.cos(sun_azimuth_rad - azimuth)
        ).clamp(min=0.0)  # no beam contribution when the sun is behind the panel

        # Isotropic-sky transposition to plane-of-array (POA) irradiance.
        poa_beam = dni * cos_aoi
        poa_sky_diffuse = dhi * (1.0 + torch.cos(tilt)) / 2.0
        poa_ground = ghi * GROUND_ALBEDO * (1.0 - torch.cos(tilt)) / 2.0
        poa = poa_beam + poa_sky_diffuse + poa_ground

        # DC power: capacity is the nameplate at the reference irradiance.
        return dc_capacity * poa / STC_IRRADIANCE

```

---

## 3. Graph Neural Network (GNN) Integration

The distribution network is fundamentally a topological graph. We couple our Differentiable Physics engine with a spatial GNN so that power balances across the grid are conserved by construction. This mirrors the GNN schematic in the Milestone 1 report (Fig. 10).

### Node definitions

Following the report's schematic, the graph uses the following node types:

1. **Substation nodes** — the measured, net blended power flow at a primary substation (the main target constraint).
2. **Metered load nodes** — demand that NGED meters directly, where such metering exists.
3. **Gross demand nodes** — the underlying, unmetered consumer load, inferred by the model.
4. **Metered PV / Wind nodes** — generators with dedicated, live generation metering.
5. **Unmetered PV / Wind fleet nodes** — aggregated behind-the-meter (BTM) solar and distributed wind, grouped by location, with no direct metering.

Each generation node feeds the substation through a **curtailment gate**: a separate multiplicative factor, driven by NGED's ANM/curtailment data feed, that represents network-enforced reductions. Keeping curtailment in its own gate (rather than inside the capacity parameter) is what lets the effective-capacity estimate stay a clean measure of physical availability — see §4 and Fig. 10.

### The fusion mechanism

The GNN handles spatial message-passing to capture regional correlations (e.g. if it is raining at Substation A, the adjacent Unmetered PV Fleet B is probably cloudy too). The GNN's spatial hidden layers feed the parameters/inputs of the **DP modules**, which compute explicit physical generation. A hard Kirchhoff balance node then aggregates the elements:

$$\text{Net substation flow} = \text{Gross demand} - \gamma_{\text{PV}}\,(\text{PV}_{\text{metered}} + \text{PV}_{\text{unmetered}}) - \gamma_{\text{wind}}\,(\text{Wind}_{\text{metered}} + \text{Wind}_{\text{unmetered}})$$

where the $\gamma$ terms are the per-asset curtailment gates. The error between predicted and measured substation flow produces a gradient that flows back through the graph, optimising the GNN weights and the physical parameter posteriors simultaneously.

### Apparent-power (MVA) metering

Some substations are metered only in apparent power (MVA), which reports the *absolute value* of flow and so cannot distinguish import from export — when embedded generation pushes power back into the grid, an MVA trace "bounces" off zero instead of going negative. Because the DP/GNN framework reconstructs signed demand and generation explicitly, it handles this natively: we compare the measured MVA reading against the *magnitude* of the reconstructed net flow,

$$\text{MVA}_{\text{measured}} \approx \bigl|\,\text{Net substation flow}\,\bigr|$$

(assuming near-unity power factor). The physics grounds the model so that a sunny-day "bounce" is correctly attributed to reverse power flow from generation, not to a spike in demand. This is one of the two capabilities the Milestone 1 report highlights for DP — the other being unmetered disaggregation.

---

## 4. Phasing Roadmap

To minimise engineering risk, DP is introduced in two sequential phases that line up with the project [roadmap](../roadmap.md): capacity estimation in v1, full disaggregation in v2.

### Phase 1: Dynamic capacity estimation (v1)

Before attempting full disaggregation, DP is deployed purely to estimate the effective capacity of the **metered PV and wind sites** (roadmap v0.6 / v0.7).

* **The problem:** A generator's effective capacity drifts over time — turbines fail, inverters drop out, panels soil and degrade (or are cleaned and replaced). A static nameplate value introduces large downstream errors.
* **The solution:** With the site's coordinates locked and live weather passed through the DP module, any residual between expected and actual power is backpropagated to update the **effective-capacity parameter** ($\mu_{\text{capacity}}$) on a rolling basis. This doubles as a real-time health and availability monitor.
* **What effective capacity must exclude:** ANM curtailment is a deliberate, network-driven reduction, not a loss of physical capability. We identify curtailed periods from NGED's curtailment/ANM feed and model them as a separate gate (§3), so the capacity estimate reflects only the asset's true availability. Folding curtailment into capacity would corrupt exactly the signal NGED needs — and §7 covers how we further stop the capacity parameter from absorbing unexplained noise.

### Phase 2: Full GNN-coupled disaggregation (v2)

Once the metered assets are accurately tracked, we unlock the full architecture. The DP modules for the *unmetered* fleets are bound to the GNN nodes. The network uses the verified metered assets and spatial weather cues to disaggregate the mixed substation signals, cleanly separating true gross demand from hidden BTM renewable generation.

---

## 5. Long-Term Vision: GB-Wide Inverse Irradiance Mapping

Once the v1 architecture has calibrated, parameter-verified "virtual sensors" across the metered fleet, we can run the inversion trick at scale. Freezing the calibrated asset parameters and running gradient descent *backward* through the DP modules — from measured generation to the weather inputs — recovers a surface-irradiance estimate (and, for wind, a wind-speed estimate) **at each metered site**. These point estimates are sparse virtual observations; the spatial GNN then interpolates between them to fill in a denser field across Great Britain. The result would be a half-hourly, physics-validated weather product, independent of the NWP, useful as a cross-check for real-time grid balancing. This is a research aspiration well beyond v2, and the density of the recovered field is fundamentally limited by the spatial coverage of the metered fleet.

---

## 6. Combining differentiable physics with weather encoder

```
+-------------------+
| Learnt parameters |
|     per site:     |
|                   |
|  • PV tilt        |                   +---------------+
|  • PV azimuth     |     <=======>     | pvlib-pytorch |----------+
|  • AC capacity    |                   +-------+-------+          |
|  • DC capacity    |                           ^                  |
|  • etc.           |                           |                  v
+-------------------+                           |           +-------------+
                                         +------+------+    |  multi-seq  |
                                         | Irradiance  |    |  alignment  |
                                         | Temperature |    |  with axial |---> [ p̂ ]
                                         +------+------+    |  attention  |
                                                ^           +-------------+
+--------------+     +-------------------+      |                  ^
| Weather data |---->|  weather encoder  |------+                  |
+--------------+     +-------------------+                  +------+------+
                                                            |   History   |
                                                            +-------------+
```

- **NWP bias** is handled by the **weather encoder**: "the weather model says it's cloudy, but historically this specific pressure pattern at this location means it's actually clear" (feature-level correction).
- **Physical constraints** are handled by the **differentiable physics**: "based on the corrected weather, the geometry of the sun and panel dictates $X$ power" (first-principles baseline).
- **Systematic / local anomalies** (the "unknown unknowns") are handled by the **retrieval / alignment** module: "on days that looked exactly like this in the past, the physics model consistently over-predicted the evening ramp-down by 5% because of that one tree on the horizon" (residual correction).

`pvlib-pytorch` is a planned Open Climate Fix open-source library — a differentiable, PyTorch-native port of [pvlib](https://pvlib-python.readthedocs.io/) — that we intend to spin out of this project. It would generalise the hand-rolled transposition and panel geometry in §2 and §8 into a reusable, tested component; treat those sketches as the prototype it grows from.


---

## 7. Estimating capacity with DP

The capacity parameter must not be free to bounce around at the data's sampling rate, or it will simply soak up whatever noise the rest of the model cannot explain. The right regulariser depends on whether the capacity can physically *fall* as well as rise.

### Metered effective capacity (can go up *or* down)

The effective capacity of a metered generator changes in both directions: it drops when turbines fail or inverters trip, and recovers when they are repaired. We therefore represent it as a smoothly-varying latent series and penalise high-frequency movement — for example a total-variation (or random-walk) penalty on the step-to-step change. This lets capacity track genuine, persistent changes (a turbine offline for a fortnight) while refusing to chase half-hourly noise. Sudden, sustained drops are exactly the generator-fault signal we want to surface.

### Unmetered installed capacity (grows monotonically)

The *installed* capacity of an unmetered fleet behaves differently: it essentially only ever grows, as more households and businesses fit panels. Here a monotonic representation is the right prior. We model capacity as a cumulative sum of per-week increments, each constrained to be **non-negative**, with an L1 (sparsity) penalty pushing most weekly increments to exactly zero — because installs happen in occasional bursts, not every week. The running total is then non-decreasing by construction.

---

## 8. `FleetSolarNode`

A single tilt/azimuth pair cannot represent a "mishmash" of hundreds or thousands of rooftops. A fleet facing east, south and west produces a broad, flat "mound" of power, whereas a single south-facing parameter produces a sharp "hill". Used on a primary substation, the single-array model will fail to fit the wide shoulders of morning and evening generation.

We fix this with a **physics-informed basis expansion** rather than simulating every individual system.

### 1. The basis-function insight

Think of the fleet not as a physical simulation of thousands of houses but as a signal-reconstruction problem. The aggregate curve of many fixed-tilt systems is a linear combination of a few fundamental shapes. East, south and west span the space of fixed-tilt orientations: by mixing them (e.g. 30% east, 70% south) you can approximate almost any aggregate fixed-tilt curve, including SE/SW.

### 2. When to add a tracking basis

Single-axis trackers produce a flat-topped "top hat" shape that no mixture of fixed-tilt "bell curves" can reproduce. So *where trackers are present*, we add tracking as a fourth basis function. This matters mainly for **commercial / utility ground-mount** sitting behind a primary; domestic rooftop fleets are almost entirely fixed-tilt, so for a purely-domestic fleet the tracking weight should learn to ≈zero (or be dropped to save parameters).

### 3. Soft clipping for diverse inverters

A fleet contains many inverters of different sizes. An individual inverter hard-clips (a brick wall), but the *aggregate* clips softly: as irradiance rises the most undersized inverters saturate first, then the average ones, then the oversized ones — a smooth, curved shoulder rather than a sharp corner.

A `tanh` clip is tempting but wrong: `P_max · tanh(P / P_max)` already attenuates *mid-range* power (at `P = P_max` it returns only ≈0.76 `P_max`), so it biases the whole curve down, not just the shoulder. Instead use a **smooth-min** that stays roughly linear until the limit and only then rolls off:

$$\text{smin}(P, P_{\max}) = P_{\max} - \frac{1}{\beta}\,\operatorname{softplus}\!\bigl(\beta\,(P_{\max} - P)\bigr)$$

The learnable $P_{\max}$ is the effective aggregate inverter (AC) capacity; the learnable sharpness $\beta$ sets the curvature of the shoulder.

### 4. Implementation: `UniversalFleetNode`

This upgrades the node to handle orientation mix, tracking, and soft clipping in a single differentiable module.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalFleetNode(nn.Module):
    """Aggregate solar fleet behind one substation: orientation mix + soft inverter clip."""

    def __init__(self, n_weeks: int) -> None:
        super().__init__()
        # Orientation mix over [east, south, west, tracking], tracked per week because the
        # fleet composition drifts over time as new systems are installed.
        self.mix_logits = nn.Parameter(torch.zeros(n_weeks, 4))

        # Effective aggregate inverter (AC) capacity and the curvature of the clip shoulder.
        # Both pushed through softplus so they stay strictly positive.
        self.raw_inverter_capacity = nn.Parameter(torch.tensor(2.3))  # softplus(2.3) ~ 10
        self.raw_clip_sharpness = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, sun_vec: torch.Tensor, weather: torch.Tensor, week_idx: torch.Tensor
    ) -> torch.Tensor:
        # 1. The four physical basis curves (ideal shapes from geometry + weather).
        #    calc_fixed / calc_tracker would come from pvlib-pytorch (see section 6).
        #    Azimuth convention matches section 2: south = 0, east = -90, west = +90.
        p_east = self.calc_fixed(sun_vec, weather, azimuth_deg=-90.0)
        p_south = self.calc_fixed(sun_vec, weather, azimuth_deg=0.0)
        p_west = self.calc_fixed(sun_vec, weather, azimuth_deg=90.0)
        p_track = self.calc_tracker(sun_vec, weather)  # single-axis backtracking logic

        # 2. Mix the bases with this week's weights (the "asset identification").
        basis = torch.stack([p_east, p_south, p_west, p_track], dim=-1)
        weights = torch.softmax(self.mix_logits[week_idx], dim=-1)
        p_raw = (basis * weights).sum(dim=-1)

        # 3. Aggregate soft clip. Smooth-min stays ~linear until p_raw nears the limit,
        #    then rolls off — unlike tanh it does not suppress mid-range power.
        limit = F.softplus(self.raw_inverter_capacity)
        beta = F.softplus(self.raw_clip_sharpness) + 1e-3
        return limit - F.softplus(beta * (limit - p_raw)) / beta
```
