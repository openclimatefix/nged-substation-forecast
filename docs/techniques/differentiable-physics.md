# Differentiable Physics

> **Status: durable method explainer. 🚧 Planned / 🔬 Research — none of this is implemented
> yet.** This page explains the *method*: what differentiable physics (DP) is, the inversion idea
> at its core, and the two reusable building blocks (a single-site solar model and an
> aggregate-fleet model). The plans for *applying* it live in the roadmap: DP is
> [one of the candidate estimators](../roadmap/capacity-estimation.md#candidate-b-the-differentiable-physics-estimator)
> for metered-generator effective capacity
> ([v0.7](../roadmap/index.md#v07-dynamic-generator-capacity)), and the engine behind the
> [graph-structured disaggregation of net substation power](../roadmap/disaggregation.md)
> ([v2 research](../roadmap/index.md#v20-scale-up-future-research)). The tooling counterpart to
> this page — which estimation problems belong in convex optimisation (CVXPY) rather than
> PyTorch, and the `cvxpylayers` bridge between the two — is
> [Convex Optimisation](convex-optimisation.md). The Python in this document is illustrative
> sketch code, not the implementation. See the [roadmap index](../roadmap/index.md) for status
> conventions.

## Why differentiable physics?

Pure black-box models (like standard neural networks or XGBoost) require vast amounts of data to
learn basic physical invariants and are prone to overfitting or hallucinating non-physical
behaviour. Differentiable physics (DP) instead injects a mechanistic physical model directly into
the computational graph, for three core reasons:

- **Data and sample efficiency:** The model does not have to spend capacity re-learning solar
  geometry (where the sun is) or the shape of a turbine power curve (roughly cubic in wind speed
  between cut-in and rated, then flat to rated power, then zero beyond cut-out). The physics engine
  supplies these constraints for free, letting the ML components focus entirely on atmospheric and
  behavioural nuances.
- **True invertibility (inputs as parameters):** Because every operation in the physics engine
  preserves gradients, we can backpropagate errors all the way to the *inputs*. This lets us treat
  unobserved variables (like local irradiance) or system configurations as parameters that can be
  solved via gradient descent.
- **Interpretability:** Instead of inspecting uninterpretable latent layers, the system updates
  explicit physical parameters like tilt, azimuth, or capacity. This lets engineers immediately
  audit the model's assumptions.

## The core idea: inversion through a differentiable forward model

The fundamental insight is to treat a meter reading as the output of a **forward model** that
takes the unobserved quantities as inputs. For a substation:

$$P_{\text{obs}}(t) = P_{\text{demand}}(t) - P_{\text{pv}}(t) - P_{\text{wind}}(t) + P_{\text{loss}}(t)$$

where $P_{\text{obs}}$ is the observed meter reading, $P_{\text{demand}}$ the latent (unobserved)
demand, and $P_{\text{pv}}$, $P_{\text{wind}}$, $P_{\text{loss}}$ are the PV generation, wind
generation, and network losses respectively.

Each right-hand-side term is modelled explicitly as a physical function of weather and time, with
its own latent parameters (capacities, orientations, power curves). The forward model is
**differentiable end-to-end**: every component is implemented in a differentiable framework (JAX
or PyTorch), so that gradients of the reconstruction error with respect to all latent parameters
and states can be computed and used for learning. Training minimises the discrepancy between the
forward model's output and the observed meter readings, jointly optimising every latent term
simultaneously. This is the "inversion" step — running the forward model backwards, constrained
by physics.

The pay-off is not just the parameter estimates but the latent terms themselves: subtract the
reconstructed generation and what remains is the latent demand, free of the confounding effect of
distributed energy resources (DERs). How this project deploys the idea — which terms are metered
vs unmetered, which DER types are tractable, and the full graph-structured engine over the
substation network — is planned in
[Net-demand disaggregation](../roadmap/disaggregation.md).

A note on the name: "differentiable physics" is shorthand for *a physics simulator written in an
autodiff framework so that it composes with gradient-based learning* — the differentiability is
the enabling property, not the point. The underlying activity is **inverse modelling**, and it is
shared with the [convex route](convex-optimisation.md): both toolchains write a forward model and
invert it against observations, and both host physics. What actually separates them is the
expressiveness of the forward model versus the guarantees on the inversion — unpacked in
[Two routes to the same inverse problem](convex-optimisation.md#two-routes-to-the-same-inverse-problem).

---

## The core building block: `DifferentiableSolarPlant`

We model each physical parameter as a learnable Normal distribution $\mathcal{N}(\mu, \sigma^2)$ — a mean-field variational posterior — and train with the reparameterisation trick (`rsample()`) so gradients flow through the sampling step.

Crucially, the training objective is an **ELBO**, not a bare reconstruction loss: a power-reconstruction term *plus* a KL term that pulls each posterior toward a fixed physical prior. The KL term is not optional. Minimising power error alone always rewards shrinking $\sigma \to 0$, so the parameter "uncertainty" we are trying to capture would simply collapse. The prior does double duty: it keeps the posterior spreads honest, and it injects weak domain knowledge (e.g. "panels point roughly south at a typical UK roof pitch") that regularises sites with little data.

Two practical details the ELBO must get right — both classic failure modes of hand-rolled variational inference:

- **The reconstruction term must be a proper likelihood, not a bare MSE.** Bare MSE silently assumes an observation-noise scale of 1 in whatever units the power happens to be in, which makes the balance between reconstruction and KL arbitrary — the posterior then either collapses onto the prior or ignores it, depending on nothing but the units. Use a Gaussian likelihood with a **learnable observation-noise scale** $\sigma_{\text{obs}}$ (see `negative_elbo` in the sketch).
- **Scale the KL for minibatches.** The KL term regularises the *whole-dataset* objective once, so on a minibatch it must be weighted by `batch_size / dataset_size` — otherwise the effective prior strength depends on the batching.

Three physics details the sketch gets right:

- **Irradiance transposition.** Plane-of-array (POA) irradiance is *not* GHI scaled by the angle of incidence — GHI already bakes in a cosine-of-zenith projection. We decompose the resource into beam, sky-diffuse and ground-reflected components (an isotropic sky model) and transpose each correctly. The beam term uses DNI (not GHI) projected by the angle of incidence.
- **DC and AC capacity.** The learnable `dc_capacity` is the DC nameplate in power units; POA is normalised by the reference irradiance (1000 W/m²) so that capacity falls out in MW at standard test conditions. `ac_capacity` is a separate learnable parameter that clips the inverter output via `torch.minimum` — a single plant clips hard, unlike [the fleet soft clip](#scaling-to-aggregate-fleets-universalsolarfleetnode).
- **Panel temperature derate.** Cell temperature sits above ambient in proportion to absorbed POA irradiance; efficiency then falls roughly linearly with temperature above 25 °C. The sketch adds a steady-state derate using two module-level constants; in the full variational model these become learnable posteriors — and the [subsection below](#panel-temperature) extends this to the broken-cloud effect.

Angle convention: azimuth is measured from due south, with east negative and west positive (east = −90°, south = 0°, west = +90°). Beware that this differs from [pvlib](https://pvlib-python.readthedocs.io/)'s convention (north = 0°, clockwise, in degrees); `pvlib-pytorch` should adopt pvlib's own convention and convert at the boundary — mismatched angle conventions are the classic silent bug in PV modelling.

One detail the sketch deliberately ignores — **interval averaging**. NWP and satellite irradiance are half-hourly *period means* (period-ending in our NWP schema), but the sketch evaluates the solar geometry at an instant. Over 30 minutes the sun's hour angle moves ~7.5°, and near sunrise/sunset the transposition is strongly non-linear, so evaluating at a single timestamp biases the fit. Worse, a timestamp-convention error (period-start vs period-end vs mid-point) masquerades as an azimuth shift that the optimiser will happily absorb into the fitted parameters. The fix is cheap: evaluate the geometry at ~5-minute sub-steps across each half-hour and average the modelled power to the metered interval — and unit-test the timestamp convention end-to-end before trusting any fitted azimuth.

Here's a quick Python code sketch of the rough idea for applying differentiable physics to solar:

```python
import torch
import torch.nn as nn
import torch.distributions as dist

STC_IRRADIANCE = 1000.0    # reference plane-of-array irradiance at standard test conditions (W/m^2)
GROUND_ALBEDO = 0.2        # typical broadband ground reflectance
STC_CELL_TEMP = 25.0       # cell temperature at standard test conditions (°C)
TEMP_COEFF_POWER = -0.004  # relative power loss per °C above STC for crystalline silicon (1/°C)
NOCT_TEMP_RISE = 0.03      # steady-state cell-temp rise per W/m² of POA (°C·m²/W)


class DifferentiableSolarPlant(nn.Module):
    """Differentiable physical model of a single metered solar site.

    Each physical parameter is a mean-field variational posterior N(mu, sigma^2).
    Training minimises `negative_elbo`: a Gaussian reconstruction likelihood (learnable
    observation noise) plus a KL term against a fixed prior (see `kl_divergence`). The KL
    term is what stops the posterior spreads from collapsing to zero under a pure
    reconstruction loss.

    Azimuth convention: measured from due south, east negative, west positive.
    """

    def __init__(self, prior_tilt: float, prior_azimuth: float, prior_dc_capacity: float, prior_ac_capacity: float) -> None:
        super().__init__()
        # Variational posteriors, each parameterised in an unconstrained space so no
        # gradient-breaking clamp is needed and the Normal posterior/prior stay
        # self-consistent: tilt in logit-space (squashed to (0, pi/2) in `forward`),
        # azimuth in radians, capacities in log-space (strictly positive).
        self.raw_tilt_mu = nn.Parameter(torch.special.logit(torch.tensor(prior_tilt) / (torch.pi / 2)))
        self.raw_tilt_log_std = nn.Parameter(torch.tensor(-2.0))

        self.azimuth_mu = nn.Parameter(torch.tensor(prior_azimuth))
        self.azimuth_log_std = nn.Parameter(torch.tensor(-2.0))

        self.log_dc_capacity_mu = nn.Parameter(torch.log(torch.tensor(prior_dc_capacity)))
        self.log_dc_capacity_log_std = nn.Parameter(torch.tensor(-2.0))

        self.log_ac_capacity_mu = nn.Parameter(torch.log(torch.tensor(prior_ac_capacity)))
        self.log_ac_capacity_log_std = nn.Parameter(torch.tensor(-2.0))

        # Observation-noise scale for the Gaussian reconstruction likelihood in
        # `negative_elbo`: learnable, in log-space to stay strictly positive.
        self.log_obs_noise = nn.Parameter(torch.tensor(0.0))

        # Fixed priors. Weakly-informative: "panels point roughly south at a UK roof pitch,
        # with a capacity near nameplate". These also keep the posterior spreads from collapsing.
        self.priors = {
            "raw_tilt": dist.Normal(torch.special.logit(torch.tensor(prior_tilt) / (torch.pi / 2)), torch.tensor(0.3)),
            "azimuth": dist.Normal(torch.tensor(prior_azimuth), torch.tensor(0.5)),
            "log_dc_capacity": dist.Normal(torch.log(torch.tensor(prior_dc_capacity)), torch.tensor(0.5)),
            "log_ac_capacity": dist.Normal(torch.log(torch.tensor(prior_ac_capacity)), torch.tensor(0.3)),
        }

    def posteriors(self) -> dict[str, dist.Normal]:
        """The current variational posterior for each physical parameter."""
        return {
            "raw_tilt": dist.Normal(self.raw_tilt_mu, self.raw_tilt_log_std.exp()),
            "azimuth": dist.Normal(self.azimuth_mu, self.azimuth_log_std.exp()),
            "log_dc_capacity": dist.Normal(self.log_dc_capacity_mu, self.log_dc_capacity_log_std.exp()),
            "log_ac_capacity": dist.Normal(self.log_ac_capacity_mu, self.log_ac_capacity_log_std.exp()),
        }

    def kl_divergence(self) -> torch.Tensor:
        """Sum of KL(posterior || prior) over all parameters — the regulariser in the ELBO."""
        q, p = self.posteriors(), self.priors
        return sum(dist.kl_divergence(q[key], p[key]) for key in q)

    def negative_elbo(
        self, predicted_power: torch.Tensor, observed_power: torch.Tensor, dataset_size: int
    ) -> torch.Tensor:
        """Minibatch training loss: Gaussian reconstruction likelihood + scaled KL.

        The KL term regularises the whole-dataset objective once, so on a minibatch it is
        weighted by batch_size / dataset_size — otherwise the effective prior strength
        would depend on how the data happens to be batched.
        """
        obs_noise = self.log_obs_noise.exp()
        log_lik = dist.Normal(predicted_power, obs_noise).log_prob(observed_power).sum()
        kl_weight = observed_power.numel() / dataset_size
        return kl_weight * self.kl_divergence() - log_lik

    def forward(
        self,
        dni: torch.Tensor,             # direct normal irradiance (W/m^2)
        dhi: torch.Tensor,             # diffuse horizontal irradiance (W/m^2)
        ghi: torch.Tensor,             # global horizontal irradiance (W/m^2)
        sun_zenith_rad: torch.Tensor,
        sun_azimuth_rad: torch.Tensor,
        air_temp: torch.Tensor,        # ambient air temperature (°C)
    ) -> torch.Tensor:
        """Predict DC power with steady-state temperature derate. All inputs and operations preserve gradients."""
        q = self.posteriors()

        # Reparameterised samples (one per forward pass). tilt is sampled in logit-space
        # and squashed to (0, pi/2) — a sigmoid, not a clamp, so gradients flow everywhere
        # and the sample stays consistent with the Normal posterior used in the KL.
        tilt = torch.sigmoid(q["raw_tilt"].rsample()) * (torch.pi / 2)
        azimuth = q["azimuth"].rsample()
        dc_capacity = q["log_dc_capacity"].rsample().exp()  # strictly positive by construction

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

        # DC power: capacity is the nameplate at the reference irradiance, derated for cell temperature.
        cell_temp = air_temp + NOCT_TEMP_RISE * poa
        temp_derate = 1.0 + TEMP_COEFF_POWER * (cell_temp - STC_CELL_TEMP)
        dc_power = dc_capacity * poa / STC_IRRADIANCE * temp_derate
        ac_capacity = q["log_ac_capacity"].rsample().exp()
        return torch.minimum(dc_power, ac_capacity)

```

### Panel temperature

**Steady-state derate.** Cell temperature is not directly measured, and it is not a simple function of air temperature. On a hot, still, clear-sky summer day a panel can reach 60–70 °C — hot enough for efficiency to fall noticeably below its standard-test-condition (STC) value. What drives the cell above ambient is absorbed irradiance, moderated by convective cooling from wind. The standard Faiman relation captures this:

$$T_{\text{cell}} \approx T_{\text{air}} + \frac{\text{POA}}{U_0 + U_1\,v_{\text{wind}}}$$

Efficiency then falls roughly linearly with temperature above the 25 °C STC reference. The correction factor applied to DC output is:

$$\eta_T = 1 + \gamma\,(T_{\text{cell}} - 25\,^\circ\text{C})$$

with $\gamma \approx -0.004\,/\,^\circ\text{C}$ ($-0.4\,\%/^\circ\text{C}$) for crystalline silicon. The constant `NOCT_TEMP_RISE` in the sketch is $1/U_0$ with wind neglected — a useful simplification that removes the need for a wind-speed input in the code example. In the full variational model, $U_0$, $U_1$, and $\gamma$ become learnable posteriors with tight physical priors, exactly like `tilt`, `azimuth`, and `dc_capacity`; the two new inputs (air temperature and POA) are already available: NWP temperature, and POA computed midway through the forward pass.

**Thermal-mass upgrade and the broken-cloud effect.** The steady-state model assumes the panel equilibrates to the current irradiance instantaneously. Real panels have thermal mass and lag by several minutes — and this is exactly what produces the *broken-cloud effect*: a panel emerging cool from beneath cloud cover is briefly more efficient than one that has been baking under a clear sky, so the power peak immediately after cloud clearance can transiently *exceed* the steady-state clear-sky peak. This is also why peak daily yield on a partly-cloudy summer day can occasionally beat that on a fully clear day.

Capture this by promoting cell temperature from an instantaneous quantity to a dynamic latent state — first-order relaxation toward the equilibrium temperature with a learnable thermal time constant $\tau$:

$$\frac{dT_{\text{cell}}}{dt} = \frac{T_{\text{eq}} - T_{\text{cell}}}{\tau}, \qquad T_{\text{eq}} = T_{\text{air}} + \frac{\text{POA}}{U_0 + U_1\,v_{\text{wind}}}$$

This is a state-space recurrence — the same pattern as a battery charge/discharge component.

**Resolution caveat.** The thermal time constant $\tau$ is on the order of minutes, and the cloud transients that drive the broken-cloud effect occur on the same sub-half-hourly timescale. Half-hourly-mean POA discards exactly the sub-grid variability the dynamic model needs. This upgrade therefore only earns its keep with higher-frequency irradiance inputs — satellite-derived irradiance at ~5-minute intervals, or sub-hourly metering — and the steady-state derate remains the right choice until that data is available.

---

## Scaling to aggregate fleets: `UniversalSolarFleetNode`

A single tilt/azimuth pair cannot represent a "mishmash" of hundreds or thousands of rooftops. A fleet facing east, south and west produces a broad, flat "mound" of power, whereas a single south-facing parameter produces a sharp "hill". Used on a primary substation, the single-array model will fail to fit the wide shoulders of morning and evening generation.

We fix this with a **physics-informed basis expansion** rather than simulating every individual system. (Aggregate, unmetered fleets behind a primary are a v2 concern — this node becomes one of the node types in
[the graph-structured engine](../roadmap/disaggregation.md#the-graph-structured-engine).)

### 1. The basis-function insight

Think of the fleet not as a physical simulation of thousands of houses but as a signal-reconstruction problem. The aggregate curve of many fixed-tilt systems is a linear combination of a few fundamental shapes. East, south and west span the space of fixed-tilt orientations: by mixing them (e.g. 30% east, 70% south) you can approximate almost any aggregate fixed-tilt curve, including SE/SW.

### 2. When to add a tracking basis

Single-axis trackers produce a flat-topped "top hat" shape that no mixture of fixed-tilt "bell curves" can reproduce. So *where trackers are present*, we add tracking as a fourth basis function. This matters mainly for **commercial / utility ground-mount** sitting behind a primary; domestic rooftop fleets are almost entirely fixed-tilt, so for a purely-domestic fleet the tracking weight should learn to ≈zero (or be dropped to save parameters).

### 3. Soft clipping for diverse inverters

A fleet contains many inverters of different sizes. An individual inverter hard-clips (a brick wall), but the *aggregate* clips softly: as irradiance rises the most undersized inverters saturate first, then the average ones, then the oversized ones — a smooth, curved shoulder rather than a sharp corner.

A `tanh` clip is tempting but wrong: `P_max · tanh(P / P_max)` already attenuates *mid-range* power (at `P = P_max` it returns only ≈0.76 `P_max`), so it biases the whole curve down, not just the shoulder. Instead use a **smooth-min** that stays roughly linear until the limit and only then rolls off:

$$\text{smin}(P, P_{\max}) = P_{\max} - \frac{1}{\beta}\,\operatorname{softplus}\!\bigl(\beta\,(P_{\max} - P)\bigr)$$

The learnable $P_{\max}$ is the effective aggregate inverter (AC) capacity; the learnable sharpness $\beta$ sets the curvature of the shoulder.

### 4. Implementation: `UniversalSolarFleetNode`

This upgrades the node to handle installed-capacity growth, orientation mix, tracking, and soft clipping in a single differentiable module. Note the division of labour: the softmax mix weights sum to 1, so the mixture alone is magnitude-free ("which way does the fleet face"); the magnitude ("how much is installed") is carried by a separate per-week capacity series, built as a cumulative sum of non-negative weekly increments so it is non-decreasing by construction — exactly the monotone representation from
[the disaggregation plan](../roadmap/disaggregation.md#unmetered-installed-capacity-grows-monotonically).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalSolarFleetNode(nn.Module):
    """Aggregate solar fleet behind one substation: capacity growth + orientation mix + soft clip."""

    def __init__(self, n_weeks: int) -> None:
        super().__init__()
        # Installed DC capacity, tracked per week as a cumulative sum of non-negative
        # increments (installs only ever add capacity — see the disaggregation
        # roadmap page). An L1 penalty on the increments (not shown) pushes most weeks
        # to exactly zero growth, because installs arrive in occasional bursts.
        self.raw_capacity_increments = nn.Parameter(torch.full((n_weeks,), -2.0))

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
        # 1. The four physical basis curves (ideal shapes from geometry + weather), each
        #    normalised to unit DC capacity.
        #    calc_fixed / calc_tracker would come from pvlib-pytorch (see below).
        #    Azimuth convention matches the single-site model: south = 0, east = -90, west = +90.
        p_east = self.calc_fixed(sun_vec, weather, azimuth_deg=-90.0)
        p_south = self.calc_fixed(sun_vec, weather, azimuth_deg=0.0)
        p_west = self.calc_fixed(sun_vec, weather, azimuth_deg=90.0)
        p_track = self.calc_tracker(sun_vec, weather)  # single-axis backtracking logic

        # 2. Mix the unit bases with this week's weights (the "asset identification"),
        #    then scale by this week's installed capacity — non-decreasing by construction.
        basis = torch.stack([p_east, p_south, p_west, p_track], dim=-1)
        weights = torch.softmax(self.mix_logits[week_idx], dim=-1)
        capacity = F.softplus(self.raw_capacity_increments).cumsum(dim=0)
        p_raw = capacity[week_idx] * (basis * weights).sum(dim=-1)

        # 3. Aggregate soft clip. Smooth-min stays ~linear until p_raw nears the limit,
        #    then rolls off — unlike tanh it does not suppress mid-range power.
        limit = F.softplus(self.raw_inverter_capacity)
        beta = F.softplus(self.raw_clip_sharpness) + 1e-3
        return limit - F.softplus(beta * (limit - p_raw)) / beta
```

### A convex twin, for initialisation and sanity checking

With the four basis curves held fixed, most of this node is
[fixed shapes × unknown coefficients](convex-optimisation.md#the-recurring-pattern-fixed-shapes-unknown-coefficients):
reparameterise (mix weights × capacity) as four non-negative per-basis capacity series, and the
fleet's below-clip output is *linear* in them, with monotone growth and sparse increments
available as hard convex constraints and an $\ell_1$ penalty. The one part that breaks convexity
is the soft clip, so the convex twin is exact only below the clip (or with the clip frozen). That
is still enough for two jobs: a principled **initialiser** for the PyTorch fit, and an
independent **sanity check** — if the convex twin and the trained node disagree materially on
installed capacity, the neural fit deserves investigation before its answer is trusted.

## `pvlib-pytorch`

`pvlib-pytorch` is a planned Open Climate Fix open-source library — a differentiable,
PyTorch-native port of [pvlib](https://pvlib-python.readthedocs.io/) — that we intend to spin out
of this project. It would generalise the hand-rolled transposition and panel geometry in the
[single-site](#the-core-building-block-differentiablesolarplant) and
[fleet](#scaling-to-aggregate-fleets-universalsolarfleetnode) sketches into a reusable, tested
component; treat those sketches as the prototype it grows from.

## Applications in this project

- **Metered-generator effective capacity ([roadmap v0.7](../roadmap/index.md#v07-dynamic-generator-capacity))** —
  the single-site model, with orientation and capacity as variational posteriors, is
  [Candidate B](../roadmap/capacity-estimation.md#candidate-b-the-differentiable-physics-estimator)
  in the capacity-estimation head-to-head.
- **Net-demand disaggregation ([v2 research](../roadmap/index.md#v20-scale-up-future-research))** —
  the fleet node and its wind/demand/heat-pump siblings become node types in the
  [graph-structured engine](../roadmap/disaggregation.md#the-graph-structured-engine); the fuller
  forecasting architecture combining DP with
  [learned weather encoders](encoders.md) is also described on
  [that page](../roadmap/disaggregation.md#combining-the-physics-with-the-weather-encoder).
- **Switching events** — the type-resolved mixture routes the outputs of per-type DP modules
  over the neighbourhood graph; see
  [Switching events & latent demand](../roadmap/switching-events.md).
