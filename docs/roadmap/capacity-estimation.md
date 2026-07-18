# Effective-capacity estimation

> **Status: 🚧 Planned (v0.7).** Epic:
> [#141](https://github.com/openclimatefix/nged-substation-forecast/issues/141); issues:
> [#157](https://github.com/openclimatefix/nged-substation-forecast/issues/157) (solar),
> [#158](https://github.com/openclimatefix/nged-substation-forecast/issues/158) (wind). This
> page is the plan for estimating the time-varying **effective capacity of metered generators**
> ([roadmap v0.7](index.md#v07-dynamic-generator-capacity)) — run as a **head-to-head between
> candidate estimators**, with the winner shipping in v1. The methods the candidates build on
> are explained in the techniques pages:
> [Convex Optimisation](../techniques/convex-optimisation.md) and
> [Differentiable Physics](../techniques/differentiable-physics.md). The v2 work this feeds —
> full disaggregation of unmetered DERs — has its own canonical page,
> [Net-demand disaggregation](disaggregation.md). The Python in this document is illustrative
> sketch code, not the implementation. See the [roadmap index](index.md) for status conventions.

## The problem

A generator's effective capacity drifts over time — turbines fail, inverters drop out, panels
soil and degrade (or are cleaned and replaced), arrays are extended. A static nameplate value
introduces large downstream errors. The v0.7 deliverable is a **time-varying effective-capacity
series per metered generator**, feeding the
[`effective_capacity`](delivery-tables.md#table-4-effective_capacity) delivery table. Because
this turns v0.1's single scalar-per-series capacity into a time-varying series, the metrics
pipeline must also swap its `time_series_id`-only NMAE-denominator join for a temporal as-of
join — see
[Normalising NMAE by `effective_capacity`](metrics-and-leaderboard.md#normalising-nmae-by-effective_capacity).

**How capacity feeds the forecast:** a **two-pass** approach. The first pass estimates effective
capacity (this page); the second normalises each generator's time series by its effective
capacity before the power-forecast model trains on it. XGBoost continues to do the actual power
forecasting in v1; the estimators here are responsible only for *estimating capacity* of
*metered* generators. None of the "clever" latent-demand or switching inversion happens here —
that is [v2 research](disaggregation.md).

The changepoints in the fitted capacity series are a deliverable in their own right: sudden,
sustained drops are exactly the generator-fault signal worth surfacing to NGED, and a capacity
series that names its change dates doubles as a real-time health and availability monitor.

## Several estimators, one winner

There is more than one credible way to estimate effective capacity, and they occupy genuinely
different corners of the tooling space. Rather than pick on paper, **v0.7 races the candidates
head-to-head on the same data and the same judging criteria, and the winner ships in v1.** The
losers do not disappear: they stay on the leaderboard as permanent baselines and honesty checks.

The contest has a second, deliberate purpose beyond picking the best estimator: **building
hands-on experience with convex optimisation (CVXPY) during v1**, so that its fit for the v2
problems — and our advice to NGED about tooling — rests on first-hand evidence rather than
paper argument. The
[Convex Optimisation](../techniques/convex-optimisation.md) page makes a strong prior case;
v0.7 is where that case meets NGED data.

The candidates:

- **[Candidate A — the convex estimator](#candidate-a-the-convex-estimator-cvxpy)**: a censored
  quantile-envelope fit with fused-lasso changepoints, solved exactly by CVXPY, with panel
  orientation found by grid search.
- **[Candidate B — the differentiable-physics estimator](#candidate-b-the-differentiable-physics-estimator)**:
  the variational PyTorch model, fitting orientation and capacities as posteriors.
- **[Cheap baselines](#cheap-baselines-to-beat)**: a rolling quantile of clear-sky-normalised
  output, and SLAC's off-the-shelf convex capacity-change detection.

Despite coming from different toolchains, the two serious candidates are the *same kind of
thing*: both are **inverse modelling** — write down a forward model mapping unknown parameters
(capacities, orientation) to predicted power, then invert it against the observed power to
recover the parameters. Neither is "more physical" than the other; a fixed pvlib per-unit curve
inside a convex problem is exactly as much a physics model as a transposition calculation inside
a PyTorch module. What genuinely separates them is **expressiveness versus guarantees**:
Candidate A restricts its forward model to what convexity can certify and gets the exact,
reproducible global optimum in return; Candidate B may write down any differentiable physics it
likes — and gives up the certificates. The head-to-head is therefore also a live measurement of
that trade-off on NGED data; the framing is developed fully in
[Two routes to the same inverse problem](../techniques/convex-optimisation.md#two-routes-to-the-same-inverse-problem).

There is a testable prediction to settle, made when this contest was designed: **the convex
estimator wins per-site, and the PyTorch route only pulls ahead when pooling many sites to
*learn* shared physics corrections** (soiling, snow, systematic irradiance-model bias) — which is
fleet-scale work beyond v0.7. The
[head-to-head protocol](#the-head-to-head-protocol) is shaped to confirm or kill that
prediction.

## What every candidate must get right

The requirements in this section belong to the *problem*, not to any particular estimator. Each
candidate meets them with its own machinery, and the judging checks each one.

### What effective capacity must exclude

ANM curtailment is a deliberate, network-driven reduction, not a loss of physical capability.
Folding curtailment into capacity would corrupt exactly the signal NGED needs. We identify
curtailed periods from NGED's curtailment/ANM feed and keep them out of the capacity estimate —
in the physics-model formulation this is a separate multiplicative **curtailment gate** on the
generator's output (see
[the v2 engine's node definitions](disaggregation.md#node-definitions) for where the same gate
reappears at scale); in the convex formulation it amounts to masking or down-weighting flagged
periods.

**The ANM feed is imperfect, in both directions.** NGED have told us plainly: there will be
periods when generators are curtailed with nothing in the ANM logs, and logged events that do
not match reality. So the feed is a *noisy label*, not ground truth — use it, but do not lean on
it:

- **Unlogged curtailment** (including **economic self-curtailment** during negative-price
  periods, which never appears in the ANM feed) would read as capacity loss to any estimator
  that fits the *middle* of the data. The structural defence is to fit an **upper envelope**
  instead — an asymmetric, quantile-flavoured loss under which curtailed samples fall below the
  fitted capacity without dragging it down. This defence matters for *every* candidate: it is
  native to the convex estimator's
  [pinball loss](../techniques/convex-optimisation.md#fitting-an-envelope-with-the-quantile-pinball-loss),
  and the differentiable-physics candidate should adopt the same asymmetric loss in place of a
  plain Gaussian likelihood for the same reason. Masking periods flagged as negative-price from
  public market data is a complementary mitigation.
- **Spuriously-logged curtailment** merely discards good samples (the period is masked though
  the generator ran freely) — a milder failure, but one more reason the estimate should not
  *depend* on the feed being right.

### The regularisation prior: piecewise-constant capacity

The capacity series must not be free to bounce around at the data's sampling rate, or it will
simply soak up whatever noise the rest of the model cannot explain.

#### Metered effective capacity (can go up *or* down)

The effective capacity of a metered generator changes in both directions: it drops when turbines
fail or inverters trip, and recovers when they are repaired. The right prior is
**piecewise-constant**: capacity holds a level, then steps to a new one — which is expressed as a
penalty on step-to-step change (a total-variation penalty, equivalently the **fused lasso**: an
$\ell_1$ penalty on successive differences). This lets capacity track genuine, persistent changes
(a turbine offline for a fortnight) while refusing to chase half-hourly noise.

The prior is shared; how exactly each candidate realises it is part of the contest. A proximal
convex solver produces **exactly zero** change on most days — so the nonzero steps *are* a
literal event log — while gradient descent produces approximately-zero changes that need a
threshold to read as events (see
[the exact-zeros property](../techniques/convex-optimisation.md#the-corners-are-a-feature-exact-zeros)).

(The *installed* capacity of an **unmetered** fleet behaves differently — it essentially only
grows — and its monotone prior is documented with the v2 work:
[Unmetered installed capacity grows monotonically](disaggregation.md#unmetered-installed-capacity-grows-monotonically).)

### Identifiability: the data goes silent at night

At night, and deep in winter gloom, the observed power says almost nothing about capacity — a
5 MW site and a 50 MW site both meter ≈0. Fitting those samples adds noise, not information.
Every candidate should weight the fit by the per-unit physics output (or drop low-irradiance /
low-wind samples outright), and let the piecewise-constant prior carry the estimate across the
uninformative gaps — that is precisely what the prior is for.

### Keeping weather bias out of capacity

The irradiance driving any physics-based estimator is itself biased: NWP and satellite products
carry regional, *seasonal* error (satellite retrievals degrade at low UK winter sun angles, for
example). Per site, that bias is indistinguishable from slow capacity drift — an unconstrained
fit will alias it into exactly the signal we deliver to NGED. The mitigation exploits structure:
every site in a region sees the **same** weather bias, while genuine capacity changes are
site-specific. Fitting a **shared regional irradiance-bias term jointly across the metered
fleet** separates the two. This is the v0.7-sized version of what
[the weather encoder](disaggregation.md#combining-the-physics-with-the-weather-encoder) does in
v2.

How naturally each candidate accommodates this term differs sharply — it is a genuine structural
advantage of the differentiable-physics route and a documented
[caveat of the convex route](#honest-caveats-of-the-convex-route).

### Causal vs smoothed capacity — a lookahead trap in the two-pass scheme

A capacity series regularised over the whole record is a *smoother*: it uses future
observations, so the estimate for a given day changes once a later fault is seen. That is
correct for the historical
[`effective_capacity`](delivery-tables.md#table-4-effective_capacity) table and the NMAE
denominator — but the capacity used to normalise at forecast init time, in live running *and in
backtests*, must be the **causal (filtered) estimate available at that init time**, or backtest
skill is quietly inflated by lookahead. This is the same no-lookahead invariant the feature
pipeline enforces for power lags. It binds every candidate equally.

## Candidate A — the convex estimator (CVXPY)

At first sight capacity estimation looks non-convex: predicted power is
capacity × per-unit output, and the per-unit output depends non-linearly on panel tilt and
azimuth. But the non-convexity lives *entirely* in two angles. **Conditional on orientation, the
whole problem is convex** — and two angles are cheap to enumerate. That decomposition is the
whole design:

1. **Outer loop:** grid-search tilt × azimuth (~30 tilts × ~70 azimuths), computing the per-unit
   output curve $u_t$ with plain [pvlib](https://pvlib-python.readthedocs.io/) at each grid
   point. (No PyTorch, and no `pvlib-pytorch`, anywhere in this candidate.)
2. **Inner solve:** given $u_t$, fit the capacity traces by solving one convex problem —
   certified global optimum, deterministic, warm-started from the previous grid point.

An exhaustive outer search wrapped around a certified inner solve is fully deterministic and, up
to the grid resolution, leaves no valley unvisited — see
[the tooling page](../techniques/convex-optimisation.md#where-pytorch-is-the-right-tool) for
where this trick sits in the convex/non-convex landscape.

### The model

Two piecewise-constant traces per site, in daily blocks: **DC effective capacity**
$c^{\text{dc}}_{d}$ (the array side) and **AC effective capacity** $c^{\text{ac}}_{d}$ (the
inverter side). Predicted power is the DC output hard-clipped at the AC limit:

$$
\hat{y}_t \;=\; \min\!\bigl(c^{\text{dc}}_{d(t)} \cdot u_t,\;\; c^{\text{ac}}_{d(t)}\bigr)
$$

where $u_t$ is the fixed per-unit physics output and $d(t)$ maps each half-hour to its day.
Daily blocks keep the problem small — roughly 730 unknowns per trace per year, which solves in
well under a second.

### The clip is censoring, not an obstacle

The hard $\min$ looks like it breaks convexity, but clipped samples are *observable* in the
data — flat plateaus at a common ceiling on bright days. Pre-classify each timestep as
clipped/unclipped and the problem splits into convex pieces, exactly the
[censoring pattern](../techniques/convex-optimisation.md#hard-limits-as-censoring) from the
tooling page:

- **Unclipped samples**: observed power ≈ $c^{\text{dc}}_{d(t)} u_t$ — linear in the unknowns.
- **Clipped samples**: the plateau level is a direct noisy measurement of $c^{\text{ac}}$;
  *and* the DC-side output must have exceeded the AC limit for clipping to occur — a linear
  inequality, added as a hinge penalty. Even censored points carry information.

This is Tobit regression in energy clothing — and the plateau classifier is the formulation's
weakest joint; see [the caveats](#honest-caveats-of-the-convex-route).

### Loss and penalties

- **Fit: a quantile envelope, not least squares.** Curtailment appears as excursions *below* the
  physics prediction; least squares would drag capacity down. A
  [pinball loss](../techniques/convex-optimisation.md#fitting-an-envelope-with-the-quantile-pinball-loss)
  at a high quantile $\tau$ fits the envelope: capacity = "what the site delivers when nothing
  holds it back", with curtailed samples falling below the envelope without biasing it. $\tau$
  is a real tuning choice (too low: curtailment leaks in; too high: the fit chases meter
  glitches) and should be validated on synthetic injections.
- **Penalty: fused lasso on both traces.**
  $\lambda \bigl( \lVert \Delta c^{\text{dc}} \rVert_1 + \lVert \Delta c^{\text{ac}} \rVert_1 \bigr)$
  gives sparse, *exactly-zero* day-to-day changes. The payoff is the deliverable itself: the
  fitted changepoints — each with a date and a magnitude — **are** the site's event log
  (inverter trips, feeder outages, array expansions, deratings), directly cross-checkable
  against NGED's maintenance records.
- **Priors, where records exist.** A registered capacity, a previous year's fit, or a connection
  record enters as one more convex penalty
  ([priors as penalties](../techniques/convex-optimisation.md#priors-as-convex-penalties-and-the-uncertainty-you-dont-get)) —
  including **asymmetric** priors (cheap to sit below the registered value, expensive to exceed
  it, since registers overstate more than they understate) and **timing** priors (a known March
  expansion makes jumps cheap at that date, expensive elsewhere).

### Wind: same structure, simpler

A turbine has no separate inverter limit, but its power curve's **rated plateau** plays the
identical censoring role: saturated high-wind samples measure capacity directly, and the same
envelope + fused-lasso machinery applies with $u_t$ from a reference power curve. Curtailment is
a larger fraction of life for wind than for solar, which makes the quantile-envelope loss even
more clearly right there.

### Sketch

Illustrative, untested sketch, not the implementation — one site, one candidate orientation
(the inner solve of the grid search):

```python
import cvxpy as cp

# Inputs for one site at one candidate orientation:
#   u:       (T,) per-unit output from pvlib at that orientation
#   y:       (T,) observed power (MW)
#   clipped: (T,) bool — the plateau classifier's verdict (see caveats)
#   day:     (T,) int — maps each half-hour to its daily capacity block
c_dc = cp.Variable(n_days, nonneg=True)  # DC effective capacity (MW)
c_ac = cp.Variable(n_days, nonneg=True)  # AC (inverter) effective capacity (MW)

# Unclipped, informative samples: pinball loss at a high quantile fits the envelope.
informative = ~clipped & (u > 0.15)
r = y[informative] - cp.multiply(u[informative], c_dc[day[informative]])
fit_dc = cp.sum(TAU * cp.pos(r) + (1 - TAU) * cp.pos(-r))

# Clipped samples: the plateau level directly measures AC capacity...
fit_ac = cp.sum_squares(y[clipped] - c_ac[day[clipped]])
# ...and the DC-side output must have *exceeded* the AC limit for clipping to occur.
censor = cp.sum(cp.pos(c_ac[day[clipped]] - cp.multiply(u[clipped], c_dc[day[clipped]])))

# Fused lasso: sparse, exactly-zero day-to-day changes -> piecewise-constant traces.
fused = cp.norm1(cp.diff(c_dc)) + cp.norm1(cp.diff(c_ac))

problem = cp.Problem(cp.Minimize(fit_dc + fit_ac + mu * censor + lam * fused))
problem.solve(solver=cp.CLARABEL)  # certified global optimum, warm-startable
```

### Honest caveats of the convex route

- **Effective capacity, not nameplate truth.** Any systematic bias in the pvlib per-unit model —
  soiling, unmodelled horizon shading — is absorbed into the capacity estimate. For forecasting
  that is a feature (it *is* effective capacity), but the fitted MW must not be read as nameplate
  truth against capacity registers.
- **The plateau classifier sits outside the guarantee.** The censoring split is a heuristic
  pre-step, and at half-hourly resolution with real meter noise it is fiddly: misclassified
  samples poison both the AC fit and the censoring constraint. It needs its own validation
  (synthetic clipping injected into unclipped sites is the obvious test).
- **The shared regional irradiance-bias term breaks the one-shot convexity.** A multiplicative
  regional bias × per-site capacity is bilinear — the same structure that makes
  [the mixture model alternation-only](switching-events.md#approach-4-the-magnitude-only-mixture-model-the-workhorse).
  The convex route must then choose between: **alternation** (bias given capacities, capacities
  given bias — each step convex, but the certified-global-optimum headline no longer applies to
  the joint answer); a **plug-in pre-estimate** (e.g. the fleet-median clear-sky-normalised
  residual per region, fixed before the capacity solve — every solve stays certified but the
  two-step answer has no joint certificate); or an **additive-only bias** (jointly convex, but a
  physically weaker model of what irradiance error does). None is free. The
  differentiable-physics candidate fits the shared bias jointly without contortion — a genuine
  structural advantage to weigh in the head-to-head.
- **Uncertainty is a bolt-on.** The solve returns a point estimate; error bars come from
  bootstrap or sensitivity re-solves, with
  [known weaknesses](../techniques/convex-optimisation.md#priors-as-convex-penalties-and-the-uncertainty-you-dont-get).
  See [the uncertainty criterion below](#uncertainty-a-first-class-judging-criterion).
- **Grid-search compute grows with the fleet.** ~2,000 warm-started sub-second solves per site
  is trivial for the trial area's metered generators; at V2 scale (~2,500 series, though far
  fewer *metered generators*) it wants revisiting — e.g. a coarse-to-fine grid.

## Candidate B — the differentiable-physics estimator

The variational single-site model
([`DifferentiableSolarPlant`](../techniques/differentiable-physics.md#the-core-building-block-differentiablesolarplant)
and its wind analogue): site coordinates locked, live weather passed through the physics, and
tilt, azimuth, DC and AC capacity fitted as mean-field posteriors by gradient descent on an
ELBO. Its distinct strengths in this contest:

- **Native posteriors.** Every parameter carries a fitted spread, so capacity uncertainty comes
  out of the estimator rather than being bolted on — directly relevant to
  [the uncertainty criterion](#uncertainty-a-first-class-judging-criterion).
- **Joint fleet-wide corrections without contortion.** The shared regional irradiance-bias term —
  and, later, learned corrections for soiling, snow, or systematic pvlib bias — drop into the
  same training loop. This is the "fleet-wide-with-learning" regime where the
  [crossover prediction](#several-estimators-one-winner) expects PyTorch to win.
- **Continuity with v2.** The fitted modules and the experience of training them carry straight
  into [the v2 engine](disaggregation.md), where PyTorch is unavoidable.

And its costs, mirror-images of Candidate A's strengths: gradient descent brings learning rates,
schedules, seeds and stopping criteria for a per-site problem the convex route solves exactly;
the fused-lasso-style penalty yields approximately-zero changes, so reading changepoints off the
capacity trace needs a threshold; and there is no global-optimum certificate. Candidate B should
also adopt the envelope-flavoured asymmetric reconstruction loss
([above](#what-effective-capacity-must-exclude)) — a plain Gaussian likelihood fits the middle
of the data and inherits the curtailment bias.

## Cheap baselines to beat

Both headline candidates must beat deliberately simple baselines on the same leaderboard:

- a **rolling quantile of clear-sky-normalised output** (observed power ÷ physics-predicted
  clear-sky power);
- the convex capacity-change detection in SLAC's
  [solar-data-tools](https://github.com/slacgismo/solar-data-tools).

If neither candidate can beat the cheap estimators, that is a finding worth surfacing early —
and the baselines slot into the same leaderboard discipline used for the forecasting models.

## Uncertainty — a first-class judging criterion

Our working hypothesis (a hunch, stated so the contest can test it): **capacity-estimate error
will contribute a share of total energy-forecast error comparable to every other source
combined.** If that is even half right, a capacity estimate without honest uncertainty quietly
launders one of the largest error sources in the system into numbers that look exact.

The hypothesis is cheap to test, and the contest should: perturb the capacity series by its
plausible error band, run the perturbed series through the two-pass normalisation, and measure
the change in downstream forecast NMAE. That experiment directly quantifies how much forecast
error flows through the capacity channel.

What we ideally want from the winning estimator, in order:

1. **Uncertainty captured in the capacity estimate itself** — a spread, not a point.
2. **Propagated through the two-pass scheme** — run the normalisation per capacity-sample
   (posterior draws for Candidate B; bootstrap ensemble for Candidate A), so the forecast
   inherits the capacity uncertainty rather than ignoring it.
3. **Decomposable** — the final forecast's spread should be attributable: this much from
   capacity, this much from weather, this much from the model. Per-sample propagation is what
   makes that attribution possible at all.

**The judging stance, stated plainly:** calibrated, decomposable uncertainty is a first-class
criterion, *not* a hard shipping gate — a point-estimate-only winner may still ship in v1 with
uncertainty deferred. But between candidates of comparable point accuracy, the one with honest,
decomposable uncertainty wins — **even at a slight cost in mean accuracy**. This criterion
structurally favours Candidate B's native posteriors over Candidate A's bootstrap; the docs say
so openly rather than pretending the criteria are neutral, and Candidate A can close the gap by
demonstrating that its bootstrap intervals are actually calibrated (see the evaluation hooks
below).

## The head-to-head protocol

All candidates run on the same sites, the same weather inputs, and the same folds. There is no
direct ground truth for capacity, so judging combines proxies, each targeting a claim a
candidate makes:

1. **Downstream forecast skill** — the deciding metric. Each candidate's capacity series feeds
   the two-pass normalisation; the resulting forecasts compete on the existing leaderboard
   (NMAE and pinball, per
   [metrics-and-leaderboard](metrics-and-leaderboard.md)).
2. **Synthetic fault injection** — scale a known period of a healthy site's output down by a
   known factor and check: does the estimator find the changepoint, at the right date, with the
   right magnitude? Does its uncertainty interval cover the truth? (The same injection
   discipline the [switching detector](switching-events.md) uses.)
3. **Event-log quality** — precision/recall of fitted changepoints against NGED maintenance and
   outage records, where records exist.
4. **Robustness to curtailment label noise** — inject unlogged synthetic curtailment and measure
   how far each estimate is dragged.
5. **Uncertainty calibration** — coverage of the stated intervals under (2) and on held-out
   periods, per [the criterion above](#uncertainty-a-first-class-judging-criterion).
6. **Runtime and operability** — wall-clock per site, determinism across re-runs, and the count
   of tunable knobs that had to be tuned.

The winner ships in v1 and populates the
[`effective_capacity`](delivery-tables.md#table-4-effective_capacity) table; the rest remain on
the leaderboard as standing baselines. Whichever way it goes, the result settles the
[crossover prediction](#several-estimators-one-winner) with evidence — which is itself a
deliverable of the contest, feeding the v2 tooling choice and our advice to NGED.

## Irradiance inputs

The beam/diffuse decomposition the physics needs (for either candidate — pvlib's transposition
wants the same inputs as
[the differentiable model](../techniques/differentiable-physics.md#the-core-building-block-differentiablesolarplant))
is covered by the v0.7 weather ingests. **CM SAF SARAH-3** provides global (SIS), direct (SID)
and direct-normal (DNI) irradiance at 0.05° / 30-minute resolution from 1983 (diffuse =
SIS − SID) — the primary input, matching the half-hourly metering. **ERA5** provides global plus
**direct** (`fdir`) short-wave (diffuse by subtraction), and its near-real-time ERA5T stream
(~5 days behind) suits the near-real-time capacity estimate — unlike CERRA, whose ~3.5-month
latency [rules it out here](data-sources.md#weather-data). The live **ECMWF ENS** feed carries only GHI — fine for
v0.7, but v2 physics *forecasting* of PV needs a differentiable GHI → DNI/DHI decomposition
model (or `fdir` added to the upstream dataset). See
[data sources](data-sources.md#weather-data).

## What comes after v0.7

Once the metered assets are accurately tracked, the harder v2 goal — disaggregating the
*unmetered* DERs behind every substation — builds directly on this work. That plan lives on its
own canonical page: [Net-demand disaggregation](disaggregation.md).
