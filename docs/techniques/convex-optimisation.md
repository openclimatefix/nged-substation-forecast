# Convex Optimisation — When the Best Answer Is Guaranteed

> **Status: durable method explainer.** This page explains what convex optimisation is, why a
> convex formulation is a fundamentally better deal than gradient-descent training wherever it is
> available, and the project's practical tooling rule — **CVXPY for convex estimation
> subproblems, PyTorch for physics + learning**. The first planned applications are the
> [joint edge-flow estimator](../roadmap/switching-events.md#approach-3-the-joint-edge-flow-estimator)
> for switching-event detection and the
> [convex candidate](../roadmap/capacity-estimation.md#candidate-a-the-convex-estimator-cvxpy) in
> the v0.7 effective-capacity head-to-head; further applications are expected. The Python in this
> document is illustrative sketch code, not the implementation.

## The problem class

An optimisation problem has three parts: **unknowns** (numbers we want to find), a **score** (a
function that says how bad any candidate setting of the unknowns is), and optionally
**constraints** (rules a candidate must obey). "Solving" means finding the setting with the
lowest score. Most estimation tasks in this project fit this shape once said out loud: *find the
edge flows that best explain every substation's residual, preferring stories with few active
pipes and few changes* (the switching-events estimator); *find the latent capacities that make
the physics model reproduce the meter readings*
([differentiable physics](differentiable-physics.md)).

The two examples differ not in the question but in the **shape of the score landscape** — and
that shape decides which tool can answer.

## Convexity: the one-bowl property

Picture the score as a landscape over every possible setting of the unknowns. A problem is
**convex** when that landscape is a single bowl: pick any two points on the surface, draw the
straight line between them, and the surface never rises above that line. The payoff: **a convex
landscape has no false valleys.** Anywhere that looks like a bottom locally *is* the bottom
globally. (If the score is *strictly* convex the bottom is a single point; with penalties like
the absolute value the bottom can be a small flat face — but every point on it scores
identically, so any answer the solver returns is a best answer.)

Formally, a function $f$ is convex when, for all $x$, $y$ and $0 \le \lambda \le 1$:

$$
f(\lambda x + (1-\lambda) y) \;\le\; \lambda f(x) + (1-\lambda) f(y)
$$

You rarely check this by hand, because convexity has a *calculus* — a small rulebook that builds
complicated convex functions from simple ones:

- **Atoms.** Squared error $x^2$; absolute value $|x|$; norms, including group norms such as the
  $\ell_2$ norm of a vector of increments; $\max$; the quantile/pinball loss; the Huber loss — all
  convex.
- **Combination rules.** A non-negative weighted sum of convex functions is convex (so
  $\text{fit} + \lambda \cdot \text{penalty}$ is safe). A convex function of an *affine* expression ($A \cdot x + b$) is convex
  (so "residual = data − incidence-matrix × flows, then sum of squares" is safe). A pointwise max
  of convex functions is convex.

Every term in the edge-flow estimator is assembled from exactly these pieces — that is *why* it
is convex — and the rulebook is mechanical enough for software to verify (two sections down).

## What a solver is, and why convexity matters

A solver is off-the-shelf software whose whole job is finding the best answer to an optimisation
problem: hand it a scoring rule and a set of unknowns, and it returns the values that score best.
Convex problems are the friendly kind — the score landscape is a single bowl with no false
valleys. The solver rolls downhill and is *guaranteed* to stop at the true bottom. Contrast training a neural network, where the landscape is full of dents and you
never know whether you found the best one. Here you always do, deterministically: no learning
rates, no convergence babysitting, no random seeds, and the identical answer every run.

That last property is quietly important for this project's deliverables: an event list handed to
NGED should not change because a fit was re-run.

## The corners are a feature: exact zeros

Lasso-family penalties ($|x|$ and its group cousins) have a sharp corner at zero, and the corner
is the point: it makes the optimum sit at *exactly* zero for most variables, rather than merely
near it. Designs like the edge-flow estimator lean on this — "this pipe's stretch of nonzero
flow" literally *is* the detected event, which only works if inactive pipes read 0.000 MW.

Specialised convex solvers (proximal and interior-point methods) handle those corners exactly.
Plain gradient descent does not: it slides around the corner and leaves every variable carrying a
tiny residual trickle — ±0.03 MW on every pipe, everywhere — forcing a threshold to decide what
counts as an event, and thereby reintroducing exactly the arbitrary cutoff the formulation was
designed to remove. Fixing this in PyTorch means hand-rolling the proximal machinery that convex
solvers provide for free.

## CVXPY: the modelling language

[CVXPY](https://www.cvxpy.org/) is a Python modelling language for convex problems. You write the
score in NumPy-like syntax; CVXPY checks *at construction time* — using the compositional
rulebook above, formalised as **disciplined convex programming (DCP)** — that the problem really
is convex, and refuses to build it otherwise. It then compiles the problem and hands it to a
solver backend (Clarabel, ECOS, SCS, …).

```python
import cvxpy as cp

x = cp.Variable(n)
score = cp.sum_squares(A @ x - b) + lam * cp.sum(cp.abs(x))
problem = cp.Problem(cp.Minimize(score), [x >= 0])
problem.solve(solver=cp.CLARABEL)  # certified global optimum in x.value
```

The construction-time check cuts both ways, and both directions are useful: it *certifies* the
formulations it accepts, and it *refuses* the ones it cannot certify — which is a feature, not an
obstacle, because it says honestly when a problem has left the convex world and needs the other
toolchain.

## Why not just PyTorch for the convex problems too?

Because the same loss implemented in PyTorch surrenders the two properties that motivated the
convex formulation in the first place:

1. **The guarantee.** CVXPY exploits convexity to deliver the certified global optimum,
   identically, every run. Gradient descent doesn't know the landscape is one bowl and inherits
   all the usual tuning — learning rates, schedules, stopping criteria, seeds — for a problem that
   required none.
2. **Exact zeros.** As above: crisp zeros come from solvers that respect the $\ell_1$ corner; gradient
   descent leaves trickles and brings thresholds back.

The residual attractions of PyTorch are real — one toolchain across the project, GPUs at
~2,500-series scale, and the freedom to later make a model non-convex — but they do not outweigh
the two properties above *for a formulation that is actually convex*. And the bridge below means
the choice is not a wall.

## Two routes to the same inverse problem

It is tempting to file CVXPY and PyTorch under different activities — "optimisation" versus
"physics modelling". That framing is wrong, and correcting it makes the tooling rule much easier
to hold in your head. Both toolchains, as this project uses them, are doing **inverse
modelling**: write down a *forward model* mapping unknown parameters to a predicted signal
(capacities and orientations to power; edge flows to substation residuals), then invert it
against observations to recover the parameters. And both host physics — a fixed per-unit
[pvlib](https://pvlib-python.readthedocs.io/) curve inside a CVXPY problem is exactly as much a
physical model as a transposition calculation inside a PyTorch module.

The genuine difference is a single trade-off:

- **CVXPY restricts what the forward model may contain** — only expressions the
  [DCP rulebook](#cvxpy-the-modelling-language) can certify as convex in the unknowns — **and in
  exchange guarantees the inversion**: the certified global optimum, identically every run, with
  [exact zeros](#the-corners-are-a-feature-exact-zeros), no initialisation, no learning rates,
  no "did I converge?" anxiety. The restriction bites on real physics: capacity × an
  orientation-dependent per-unit curve is convex only once the orientation is fixed
  ([grade 2 below](#where-pytorch-is-the-right-tool)), and capacity × a shared irradiance bias
  is bilinear, full stop.
- **PyTorch accepts any differentiable forward model** — temperature-dependent efficiency, hard
  clips, neural components, variational posteriors — **and gives up the guarantees**: the
  inversion is gradient descent on a dented landscape, managed with initialisation, restarts,
  and validation. In return it gains unrestricted expressiveness and composability —
  minibatching, joint training with neural nets, amortisation across sites.

So the axis separating the tools is **expressiveness of the forward model versus guarantees on
the inversion** — not whether physics is involved, and not whether differentiation is involved.

That last point deserves spelling out, because "CVXPY doesn't use differentiation" is a natural
misreading. The interior-point solvers behind CVXPY use derivative information heavily — that is
how they navigate to the bottom of the bowl — and
[`cvxpylayers`](#the-bridge-welding-cvxpy-into-pytorch) literally differentiates *through* a
convex solve. What CVXPY does not do is backpropagate through an arbitrary user-written
simulator, because it never sees a simulator: it sees a structured algebraic problem, and it is
precisely by exploiting that structure that it can promise the global optimum. Conversely,
"[differentiable physics](differentiable-physics.md)" is shorthand for *a physics simulator
written in an autodiff framework so that it composes with gradient-based learning* — the
differentiability is the enabling property, not the point.

One honest wrinkle in the inverse-modelling framing: it covers the *estimation* applications on
this page cleanly, but convex optimisation is broader — a future scheduling or dispatch problem
would be a *forward* optimisation (choose the best actions), not an inversion. The framing
describes the applications this project currently plans, not the whole convex world.

## The recurring pattern: fixed shapes, unknown coefficients

Many estimation problems in this project share one algebraic skeleton. Physics (or a previously
learnt model) fixes the **shape** of a signal, and the data only has to decide **how much** of
that shape is present:

$$
\text{predicted signal} \;=\; \text{known shape} \times \text{unknown coefficient}
$$

The predicted signal is then *linear* in the unknowns — and a convex loss on a linear predictor,
plus convex penalties on the coefficients, is a convex problem end to end. The pattern shows up
everywhere once you look for it:

- a metered PV site's per-unit output curve (fixed, from
  [pvlib](https://pvlib-python.readthedocs.io/) at a given orientation) × its unknown DC
  capacity — the
  [convex capacity candidate](../roadmap/capacity-estimation.md#candidate-a-the-convex-estimator-cvxpy);
- a small set of orientation basis curves × unknown per-basis installed capacity — the convex
  view of the
  [fleet node](differentiable-physics.md#scaling-to-aggregate-fleets-universalsolarfleetnode);
- a menu of candidate DER output curves × the unknown amount of each behind a substation — the
  [convex dictionary baseline](../roadmap/disaggregation.md#the-convex-dictionary-baseline) for
  disaggregation;
- a frozen dictionary of universal demand shapes × a substation's unknown "style vector"
  (see the `BasisLoadNode` in
  [the graph-structured engine](../roadmap/disaggregation.md#node-definitions)).

The coefficients in all four cases want the same *priors*, and each prior is convex:
non-negativity (a hard constraint), sparsity ($\ell_1$, giving
[exact zeros](#the-corners-are-a-feature-exact-zeros)), piecewise-constancy (an $\ell_1$ penalty on
successive differences — the **fused lasso**), and monotone growth (a hard constraint on the
differences).

It is worth comparing this with the devices the
[differentiable-physics](differentiable-physics.md) sketches use to express the *same* priors
inside PyTorch:

| Prior on a coefficient | In a PyTorch (non-convex) host | In a CVXPY (convex) host |
|---|---|---|
| Capacity never decreases | Cumulative sum of softplus-transformed increments | Hard constraint on the increments (one line) |
| Growth arrives in sparse bursts | $\ell_1$ penalty under gradient descent → *approximate* zeros | $\ell_1$ penalty under a proximal solver → *exact* zeros |
| Output cannot exceed a hard limit | Smooth clip, to keep gradients alive | Censoring split ([next section](#hard-limits-as-censoring)) |

The PyTorch devices are **not mistakes**. Inside a model that is non-convex anyway — because it
is also *learning the shapes* — they are the right way to keep gradients flowing, and CVXPY could
not host that model at all. The comparison cuts the other way: when a problem reduces entirely to
fixed shapes × unknown coefficients, the convex host expresses each prior exactly, in a line,
with the guarantees of [the sections above](#why-not-just-pytorch-for-the-convex-problems-too).
(And the two hosts converge in the end: under a
[`cvxpylayers` bridge](#the-bridge-welding-cvxpy-into-pytorch), the hard constraints come back
*inside* PyTorch.)

## Hard limits as censoring

Physical systems are full of hard limits: an inverter clips PV output at its AC rating, a wind
turbine's power curve is flat at rated power. A hard limit looks like an obstacle to convexity —
the observed power is $\min(\text{capacity} \times \text{per-unit output},\ \text{limit})$, and
fitting *through* a min of unknowns is not a convex operation.

The way out is old and comes from econometrics (**Tobit regression**, 1958): treat the limit as
**censoring**, and classify each sample *before* fitting. A clipped sample is usually observable
directly in the data — on a bright day a clipped PV site sits on a flat plateau at a common
ceiling. Once each timestep is labelled clipped / unclipped, the problem falls apart into convex
pieces:

- **Unclipped samples** behave linearly: observed power ≈ shape × coefficient, and any convex
  loss applies.
- **Censored (clipped) samples** still carry information — two pieces of it. The plateau level is
  a direct, noisy *measurement of the limit itself*. And the un-clipped signal must have
  *exceeded* the limit for clipping to occur at all, which is a linear inequality on the
  unknowns, enforceable softly as a hinge penalty:

$$
\max\!\bigl(0,\; \text{limit} - \text{shape} \times \text{coefficient}\bigr)
$$

Both pieces are convex, so the censored formulation keeps every guarantee on this page while
extracting information even from the samples the limit has flattened.

One honest caution: the pre-classification step is a *heuristic that sits outside the convex
problem*, and the guarantee downstream is conditional on it. Samples labelled wrongly poison both
terms — an unclipped sample called clipped drags the limit estimate down; a clipped sample called
unclipped biases the coefficient fit. Where this pattern is applied, the classifier needs its own
validation (see
[the capacity-estimation caveats](../roadmap/capacity-estimation.md#candidate-a-the-convex-estimator-cvxpy)
for the concrete case).

## Fitting an envelope with the quantile (pinball) loss

Least squares fits the *middle* of the data. Sometimes the quantity we want is the *top*: a
generator's capacity is "what the site delivers when nothing holds it back", and the data is
littered with samples where something *did* hold it back — curtailment, outages, faults — all of
which push observations **down**, never up. Least squares would split the difference and drag the
estimate below the truth.

The **quantile loss** (also called the **pinball loss**) fits any chosen quantile instead of the
mean. For residual $r = \text{observed} - \text{predicted}$ and quantile level
$\tau \in (0, 1)$:

$$
\rho_\tau(r) \;=\; \tau \,\max(r, 0) \;+\; (1 - \tau)\,\max(-r, 0)
$$

At $\tau = 0.5$ the two sides weigh equally and the fit is the median. At a high $\tau$ (say
0.95), an observation *above* the fitted curve costs $19\times$ more than one the same distance
below — so the curve rises until only ~5% of points sit above it. The fit becomes an **upper
envelope**: curtailed and faulted samples fall below it without biasing it. The loss is a
non-negative weighted sum of $\max$ atoms, so it is convex and CVXPY-native.

$\tau$ is a genuine tuning choice with consequences in both directions: too low and curtailment
leaks back into the estimate; too high and the envelope chases meter glitches and noise spikes.
(Where the corruption is symmetric outliers rather than one-sided suppression, the **Huber
loss** — quadratic near zero, linear in the tails — is the usual convex alternative.)

## Priors as convex penalties — and the uncertainty you don't get

Prior knowledge slots into a convex problem as one more penalty term charging deviation from the
prior — the optimum is then a *maximum a posteriori* (MAP) estimate, and the rulebook keeps
everything convex:

- **A prior value** (a capacity register entry, last year's fit, a connection record): penalise
  distance from it, e.g. $\lambda \, |c - c_{\text{register}}|$.
- **An asymmetric prior**: registers overstate more than they understate, so make it cheap for
  the estimate to sit *below* the registered value and expensive to exceed it — two hinge terms
  with different weights.
- **A timing prior**: if works are known to have happened in a given week, make changes cheap at
  that date and expensive elsewhere — simply weight a fused-lasso penalty per step.

What this machinery **cannot** give is a *posterior*. A convex solve returns the single most
plausible answer under the prior and the data; it does not say how confident to be, and there are
no error bars. The available bolt-ons are all imperfect:

- **Bootstrap** (re-solve on resampled residuals): mechanically easy, but bootstrap-after-lasso
  is known to be statistically shaky exactly where we care — the uncertainty of *which*
  changepoints exist.
- **Sensitivity re-solves** (perturb the penalty strengths, watch the answer move): a cheap
  honesty check, not a calibrated interval.
- **Bayesian reinterpretation** (the fused lasso is the MAP estimate under Laplace priors):
  obtaining the full posterior means leaving CVXPY for MCMC or variational inference.

By contrast, the [variational differentiable-physics route](differentiable-physics.md) produces
(approximate) posteriors natively. Where calibrated, decomposable uncertainty is a first-class
requirement — as it is for
[effective-capacity estimates](../roadmap/capacity-estimation.md#uncertainty-a-first-class-judging-criterion) —
this is the honest structural con of the convex route, to be weighed against its guarantees.

## Where PyTorch is the right tool

"Non-convex" is not one condition — it comes in grades, and the grade decides the tool:

1. **Convex outright.** Everything above:
   [fixed shapes × unknown coefficients](#the-recurring-pattern-fixed-shapes-unknown-coefficients)
   under convex losses, penalties and constraints. CVXPY, no contest.

2. **Jointly non-convex, but convex once you condition on a small set of unknowns.** Estimating a
   metered PV site's capacity *and* its panel orientation is non-convex — capacity multiplies a
   per-unit output curve that depends non-linearly on tilt and azimuth. But *given* the
   orientation, the per-unit curve is a fixed shape and the capacity fit is convex. When the
   conditioning set is low-dimensional — here just two angles — the right move is to **enumerate
   it**: grid-search tilt × azimuth on the outside, and solve the fast, warm-started convex
   capacity problem at each grid point. An exhaustive outer search wrapped around a certified
   inner solve is deterministic and leaves no valley unvisited (up to the grid resolution) — no
   gradients required. This is
   [Candidate A in the v0.7 capacity head-to-head](../roadmap/capacity-estimation.md#candidate-a-the-convex-estimator-cvxpy).
   The trick's limits are its dimensions: tens of sites × a few thousand grid points is cheap;
   it does not scale to genuinely high-dimensional parameter spaces.

3. **Intrinsically non-convex.** Anything where the *shapes themselves* are being learnt —
   wind-turbine power curves, heat-pump COP rolloff, thermal time constants, neural-network
   layers — and all variational machinery (CVXPY has no notion of a posterior; the ELBO training
   in [differentiable physics](differentiable-physics.md) is outside its world). Some problems
   are non-convex *by construction*: reconstructing signed flow from magnitude-only MVA metering
   has a sign ambiguity, i.e. two valleys. CVXPY's DCP check refuses all of these at construction
   time — correctly. PyTorch offers the opposite deal: write down almost anything, gradients flow
   through it, GPUs make it fast, and the dented landscape is managed with initialisation,
   restarts, and validation.

**Rule of thumb:** convex estimation subproblems → **CVXPY**, for certainty, reproducibility, and
little code. Learning shapes, or anything needing posteriors → **PyTorch**, because nothing else
can express it. A low-dimensional non-convex remainder wrapped around a convex core → enumerate
the remainder, solve the core.

## The bridge: welding CVXPY into PyTorch

There is a bridge between the two worlds: **differentiable convex optimisation layers**
([`cvxpylayers`](https://github.com/cvxgrp/cvxpylayers), from the same research group as CVXPY).
It embeds a convex solve inside a PyTorch model as if it were another layer, with gradients
flowing through the solve itself. Concretely for this project: the switching-event estimator need
not remain a separate preprocessing step forever — the routing/switching solve can eventually sit
*inside* the type-resolved mixture as a convex layer, with the differentiable-physics modules
feeding it and gradients passing straight through, keeping exact zeros and built-in conservation
where free tensors would lose both. (The magnitude-only mixture model, by contrast, needs no
PyTorch at all: it is buildable as alternating CVXPY solves.) When each rung earns its place is
documented in the
[mixture-model tooling note](../roadmap/switching-events.md#approach-4-the-magnitude-only-mixture-model-the-workhorse)
on the switching-events roadmap page. Not a day-one build — but it means choosing CVXPY for
switching now does not wall that code off from the PyTorch future.

## Applications in this project

- **Switching-event detection — the
  [joint edge-flow estimator](../roadmap/switching-events.md#approach-3-the-joint-edge-flow-estimator)**
  (first planned application): a group fused lasso on signed edge flows; implementation sketch on
  the roadmap page.
- **Metered-generator effective capacity — the
  [convex candidate](../roadmap/capacity-estimation.md#candidate-a-the-convex-estimator-cvxpy)**
  in the v0.7 head-to-head: censored quantile-envelope fit with fused-lasso changepoints, and
  orientation by grid search.
- **The magnitude-only mixture model** — bilinear, so not one convex problem, but buildable entirely as
  *alternating* CVXPY solves (each step convex; global-optimum guarantee holds per step, not
  overall). See the
  [mixture-model tooling note](../roadmap/switching-events.md#approach-4-the-magnitude-only-mixture-model-the-workhorse).
- **The [convex dictionary baseline](../roadmap/disaggregation.md#the-convex-dictionary-baseline)**
  for unmetered-DER disaggregation: convex selection from a menu of precomputed candidate
  systems — the transparent baseline the v2 engine must beat.
- **Convex companions to the differentiable-physics modules**: a below-the-clip initialiser and
  sanity check for the
  [fleet node](differentiable-physics.md#scaling-to-aggregate-fleets-universalsolarfleetnode),
  and cheap style-vector fitting against a frozen demand dictionary
  ([`BasisLoadNode`](../roadmap/disaggregation.md#node-definitions)).
- More are expected to follow. Roadmap pages that adopt a convex formulation should cite this
  page for the rationale and tooling choice rather than re-deriving them.
