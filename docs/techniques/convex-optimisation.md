# Convex Optimisation — When the Best Answer Is Guaranteed

> **Status: durable method explainer.** This page explains what convex optimisation is, why a
> convex formulation is a fundamentally better deal than gradient-descent training wherever it is
> available, and the project's practical tooling rule — **CVXPY for convex estimation
> subproblems, PyTorch for physics + learning**. The first planned application is the
> [joint edge-flow estimator](../roadmap/switching-events.md#v061-the-joint-edge-flow-estimator)
> for switching-event detection; further applications are expected. The Python in this document
> is illustrative sketch code, not the implementation.

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

## Where PyTorch is the right tool

For the differentiable-physics disaggregation work — estimating unmetered solar, wind, EVs and
heat pumps, and the effective capacity of metered generators — CVXPY is not marginally wrong but
*cannot-express-the-problem* wrong. Convexity is fragile: panel-orientation trigonometry,
wind-turbine power curves (S-shaped ramp, cut-out), thermal heat-pump models, even multiplying
two unknowns together (unknown capacity × unknown per-unit output) all break it, and GNN layers
are as non-convex as it gets. CVXPY's DCP check would refuse these models at construction —
correctly. PyTorch offers the opposite deal: write down almost anything, gradients flow through
it, GPUs make it fast, and the dented landscape is managed with initialisation, restarts, and
validation. The [differentiable-physics](differentiable-physics.md) plan is right to live there.

**Rule of thumb:** convex estimation subproblems (the switching-events edge-flow estimator, and
similar) → **CVXPY**, for certainty, reproducibility, and little code. Physics + learning
(capacity estimation, disaggregation) → **PyTorch**, because nothing else can express it.

## The bridge: welding CVXPY into PyTorch

There is a bridge between the two worlds: **differentiable convex optimisation layers**
([`cvxpylayers`](https://github.com/cvxgrp/cvxpylayers), from the same research group as CVXPY).
It embeds a convex solve inside a PyTorch model as if it were another layer, with gradients
flowing through the solve itself. Concretely for this project: the switching-event estimator need
not remain a separate preprocessing step forever — the routing/switching solve can eventually sit
*inside* the v2.6 type-resolved model as a convex layer, with the differentiable-physics modules
feeding it and gradients passing straight through, keeping exact zeros and built-in conservation
where free tensors would lose both. (v2.5, by contrast, needs no PyTorch at all: it is buildable
as alternating CVXPY solves.) When each rung earns its place is documented in the
[v2.5 tooling note](../roadmap/switching-events.md#v25-magnitude-only-mixture-model-the-workhorse)
on the switching-events roadmap page. Not a day-one build — but it means choosing CVXPY for
switching now does not wall that code off from the PyTorch future.

## Applications in this project

- **Switching-event detection — the
  [joint edge-flow estimator](../roadmap/switching-events.md#v061-the-joint-edge-flow-estimator)**
  (first planned application): a group fused lasso on signed edge flows; implementation sketch on
  the roadmap page.
- **The v2.5 mixture model** — bilinear, so not one convex problem, but buildable entirely as
  *alternating* CVXPY solves (each step convex; global-optimum guarantee holds per step, not
  overall). See the
  [v2.5 tooling note](../roadmap/switching-events.md#v25-magnitude-only-mixture-model-the-workhorse).
- More are expected to follow. Roadmap pages that adopt a convex formulation should cite this
  page for the rationale and tooling choice rather than re-deriving them.
