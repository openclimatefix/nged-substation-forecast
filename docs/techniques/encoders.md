# Learned Encoders

> **Status: 🔬 Research.** Encoders are a v2 research direction. V1 uses raw NWP features directly.

In the full graph-structured architecture, many components need to transform the same raw inputs — NWP
grid values, timestamps, substation location — into useful representations. Rather than re-learning these
transformations independently inside each node type, **shared encoder modules** learn a single
compact embedding that every node can use.

**The case for a pre-trained encoder rests on results from computer vision and Earth observation
rather than from energy forecasting**, the [energy-forecasting
review](../background/energy-forecasting-review.md#pre-trained-encoders) found. [Siméoni et al.
(2025)](https://arxiv.org/abs/2508.10104) and [Brown et al.
(2025)](https://arxiv.org/abs/2507.22291) each show a single frozen encoder serving many downstream
tasks, which is the arrangement this page plans. Neither result promises that a pre-trained encoder
beats hand-designed features: Brown et al. report that learned featurisations "don't always
outperform designed featurization methods in scarce data regimes". The gradient-boosted tree on
hand-designed features is therefore the bar these encoders have to clear, rather than a floor they
can be assumed to sit above.

## Why encoders are a natural fit with differentiable physics

This is the key insight: the differentiable physics layer handles DER-specific physical relationships
(how much irradiance a panel converts to power given its geometry; how wind speed maps to turbine
power via the cubic law). This means the **weather encoder does not need to learn anything about
solar panels or wind turbines** — it just needs to produce a good representation of the atmospheric
state. The physics layer then interprets that shared representation through the appropriate equations
for each DER type.

The practical payoff: a single shared weather encoder can be trained jointly across all DER types and
all substations, benefiting from the full dataset. The encoder learns weather; the physics handles
DER specifics.

**The [energy-forecasting review](../background/energy-forecasting-review.md#pre-trained-encoders)
found nobody pre-training a weather encoder against observations and then reading a substation's
probabilistic load forecast off it, nor anybody using a differentiable model of a solar or wind farm
to strip out the variance the engineering explains so that the weather encoder trains on a clean
weather signal.** Both are what this page describes. The nearest precedent the review found for
joining a pre-trained weather representation to a downstream forecast is one entrant in HEFTCom, a
competition to forecast a GB wind-and-solar portfolio day-ahead: [Browell et al.
(2026)](https://doi.org/10.1016/j.ijforecast.2025.10.005) report that team Rnt fed embeddings from
their own AI weather models into downstream neural networks and finished third of the ranked
entrants.

## Encoder types

### WeatherEncoder

Maps NWP grid-cell values at a given (location, time) to a compact weather embedding.

Candidate inputs: temperature, wind speed and direction, cloud cover, GHI, DNI, DHI, humidity,
pressure. May need to capture temporal context (a persistent pressure pattern carries different
meaning than a transient one) and spatial context across nearby NWP grid cells.

Training signal: the reconstruction error from the full forward model — if the weather embedding is
poor, the DP modules cannot reconstruct observed substation power correctly.

**Each half of this design has already been built separately, by different authors.** [Rasp and
Lerch (2018)](https://arxiv.org/abs/1805.09091) post-process a 50-member ECMWF ensemble into
calibrated probabilistic 2-metre temperature at 537 German weather stations 48 hours ahead, cutting
mean continuous ranked probability score from 1.16 for the raw ensemble to 0.78. [Mitra and
Ramavajjala (2023)](https://arxiv.org/abs/2312.00290) freeze a weather autoencoder and train small
models on the frozen representation alone, at accuracy comparable to purpose-built models. Their
targets are further weather variables rather than power on an electricity network, though. The
[energy-forecasting review](../background/energy-forecasting-review.md#pre-trained-encoders) sets
out both.

### TimeEncoder

Maps a timestamp to an embedding capturing periodic structure: time-of-day, day-of-week, month, bank
holidays, UK calendar effects (Christmas, Easter). Shared across all node types since every component
of the forward model has some time-of-day / time-of-year structure.

### SpaceEncoder (possible future addition)

A static per-substation embedding capturing geographic context — latitude, altitude, local terrain,
proximity to coast — that does not change over time. The DP modules already use lat/lon directly for
solar geometry, so this encoder would capture residual structure the hard-coded geometry does not
explain.

## Handling missing inputs: remove the token, don't zero-fill

Encoder inputs go missing in production — a missed NWP run, a variable absent from a slice, a
stalled meter — so how absence is represented is an architectural decision, not an afterthought.

**Do not zero-fill.** Zero is a meaningful value in physical units: 0 MW, 0 W/m² and 0 °C are all
real physical states. Substituting zero for an unknown therefore asserts something false, and the
network cannot tell the two apart, so it learns a conditional mean contaminated by fabricated data.

**Treat inputs as a set of tokens and simply omit the absent ones.** Each token carries a value
embedding, a feature-identity embedding and a time embedding; attention is natively
permutation-invariant and variable-length, so a missing input is *structurally* absent rather than
encoded as a sentinel. Mask the attention matrix for padding only. The dense alternative, for
architectures that need a fixed-width input, is **value + mask channels**, so the network can still
distinguish "zero" from "unknown" — GRU-D is the standard precedent, pairing masks with a learned
decay of the last observation toward an empirical mean.

**Do not train for missingness with random dropout alone.** Random dropout simulates data that is
*missing completely at random*, and production missingness is not: outages correlate with time of
day, weather systems and provider incidents, and a meter that drops out during the storm that caused
an extreme reading is missing *because* the value was extreme. A model trained on random dropout is
calibrated for a world it does not live in, and the miscalibration shows up as over-confident
predictions during a real outage — the worst possible moment. Use **structured, outage-shaped**
dropout drawn from the same failure-scenario vocabulary the rest of the project scores against.

See [Inherent Stability](../design-philosophy/inherent-stability.md) for the whole principle.

## What the encoders do *not* need to learn

Because the DP layer hard-codes solar geometry, the weather encoder does not need to learn that
"noon → peak irradiance" or "winter → low sun angle." The time encoder does not need to represent
seasonality for PV — that is handled by the ephemeris computation in the DP module. The encoders can
focus entirely on the residual structure the physics does not explain: NWP biases, local
microclimatic effects, and behavioural anomalies.
