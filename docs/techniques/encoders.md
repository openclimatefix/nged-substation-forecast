# Learned Encoders

> **Status: 🔬 Research.** Encoders are a v2 research direction. V1 uses raw NWP (Numerical Weather Prediction) features directly.

In the full graph-structured architecture, many components need to transform the same raw inputs — NWP
grid values, timestamps, substation location — into useful representations. Rather than re-learning these
transformations independently inside each node type, **shared encoder modules** learn a single
compact embedding that every node can use.

**The case for a pre-trained encoder rests on results from computer vision and Earth observation
rather than from energy forecasting, and neither result promises a pre-trained encoder beats
hand-designed features** — see the [energy-forecasting
review](../background/energy-forecasting-review.md#pre-trained-encoders) for DINOv3, AlphaEarth
Foundations, and the evidence on both sides. The gradient-boosted tree on hand-designed features
is therefore the bar these encoders have to clear, rather than a floor they can be assumed to sit
above.

## Why encoders are a natural fit with differentiable physics

This is the key insight: the differentiable physics (DP) layer handles distributed energy resource
(DER)-specific physical relationships (how much irradiance a panel converts to power given its
geometry; how wind speed maps to turbine power via the cubic law). This means the **weather encoder
does not need to learn anything about solar panels or wind turbines** — it just needs to produce a
good representation of the atmospheric state. The physics layer then interprets that shared
representation through the appropriate equations for each DER type.

The practical payoff: a single shared weather encoder can be trained jointly across all DER types and
all substations, benefiting from the full dataset. The encoder learns weather; the physics handles
DER specifics.

**The [energy-forecasting
review](../background/energy-forecasting-review.md#pre-trained-encoders) found nobody
pre-training a weather encoder against observations and then reading a substation's probabilistic
load forecast off it, nor anybody using a differentiable model of a solar or wind farm to strip
out the variance the engineering explains so that the weather encoder trains on a clean weather
signal.** Both are what this page describes.

## Encoder types

### WeatherEncoder

Maps NWP grid-cell values at a given (location, time) to a compact weather embedding.

Candidate inputs: temperature, wind speed and direction, cloud cover, GHI, DNI, DHI, humidity,
pressure. May need to capture temporal context (a persistent pressure pattern carries different
meaning than a transient one) and spatial context across nearby NWP grid cells.

Training signal: the reconstruction error from the full forward model — if the weather embedding is
poor, the DP modules cannot reconstruct observed substation power correctly.

**Each half of this design — turning a raw ensemble into a calibrated forecast, then freezing it
as a shared representation — has already been built separately, by different authors,** as the
[energy-forecasting review](../background/energy-forecasting-review.md#pre-trained-encoders) sets
out: [Rasp and Lerch (2018)](https://arxiv.org/abs/1805.09091) post-process an ECMWF ensemble
into calibrated station-level temperature; [Mitra and Ramavajjala
(2023)](https://arxiv.org/abs/2312.00290) freeze a weather autoencoder and train small models on
the frozen representation, though on further weather variables rather than power.

#### Blending several NWP sources

**The WeatherEncoder can take several NWP sources at once and learn when to trust each source.**
The candidate sources are ECMWF ENS, ECMWF's machine-learned AIFS-ENS, and DWD's deterministic
ICON-EU. Each source enters as tokens carrying a source-identity embedding, so a missing source
simply [drops out](#handling-missing-inputs-remove-the-token-dont-zero-fill).

**Feed the encoder ensemble members, not statistics summarising the ensemble, because the
differentiable-physics layer is nonlinear.** Wind power rises with the cube of wind speed between
cut-in and rated speed, inverters clip PV output, and PV output depends on the sun's position. So
the power computed from the ensemble-mean weather is not the mean of the power computed from each
member. Pushing weather quantiles through the physics layer works for one weather variable at one
generator, whose power curve is monotone below cut-out. A substation, though, sums many generators
and a demand driven by several correlated weather variables. Each member carries that joint
structure across variables and sites, and a summary statistic discards the joint structure. The
gradient-boosted tree has no physics layer, which is why [ensemble statistics as
features](../roadmap/xgboost-improvements.md#ensemble-statistics-as-features-instead-of-member-by-member-rows)
remains a fair experiment for the tree.

**Feed ECMWF ENS member *n* and AIFS-ENS member *n* into the same forward pass, because the two
members start from the same initial conditions.** ECMWF starts AIFS-ENS member *n* from the
[initial conditions of ECMWF ENS member
*n*](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+ENS+v1). The pair is
therefore one perturbed starting state run through two models, and the difference between the two
members measures model error with initial-condition error held fixed. A forward pass that sees
only one source can learn to correct that source's biases, but cannot learn how far to trust that
source, because trust is relative to the alternatives. ICON-EU is a single deterministic run, so
the same ICON-EU tokens enter all 51 forward passes. Where training finds ICON-EU reliable, the
ICON-EU tokens pull every member towards the ICON-EU solution, without adding spread of their own.
The output stays 51 equiprobable members, so the [linear
pool](probabilistic-forecasting.md#the-fix-formally-a-mixture-of-conditional-distributions) and
the [fair CRPS](evaluation-metrics.md#crps-continuous-ranked-probability-score) apply unchanged.

**A weighted mixture of the 51 ECMWF ENS members and the 51 AIFS-ENS members is the weaker
alternative.** The mixture is a valid forecast, but the mixture discards the pairing, which is the
most direct signal of model error. The mixture also changes the member count, and the spread-skill
ratio and PICP [shift with the member count](evaluation-metrics.md#probabilistic-metrics).

**The encoder infers the weather situation from its inputs, with no hand-labelled weather
regimes.** We expect ensemble spread and the disagreement between sources to be the strongest
indicators of which source to trust, and both are already inputs. Checking what the encoder has
learned needs no labels either: group the results by pressure gradient, ensemble spread, or
disagreement between sources, or cluster the encoder's embeddings after training. The bar is the
gradient-boosted tree given the same sources, and the [multi-source XGBoost
experiment](../roadmap/xgboost-improvements.md#several-nwp-sources-as-features-v21) also tests,
more cheaply, whether trust should depend on the weather situation at all.

**An irradiance field recovered by inverting the metered DER fleet would be a different kind of
input from an NWP source.** The [GB-wide
inversion](../roadmap/disaggregation.md#gb-wide-inverse-irradiance-mapping) estimates the
irradiance at the present time, not a forecast. At the same time step, the encoder would be
reconstructing the very fleet power the field was inverted from, so the field could enter only as
a lagged observation. Whether interpolating the fleet's point estimates into a field adds
information beyond the points themselves is an open question: interpolating irradiance between
[ground stations stops helping beyond roughly 20 to 34
km](../roadmap/disaggregation.md#correcting-satellite-irradiance-over-great-britain), and the
disaggregation page does not yet check the metered fleet's spacing against that distance. Until that question is
settled, the point estimates can enter as tokens at their own locations.

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
real physical states. Substituting zero for an unknown therefore asserts that the reading was zero,
which is often false. The network cannot tell a true zero from a missing one, so it learns a
conditional mean contaminated by fabricated data.

**Treat inputs as a set of tokens and simply omit the absent tokens.** Each token carries a value
embedding, a feature-identity embedding, and a time embedding. Attention is natively
permutation-invariant and variable-length, so a missing input is *structurally* absent rather than
encoded as a sentinel. Mask the attention matrix for padding only. The dense alternative, for
architectures that need a fixed-width input, is **value + mask channels**, so the network can still
distinguish "zero" from "unknown" — GRU-D is the standard precedent, pairing masks with a learned
decay of the last observation toward an empirical mean.

**Do not train for missingness with random dropout alone.** Random dropout simulates data that is
*missing completely at random*, and production missingness is not. Outages correlate with time of
day, weather systems, and provider incidents, and a meter that drops out during the storm that
caused an extreme reading is missing *because* the value was extreme. A model trained on random
dropout is calibrated for a world it does not live in, and the miscalibration shows up as
over-confident predictions during a real outage — the worst possible moment. Use **structured,
outage-shaped** dropout drawn from the same failure-scenario vocabulary the rest of the project
scores against.

See [Inherent Stability](../design-philosophy/inherent-stability.md) for the whole principle.

## What the encoders do *not* need to learn

Because the DP layer hard-codes solar geometry, the weather encoder does not need to learn that
"noon → peak irradiance" or "winter → low sun angle." The time encoder does not need to represent
seasonality for solar photovoltaic (PV) power — that is handled by the ephemeris computation in the
DP module. The encoders can focus entirely on the residual structure the physics does not explain:
NWP biases, local microclimatic effects, and behavioural anomalies.
