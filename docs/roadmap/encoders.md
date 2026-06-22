# Learned Encoders

> **Status: 🔬 Research.** Encoders are a v2 research direction. V1 uses raw NWP features directly.

In the full GNN architecture, many components need to transform the same raw inputs — NWP grid
values, timestamps, substation location — into useful representations. Rather than re-learning these
transformations independently inside each node type, **shared encoder modules** learn a single
compact embedding that every node can use.

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

## Encoder types

### WeatherEncoder

Maps NWP grid-cell values at a given (location, time) to a compact weather embedding.

Candidate inputs: temperature, wind speed and direction, cloud cover, GHI, DNI, DHI, humidity,
pressure. May need to capture temporal context (a persistent pressure pattern carries different
meaning than a transient one) and spatial context across nearby NWP grid cells.

Training signal: the reconstruction error from the full forward model — if the weather embedding is
poor, the DP modules cannot reconstruct observed substation power correctly.

### TimeEncoder

Maps a timestamp to an embedding capturing periodic structure: time-of-day, day-of-week, month, bank
holidays, UK calendar effects (Christmas, Easter). Shared across all node types since every component
of the forward model has some time-of-day / time-of-year structure.

### SpaceEncoder (possible future addition)

A static per-substation embedding capturing geographic context — latitude, altitude, local terrain,
proximity to coast — that does not change over time. The DP modules already use lat/lon directly for
solar geometry, so this encoder would capture residual structure the hard-coded geometry does not
explain.

## What the encoders do *not* need to learn

Because the DP layer hard-codes solar geometry, the weather encoder does not need to learn that
"noon → peak irradiance" or "winter → low sun angle." The time encoder does not need to represent
seasonality for PV — that is handled by the ephemeris computation in the DP module. The encoders can
focus entirely on the residual structure the physics does not explain: NWP biases, local
microclimatic effects, and behavioural anomalies.
