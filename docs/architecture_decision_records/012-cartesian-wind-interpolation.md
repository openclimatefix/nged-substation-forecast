# ADR-001: Cartesian Wind Vector Interpolation

## Status
Accepted

## Context
Previously, wind speed and direction were interpolated using circular logic. This led to "phantom high wind" artifacts when the wind direction shifted rapidly (e.g., from 359 to 1 degree). For example, if the wind direction shifted from 359 to 1 degree, the circular interpolation might incorrectly calculate a large change in direction, leading to an artificially high wind speed during the transition.

## Decision
We have shifted from circular interpolation of wind speed and direction to linear interpolation of Cartesian `u` and `v` components.

## Rationale
1. **Physical Correctness:** Linear interpolation of `u` and `v` components is physically realistic and accurately represents the vector nature of wind.
2. **Elimination of Artifacts:** This approach completely avoids the "phantom high wind" artifacts associated with circular interpolation of direction.
3. **Simplicity:** It simplifies the codebase by removing complex circular interpolation logic and replacing it with standard linear interpolation of Cartesian components.
4. **Consistency:** This is the standard approach used in meteorological data processing and NWP post-processing.

## Consequences
- Wind speed and direction must be reconstructed from the interpolated `u` and `v` components after the interpolation step.
- The `Nwp` schema now explicitly includes `wind_u` and `wind_v` components as `Float32` to support this logic.
- The `lead_time_hours` feature is now explicitly tracked to allow the model to learn the decay in NWP skill over time, which is particularly important for the 14-day (336h) forecast horizon.
