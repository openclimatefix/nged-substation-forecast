---
status: "Accepted"
date: "2026-04-02"
author: "Software Architect (gemini-3.1-pro-preview)"
tags: ["ml", "interpolation", "wind", "physics"]
---

# ADR-012: Cartesian Wind Vector Interpolation

## 1. Context & Problem Statement
Previously, wind speed and direction were interpolated using circular logic. This led to "phantom high wind" artifacts when the wind direction shifted rapidly (e.g., from 359 to 1 degree). For example, if the wind direction shifted from 359 to 1 degree, the circular interpolation might incorrectly calculate a large change in direction, leading to an artificially high wind speed during the transition.

## 2. Options Considered

### Option A: Circular Interpolation of Speed and Direction
* **Description:** Interpolate wind speed linearly and wind direction using circular math.
* **Why it was rejected:** This approach is physically unrealistic and prone to interpolation artifacts (phantom winds) during rapid direction shifts. It also requires complex custom logic for circular means.

### Option B: Cartesian Interpolation of U and V Components
* **Description:** Decompose wind into its orthogonal `u` and `v` components, interpolate these linearly, and reconstruct the vector.
* **Why it was selected:** It is the standard meteorological approach. It is physically correct, eliminates artifacts, and simplifies the interpolation logic by using standard linear methods.

## 3. Decision
We have shifted from circular interpolation of wind speed and direction to linear interpolation of Cartesian `u` and `v` components.

## 4. Consequences
* **Positive:** Physical correctness is ensured, and phantom wind artifacts are eliminated.
* **Positive:** Simplification of the core interpolation codebase.
* **Negative/Trade-offs:** Wind speed and direction must be reconstructed from the interpolated `u` and `v` components after the interpolation step.
* **Note:** The `Nwp` schema now explicitly includes `wind_u` and `wind_v` components as `Float32`.
* **Note:** The `lead_time_hours` feature is now explicitly tracked to allow the model to learn the decay in NWP skill over time.
