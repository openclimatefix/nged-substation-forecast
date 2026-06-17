# NGED Flexpectation

**NGED Flexpectation** is an NIA-funded project by [Open Climate Fix](https://openclimatefix.org/) to deliver state-of-the-art, probabilistic power forecasts for National Grid Electricity Distribution (NGED). The forecasts help NGED optimise flexibility procurement and manage network congestion.

![Example power forecast](example_power_forecast.png)

## What the forecasts look like

Each forecast is:

- **Probabilistic** — expressed as an ensemble of 51 members (one per ECMWF ENS member), or as percentiles
- **14-day horizon**, half-hourly temporal resolution
- **Updated every 6 hours**
- **Scaled to [−1, +1]** — a normalised value that NGED multiplies by the site's capacity to get MW/MVA
- **Sign convention**:
    - *Substations*: positive = power flowing towards end-users; negative = excess generation flowing back into the grid
    - *Customer meters (generators)*: positive = generator sending power to NGED's grid; negative = customer consuming power

## Scope

**Version 1** (current focus): 32 time series in NGED's trial area — 16 primary substations, 6 solar PV farms, 3 wind farms, 2 GSPs, 2 BSPs, 1 biofuel generator, 1 BESS, and 1 reciprocating gas generator.

**Version 2** (future): Scale to approximately 2,500 time series covering all of NGED's primary substations and most customer meters.

## Documentation

- [Architecture & Philosophy](architecture/philosophy.md) — design goals and principles
- [Architecture Overview](architecture/overview.md) — technical components and data flow
- [Code Style](architecture/code-style.md) — code conventions
- [Guides](guides/create-forecast.md) — how to run forecasts, backtests, and add models
- [Roadmap](roadmap.md) — planned future work
