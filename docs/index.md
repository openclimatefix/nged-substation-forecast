# NGED Flexpectation

**NGED Flexpectation** is an NIA-funded project by [Open Climate Fix](https://openclimatefix.org/) to deliver state-of-the-art, probabilistic power forecasts for National Grid Electricity Distribution (NGED). The forecasts help NGED optimise flexibility procurement and manage network congestion.

![Example power forecast](example_power_forecast.png)

## What the forecasts look like

Each forecast is:

- **Probabilistic** — expressed as an ensemble of 51 members (one per ECMWF ENS member), or as percentiles
- **14-day horizon**, half-hourly temporal resolution
- Refreshed **every 6 hours**
- **Scaled to [−1, +1]** — a normalised value that NGED multiplies by the site's capacity to get MW/MVA
- **Sign convention** depends on the time series type:
    - *Substations*: positive = power flowing towards end-users; negative = excess generation flowing back into the grid
    - *Customer meters (generators)*: positive = generator sending power to NGED's grid; negative = customer consuming power

## Scope

**Version 1** (current focus): 32 time series in NGED's trial area — 16 primary substations, 6 solar PV farms, 3 wind farms, 2 GSPs, 2 BSPs, 1 biofuel generator, 1 BESS, and 1 reciprocating gas generator.

**Version 2** (future): Scale to approximately 2,500 time series covering all of NGED's primary substations and most customer meters.

## More than a forecast

A large part of this project is building a production forecasting system and researching novel
forecasting methods. But NGED's interest goes beyond the forecasts themselves: they also want
**information** — to learn which forecasting approaches actually work well on their data (a major
reason we invest in a rigorous [leaderboard](ml_experimentation/index.md)), and to understand the
underlying issues involved in forecasting their network.

That means a negative result can be just as valuable as a positive one. For example, if we try
hard to detect [switching events](background/switching-events.md) unsupervised and conclude it
isn't reliably possible from power readings alone, that's a useful finding in its own right — NGED
can use it as evidence to justify investing in extracting switching-event labels from their own
operational systems, rather than us silently working around the gap.

## Documentation

- [Background & Challenges](background/network.md) — NGED's network, project requirements, and data quality challenges
- [Architecture Overview](architecture/overview.md) — design philosophy, technical components, and data flow
- [Code Style](architecture/code-style.md) — code conventions
- [ML Experimentation](ml_experimentation/index.md) — methodology for our implemented ML experimentation: cross-validation folds, the leaderboard, and how we evaluate models
- [Live Forecasts](live_forecasts/index.md) — operating the live, 6-hourly production service: promoting a champion model and backfilling missed runs
- [Roadmap](roadmap/index.md) — planned future work, plus detailed design docs for the delivery tables, forecast building blocks, metrics & leaderboard, data sources, differentiable physics, switching events, disaggregation evaluation, and encoders

> Documentation convention: `roadmap/` holds **only not-yet-implemented** design. Once a feature ships, its docs move out to a permanent home (e.g. `architecture/` or `ml_experimentation/`), leaving a one-line pointer behind.
