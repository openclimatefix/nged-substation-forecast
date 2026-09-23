# Background

**This section is the context every other page assumes: NGED's electricity network, what NGED has
asked for, how distribution network operators have traditionally forecast, and the two limits —
measurement artefacts in the telemetry and a fragmented literature — that shape the rest of the
design.** [Design philosophy](../design-philosophy/index.md) states the principles that answer to
this problem; [architecture](../architecture/overview.md) describes what is built to solve it.

- [NGED's network and its data](network.md) — the primary substations, bulk supply points, and grid
  supply points NGED operates, the generation connected to them, and the false zeros, stuck values,
  missing-data gaps, and months-long commissioning ramps in the trial-area telemetry.
- [Weather products surveyed for the weather-product studies](weather-products-survey.md) — which
  weather products the project uses, and the forecast models, reanalyses, and satellite retrievals
  that could join the weather-product studies, with what each carries. A survey, not a list of what
  the project will ingest.
- [Requirements](requirements.md) — the phased rollout from 32 time series to roughly 2,500, and the
  objectives NGED prioritises.
- [The manual heuristic forecast](manual-heuristic-forecast.md) — the historical-analogue method
  that, until recently, was the normal approach among distribution network operators, with no
  weather model and no machine learning, and the baseline our own forecasts are measured against.
- [Switching events](switching-events.md) — why NGED's meshed network is operated as a radial tree,
  and why reconfiguring that tree moves load between substations without warning.
- [The state of the art in energy forecasting](energy-forecasting-review.md) — what the published
  literature does and does not settle, and where Flexpectation's plan sits against it.
