# Background

**This section is the context every other page assumes: NGED's electricity network, what NGED
has asked for, how NGED forecasts today, and the two limits — messy data and a fragmented
literature — that shape the rest of the design.** [Design philosophy](../design-philosophy/index.md)
states the principles that answer to this problem; [architecture](../architecture/overview.md)
describes what is built to solve it.

- [NGED's network](network.md) — the primary substations, bulk supply points, and grid supply
  points NGED operates, and where the trial area sits inside that network.
- [Requirements](requirements.md) — the phased rollout from 32 time series to roughly 2,500, and
  the objectives NGED prioritises.
- [NGED's incumbent forecast](nged-incumbent-forecast.md) — the historical-analogue method NGED
  uses today, with no weather model and no machine learning, and the baseline our own forecasts
  are measured against.
- [Data quality](data-quality.md) — the false zeros, stuck values, and missing-data gaps in
  NGED's trial-area telemetry.
- [Switching events](switching-events.md) — why NGED's meshed network is operated as a radial
  tree, and why reconfiguring that tree moves load between substations without warning.
- [The state of the art in energy forecasting](energy-forecasting-review.md) — what the
  published literature does and does not settle, and where Flexpectation's plan sits against it.
