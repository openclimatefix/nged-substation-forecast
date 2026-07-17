# Techniques

This section holds durable explainers of the **solution methods** we bring to the forecasting
problem — what each technique is, how it works, and why it fits. It complements the other
sections: [background](../background/network.md) describes the *problem*, the
[roadmap](../roadmap/index.md) says *what* we plan to build and in what order, and
[architecture](../architecture/overview.md) documents what is *already built*. Unlike roadmap
pages, techniques pages are permanent: they stay put (and stay linkable — including from code
docstrings) as the work that applies them ships.

- [Convex optimisation](convex-optimisation.md) — the one-bowl property, why a certified global
  optimum beats gradient descent wherever it is available, exact zeros from ℓ1 penalties,
  hard limits as censoring, quantile envelopes, priors as penalties, and the project tooling rule
  (CVXPY for convex estimation subproblems; PyTorch for physics + learning). First applications:
  the edge-flow estimator on the
  [switching-events roadmap page](../roadmap/switching-events.md), and the convex candidate in
  the [v0.7 capacity head-to-head](../roadmap/capacity-estimation.md).
- [Probabilistic forecasting from NWP ensembles](probabilistic-forecasting.md) — why pushing an
  NWP ensemble through a deterministic power model is systematically overconfident
  (under-dispersed), the three-term uncertainty decomposition, the fix (per-member conditional
  quantile forecasts recombined by the linear-pool mixture), the pooling recipe, why
  quantile-averaging (Vincentization) re-creates the under-dispersion, and the double-counting
  caveat. Applied by the
  [probabilistic evaluation & calibration plan](../roadmap/metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
  and the [delivery tables](../roadmap/delivery-tables.md#table-1-power_forecast).
- [Evaluation metrics](evaluation-metrics.md) — every metric `compute_metrics` produces, each
  with a plain-language summary, the equation, and the design decisions behind it: the
  deterministic four (MAE, NMAE, RMSE, MBE), fair CRPS, the Fortin-corrected spread-skill
  ratio, pinball loss at the NGED delivery quantiles, PICP with its finite-ensemble calibrated
  references, and interval width. Applied by the
  [metrics & leaderboard plan](../roadmap/metrics-and-leaderboard.md) and computed by the
  `metrics` Dagster asset.
- [Differentiable physics](differentiable-physics.md) — inversion through differentiable forward
  models: the single-site solar plant and the aggregate fleet node. Applied by the
  [capacity-estimation](../roadmap/capacity-estimation.md) and
  [net-demand disaggregation](../roadmap/disaggregation.md) roadmap pages.
- [Learned encoders](encoders.md) — shared WeatherEncoder / TimeEncoder modules and why they pair
  naturally with differentiable physics.
- [Disaggregation evaluation](disaggregation-evaluation.md) — the multi-pronged protocol for
  evaluating disaggregation when no single clean ground truth exists.
