# Production reference and best guess

**The summary of this theme is in progress.** This theme covers how the production forecast compares
with the best-guess recipes and with a perfect weather input. The [Experiments](experiments.md) page
lists every experiment tried, with its verdict and a take-home message, and the [Research
plan](plan.md) lists what is planned next.

## What would change in production

**N1 is the single-stage XGBoost quantile model trained on ECMWF ensemble (ENS) weather, and the
best guess is the project's current best forecasting recipe.**

- **[00019 - Retraining and error correction, part
  2](../../00019-primary-demand-stack-vs-manual-heuristic/result.md):** Add a 14-day error
  correction and a spread factor of 1.15 to N1 for Primary demand. The correction supplies nearly
  all of the gain: it lowers the quantile score by 0.99 points of capacity. The experiment tested
  the correction on forecasts made 5 days apart and on a model that was never retrained.
- **[00033 - Best guess against
  production](../../00033-best-guess-architecture-vs-production/result.md):** Replace the production
  recipe with the best guess: N1 with the 14-day correction and spread factor 1.15 for Primary
  demand, N1 for bulk supply point (BSP) and grid supply point (GSP) demand, and ERA5-trained
  mixtures for solar and wind. The experiment ranks the candidates against production. Monthly
  retraining without the correction scored 5.12 on one seed against 5.32 for the best guess, and
  needs a three-seed test.
- **[00087 - Best guess assembled
  (BG2)](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md):** Serve BG2, the
  second assembled best guess: weekly refits, a five-model blend for Primary demand, and weekly
  refits with no correction for supply points. BG2 lowers the quantile score by 0.283 points at
  Primaries and 0.634 at supply points compared with the previous best forecast. The farm models
  cannot yet be ranked.
