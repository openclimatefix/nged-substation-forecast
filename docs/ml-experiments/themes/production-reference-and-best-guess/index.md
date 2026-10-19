# Production reference and best guess

**The summary of this theme is in progress.** This theme covers how the production forecast compares
with the best-guess recipes and with a perfect weather input. The [Experiments](experiments.md) page
lists every experiment tried, with its verdict and a take-home message, and the [Research
plan](plan.md) lists what is planned next.

## What would change in production

**These experiments are flagged `candidate`, and none is adopted yet.** The flags are proposals for
the maintainer to confirm.

- **[00019 - Retraining and error correction, part
  2](../../00019-primary-demand-stack-vs-manual-heuristic/result.md):** Add a 14-day error
  correction and a spread factor of 1.15 to the single-stage ENS-trained quantile model N1 for
  Primary demand. The correction supplies nearly all of the gain (-0.99); it is tested on forecasts
  5 days apart and on a model that is never retrained.
- **[00033 - Best guess against
  production](../../00033-best-guess-architecture-vs-production/result.md):** Replace the production
  recipe with the best guess: N1 with the 14-day correction and spread factor 1.15 for Primary
  demand, N1 for BSP and GSP demand, ERA5-trained mixtures for solar and wind. The comparison that
  ranks the candidates against production; monthly retraining without the correction scored 5.12 on
  one seed and needs a three-seed test.
- **[00087 - Best guess assembled
  (BG2)](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md):** Serve the BG2
  recipe: weekly refits, a five-model blend for Primary demand, weekly refits with no correction for
  supply points. Lowers the score by 0.283 points at Primaries and 0.634 at supply points compared
  with the previous best; the farm models cannot yet be ranked.
