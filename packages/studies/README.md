# `studies`

**The machinery a one-off study calls, held to the repository's normal standard while the studies
themselves are not.** `studies/README.md` says which code moves here, and why.

## What this package owns, and what it does not

It owns the pieces that more than one study needs and that fail silently when wrong: the
out-of-fold XGBoost loop and the paired bootstrap every comparison is read through, solar geometry,
the checks that establish what temporal object a downloaded column holds, the half-hourly-to-hourly
power aggregation, the mapping from a meter to an anonymous label, sampling a projected weather grid
at a set of coordinates, the Fractions Skill Score, and the dot-and-interval chart form the study
pages share, with the parser that reads a study report's contrast tables into it.

It does not own the question, the arms, which charts a page draws, or the write-up. Those stay in
the study, as scripts, because a one-off comparison is not a library and should not be dressed as
one. It does not own downloads either: a fetcher needs the network to exercise, which is what the `--run-network`
gate covers, so the fetchers stay in the study alongside the arms they serve.

Two neighbouring packages own things a reader might look for here. `geo` owns H3 indexing, including
the mapping from a coordinate to a cell. `contracts` owns every data schema, including
`POWER_TIMESTAMPS_CORRECTED_BEFORE` and the account of which window a power timestamp names.

## The modules

- `cross_validation` — per-site, out-of-fold XGBoost fits and their per-row losses, with the fold
  scheme and the fixed hyperparameters they use.
- `bootstrap` — the paired arm-to-arm difference and its interval, resampling whole months and a
  seed.
- `anonymise` — the one mapping from a meter's `time_series_id` to the anonymous label a chart or a
  write-up may carry.
- `solar` — solar position and the extraterrestrial flux, for a series of timestamps at one
  coordinate.
- `served_column_checks` — two assertions about a downloaded irradiance column: that the hourly
  value is a backward mean over the hour ending at its label, and that a published direct fraction
  carries information a separation model applied to the total would not.
- `power` — the half-hourly-to-hourly aggregation, on the period-ending convention
  `contracts.PowerTimeSeries` states.
- `grid_sampling` — the nearest grid cell's value at each site, on a projected grid.
- `fractions_skill_score` — a timing-tolerant score, which asks whether a forecast put a threshold
  exceedance near the right hour rather than exactly on it.
- `charts` — the dot-and-interval chart panel, the figure caption, the colour of each product
  family, and the parser that reads a study report's contrast tables.
