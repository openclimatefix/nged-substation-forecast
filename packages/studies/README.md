# `studies`

**The machinery a one-off study calls, held to the repository's normal standard while the studies
themselves are not.** A study under `studies/` answers a question once, and the tier it belongs to
promises no tests, no maintenance and no backwards compatibility. Code moves into this package when
that promise is too weak for it: when a study already got this wrong once, or when getting it wrong
would produce a plausible-looking number rather than an error.

## What this package owns, and what it does not

It owns the pieces that more than one study needs and that fail silently when wrong: solar geometry,
the checks that establish what temporal object a downloaded column holds, the half-hourly-to-hourly
power aggregation, the mapping from a meter to an anonymous label, and the Fractions Skill Score.

It does not own the question, the arms, the charts, or the write-up. Those stay in the study, as
scripts, because a one-off comparison is not a library and should not be dressed as one. It does not
own downloads either: a fetcher needs the network to exercise, which is what the `--run-network`
gate covers, so the fetchers stay in the study alongside the arms they serve.

Two neighbouring packages own things a reader might look for here. `geo` owns H3 indexing, including
the mapping from a coordinate to a cell. `contracts` owns every data schema, including
`POWER_TIMESTAMPS_CORRECTED_BEFORE` and the account of which window a power timestamp names.

## The modules

- `anonymise` — the one mapping from a meter's `time_series_id` to the anonymous label a chart or a
  write-up may carry. NGED's generator data leaves the project anonymised only, and a second
  derivation of the mapping is a second chance to disagree with the first.
- `solar` — solar position, the extraterrestrial flux, and the clearness index, for a frame of
  timestamps at a set of coordinates.
- `served_column_checks` — two assertions about a downloaded irradiance column: that the hourly
  value is a backward mean over the hour ending at its label, and that a published direct fraction
  carries information a separation model applied to the total would not. Each raises rather than
  warns, because a study that trains on a misread column produces a number nobody can distinguish
  from a real one.
- `power` — the half-hourly-to-hourly aggregation, on the period-ending convention
  `contracts.PowerTimeSeries` states.
- `fractions_skill_score` — a timing-tolerant score, which asks whether a forecast put a threshold
  exceedance near the right hour rather than exactly on it. [Evaluation
  metrics](https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/#fractions-skill-score-fss)
  explains what the score measures and what limits it.
