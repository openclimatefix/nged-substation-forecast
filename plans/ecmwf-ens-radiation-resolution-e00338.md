# Plan: NWP variable conventions and the 3 h → 6 h resolution change

**Status: docs landed on this branch. No code changed.** Two issues filed:
[#525](https://github.com/openclimatefix/nged-substation-forecast/issues/525) (store wind as u/v)
and [#526](https://github.com/openclimatefix/nged-substation-forecast/issues/526) (fix the
resample), both attached to the v0.5 epic #145 at positions 1 and 2.

## What was established

Verified against the archive, not inferred from the schema:

- ECMWF ENS steps are 3-hourly to 144 h and 6-hourly to 360 h.
- The three de-accumulated variables are period-ending means over the preceding step. Proved by
  forming the clear-sky index two ways over 7.16 M rows: dividing by the instantaneous clear-sky
  value gives a p99 of 2.02/2.27 (physically impossible); dividing by the clear-sky mean over the
  preceding step gives 0.99/0.96, capped at ~1.0 in both halves of the horizon.
- Wind direction is linearly interpolated in degrees, so 3.7 % (3-hourly) and 8.3 % (6-hourly) of
  native step pairs straddle North and are interpolated the long way round — exactly 180 ° wrong at
  the midpoint, 6.57 % of half-hourly rows. Live in the promoted feature list.
- Temperature loses 11.6 % of its mean daily range beyond day 6.
- Pressure, MSLP and geopotential lose essentially nothing (MAE/SD 0.02–0.09) — no action.
- Storing wind as u/v costs **+5.7 % to +6.3 %** of the whole table, measured through the production
  write path on two full runs. The round trip between the two representations is exact.
- The H3 spatial aggregation already works in u/v space, so there is no spatial wrap defect.

Scripts are in the session scratchpad; the storage benchmark is superseded by the larger experiment
specified in #525.

## Docs changed

| File | Change |
|---|---|
| `docs/architecture/nwp-variable-conventions.md` | **New.** Per-variable conventions, the three defects with measurements, the storage benchmark, the vector-vs-scalar mean note. Opens by delimiting itself from the known-issues page. |
| `docs/architecture/ecmwf-ens-known-issues.md` | Scope statement at the top: that page is upstream's faults, the new page is ours. |
| `docs/design-philosophy/design-principles.md` | New **principle 15**, appended (never inserted — the numbering is cross-referenced by position). Classified in the "before copying" section as general in reasoning, contingent in arithmetic. |
| `docs/roadmap/xgboost-improvements.md` | New lead section "Before anything else"; new Tier-1 item for raw u/v as features; stage (b) of the physics item reduced to a cross-reference; stage (c) gains the clear-sky-index denominator requirement and the PV-proxy convention note; horizon-focus preamble now says the 144 h change is a phase error, not only a resolution loss; gap-filling item cross-linked. |
| `docs/techniques/differentiable-physics.md` | Its existing timestamp-convention warning now points at the live instance of the same trap. |
| `packages/contracts/src/contracts/weather_schemas.py` | Docstrings only: `valid_time` gains the resampling requirement, both directions are marked circular, `wind_speed_10m` records that it is a vector mean. |
| `mkdocs.yml` | Nav entry. |

Verified: `pymarkdown` clean, `ruff check`/`format` clean, docstring markdown lint clean,
`mkdocs build --strict` clean, and every cross-referenced anchor confirmed present in the built
HTML.

## Recommended ordering, and why it is reversible

**#525 before #526.** Once wind is stored as components there is no wrapped angle for the resample
to handle, so #526 shrinks to radiation plus the diurnal question. Reversed, #526 has to build
convert-to-components-and-back machinery inside the resample that #525 then deletes.

The argument against, should you want it: #525 is gated on the larger storage experiment and an
overnight re-ingest, whereas #526's radiation half is pure feature-engineering code and could start
today. Each issue states the dependency in both directions, so the order can be flipped by editing
the two "Related" sections and the sub-issue positions.

## Open questions recorded rather than answered

- **Does a scalar-mean wind speed column earn its storage?** Today's stored speed is the magnitude
  of the cell's vector mean; a power curve wants the mean of the point-wise scalar speeds. Not
  recoverable from the archive. Jack's preference is u/v only; the gap needs measuring before that
  is settled. Recorded as the open question in #525.
- **Is `categorical_precipitation_type_surface` period-ending?** Unconfirmed, and not settleable
  from our own data. If it is, forward-filling it carries a full-step phase error. The new
  architecture page says "unconfirmed" rather than guessing.
