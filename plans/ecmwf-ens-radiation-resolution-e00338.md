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
  native step pairs straddle North and are interpolated the long way round — 180 ° wrong at the
  midpoint, ~100 ° on average across the interval, affecting 6.57 % of interpolated rows (5.80 % of
  all half-hourly rows). Live in the promoted feature list.
- Temperature loses 11.9 % of its mean daily range beyond day 6 (6.09 °C to 5.37 °C).
- Pressure, MSLP and geopotential lose essentially nothing (MAE/SD 0.02–0.09) — no action.
- Storing wind as u/v costs **+5.7 % to +6.3 %** of the whole table, measured through the production
  write path on two full runs. The round trip loses at most 6.8 × 10⁻³ ° once the components are
  rounded to the significand budget the table already applies.
- The H3 spatial aggregation already works in u/v space, so there is no spatial wrap defect.

Scripts are in the session scratchpad; the storage benchmark is superseded by the larger experiment
specified in #525.

## Docs changed

| File | Change |
|---|---|
| `docs/architecture/nwp-variable-conventions.md` | **New.** Per-variable conventions, the three defects with measurements, the storage benchmark, the vector-vs-scalar mean note. Opens by delimiting itself from the known-issues page. |
| `docs/architecture/ecmwf-ens-known-issues.md` | Scope statement at the top: that page is upstream's faults, the new page is ours. |
| `docs/design-philosophy/design-principles.md` | New **principle 15** — *transform data in feature engineering, not in the ingest, unless it saves a lot of storage* — appended (never inserted; the numbering is cross-referenced by position). Classified in the "before copying" section as general in reasoning, contingent in arithmetic. |
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

## Second review pass

A fresh sub-agent reviewed the branch and found a batch of real defects, all now fixed: an
overstated 180 ° wind-direction error (it is ~100 ° on average, 180 ° only at the interval
midpoint), the wrong denominator for the 6.57 % figure, a "three times worse" claim that the page's
own table put at 1.5×, an unreproducible direction MAE, a Nyquist claim that was wrong by a factor
of two, an overclaim that all four clear-sky requirements had been verified when one cannot be
tested on a clear-sky day, present-tense "are fixed" on a durable page, and two internal
contradictions in principle 15 (its *Decided* named an undecided decision, and its *Without it*
listed a spatial-averaging defect the architecture page says does not exist). It also found that
`continuous_var_names()` has two callers beyond the resample, which materially changes #526's scope.

One of its findings was rejected after re-measurement — the temperature compression was not
confounded across lead ranges; controlling for that gives 11.9 %, not the 13.5 % it reported. The
other rejection was wrong: I dismissed its per-column byte finding on the strength of two 40-file
samples, and summing every column chunk in the table (1,726 files, 6.25 billion rows, 98.85 GB)
gives **1.53 bytes/row**, not the 1.57 the page carried. The 19.3 % share and "most expensive
weather column" were right; the per-row figure was not.

## Third pass

A second independent review, told only that a first round had happened, found the class of defect
that revision introduces: a number corrected on one page and left stale on another. The roadmap
still claimed the solar day "peaks at 18:00 UTC" after the architecture page had been corrected to
call that framing dishonest; #525 pointed at a principle-15 anchor renamed on this branch; the
opening sentence said "six hours" where the window is three or six, and "one angle" where there are
two; the 11.9 % figure was labelled "beyond day 6" when it is a controlled emulation over days 1-6;
and "a clear-sky index above 1 is physically impossible" is simply false, because cloud enhancement
is real — it also sat in the same bullet as an instruction to clip at 1.2. It also found a third
dependent of `continuous_var_names()` (the contract's partition test) and argued that defect (c) is
a limit of ECMWF's 6-hourly output rather than a bug in our code, which is right and is now how both
the roadmap and #526 describe it.

## Left for Jack's judgement

Principle 15's admission. The page's own test asks a principle to name a decision it actually
decided, and principle 15's flagship decision (#525) is deliberately still open pending the storage
experiment; the two decisions it does name (H3 aggregation, significand rounding) were already made
and are already claimed by principles 6 and 12. The counter-argument is that it decides the
*default* for every future ingest transform, and that it is what put the wind conversion up for
review at all. Worth deciding whether that clears the bar.
