# Plan: which ERA5 variables help explain how much sunlight reaches a solar farm?

Status: signed off by the maintainer after two agentic reviews. A `Spike` issue under the studies
epic carries this plan as its body. No code has been written and nothing has been
downloaded.

## Question

**If an XGBoost model is given every ERA5 variable that could plausibly matter, does it predict
solar PV output better than an XGBoost model given the minimal ERA5 set?** The same question is
asked of a second target, CAMS global horizontal irradiance (GHI) at the same six solar farms,
because CAMS is a satellite retrieval and shows how much sunlight reached the ground without the
panel, inverter, or curtailment in the way. The scientific question behind both is which ERA5
variables carry information about how much sunlight gets through the atmosphere that ERA5's own
downward short-wave radiation (`ssrd`) does not.

## Why ERA5 at all, when the aim is an IFS-driven forecast

**The study's ultimate aim is to choose which ECMWF IFS variables to feed a solar power forecast.**
ERA5 is the stand-in because it is the only archive that carries all the candidate variables for
2019-09 to 2026-09. The Data DOWNLOAD COORDINATOR checked what IFS products we can get:

- **16 of the 34 candidate variables** are in the free ECMWF feed (open data, and the Dynamical.org
  archive built from it): `ssrd`, `strd`, `t2m`, `d2m`, `u10`, `v10`, `sp`, `tp`, `tcc`, `tcwv`,
  `skt`, `sd`, `sf`, `asn`, `mucape` (not `cape`), and the gust field (`10fg`, not `i10fg`).
- **6 more** are on Open-Meteo's 9 km IFS (`lcc`, `mcc`, `hcc`, `blh`, `cin`, `fdir`; whether the
  cloud layers and `fdir` are native or derived is unverified).
- **12 are only in the full MARS archive:** `ssrdc`, `cdir`, `tclw`, `tciw`, `tcslw`, `cbh`, `tcrw`,
  `tcsw`, `tco3`, `uvb`, `fal`, `deg0l`. They are "worth asking ECMWF for", not candidate features.
- **Archive depth is short:** Open-Meteo's `ecmwf_ifs` starts 2024-03-14, Dynamical.org ENS
  2024-04-01, and the Source Cooperative backfill covers about 2021-03 to 2024-03 (14 surface
  fields, no `fdir`).

**ERA5 is therefore an upper-bound screen, not the training source.** ERA5 runs a frozen 2016 model
(IFS cycle 41r2) with hourly fields from short forecasts, so its cloud and radiation are better than
an IFS forecast's at day 1 to 14 leads. A variable that helps ERA5 may add only noise at day 3. The
page reports each result against the variable's IFS availability (free feed, Open-Meteo, MARS-only),
so the reader sees which winners are usable. Confirming any winning rung on matched-lead IFS
forecasts is a follow-up study, outside this plan.

**Aerosol makes ERA5 and the IFS differ in clear-sky irradiance.** Per the Opus aerosol review, the
operational IFS uses a fixed monthly aerosol climatology and not prognostic aerosol: the CAMS
interim-reanalysis climatology (2003 to 2013, 3° grid) since cycle 43r3, revised again in cycle 50r1
(operational 12 May 2026; the note does not say which climatology). ERA5 uses the older Tegen et al.
(1997) climatology, with a CMIP5 sulphate trend. Neither model sees an individual dust or smoke
event. CAMS satellite irradiance (Heliosat-4: McClear for clear sky, McCloud for cloud extinction)
does use CAMS aerosol analyses and forecasts every 3 hours. The ERA5-versus-IFS climatology mismatch
is a second reason to train the production forecast on IFS forecasts, whatever this study finds.

## Arms: a ladder of variable groups, each adding one physical idea

The ladder comes from the Opus variable review (the brief and its report are in
`.claude/worktrees/era5-solar-variables-brief.md` and this conversation). Every rung contains the
rungs below it.

| Rung | Adds | ERA5 variables (derived features in brackets) |
|---|---|---|
| G0 minimal | the "normal" set | `ssrd`, `t2m`, plus top-of-atmosphere flux and solar geometry from `studies.pv_dataset.add_solar_geometry` (a midpoint-zenith estimate of the top-of-atmosphere flux, close to but not identical with ERA5's hour-integrated `tisr`), and the clearness index `ssrd / extraterrestrial_horizontal_w_m2` |
| G1 cloud amount | total cloud | `tcc` |
| G2 cloud layers | low, medium, high cloud | `lcc`, `mcc`, `hcc` |
| G3 clear-sky normalisation | how bright the sky would be without cloud | `ssrdc` (clear-sky index `ssrd / ssrdc`) |
| G4 cloud optical thickness | water and ice in the cloud | `tclw`, `tciw`, `tcslw`, `cbh` |
| G5 beam and diffuse | direct beam against scattered light | `fdir`, `cdir` |
| G6 panel temperature | convective cooling and thermal radiation | `u10`, `v10`, `strd` |
| G7 humidity and haze | moisture in the air column | `d2m`, `tcwv`, `blh` |
| G8 snow and albedo | snow on the panel, ground reflection | `sd`, `sf`, `asn`, `fal` |
| G9 everything plausible | the remaining ERA5 variables | `tp`, `tcrw`, `tcsw`, `cape`, `cin`, `skt`, `tco3`, `uvb`, `i10fg`, `sp`, `deg0l` |
| G10 aerosol (not ERA5) | event-level aerosol, which neither ERA5 nor the IFS carries | CAMS EAC4 total aerosol optical depth at 550 nm and dust optical depth at 550 nm |

**`ceil` and `hcct` are not ERA5 variables.** The CDS download form (checked 2026-10-08) lists
neither. `cbh` is the nearest available field to a ceiling, and no ERA5 field gives a convective
cloud-top height, so G4 and G9 cover what the maintainer's list wanted as far as ERA5 can.

**Aerosol is the biggest gap in ERA5 for this question.** ERA5 has no aerosol optical depth, and its
radiation scheme uses a climatology (see "Why ERA5 at all"). CAMS satellite irradiance uses CAMS
aerosols. G10 tests this directly: it adds CAMS EAC4 total and dust aerosol optical depth at 550 nm,
which are not ERA5 fields. The expected sign is a gain on the CAMS target (CAMS sees aerosol, ERA5
cannot) and a small gain on the PV target (aerosol optical depth over Great Britain is usually low,
about 0.1 to 0.2, from the reviewer's memory). Any ERA5-versus-CAMS gap that survives G10 is not
explained by aerosol. **G10 is exploratory, and only useful in production if the forecast also
receives CAMS forecast aerosol, a second live feed.** EAC4 ends in 2025, so G10 is compared with a
G9 arm refitted on exactly the G10 rows, with folds cut on that shorter span, and no second CAMS
product is spliced in. EAC4 is 3-hourly at 0.75°: the aerosol join interpolates linearly in time to
labels H−1 and H, takes the mean of the two, and reads the EAC4 cell nearest each farm. About 4 MB
in total.

**Every rung is a nested superset, so the number of feature columns differs between rungs.** XGBoost
runs with `colsample_bytree=1` so a wider rung gets no free win from column subsampling. Two
controls (below) measure what the extra columns do on their own.

## Targets

1. **PV target:** hourly mean output of each of the six NGED solar farms, as a percentage of the
   farm's capacity (its 99th percentile of metered output), for the hour ending at the label. One
   XGBoost model per farm, as in the past-weather solar page.
2. **CAMS target:** CAMS GHI (W m⁻², hour ending at the label) at each farm's location, read from
   the existing `reanalysis/CAMS` downloads. The target is the clearness index (GHI divided by
   top-of-atmosphere flux), and the page also reports error in W m⁻². The CAMS target has no
   curtailment and no panel, so the contrast between the two targets separates "the atmosphere" from
   "the panel".

**Gains from aerosol, `tcwv`, and `tco3` on the CAMS target are partly by construction.** McClear
computes CAMS GHI from CAMS aerosol, water vapour, and ozone, which come from the same
IFS-based assimilation family as ERA5's fields. Conclusions about those variables rest on the PV
target. Which CAMS aerosol product McClear uses (EAC4 or the operational analysis) is unverified.

## Rows, folds, and metric

- **Rows:** the daylight hours (top-of-atmosphere flux above a threshold in W m⁻², fixed in the
  script before any fit and stated on the page) on which the PV target and the CAMS target both
  exist, within the download span. Rows are set by the targets, the clock, and the span only, never
  by an ERA5 value. The build raises if any ERA5 value other than `cbh` and `cin` is missing on
  those rows. ERA5 sets `cbh` (and probably `cin`) missing where there is no cloud, so those two
  keep NaN, which XGBoost treats as missing, and each goes through its own `hourly_from_snapshots`
  call (the hourly value is NaN unless both snapshots exist). The first fetched chunk reports each
  variable's share of NaN, split by `tcc` below and above 0.05.
- **ERA5 release:** the span ends at the last month that is final ERA5 (`expver` 0001) in every
  file. ERA5T months (`expver` 0005) can be replaced by ECMWF. The fetch records `expver` per hour,
  and the build raises if `expver` differs between variables for one hour.
- **Snow censoring:** dropping hours that hold a zero half-hour also drops fully snow-covered
  panels, which read exactly zero. G8's PV result and the snow days in the surprises figure are
  therefore biased towards no effect. The page states this beside G8, and an exploratory G8 arm
  keeps zero hours where `sd > 0`.
- **PV cleaning:** reuse `studies.power`, `studies.export_cap`, and `studies.commissioning`, as the
  past-weather solar page does. Hours holding a zero half-hour are dropped, from the power table,
  for both targets.
- **Folds:** contiguous blocks of whole months (`studies.cross_validation.assign_folds`). No UKV era
  split is needed, because every input is ERA5. Recent ERA5 months can be the preliminary ERA5T
  release; the build records which months are ERA5T.
- **Clearness index:** it is unstable at low sun, so the daylight threshold above also bounds it.
  The CAMS target's clearness index uses the hour-integrated top-of-atmosphere value that the CAMS
  files carry, and W m⁻² is reported beside it.
- **Metric:** mean absolute error is the main metric: as a percentage of capacity (PV), and as a
  clearness-index error and in W m⁻² (CAMS). Pearson correlation between out-of-fold prediction and
  measured value, pooled over each fold's rows, is reported beside it for every rung (exploratory),
  with intervals from the same month-resampling. Each farm's error is normalised by its own capacity
  before any mean or difference.
- **Intervals:** `studies.bootstrap.bootstrap_difference`, 2,000 resamples of whole calendar months,
  paired across rungs, one of the three fitting seeds per resample. The page explains once what the
  test covers and what it does not.
- **Hour convention table:** every ladder variable is classed once as an accumulation (`ssrd`,
  `ssrdc`, `fdir`, `cdir`, `strd`, `tp`, `sf`, `uvb`) or as instantaneous (every other variable). A
  test checks each variable has exactly one class. The fetch checks the hour-of-day profile of the
  mean absolute hour-to-hour change for steps at two families of seams: cloud and cloud-water fields
  (from the 06 and 18 UTC forecasts) at 06/07 and 18/19 UTC, and analysed fields (`t2m`, `d2m`,
  `u10`, `v10`, `sp`, `skt`, `tcwv`, `sd`) at the 4D-Var window boundaries of 09 and 21 UTC.
- **Hour convention:** `ssrd`, `ssrdc`, `fdir`, `cdir`, `strd`, and the other accumulations are
  means over the hour ending at the label, the same as the PV and CAMS stamps. Instantaneous fields
  (clouds, water columns, `t2m`, `d2m`, `sd`, `blh`, `cape`) are averaged over labels H−1 and H so
  the hour matches, using `studies.hourly_means.hourly_from_snapshots(slot_offsets_minutes=(-60,
  0))`.
- **Pairing guard:** before each contrast the code raises unless both arms hold the identical set of
  (site, time, seed) rows, because `paired_differences` inner-joins silently. Every arm runs on the
  same device, recorded in the losses table.
- **Second hyperparameter setting** (`SENSITIVITY_HYPER_PARAMETERS`) on every planned contrast, with
  the two verdicts combined by `combine_setting_verdicts`.
- **GPU** (`device="cuda"`) if `nvidia-smi` shows one, with the device stated on the page. No CPU
  refit is run for a noise floor.

## Planned contrasts (written before any result exists)

Each is run on both targets, so there are eight planned contrasts:

1. **P0:** G9 (everything) minus G0. The study question. It equals P2 plus P3.
2. **P1:** G1 (adds `tcc`) minus G0. Does total cloud help at all beyond `ssrd` and `t2m`?
3. **P2:** G2 minus G0. Do the three cloud layers help beyond the minimal set?
4. **P3:** G9 minus G2. Does anything beyond the cloud covers help? This is the maintainer's
   headline question.

**Planned verdicts use a Bonferroni-adjusted level.** With eight planned contrasts, the family-wise
level of 5% becomes 0.625% per contrast (a 99.375% interval, `bootstrap_difference_at_level`). Every
interval is also shown at 95%, labelled exploratory. The page states each planned result as "rules
out a gain larger than X" using the upper bound, with a smallest effect of interest fixed before any
result (0.1 percentage points of capacity for PV, 0.01 for the clearness index).

Every other number is exploratory: each rung against G0, each rung against the one below it, the
drop-one-group runs, the per-farm numbers, and the regime and season splits. The page labels them so
and does not correct them for multiple comparisons.

## Controls

- **Negative control:** G9's added columns permuted by `studies.blending.climatology_permutation`
  over site, calendar month, and hour of day, so the column count and each column's distribution
  match G9. The permutation removes the hour-to-hour information and keeps each month's mean at each
  hour, which is month-level weather information, so the control is not strictly information-free.
  All G3 to G9 columns are passed as one column group, so their joint distribution survives. A
  contrast of this arm against G2 shows how large a difference the pipeline produces from nothing.
- **Positive control:** G2 plus CAMS GHI itself as an input on the PV target. This is the known
  good answer: a column that must help. If the pipeline cannot see this gain, a null for G9 is not
  evidence of no effect. (This arm is not run on the CAMS target, where it would be the target.)
- **Known-answer step for the CAMS target:** the same GHI-target model is first fitted with only
  `ssrd` and solar geometry, so the page shows how much of CAMS GHI the standard ERA5 field already
  explains before any extra variable is tried.

## Second instrument: drop one group from the full set

The ladder is order-dependent: a group that adds nothing after G2 may add a lot if it came first.
The drop-one-group run starts from G9 and removes one rung's variables at a time, so the page can
say for each group both what it adds on top of the minimal set (ladder) and what is lost without it
(drop-one). A group is called useful only if the two instruments agree. Grouped permutation
importance on out-of-fold rows is a third, exploratory view.

## Figures: the page tells its story through figures

The page is mostly figures, in the order below. Each has a bolded one-sentence lead and a few
sentences of support. Every chart is anonymised: farms are A to F, outputs are normalised by
capacity, and no coordinate appears.

1. **Headline (top of page).** The planned contrasts P0 to P3 on both targets, at the adjusted
   level, with the smallest effect of interest marked.
2. **The leaderboard.** Every rung's own mean absolute error with a 95% interval on the PV target
   and the CAMS target, best first, G0 and the two controls included. A second panel shows Pearson
   correlation for every rung.
3. **What the variables look like.** Three days at a farm, chosen by a stated rule (the clearest,
   the most variable, the dullest): stacked small multiples of `ssrd`, `ssrdc`, CAMS GHI, PV output,
   the cloud covers, `tclw` and `tciw`, `cbh`, and `fdir`. The time axis is shared.
4. **Cloud covers against a cloud index.** CAMS clearness index (or the PV capacity factor against
   clear-sky) plotted against `tcc`, and against `lcc`, `mcc`, and `hcc`.
5. **Where ERA5's `ssrd` misses CAMS.** ERA5 minus CAMS GHI as a time series for the same days, then
   binned against each candidate variable (`tclw`, `tciw`, `cbh`, `tcwv`, `d2m`, `blh`, `sd`). This
   is the scientific question, shown before any model.
6. **Cloud water against optical thickness.** `tclw + tciw` against CAMS clearness index, coloured
   by low-cloud cover.
7. **The XGBoost models work.** Out-of-fold PV against measured for the three stated-rule weeks at
   every farm, G0 against G9, and each rung's error per farm.
8. **Weather regimes.** The difference in error between G9 and G0, and between each of G1 and G2 and
   G0, split by regime: clear sky, broken cloud, and overcast. Regimes are set from the CAMS
   clear-sky index with thresholds fixed before any result, and a second panel splits by the ERA5
   cloud cover `tcc` for readers who want an ERA5-only definition. Exploratory.
9. **Seasons.** The same differences by season (winter, spring, summer, autumn), and for the
   clear-sky, broken-cloud, and overcast regimes within each season. Exploratory.
10. **Drop-one-group.** Error added when each group is removed from G9.
11. **Hour of day and snow.** Error by hour of day for G0 against G9, and the worst 20 days for G0
    with what G9 changed on them, anonymised by farm label.

## Data and code

- **Reuse the existing ERA5 download for G0.**
  `data/studies/downloads/reanalysis/ERA5/beam_diffuse/` already holds `ssrd`, `fdir`, and `t2m`
  over a 20-cell box for 2019-09 to 2026-09 from CDS, plus a CDS-versus-Open-Meteo check. CAMS GHI
  and site-level PV tables exist already (`reanalysis/CAMS/`, the `studies.power` loaders,
  `studies.pv_dataset`).
- **The new download is every other variable** in the table, from CDS, over the same 20-cell box and
  hours. CDS takes one job at a time, about 2.5 minutes per month of three variables, and 121,000
  fields per request (one variable-hour is one field, so one variable for the whole 85 months is
  about 62,000 fields). The Opus reviewer estimated about 38 hours of serial queue time for all the
  new variables (31 in the tiers below, so about 36 hours at the measured rate). The plan splits the
  fetch into two tiers so the first results do not wait for all of it:
    - **Tier 1 (needed for G1 to G7):** `tcc`, `lcc`, `mcc`, `hcc`, `ssrdc`, `cdir`, `tclw`, `tciw`,
      `tcslw`, `cbh`, `u10`, `v10`, `strd`, `d2m`, `tcwv`, `blh`. About 19 hours.
    - **Tier 2 (needed for G8 and G9):** `sd`, `sf`, `asn`, `fal`, `tp`, `tcrw`, `tcsw`, `cape`,
      `cin`, `skt`, `tco3`, `uvb`, `i10fg`, `sp`, `deg0l`. About 18 hours. `tisr` is not downloaded:
      the top-of-atmosphere flux comes from `add_solar_geometry`.
- **Tier 3 (not ERA5, from the Atmosphere Data Store):** CAMS EAC4 total and dust aerosol optical
  depth at 550 nm, 3-hourly, 2019-09 to the end of EAC4 (the Atmosphere Data Store catalogue gave 31
  December 2025 on 2026-10-08; the fetch sets the end month from the listing). A few MB and a
  handful of requests. The ADS account is separate from the CDS account; the CAMS irradiance
  download already used it.
- **The fetch is the `data-download` skill's job:** resumable, one pilot month first, with the
  `data-validation` checklist on the first chunk and again after the last. The fetch script also
  checks the hour-of-day profile of every accumulation for a step at the 07 and 19 UTC seams, where
  ERA5's accumulations change forecast run.
- **Code:** `studies/era5_solar_variables/` for the scripts (`fetch_era5_variables.py`,
  `build_dataset.py`, `fit_ladder.py`, `report.py`, charts), and anything a second study might use
  (the clearness-index and aerosol-join functions) goes into `packages/studies/src/studies/` with
  tests. The page goes under Studies > Past weather, as
  `docs/studies/past-weather/era5-solar-variables.md`.
- **Order of work:** issue (type `Spike`), then fetch script review, pilot month, tier 1 fetch,
  build and first report, first Opus science review, tier 2 fetch and full report, charts and draft
  page, second Opus science review, diff review, prose review and persona reviews, merge.

## Tests for the new `packages/studies` functions

- **Clearness index:** zero top-of-atmosphere flux gives null, not infinity, and rows below the
  threshold are excluded.
- **Accumulation to power:** 3600 J m⁻² over the hour is exactly 1 W m⁻².
- **Variable classification table:** every ladder variable has exactly one class.
- **Ladder:** each rung is a strict superset of the rung below, and G9 is the union.
- **Aerosol join:** a linear ramp in 3-hourly values gives the analytic hour-ending mean, and a
  3-hour shift of the input fails the test.
- **Pairing guard:** a contrast between arms with different row sets raises.

## The five complexity triggers (for sizing)

1. **Changes what gets stored:** yes, a published page and new files under `data/studies/`.
2. **Touches the production serving path:** no.
3. **Touches a degradation rule:** no.
4. **Admits more than one defensible design:** yes (ladder order, hour convention, the CAMS target
   definition, tiered download).
5. **Spans code whose callers could not be named without searching:** no, but the study imports
   `packages/studies` machinery that has to be read first.

Size: complex, as every study is. The study skill's process applies in full, with two Opus science
reviews, because the page will carry numbers.

## Decisions the maintainer has made

1. **Download span:** 2019-09 to 2026-09 for every variable. The experiments are expected to show
   little signal, so the study wants as much data as it can get.
2. **CAMS target:** the clearness index, with W m⁻² reported beside it.
3. **Aerosol:** include CAMS EAC4 aerosol optical depth as the eleventh rung (G10).
4. **MARS-only variables:** keep all 12. The study's purpose includes telling the team whether
   fetching them from MARS is worth the effort.
5. **Smallest effect of interest:** 0.1 percentage points of capacity on the PV target, and 0.01 on
   the clearness index on the CAMS target.
6. **Daylight threshold:** top-of-atmosphere horizontal flux above 50 W m⁻².
7. **GitHub issue:** a `Spike` issue under the studies epic, with the full plan as its body, created
   after the maintainer signs off and the plan has had its two agentic reviews.

## Review 1 (simplicity) triage

Accepted: `tisr` already in G0 through `add_solar_geometry`, so it is not downloaded and G3 keeps
`ssrdc` only; derived features cut to the clearness index; grouped permutation importance cut; the
label-H sensitivity run cut; the CPU noise-floor refit cut (also the maintainer's instruction); the
2026 aerosol splice cut; the negative control reuses `climatology_permutation`; the headline absorbs
the step-change and controls figures; the hour averaging names `hourly_from_snapshots`.

Rejected, with the reason:

- **Collapse the ladder to four rungs.** The maintainer asked for the ladder (minimal, then total
  cloud, then cloud layers, then more), so the intermediate rungs stay as exploratory rows.
- **Make the CAMS target exploratory.** The maintainer asked for both targets.
- **Cut `fdir` and the beam/diffuse features.** `fdir` is available from Open-Meteo's IFS and stays
  as a plain input in G5. The derived DNI, DHI, and plane-of-array features are cut under the
  derived- feature item above.

Decided by the maintainer: **keep the 12 MARS-only variables.** The study exists to tell the team
whether the effort of fetching these variables from MARS is worth it, so cutting them would remove
the answer.

## Review 2 (correctness and testability) triage

Accepted, all as text changes to this plan: the row-set rules for `cbh` and `cin`; the `expver`
release rule; the G10 refit on its own rows and the EAC4 end month; the shared-input caveat for the
CAMS target; the snow censoring; the clearness-index threshold and CAMS top-of-atmosphere source;
the smallest effect of interest and the Bonferroni level; P0; the hour-convention table and the two
seam families; the negative-control wording and single column group; the EAC4 interpolation with its
test; the pairing guard; the second-setting verdict rule; one device. Unverified claims the reviewer
flagged (the Tegen climatology for ERA5, the 12 May 2026 date for IFS cycle 50r1, Heliosat-4's
3-hourly aerosol, the 0.1 to 0.2 aerosol optical depth, the IFS availability lists) are labelled
unverified in the issue and checked before the page states them.
