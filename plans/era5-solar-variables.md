# Plan: which ERA5 variables help explain how much sunlight reaches a solar farm?

Status: draft for maintainer review. No issue exists yet; no code has been written; nothing has been
downloaded.

## Question

**If an XGBoost model is given every ERA5 variable that could plausibly matter, does it predict solar
PV output better than an XGBoost model given the minimal ERA5 set?** The same question is asked of a
second target, CAMS global horizontal irradiance (GHI) at the same six solar farms, because CAMS is
a satellite retrieval and shows how much sunlight reached the ground without the panel, inverter, or
curtailment in the way. The scientific question behind both is which ERA5 variables carry
information about how much sunlight gets through the atmosphere that ERA5's own downward
short-wave radiation (`ssrd`) does not.

## Arms: a ladder of variable groups, each adding one physical idea

The ladder comes from the Opus variable review (the brief and its report are in
`.claude/worktrees/era5-solar-variables-brief.md` and this conversation). Every rung contains the
rungs below it.

| Rung | Adds | ERA5 variables (derived features in brackets) |
|---|---|---|
| G0 minimal | the "normal" set | `ssrd`, `t2m` (plus solar geometry, see below) |
| G1 cloud amount | total cloud | `tcc` |
| G2 cloud layers | low, medium, high cloud | `lcc`, `mcc`, `hcc` |
| G3 clear-sky normalisation | how bright the sky would be without cloud | `ssrdc`, `tisr` (clearness index, clear-sky index) |
| G4 cloud optical thickness | water and ice in the cloud | `tclw`, `tciw`, `tcslw`, `cbh` (optical-depth proxy, liquid fraction) |
| G5 beam and diffuse | direct beam against scattered light | `fdir`, `cdir` (beam share, DNI, DHI, plane-of-array irradiance) |
| G6 panel temperature | convective cooling and thermal radiation | `u10`, `v10`, `strd` (wind speed, Faiman module temperature) |
| G7 humidity and haze | moisture in the air column | `d2m`, `tcwv`, `blh` (relative humidity) |
| G8 snow and albedo | snow on the panel, ground reflection | `sd`, `sf`, `asn`, `fal` (snow-on-panel flag) |
| G9 everything plausible | the remaining variables | `tp`, `tcrw`, `tcsw`, `cape`, `cin`, `skt`, `tco3`, `uvb`, `i10fg`, `sp`, `deg0l` |

**`ceil` and `hcct` are not ERA5 variables.** The CDS download form (checked 2026-10-08) lists
neither. `cbh` is the nearest available field to a ceiling, and no ERA5 field gives a convective
cloud-top height, so G4 and G9 cover what the maintainer's list wanted as far as ERA5 can.

**Aerosol is the biggest gap in ERA5 for this question.** ERA5 has no aerosol optical depth, and
(from the Opus reviewer's memory, to be verified before the page says it) its radiation scheme uses a
monthly aerosol climatology. CAMS satellite irradiance uses CAMS aerosols. Any ERA5-versus-CAMS gap
that survives all ten rungs is therefore a candidate for aerosol, and the page says so as a
hypothesis, not a finding. An optional exploratory extra (see "Open questions") adds CAMS EAC4
aerosol optical depth as an eleventh rung.

**Every rung is a nested superset, so the number of feature columns differs between rungs.** XGBoost
runs with `colsample_bytree=1` so a wider rung gets no free win from column subsampling. Two controls
(below) measure what the extra columns do on their own.

## Targets

1. **PV target:** hourly mean output of each of the six NGED solar farms, as a percentage of the
   farm's capacity (its 99th percentile of metered output), for the hour ending at the label. One
   XGBoost model per farm, as in the past-weather solar page.
2. **CAMS target:** CAMS GHI (W m⁻², hour ending at the label) at each farm's location, read from
   the existing `reanalysis/CAMS` downloads. The target is the clearness index (GHI divided by
   top-of-atmosphere flux), and the page also reports error in W m⁻². The CAMS target has no
   curtailment and no panel, so the contrast between the two targets separates "the atmosphere" from
   "the panel".

## Rows, folds, and metric

- **Rows:** the daylight hours (top-of-atmosphere flux above a threshold set in the script) on which
  the PV target and the CAMS target both exist, and no ERA5 variable is missing. Every rung scores
  exactly the same rows; the filters read the target and the clock only, never an ERA5 variable.
- **PV cleaning:** reuse `studies.power`, `studies.export_cap`, and `studies.commissioning`, as the
  past-weather solar page does. Hours holding a zero half-hour are dropped, from the power table, for
  both targets.
- **Folds:** contiguous blocks of whole months (`studies.cross_validation.assign_folds`). No UKV era
  split is needed, because every input is ERA5. Recent ERA5 months can be the preliminary ERA5T
  release; the build records which months are ERA5T.
- **Metric:** mean absolute error as a percentage of capacity (PV) and as a clearness-index error and
  in W m⁻² (CAMS). Each farm's error is normalised by its own capacity before any mean or difference.
- **Intervals:** `studies.bootstrap.bootstrap_difference`, 2,000 resamples of whole calendar months,
  paired across rungs, one of the three fitting seeds per resample. The page explains once what the
  test covers and what it does not.
- **Hour convention:** `ssrd`, `ssrdc`, `fdir`, `cdir`, `strd`, `tisr`, and the other accumulations
  are means over the hour ending at the label, the same as the PV and CAMS stamps. Instantaneous
  fields (clouds, water columns, `t2m`, `d2m`, `sd`, `blh`, `cape`) are averaged over labels H−1 and
  H so the hour matches. A sensitivity run uses the label-H value only.
- **Second hyperparameter setting** (`SENSITIVITY_HYPER_PARAMETERS`) on every planned contrast and
  any result near the 5% line.
- **GPU** (`device="cuda"`) if `nvidia-smi` shows one, with one published rung refit on CPU to report
  the device noise floor.

## Planned contrasts (written before any result exists)

Each is run on both targets, so there are six planned contrasts:

1. **P1:** G1 (adds `tcc`) minus G0. Does total cloud help at all beyond `ssrd` and `t2m`?
2. **P2:** G2 minus G0. Do the three cloud layers help beyond the minimal set?
3. **P3:** G9 (everything) minus G2. Does anything beyond the cloud covers help? This is the
   maintainer's headline question.

Every other number is exploratory: each rung against G0, each rung against the one below it, the
drop-one-group runs, the per-farm numbers, and the sensitivity runs. The page labels them so and does
not correct for multiple comparisons.

## Controls

- **Negative control:** G9's columns with each added column shuffled across rows within its calendar
  month, so the column count and each column's distribution match G9 and no information is added. A
  contrast of this arm against G2 shows how large a difference the pipeline produces from nothing.
- **Positive control:** G2 plus CAMS GHI itself as an input on the PV target. This is the known
  good answer: a column that must help. If the pipeline cannot see this gain, a null for G9 is not
  evidence of no effect. (This arm is not run on the CAMS target, where it would be the target.)
- **Known-answer step for the CAMS target:** the same GHI-target model is first fitted with only
  `ssrd` and solar geometry, so the page shows how much of CAMS GHI the standard ERA5 field already
  explains before any extra variable is tried.

## Second instrument: drop one group from the full set

The ladder is order-dependent: a group that adds nothing after G2 may add a lot if it came first. The
drop-one-group run starts from G9 and removes one rung's variables at a time, so the page can say
for each group both what it adds on top of the minimal set (ladder) and what is lost without it
(drop-one). A group is called useful only if the two instruments agree. Grouped permutation
importance on out-of-fold rows is a third, exploratory view.

## Figures: the page tells its story through figures

The page is mostly figures, in the order below. Each has a bolded one-sentence lead and a few
sentences of support. Every chart is anonymised: farms are A to F, outputs are normalised by
capacity, and no coordinate appears.

1. **Headline (top of page).** The ladder: mean absolute error per rung for the PV target and the
   CAMS target, with 95% intervals, G0 at the top, and below it the three planned contrasts.
2. **What the variables look like.** Three days at a farm, chosen by a stated rule (the clearest,
   the most variable, the dullest): stacked small multiples of `ssrd`, `ssrdc`, CAMS GHI, PV output,
   the cloud covers, `tclw` and `tciw`, `cbh`, and `fdir`. The time axis is shared.
3. **Cloud covers against a cloud index.** CAMS clearness index (or the PV capacity factor against
   clear-sky) plotted against `tcc`, and against `lcc`, `mcc`, and `hcc`. Shows how much of the
   scatter the layers explain.
4. **Where ERA5's `ssrd` misses CAMS.** ERA5 minus CAMS GHI as a time series for the same days, then
   binned against each candidate variable (`tclw`, `tciw`, `cbh`, `tcwv`, `d2m`, `blh`, `sd`). The
   panel for each variable shows whether the residual moves with it. This is the scientific
   question, shown before any model.
5. **Cloud water against optical thickness.** `tclw + tciw` (and the optical-depth proxy) against
   CAMS clearness index, coloured by low-cloud cover.
6. **The XGBoost models work.** Out-of-fold PV against measured for the three stated-rule weeks at
   every farm, G0 against G9, and each rung's error per farm.
7. **The ladder rung by rung.** Each rung's step change in error (PV and CAMS) with intervals.
8. **Drop-one-group.** Error added when each group is removed from G9.
9. **Where the gain comes from.** Error by clear-sky-index bin, season, and hour of day for G0
   against G9, so the page shows whether gains are in broken cloud, fog, snow, or low sun.
10. **The controls.** The negative-control and positive-control contrasts beside the planned ones.
11. **Surprises.** The worst 20 days for G0 and what G9 changed on them (snow and fog days
    especially), anonymised by farm label.

## Data and code

- **Reuse the existing ERA5 download for G0.** `data/studies/downloads/reanalysis/ERA5/beam_diffuse/`
  already holds `ssrd`, `fdir`, and `t2m` over a 20-cell box for 2019-09 to 2026-09 from CDS, plus a
  CDS-versus-Open-Meteo check. CAMS GHI and site-level PV tables exist already
  (`reanalysis/CAMS/`, the `studies.power` loaders, `studies.pv_dataset`).
- **The new download is every other variable** in the table, from CDS, over the same 20-cell box and
  hours. CDS takes one job at a time, about 2.5 minutes per month of three variables, and 121,000
  fields per request (one variable-hour is one field, so one variable for the whole 85 months is
  about 62,000 fields). The Opus reviewer estimated about 38 hours of serial queue time for all the
  new variables (31 in the tiers below, so about 36 hours at the measured rate). The plan splits the fetch into two tiers so the first results do not wait for all
  of it:
    - **Tier 1 (needed for G1 to G7):** `tcc`, `lcc`, `mcc`, `hcc`, `ssrdc`, `cdir`, `tclw`, `tciw`,
      `tcslw`, `cbh`, `u10`, `v10`, `strd`, `d2m`, `tcwv`, `blh`. About 19 hours.
    - **Tier 2 (needed for G8 and G9):** `sd`, `sf`, `asn`, `fal`, `tp`, `tcrw`, `tcsw`, `cape`,
      `cin`, `skt`, `tco3`, `uvb`, `i10fg`, `sp`, `deg0l`. About 18 hours. `tisr` is computed, not
      downloaded.
- **The fetch is the `data-download` skill's job:** resumable, one pilot month first, with the
  `data-validation` checklist on the first chunk and again after the last. The fetch script also
  checks the hour-of-day profile of every accumulation for a step at the 07 and 19 UTC seams, where
  ERA5's accumulations change forecast run.
- **Code:** `studies/era5_solar_variables/` for the scripts (`fetch_era5_variables.py`,
  `build_dataset.py`, `fit_ladder.py`, `report.py`, charts), and anything a second study might use
  (derived-feature functions such as clearness index, Faiman module temperature, and relative
  humidity) goes into `packages/studies/src/studies/` with tests. The page goes under Studies >
  Past weather, as `docs/studies/past-weather/era5-solar-variables.md`.
- **Order of work:** issue (type `Spike`), then fetch script review, pilot month, tier 1 fetch, build
  and first report, first Opus science review, tier 2 fetch and full report, charts and draft page,
  second Opus science review, diff review, prose review and persona reviews, merge.

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

## Open questions for the maintainer

1. **Download span.** Default: 2019-09 to 2026-09 for all variables, matching the existing ERA5 and
   CAMS records. If the PV record is much shorter, the span could be cut to the PV record plus a
   margin, saving up to several hours of queue time. I have not yet checked each farm's PV start
   date.
2. **CAMS target definition.** Default: clearness index, with W m⁻² reported beside it. Alternative:
   W m⁻² only.
3. **Aerosol rung.** Include CAMS EAC4 aerosol optical depth as an optional eleventh rung? It is not
   ERA5, so it is outside the question as asked. Default: leave it out, and say on the page that
   aerosol is the main gap ERA5 cannot fill.
4. **GitHub issue.** Create a `Spike` issue under the past-weather epic before work starts? Default:
   yes, once the plan is agreed.
