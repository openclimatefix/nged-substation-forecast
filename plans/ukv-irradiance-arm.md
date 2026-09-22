# Add a UKV arm to the beam/diffuse split experiment (#800)

**The problem.** The [beam/diffuse split experiment](https://openclimatefix.github.io/nged-substation-forecast/results/beam-diffuse-split/) found that a weather product's published direct-beam field cut PV power error by 1.8% on the CAMS satellite retrieval and by nothing detectable on ERA5. The page declines to say why, because CAMS and ERA5 differ in resolution, in delivery, and in production method all at once. Nothing in that experiment can tell those three apart.

**The planned solution.** Add the Met Office's UKV as a third irradiance source, taken from Open-Meteo's historical-forecast archive of `ukmo_uk_deterministic_2km`, running the same six arms on the same six meters. UKV is a numerical weather prediction model at 2 km that publishes its own direct and diffuse fields, so ERA5 against UKV is a resolution contrast inside one product class. Two checks gate the arm before any model is trained: a lineage check against the Met Office's own files on AWS, and an explicit treatment of UKV's instantaneous radiation against ERA5's hourly integral. Three departures from the issue body are set out below, the largest being that the issue's preferred temporal treatment is the wrong one.

## Verdict: worth implementing, with three departures

**The issue is worth implementing roughly as described, but it overstates what UKV settles.** Its premise checks out against the code and the published page: `docs/results/beam-diffuse-split.md` does decline to attribute the CAMS-against-ERA5 gap to resolution, and the experiment's structure — a `--source` axis, arms defined only by irradiance columns, a per-site source path already built for CAMS — takes a third source with no redesign. The three-way confound the issue names is real and UKV does narrow it. **UKV does not reduce it to resolution alone, because UKV and ERA5 also differ in how they treat aerosol**, which is set out in the next section and is a finding the issue does not contain.

**Departure 1: the primary temporal treatment is Open-Meteo's default hourly value, not the `_instant` variants the issue prefers.** The experiment's target is a period-ending hourly mean of the meter's power. ERA5 and CAMS both publish an hourly integral, which is the same temporal object. Open-Meteo's default UKV hourly value is the trapezoid of the snapshots at the start and end of the hour, which is an *estimator of that same object* — a noisy one, but the same quantity. The `_instant` variant is a point sample at one instant, which is a different quantity from the target and from the other two sources. Preferring `_instant` would introduce the mismatch the issue is trying to avoid rather than remove it. The default is therefore primary and `_instant` is the pre-registered sensitivity, which is the only way to measure how far the two-point trapezoid strays under broken cloud.

**Departure 2: the lineage check is a gate on the work, not a check worth running, and it cannot cover the whole archive.** `docs/roadmap/data-sources.md` records, from a bucket listing rather than from documentation, that the AWS UKV archive starts on 2024-09-19. Open-Meteo's archive starts on 2022-03-01. **Over half of Open-Meteo's UKV history sits in a period the Met Office's own open archive no longer holds, so no sampling against AWS can check it.** The verifiable era runs 2024-09-19 to now and does contain the PS47 boundary, so the check the issue asks for is possible on the recent half and impossible on the older half. This plan makes the verifiable era the primary UKV span and the full span a sensitivity run, and flags the choice as the first question for the human reviewer.

**Departure 3: ICON-D2 is planned as a follow-up issue rather than a second arm here.** Costed below. The short reason is that ICON-D2 cannot meet the lineage standard this plan applies to UKV, because DWD's open-data server holds about a day of ICON-D2 and there is no multi-year native archive to sample against.

**One mechanism in the issue body is kept unchanged**: the arm structure, the five load-bearing constraints (`colsample_bytree` at 1.0, the negative control, the piecewise stamp shift, the export-cap clamp, and anonymisation), and the rule that every published number comes from a script.

## What UKV assimilates, and the two confounds it leaves

Findings in this section come from a research sub-agent run on 2026-09-22, which was asked to separate Met Office documentation from its own inference and to say what it could not establish. Its open questions are carried into the risks section below.

**UKV's radiation scheme carries no time-varying aerosol, which the issue does not account for and which puts a second difference into the ERA5-against-UKV contrast.** The regional configuration's radiation uses a fixed five-species climatology, unchanged from RAL1 through the RAL3 package that went operational at PS47 — [Bush et al. (2020)](https://doi.org/10.5194/gmd-13-1999-2020) describes the climatology, and [Bush et al. (2025)](https://doi.org/10.5194/gmd-18-3819-2025) records that no radiation parameters changed between RAL2 and RAL3. UKV's only advected aerosol quantity is the Murk tracer of [Clark et al. (2008)](https://doi.org/10.1002/qj.318), which diagnoses visibility and does not reach the radiation calculation. CAMS is aerosol-informed by construction and ERA5 carries a time-varying assimilated aerosol field, so **any skill UKV's split shows can only be cloud-sourced, where the other two products' splits could be cloud-sourced or aerosol-sourced.** ERA5 against UKV is therefore a contrast in resolution *and* in aerosol treatment, rather than the clean resolution contrast the issue claims.

**The aerosol asymmetry sits mostly in the bin where the published result is weakest, and the existing machinery already reports that bin.** Aerosol sets the direct-diffuse partition most strongly under a clear sky, and the published page finds the clear-sky bin is where the split helps least. `sky_conditions.py` already splits the headline contrast by clearness index, so the run reports the bin the asymmetry would show up in without any new script. The write-up states the asymmetry rather than the plan designing around it.

**UKV assimilates a large volume of satellite-derived cloud, so at short lead times it is partly a retrieval — which makes the lead time the archive is built from a load-bearing design question rather than a quality check.** Satellite-derived cloud fraction was the single largest observation type by count in the UKV's 2013 observation table, at 650,000 a day against 39,000 for SEVIRI radiances ([Tubbs and Kelly (2013)](https://www-cdn.eumetsat.int/files/2020-04/pdf_conf_p_s7_09_tubbs_v.pdf)), entering the humidity field as a pseudo-observation by the mechanism of [Renshaw and Francis (2011)](https://doi.org/10.1002/qj.980). The Met Office's [PS43 release notes](https://www.metoffice.gov.uk/services/data/met-office-data-for-reuse/ps43_ftp) confirm the stream was still operational and being refined in December 2019, and the PS47 notes name only latent heat nudging and the adaptive vertical grid as the assimilation methods removed. The sub-agent found no document from 2020 onwards positively re-confirming the satellite cloud stream, and says so.

**What follows is that a T+0 UKV analysis and the CAMS retrieval are both downstream of the same geostationary satellite, so a UKV arm built from T+0 would not give the "forecast against retrieval" contrast the issue wants.** The contrast recovers as lead time grows and the model's own physics overwrites the initial cloud field. This is why `verify_ukv_lineage.py` below sweeps lead times rather than checking one, and why the answer it returns may change the design rather than merely annotating it.

## Size: complex — the full routine

Answering each of the five triggers rather than the one that fired:

| Trigger | Answer |
|---|---|
| Does it change what gets stored? | **No.** No Patito model, no Delta table, no asset, no contracted parquet. Everything written lands in the git-ignored `data/` tree, outside the asset graph. |
| Does it touch the production serving path? | **No.** `scripts/experiments/` is imported by nothing in `src/` or `packages/`, adds no package, and enters no Dagster asset. |
| Does it touch a degradation rule? | **No.** No asset check, no warning path, nothing in `defs/`. R&D code throughout, which fails fast by design. |
| Does more than one design defensibly satisfy it? | **Yes.** The temporal treatment has three defensible answers and the issue picks a different one from this plan. The span question has three (verifiable era, full archive, pre-PS47 only). Each changes what the experiment measures. |
| Could every caller be named without searching? | **No.** The source name is duplicated as an `argparse` `choices` literal in 14 scripts, which a search found and a reading of the two obvious entry points did not. |

Two triggers fire, so the issue is **complex**: this plan, both plan reviews, and both diff reviews in `implement-issue`.

## What changes, file by file

Everything is on branch `ukv-irradiance-arm`, cut from `beam-diffuse-split-experiment` rather than from `main`, because #785 is deliberately unmerged and its branch must be kept. The pull request's base is `beam-diffuse-split-experiment` for the same reason: based on `main` it would show all 28 of #785's scripts as additions.

**The simplicity review cut this section to about a third of its first draft.** What survived, what was cut, and what was rejected are recorded under "What the simplicity review changed" below.

### New: `sources.py`, a stdlib-only leaf module

Holds the source list, the set of sources delivered per site rather than on a grid, and a registry of the Open-Meteo models this experiment can fetch — for each one its `models=` value, its source name in paths, its archive start date, and its native temporal convention.

**It imports nothing outside the standard library, which is what makes it importable by every script in the directory.** `era5_grid.py` already sets this precedent: it holds shared constants behind `from typing import Final` alone and is imported by four scripts including standalone ones. A registry in `build_dataset` would not work, because that module pulls in `pvlib`, `xarray` and a Delta store, and `elevation_breakdown.py` and `report_results.py` document run commands supplying only `polars`.

**This reverses part of the simplicity review, on new information.** The review was right that hoisting the constant into `build_dataset` breaks two scripts, and that generalising for a caller that does not exist is waste. A second Open-Meteo source is now committed rather than hypothetical, and a stdlib-only leaf module is a mechanism the review did not consider, so the constant earns its place: adding ICON-D2 becomes one registry entry rather than a 13-place edit repeated. The 13 scripts import `SOURCE_CHOICES` from `sources` and drop their own literal.

### New: `fetch_open_meteo_point.py`, roughly 120 lines

**One fetcher parameterised by model, rather than a UKV-specific script, because ICON-D2 is planned as a follow-on.** It takes `--model` and looks the rest up in the `sources.py` registry. Open-Meteo normalises variable names across models, so `shortwave_radiation`, `direct_radiation` and `diffuse_radiation` are the same request for UKV and for ICON-D2; only the `models=` value, the output path and the temporal convention differ. It downloads at each meter's own coordinates from `https://historical-forecast-api.open-meteo.com/v1/forecast`, and for UKV requests each radiation field's `_instant` variant as well. Writes `data/<SOURCE>/beam_diffuse_<source>.parquet`, one row per `(site, time)`.

**The `_instant` request is driven by the registry's temporal-convention field rather than hard-coded.** UKV's native radiation is an instantaneous snapshot, so Open-Meteo's default hourly value is a trapezoid of two of them and the `_instant` columns are worth having as a diagnostic. **ICON-D2's native fields are accumulated since model start, so Open-Meteo de-accumulates them to a genuine hourly mean, and ICON-D2 would not carry UKV's sub-hourly sampling problem at all.** That asymmetry is why the convention belongs in the registry rather than in the fetcher's body — but it is recorded as established for UKV and *to be measured* for ICON-D2, since the follow-on issue has to verify it rather than inherit this plan's assumption.

It takes its API handling from `fetch_era5_open_meteo.py`, which already speaks this endpoint and this `models=` parameter, and its per-site anonymisation from `fetch_cams.py`: import `_pv_sites` from `build_dataset`, read each meter's coordinates at run time, send them to Open-Meteo, and write only the anonymised label. No coordinate and no identifier reaches the written frame.

**Three checks run at ingest, and each one can void the experiment on its own.** They cost no model fits and no S3 access, and all three cover the whole archive rather than the era AWS can check.

1. **`direct + diffuse − total` stays inside Open-Meteo's rounding.** A larger departure means the three fields are not one consistent split, which would void arm C.
2. **The published direct is not a separation model's output.** Correlate it against the Erbs beam, which `build_dataset._add_separation_models` already computes for every source. If Open-Meteo served a decomposition of global irradiance rather than UKV's native field, arm C would be a copy of arm B and a null result would be guaranteed by construction. This is the failure `verify_era5_sources.py` exists to rule out for ERA5, and a correlation gets it here for two lines rather than a second download. **It also covers the 2022-to-2024 era that no AWS sampling can reach**, which makes it the more important of the two lineage checks rather than the cheaper substitute for one.
3. **The hourly label is period-ending.** A lagged cross-correlation of UKV global irradiance against the ERA5 frame at −1, 0 and +1 hours. The pipeline treats every label as period-ending, and the issue's own arithmetic implies Open-Meteo's UKV is too, but that is inferred rather than measured on our own rows. **This project has already lost a fortnight to a half-hour stamp offset, and an off-by-one-hour on UKV would be the same failure in a new place.** The simplicity review contributed this check; the first draft did not have it.

### New: `verify_ukv_lineage.py`, roughly 60 lines

**This script is UKV-specific and does not generalise to the ICON-D2 follow-on, which is a fact about the data rather than about the code.** DWD's open-data server holds roughly a day of ICON-D2, so no multi-year native archive exists to sample against, and a `--model` flag here would promise a capability that cannot exist. The follow-on issue carries an unverified lineage as a stated caveat instead.

It samples a small number of valid instants, pulls the matching native files from `s3://met-office-atmospheric-model-data/uk-deterministic-2km/` with `obstore`, opens them with `xarray`, takes the nearest grid point to each meter, and compares against Open-Meteo at the same instant. One file covers the whole 970×1042 domain, so a single read serves all six meters at once.

**The comparison is against the `_instant` columns, never the default hourly ones.** The native file holds an instantaneous snapshot; Open-Meteo's default is a trapezoid of two of them. Comparing the trapezoid against the snapshot would make a faithful mirror look broken, which is the single easiest way to get this script wrong.

**The script sweeps lead time, because the assimilation finding above makes it a design question rather than a quality check.** For each sampled instant it pulls T+0, T+3 and T+6 from the AWS runs valid then and reports which lead Open-Meteo matches. **If the answer is T+0, the UKV arm is built on a cloud field anchored to the same geostationary satellite CAMS retrieves from, and the between-product question changes meaning.** The remedy in that case is to fetch from the AWS bucket at a fixed longer lead, which is available for the verifiable era and costs a heavier download. The plan does not commit to it in advance, because which lead Open-Meteo serves is not yet known.

**Sampling covers both sides of the PS47 boundary on 2026-01-21, rather than a single date.** The issue records that the AWS encoding changed at PS47, so Open-Meteo's ingest could have changed there too. That makes roughly 12 to 18 reads: a handful of instants each side, times three lead times. It is not a stratified sweep across the whole era, which is what the first draft proposed and what the review rightly called oversized.

**This stays a gate: no model is trained on UKV until it has run and been read.** That is a direct instruction rather than the plan's own preference.

### `build_dataset.py` — four edits

- `"ukv"` into `SourceType` and into the `--source` `choices` literal.
- `UKV_PATH` beside `CAMS_PATH`.
- `_read_ukv()`, mirroring `_read_cams()`, roughly 20 lines. It selects the default hourly columns, or the `_instant` columns when a variant build asks for them.
- The per-site branch generalised: `if source == "cams"` becomes a membership test over a `PER_SITE_SOURCES` tuple, and the gridded read's `"open-meteo" if source == "cams"` becomes the same test.

**Air temperature keeps coming from the gridded ERA5 frame, including for the UKV build.** UKV publishes its own `temperature_2m`, and using it would change a shared non-irradiance feature between sources, breaking the invariant that arms and sources differ only in the irradiance columns. CAMS already takes ERA5's temperature for exactly this reason.

**No `--ukv-temporal` flag.** `build_dataset.py` already has `--suffix` for precisely this, and it already composes into the run paths of all six downstream scripts — the `cams-allhours` variant on disk is the same mechanism in use. A second temporal build, if ever wanted, is `--source ukv --suffix -instant`.

**Nothing restricts or widens the UKV span in code.** `_read_era5` trims at `era5_grid.LAST_DATE`, and `main()` inner-joins power to the gridded frame before the per-site irradiance join, so the UKV dataset is automatically ERA5's rows intersected with whatever span the download covers. Both span runs are a matter of which download is on disk, not of a code path.

### The 13 scripts carrying the duplicated `choices` literal

Each drops its `("cds", "open-meteo", "cams")` tuple for `from sources import SOURCE_CHOICES`. Every one of them already imports a sibling or can, and `sources.py` adds no third-party dependency, so the two scripts documented to run on `polars` alone keep working. Each is run once under its own documented command as part of the verification set, because that is the claim being relied on.

### No `ps47_breakdown.py`

Cut. PS47 landed on 2026-01-21 and the record ends on 2026-09-10, so the post-upgrade era is 7.7 months of a 54.3-month archive — about 14% of rows, giving an interval roughly three times the full-span width against a published effect of 0.096 points. **And the split would not be attributable even if it were powered**, because the pre-upgrade era spans whole years while the post-upgrade era runs January to September with no autumn and no early winter, so a pre-and-post difference confounds the science upgrade with season. The write-up names PS47, its date, the 14% share, and the seasonal imbalance in one sentence instead.

### No change to `compare_sources.py`

Its source arguments carry no `choices`, so `--first-source ukv --second-source open-meteo` and `--first-source ukv --second-source cams` both run today. Two pairwise invocations answer the issue's between-product question with no edit. **Generalising to a three-way intersection would change the row set under the already-published CAMS-against-ERA5 comparison**, so the write-up would have to restate those numbers on a new basis or carry two row sets and explain the difference. The script already prints each pair's shared-hour count.

### `README.md` in the experiment directory

The two new scripts in the run-order table, UKV in the sources table, and a short section on the temporal difference, the aerosol asymmetry, and the span the lineage check cannot cover.

### The write-up is a separate pull request against `main`

`docs/results/beam-diffuse-split.md` lives on `main` and not on this branch, exactly as #786 was separate from #785. The scripts land here; the page is updated in its own pull request once the numbers exist.

## Design-philosophy check

**This is R&D code, so it fails fast rather than degrading.** None of it runs in production, none of it enters the Dagster asset graph, and the inherent-stability rules in `docs/design-philosophy/inherent-stability.md` govern `defs/` rather than `scripts/`. A missing download raises `FileNotFoundError` naming the script to run first, which is what every existing fetcher in the directory does. No asset check is added or edited, so the `WARN`/`blocking=False` rule has nothing to bind to here.

**The change is not aimed at an engineering hypothesis.** It measures the information content of a weather product's published field, which is an input-selection question rather than a claim about the serving architecture, so no label in `docs/design-philosophy/engineering-hypotheses.md` applies.

**No principle in `design-principles.md` is traded away.** The one point of contact is the preference for a single source of truth, and hoisting `SOURCE_CHOICES` moves towards it rather than away.

**The anonymisation rule is the binding constraint and it is honoured in three places**: the fetcher sends coordinates and writes only labels, the lineage script writes differences rather than site-keyed values beside anything identifying, and every table and chart continues to carry the seeded shuffle `build_dataset._pv_sites` applies. No site name, `time_series_id`, coordinate, or per-site megawatt figure reaches any output.

## Tests

**This directory carries no pytest tests and should not gain any.** `scripts/` is linted by ruff and checked by `ty` — neither excludes it — but the repository has no test for any of #785's 28 scripts, because they are throwaway code whose output is audited by other scripts rather than by assertions. Adding a suite here would be the first of its kind and would outlive nothing.

**What stands in for tests is three checks that would fail today, each printed by a script:**

1. **The lineage check.** `verify_ukv_lineage.py` compares Open-Meteo's UKV against the Met Office's own files. It has no counterpart on `main` or on the branch, and the claim it settles — that the mirror serves UKV's native fields rather than a decomposition — is currently supported by one timestamp.
2. **The split's internal consistency.** The fetcher asserts `direct + diffuse − total` stays inside Open-Meteo's rounding. A source failing it would void arm C, and nothing checks it today.
3. **The negative control, re-measured on UKV.** Arm B is a deterministic function of arm A's features, so any B−A gap on UKV is that pipeline's re-encoding floor and the band the UKV headline has to clear. The existing run measures this on ERA5 and CAMS and has never measured it on UKV.

**The temporal sensitivity is the fourth, and it is a comparison rather than an assertion**: the trapezoid run against the `_instant` run, which no existing output contains.

## Docs to update

- `scripts/experiments/beam_diffuse_split/README.md` — the two new scripts, the UKV source, the temporal axis.
- `docs/results/beam-diffuse-split.md` — in its own pull request against `main`, as set out above.
- `docs/roadmap/data-sources.md` — currently records UKV's AWS archive start and its licence. If the lineage check establishes what Open-Meteo's archive is built from and over what span it can be trusted, that belongs on this page, written as present-tense fact rather than as a record of what this issue found.
- No ship-time triage applies: this issue completes no roadmap item and the experiment directory carries no "Implementation details" section.

## Verification commands

The green-before-push set, all from the worktree root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

Plus, for this change specifically:

- `uv run pymarkdown scan scripts/experiments/beam_diffuse_split/README.md`, which the default scan's path list does not reach.
- Each of the three ingest checks in `fetch_ukv_open_meteo.py`, whose output goes in the pull request body.
- `verify_ukv_lineage.py` run to completion, with its table pasted into the pull request body. **This is the gate: no model is trained on UKV until it has run and been read.**
- The experiment runs, which the timestamps of the existing outputs put at roughly 12 minutes for the XGBoost instrument, 29 for the physical model, and 15 for the hybrid, so about an hour of compute per span on the 32-core workstation, plus the dataset build and the diagnostics. Two spans means about three hours in total, unattended.
- Each script this change touches run once under its own documented command, since 13 of them are edited and two of them deliberately run without their siblings on the path.

`mkdocs build --strict` is not needed on this branch, which touches no page MkDocs renders; it is needed on the write-up pull request.

## Risks and open questions

**Which UKV span should the primary run use, and do the two runs agree?** The question is less which run is primary than whether the two agree, and running both settles it for about 90 minutes of unattended compute and no code, because the span is a date filter on the download rather than a code path.

Counted on the existing dataset — month counts are exact and are what the block bootstrap resamples, while row counts come from the CAMS build and so run about 20% low for UKV, which carries no reliability filter:

| Span | Rows | Months | Detection threshold |
|---|---|---|---|
| Published CAMS span, 2019-09 to 2026-09 | 128,033 | 85 | 0.018 |
| UKV full archive, 2022-03 to 2026-09 | 90,968 | 55 | ~0.023 |
| UKV AWS-verifiable era, 2024-09-19 to 2026-09 | 41,041 | 25 | ~0.033 |

The threshold column scales the published headline's half-width of 0.018 points by the square root of the ratio of months.

**If UKV behaves like CAMS the span does not matter**, because an effect of 0.096 points excludes zero on either span. The spans diverge only for small effects, which is the outcome most worth planning for: "UKV's split is worth something, but less than the 5 km retrieval's" lands somewhere around 0.03 to 0.06 points. **The verifiable era's threshold of 0.033 sits above the re-encoding floor of 0.029 that arm B minus arm A measures**, so on 25 months an effect at floor size cannot be told from zero and a null there would be uninterpretable. The full archive's 0.023 sits below the floor and can tell them apart.

*Recommendation: report the full archive as the headline, the verifiable era as the lineage check, and the agreement between them as the evidence that the headline is safe to read.* If the two point estimates agree, the era no AWS sampling can reach is not behaving differently from the era that can be checked, which is evidence about the risk obtained for nothing. **If they disagree materially, that is the finding**, because it would mean the pre-2024 archive is a different product — the failure this whole exercise exists to avoid.

**Is Open-Meteo's free tier compatible with this project's use?** It is non-commercial only. This is not new to this issue — Open-Meteo already feeds the *published* ERA5 numbers on the docs page, since `DEFAULT_SOURCE` is `open-meteo` — so it is a live question about work already shipped rather than a gate on this one. *Recommendation: settle it separately, and note that UKV has a licence-clean fallback the ERA5 arm does not, because the AWS bucket carries UKV under CC BY-SA 4.0 for the same two years the lineage check covers.*

**Should ICON-D2 be a second convection-permitting arm in this issue?** Costed in the next section. *Recommendation: a follow-up issue.* Two reasons beyond the cost: ICON-D2 cannot meet this plan's lineage standard, because no multi-year native archive exists to sample against; and what a second fine-resolution arm means depends on UKV's answer, since a null UKV result would make ICON-D2 a test of a different question.

**Does the aerosol asymmetry weaken the issue enough to change what gets built?** UKV carries a fixed aerosol climatology where ERA5 and CAMS carry time-varying aerosol, so the ERA5-against-UKV contrast is not the clean single-variable comparison the issue describes. *Recommendation: build it anyway and state the asymmetry.* The contrast still removes two of the three differences that made the CAMS result unreadable, the remaining aerosol difference bites hardest in the clear-sky bin where the published effect is smallest, and `sky_conditions.py` already reports that bin. The alternative — waiting for an aerosol-aware fine-resolution product — has no candidate.

**What the Met Office has not re-confirmed since 2019 is carried as a caveat, not resolved.** The research found no document from 2020 onwards positively re-confirming that UKV's satellite cloud-fraction assimilation still runs, only that PS47's list of removed assimilation methods does not include it. It also could not establish whether surface solar irradiance is assimilated anywhere in the Met Office's systems, or whether UKV has moved from SEVIRI to its successor instrument. None of these blocks the work, and the write-up states each as unestablished rather than assuming the convenient answer.

**`ARM_FEATURES` calls arm C `C_era5_split`, which has been wrong since CAMS was added and will be wrong a third time.** Renaming it to `C_source_split`, which the physical runner already uses, would touch eight scripts and invalidate every stored result parquet, since `arm` is a data column. *Recommendation: leave it, and record the wart here.* Flagged rather than fixed, per the out-of-scope rule.

**`output_path_for` writes every source's dataset under `data/ERA5/`, including the CAMS one and now the UKV one.** The same reasoning applies: changing it would strand every existing result directory. Flagged, not fixed.

## What the simplicity review changed

**The review cut the plan to roughly a third of its first draft, and every cut above was verified against the code before being taken.** Accepted: the `--ukv-temporal` flag, which re-invented the existing `--suffix` (confirmed composing through all six runners); the second full experiment run on the `_instant` columns, whose result is predictable because the trapezoid is the better estimator of an hourly mean in every signal regime; the `SOURCE_CHOICES` hoist; `ps47_breakdown.py`; and the `compare_sources.py` generalisation. The review also *added* the hourly-label check, which the first draft lacked and which guards against the same class of fault as the half-hour stamp offset.

**The `SOURCE_CHOICES` cut was taken and has since been partly reversed, on new information.** The review was right that the first draft's mechanism was broken: `elevation_breakdown.py` and `report_results.py` import only the standard library and `polars` and document a run command supplying only `polars`, while `build_dataset` pulls in `pvlib`, `xarray` and a Delta store, so hoisting the constant there would break both. The first draft asserted the opposite and was wrong. A second Open-Meteo source is now committed rather than hypothetical, and a stdlib-only leaf module — the shape `era5_grid.py` already uses — carries the constant without the dependency that broke it. `sources.py` above is that module.

Three findings were rejected, each with its reason:

- **"Cut the verifiable-era run and run the full archive once."** Rejected as a cut, accepted as an argument: the recommendation on which span is *primary* is reversed above, but both runs stay, because the span costs no code and the second run is 90 minutes of unattended compute against a risk this project has paid for once already.
- **"Drop the stratification across PS47 from the lineage check."** Rejected. The review's premise is that the archive's lead-time construction rule does not change with the science package, which is an assumption rather than a finding — the issue records that the AWS encoding changed at PS47, so Open-Meteo's ingest could have changed there too. The sweep is cut from a full stratification to roughly 12 to 18 reads, which is most of the saving the review wanted.
- **"Ask the resolution question of CAMS against itself, by averaging the point service over a 31 km stencil."** Declined for now by the human reviewer, and not carried as an open question. The design is a genuinely cleaner test of the issue's stated averaging mechanism — same product, same retrieval, same span, paired row for row, and crossing no product boundary — but it answers a different question from the one the page exists to feed, since CAMS is a retrieval and the procurement decision is about a forecast product's published split. It is also unbounded in cost until the Atmosphere Data Store's per-request latency is measured: a 3-by-3 stencil is 432 requests against the current 48, and the timestamps under `data/CAMS/` all fall inside one second, so they record a bulk rewrite rather than 48 downloads.
- **"Demote the lineage check from a gate to a measurement, and size the reduced change medium."** Rejected on both halves. The gate is a direct instruction. And the sizing still turns on more than one defensible design: the lead time the archive serves can force a switch from Open-Meteo to the AWS bucket, which is a different data path end to end, and the span question is still open. A change that can still change its own data source after a measurement is not a medium change.

## What adding ICON-D2 alongside UKV would cost

**Roughly half a day of agent work and 90 minutes of compute, if ICON-D2 ships with an unverified lineage.** If it has to meet UKV's lineage standard, it cannot be done from an archive at all. The estimate has halved from this plan's first draft, because ICON-D2 is now a committed follow-on and the scripts above are built for it rather than around UKV alone.

Most of the machinery is paid for here and costs nothing again:

- The fetcher already takes `--model`, so ICON-D2 is one entry in the `sources.py` registry rather than a second script.
- `SOURCE_CHOICES` lives in one place, so the 13 argparse sites need no edit at all.
- `build_dataset.py`'s per-site branch is a membership test, so a second per-site source is one more tuple member.
- Every runner, the report script, and the figure scripts take `--source` and need no change.
- The compute is another pass of the same runs, about 90 minutes, unattended.

What actually costs time is specific to ICON-D2:

- **Its temporal convention has to be measured, not inherited.** DWD's `ASWDIR_S` and `ASWDIFD_S` are averages since model start in the native output, which Open-Meteo de-accumulates to hourly means — so ICON-D2 likely matches ERA5's and CAMS's hourly integral more closely than UKV does, and likely carries none of UKV's sub-hourly sampling penalty. That is a reason to expect an easier arm, not a reason to skip the check: the `sources.py` registry records the convention as measured for UKV and unmeasured for ICON-D2, and the follow-on issue establishes it. A few hours.
- **The lineage check has no counterpart.** DWD's open-data server holds roughly a day of ICON-D2, so there is no multi-year archive to sample against. Either the arm ships with a caveat where UKV ships with a measurement, or the check becomes a live-forward capture run over several days.
- **A third discontinuity audit**, for ICON-D2's own upgrades inside the window.
- **The common row set shrinks again.** ICON-D2's archive starts 2022-11-24 against UKV's 2022-03-01, so a four-way comparison on shared hours loses another eight months on top of whatever the span decision above costs.
- **The write-up grows a third product column in every table**, and the interpretation of a four-source common-row comparison is harder to write than a three-source one.

**The strongest argument for including it is that one fine-resolution model against one coarse one is a sample of one on each side**, so a UKV result could be a UKV quirk rather than a resolution effect. That argument is real. It is also exactly as strong after UKV's result exists as before, and a day cheaper to act on then.
