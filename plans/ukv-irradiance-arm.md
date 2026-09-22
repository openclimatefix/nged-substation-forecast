# Add a UKV arm to the beam/diffuse split experiment (#800)

**The problem.** The [beam/diffuse split experiment](https://openclimatefix.github.io/nged-substation-forecast/results/beam-diffuse-split/) found that a weather product's published direct-beam field cut PV power error by 1.8% on the CAMS satellite retrieval and by nothing detectable on ERA5. The page declines to say why, because CAMS and ERA5 differ in resolution, in delivery, and in production method all at once. Nothing in that experiment can tell those three apart.

**The planned solution.** Add the Met Office's UKV as a third irradiance source, taken from Open-Meteo's historical-forecast archive of `ukmo_uk_deterministic_2km`, running the same six arms on the same six meters. UKV is a numerical weather prediction model at 2 km that publishes its own direct and diffuse fields, so ERA5 against UKV is a resolution contrast inside one product class. Two checks gate the arm before any model is trained: a lineage check against the Met Office's own files on AWS, and an explicit treatment of UKV's instantaneous radiation against ERA5's hourly integral. Three departures from the issue body are set out below, the largest being that the issue's preferred temporal treatment is the wrong one.

## Verdict: worth implementing, with three departures

**The issue is worth implementing roughly as described.** Its premise checks out against the code and the published page: `docs/results/beam-diffuse-split.md` does decline to attribute the CAMS-against-ERA5 gap to resolution, and the experiment's structure — a `--source` axis, arms defined only by irradiance columns, a per-site source path already built for CAMS — takes a third source with no redesign. The confound the issue names is real and UKV does break it.

**Departure 1: the primary temporal treatment is Open-Meteo's default hourly value, not the `_instant` variants the issue prefers.** The experiment's target is a period-ending hourly mean of the meter's power. ERA5 and CAMS both publish an hourly integral, which is the same temporal object. Open-Meteo's default UKV hourly value is the trapezoid of the snapshots at the start and end of the hour, which is an *estimator of that same object* — a noisy one, but the same quantity. The `_instant` variant is a point sample at one instant, which is a different quantity from the target and from the other two sources. Preferring `_instant` would introduce the mismatch the issue is trying to avoid rather than remove it. The default is therefore primary and `_instant` is the pre-registered sensitivity, which is the only way to measure how far the two-point trapezoid strays under broken cloud.

**Departure 2: the lineage check is a gate on the work, not a check worth running, and it cannot cover the whole archive.** `docs/roadmap/data-sources.md` records, from a bucket listing rather than from documentation, that the AWS UKV archive starts on 2024-09-19. Open-Meteo's archive starts on 2022-03-01. **Over half of Open-Meteo's UKV history sits in a period the Met Office's own open archive no longer holds, so no sampling against AWS can check it.** The verifiable era runs 2024-09-19 to now and does contain the PS47 boundary, so the check the issue asks for is possible on the recent half and impossible on the older half. This plan makes the verifiable era the primary UKV span and the full span a sensitivity run, and flags the choice as the first question for the human reviewer.

**Departure 3: ICON-D2 is planned as a follow-up issue rather than a second arm here.** Costed below. The short reason is that ICON-D2 cannot meet the lineage standard this plan applies to UKV, because DWD's open-data server holds about a day of ICON-D2 and there is no multi-year native archive to sample against.

**One mechanism in the issue body is kept unchanged**: the arm structure, the five load-bearing constraints (`colsample_bytree` at 1.0, the negative control, the piecewise stamp shift, the export-cap clamp, and anonymisation), and the rule that every published number comes from a script.

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

### New: `fetch_ukv_open_meteo.py`

Downloads UKV at each meter's own coordinates from `https://historical-forecast-api.open-meteo.com/v1/forecast` with `models=ukmo_uk_deterministic_2km`, requesting `shortwave_radiation`, `direct_radiation`, `diffuse_radiation` and each one's `_instant` variant. Writes `data/UKV/beam_diffuse_ukv.parquet`, one row per `(site, time)` carrying both temporal treatments as separate columns, so a single download serves the primary run and the sensitivity run.

Modelled on `fetch_cams.py` rather than on `fetch_era5_open_meteo.py`, because UKV is delivered per site rather than on a shared grid. It imports `_pv_sites` from `build_dataset`, reads each meter's coordinates at run time, sends them to Open-Meteo, and writes only the anonymised label — the pattern `fetch_cams.py` already uses for the same reason. No coordinate and no identifier reaches the written frame.

`direct + diffuse − total` is checked on ingest and logged, since the issue reports it holding to 0.062 W m⁻² on the native files. A departure from zero larger than Open-Meteo's own rounding would mean the three fields are not one consistent split, which would void arm C.

### New: `verify_ukv_lineage.py` — the gate

The counterpart of `verify_era5_sources.py`, and the script this plan hangs on. It samples timestamps stratified across the AWS-verifiable era, deliberately covering both sides of the PS47 boundary on 2026-01-21 and the AWS encoding change at the same date, pulls the matching native files from `s3://met-office-atmospheric-model-data/uk-deterministic-2km/`, extracts the grid point nearest each meter, and compares against Open-Meteo at the same valid time.

**The comparison is against the `_instant` columns, never the default hourly ones.** The native file holds an instantaneous snapshot; Open-Meteo's default is a trapezoid of two of them. Comparing the trapezoid against the snapshot would make a faithful mirror look broken, which is the single easiest way to get this script wrong.

**The same download settles the spin-up question for free.** Rather than comparing one lead time, the script pulls T+0, T+3 and T+6 from the AWS runs valid at each sampled instant and reports which lead Open-Meteo's value matches. That establishes what the archive is built from, which the issue lists as an open question and which one matching timestamp does not settle.

Output is a markdown table of mean and maximum absolute difference per era and per lead, plus an explicit statement of the span it could not check. It writes differences and counts, never a coordinate and never a site-keyed irradiance value paired with anything identifying.

Reading the native files needs an S3 client and a chunked-HDF5 reader; `obstore` is already a dependency of `contracts` and `nged_data`, and `xarray` of `dynamical_data`. The download is a few dozen files rather than the 79 GB a full pull would need.

### `build_dataset.py`

- Hoist the duplicated `("cds", "open-meteo", "cams")` literal into one `SOURCE_CHOICES: Final[tuple[str, ...]]` beside the existing `SourceType`, and add `"ukv"` to both. This is the minimum change that adds a source without a 14-way find-and-replace, and every script that carries the literal already imports from `build_dataset` or from `run_experiment`.
- Add `UKV_PATH` and `_read_ukv(*, temporal)`, mirroring `_read_cams`.
- Generalise the `if source == "cams"` branch in `main()` to cover any per-site source, so UKV joins on `(site, time)` the same way.
- Add `--ukv-temporal {trapezoid,instant}`, defaulting to `trapezoid`, and fold the value into the output filename so the two builds cannot overwrite each other.
- **Air temperature keeps coming from the gridded ERA5 frame, including for the UKV build.** UKV publishes its own `temperature_2m`, and using it would change a shared non-irradiance feature between sources, breaking the invariant that arms and sources differ only in the irradiance columns. CAMS already takes ERA5's temperature for exactly this reason.

### `compare_sources.py`

Generalise from `--first-source`/`--second-source` to a repeatable `--source`, intersecting rows across every source named rather than across two. Three pairwise runs work with today's code but each restricts to a different row set, so the three tables would not be mutually comparable. The intersection of all three is bounded by UKV's span and by CAMS's reliability filter at once, and the script logs the resulting row count so the write-up can state it.

### New: `ps47_breakdown.py`

Splits the UKV headline contrast either side of 2026-01-21, in the shape `elevation_breakdown.py` and `sky_conditions.py` already use. The post-PS47 era runs to about 7.5 months, so its interval will be wide, and the script prints the row count on each side so that width is readable rather than surprising.

### The 12 scripts carrying the duplicated `choices` literal

`run_experiment.py`, `run_physics_experiment.py`, `run_hybrid_experiment.py`, `report_results.py`, `sky_conditions.py`, `elevation_breakdown.py`, `inverter_clipping.py`, `capacity_denominator.py`, `oracle_capacity.py`, `restart_basins.py`, `shared_geometry.py`, `anm_curtailment.py` and `anm_setpoints.py` each import `SOURCE_CHOICES` and drop their own literal. No other behaviour changes in any of them: `dataset_path_for` and `results_dir_for` take the source as a plain `str` and format it into a path, so the runners need nothing else.

### `README.md` in the experiment directory

The two new scripts in the run-order table, UKV in the sources table, and a section on the temporal axis and what it does and does not confound.

### The write-up is a separate pull request against `main`

`docs/results/beam-diffuse-split.md` lives on `main` and not on this branch, exactly as #786 was separate from #785. The scripts land here; the page is updated in its own pull request once the numbers exist. That page needs a UKV row in the sources table, the two questions reported separately, the lineage result including the span it could not cover, the temporal caveat, the PS47 breakdown, and a revision of "What this says about asking a supplier for the beam", which currently reasons from two products and will then reason from three.

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
- `verify_ukv_lineage.py` run to completion, with its table pasted into the pull request body. **This is the gate: no model is trained on UKV until it has run and been read.**
- The experiment run itself, which the timestamps of the existing outputs put at roughly 12 minutes for the XGBoost instrument, 29 for the physical model, and 15 for the hybrid, so about an hour of compute per source on the 32-core workstation, plus the dataset build and the diagnostics.

`mkdocs build --strict` is not needed on this branch, which touches no page MkDocs renders; it is needed on the write-up pull request.

## Risks and open questions

**Which UKV span should the primary run use?** The AWS-verifiable era from 2024-09-19 gives about 24 monthly bootstrap blocks and roughly a third of the row count the CAMS comparison used. The full archive from 2022-03-01 gives 54 blocks, with over half of them unverifiable against any Met Office source. *Recommendation: primary on the verifiable era, sensitivity on the full span.* The project has just been reminded what an unverified weather lineage cost it once, and the cheap order is to make the defensible number the headline and the larger sample the check, rather than the reverse.

**Is Open-Meteo's free tier compatible with this project's use?** It is non-commercial only. This is not new to this issue — Open-Meteo already feeds the *published* ERA5 numbers on the docs page, since `DEFAULT_SOURCE` is `open-meteo` — so it is a live question about work already shipped rather than a gate on this one. *Recommendation: settle it separately, and note that UKV has a licence-clean fallback the ERA5 arm does not, because the AWS bucket carries UKV under CC BY-SA 4.0 for the same two years the lineage check covers.*

**Should ICON-D2 be a second convection-permitting arm in this issue?** Costed in the next section. *Recommendation: a follow-up issue.* Two reasons beyond the cost: ICON-D2 cannot meet this plan's lineage standard, because no multi-year native archive exists to sample against; and what a second fine-resolution arm means depends on UKV's answer, since a null UKV result would make ICON-D2 a test of a different question.

**Does UKV assimilate enough satellite cloud imagery to blur the "forecast against retrieval" contrast?** A research sub-agent is establishing what the Met Office documents about UKV's 4D-Var, its aerosol treatment, and any cloud-observation assimilation. **The answer is folded into this plan's next revision and the finding may weaken the issue's central claim**, since a model heavily informed by near-real-time satellite cloud observation is partly a retrieval at short lead times.

**`ARM_FEATURES` calls arm C `C_era5_split`, which has been wrong since CAMS was added and will be wrong a third time.** Renaming it to `C_source_split`, which the physical runner already uses, would touch eight scripts and invalidate every stored result parquet, since `arm` is a data column. *Recommendation: leave it, and record the wart here.* Flagged rather than fixed, per the out-of-scope rule.

**`output_path_for` writes every source's dataset under `data/ERA5/`, including the CAMS one and now the UKV one.** The same reasoning applies: changing it would strand every existing result directory. Flagged, not fixed.

## What adding ICON-D2 alongside UKV would cost

**Roughly one extra day of agent work and 90 minutes of compute, if ICON-D2 ships with an unverified lineage.** If it has to meet UKV's lineage standard, it cannot be done from an archive at all.

Most of the machinery is paid for by UKV and costs nothing again:

- The fetcher becomes a parameter rather than a second script, because Open-Meteo normalises variable names across models: a different `models=` value on the same endpoint.
- `build_dataset.py`'s per-site branch is already generalised by this plan, so a second per-site source is a member of `SOURCE_CHOICES`.
- Every runner, the report script, and the figure scripts take `--source` and need no change.
- The compute is another pass of the same runs, about 90 minutes, unattended.

What actually costs time is specific to ICON-D2:

- **Its radiation carries a third temporal convention.** DWD's `ASWDIR_S` and `ASWDIFD_S` are averages since model start in the native output, which Open-Meteo de-accumulates to hourly means. That is neither UKV's snapshot nor ERA5's integral, and it has to be established rather than assumed — the same species of defect this plan spends its care on for UKV. Half a day.
- **The lineage check has no counterpart.** DWD's open-data server holds roughly a day of ICON-D2, so there is no multi-year archive to sample against. Either the arm ships with a caveat where UKV ships with a measurement, or the check becomes a live-forward capture run over several days.
- **A third discontinuity audit**, for ICON-D2's own upgrades inside the window.
- **The common row set shrinks again.** ICON-D2's archive starts 2022-11-24 against UKV's 2022-03-01, so a four-way comparison on shared hours loses another eight months on top of whatever the span decision above costs.
- **The write-up grows a third product column in every table**, and the interpretation of a four-source common-row comparison is harder to write than a three-source one.

**The strongest argument for including it is that one fine-resolution model against one coarse one is a sample of one on each side**, so a UKV result could be a UKV quirk rather than a resolution effect. That argument is real. It is also exactly as strong after UKV's result exists as before, and a day cheaper to act on then.
