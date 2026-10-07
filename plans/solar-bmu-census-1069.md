# Plan: census of GB solar BMUs (issue #1069)

**Problem.** No Elexon field says which Balancing Mechanism Units (BMUs) are solar. The `fuelType` field of the BMU reference data has no solar value (null for 2,525 of 3,115 BMUs, `OTHER` for 103), and report B1420 (dataset IGCPU) types only 9 BMUs as "Solar". A throwaway prototype that selected solar BMUs from IGCPU's type found those 9 and nothing else. The true number of BM-registered solar BMUs is unknown, and so is how many the prototype missed.

**Planned solution.** Classify every candidate BMU by what it does, then check the classifier against a blind hand-labelled sample. The candidate pool is small: 230 BMUs have a generation capacity above zero, a `T_`, `E_`, or `M_` identifier, and a null or `OTHER` fuel type (121 transmission-connected, 109 embedded). For each candidate, the study downloads a year or more of settled half-hourly output (dataset B1610) and tests whether the output behaves like solar: near zero when the sun is down everywhere in Great Britain, positive around midday, and larger in summer than winter. The study combines that behavioural evidence with the name and type evidence (IGCPU type, a "solar" or "PV" name, a match in the Renewable Energy Planning Database or the Transmission Entry Capacity register) into one class per BMU. It then reports the count of solar BMUs together with a recall estimate, and writes the capacity table (one column per published capacity figure, never summed across columns). Nothing is published per generator.

## Verdict, size and departures

**Verdict: worth doing, and the issue's scope stands.** There is hope of a much better answer than 9, but the ceiling is the BM-registered solar fleet, not all GB solar, because most solar is small embedded capacity that has no BMU. The issue already says so.

**Size: complex.** The five triggers:

1. **What gets stored:** the study writes datasets and a result table under `data/studies/solar_bmu_census/`, and a page or report is published. No Patito model, no Delta table, no asset. (The `study` skill counts a published result as stored.)
2. **Production serving path:** not touched. Nothing under `src/` or any package other than `packages/studies/` changes.
3. **Degradation rule:** none touched.
4. **More than one defensible design:** yes, which fires the trigger. The three designs are (a) join by name and type only, (b) classify by behaviour only, (c) combine both and measure recall on a labelled sample. This plan takes (c). Step 5's reviewer may still find a simpler variant of (c).
5. **Callers not nameable without searching:** no. The code is new and has no existing callers.

Trigger 4 fires, so all four reviews run: two plan reviews and two diff reviews (the second diff review, the mutation pass, because `packages/studies/` changes).

**Departures from the issue body:** none. The issue offers routes, and this plan picks among them.

## What changes, file by file

Study scripts, under `studies/solar_bmu_census/` (new folder, unit-tested only where noted):

- **`fetch_sources.py`**: downloads and caches, per the `data-download` skill, one cache file per request so a crash costs one request: the BMU reference data; IGCPU in 700-day windows (the endpoint rejects windows over 731 days); B1610 per candidate BMU over the study window (`datasets/B1610/stream` with `bmUnit`, about 0.9 MB and 0.15 s per BMU-year, measured on one BMU); the TEC register CSV (its link is found on the NESO page); and the REPD CSV (its link is found on the gov.uk page; cp1252 encoded). Writes to `data/studies/solar_bmu_census/downloads/`, each file with its retrieval timestamp.
- **`build_candidates.py`**: builds the candidate pool (rules above) plus every BMU IGCPU types "Solar" or "Generation", and prints the pool's size by connection type. A BMU with a fuel type of WIND, CCGT, and so on is excluded; hybrid sites registered under such a type are a stated limitation.
- **`classify.py`**: computes the behavioural features for each candidate with `studies.bmu_behaviour` and the name evidence, and assigns one class: `solar_behaviour_and_name`, `solar_behaviour_only`, `name_only`, `not_solar`, or `no_data`. Writes `data/studies/solar_bmu_census/classes.csv` (private store, with identifiers).
- **`label_sample.py`**: draws the blind sample (below) with a fixed seed, writes `sample_to_label.csv` with the name, lead party, and capacity but no classifier output, and later reads the filled labels.
- **`report.py`**: prints every number the write-up quotes into `report.md`: pool size, counts per class and connection type, per-column capacity sums and missing counts, recall and precision with Wilson 95% intervals, and the REPD cross-check.

Shared machinery, in `packages/studies/src/studies/bmu_behaviour.py` (new, with tests), because a wrong night mask or a wrong clock convention gives a plausible-looking class rather than an error:

- **`behaviour_features(*, output: pl.DataFrame, capacity_mw: float) -> BehaviourFeatures`**: takes a BMU's half-hourly B1610 rows (`halfHourEndTime` in UTC, `quantity` in MWh) and returns: the median absolute night output divided by capacity (night means the solar zenith exceeds 100 degrees at both a southern and a northern reference point, so the sun is down everywhere in Great Britain); the share of days with output above 5% of capacity in the three hours around solar noon; the ratio of mean June-to-August midday output to mean December-to-February midday output; and the Spearman correlation between half-hourly output and the cosine of the zenith at a central reference point. It converts MWh per half-hour to MW, and takes its zenith from `studies.solar`.
- **`class_from_features(*, features, name_evidence) -> SolarClass`**: the thresholds, as named constants with docstrings, applied in one place.

Chosen thresholds are set on the nine IGCPU-typed solar BMUs and on 10 clearly non-solar candidates (wind and gas, from the pool's neighbours in the reference data), before the labelled sample is drawn, so the sample measures the classifier rather than tuning it.

**The recall check.** A BMU has no coordinates, so no independent labelled list exists. The plan builds one:

1. Draw a stratified random sample of 60 BMUs from the pool, 30 transmission-connected and 30 embedded, with a fixed seed. The sample is drawn from the pool, not from the classifier's positives, so it can reveal missed solar BMUs.
2. Label each BMU solar, not solar, or unknown from public sources (lead party, site name, a web search of the site's technology), without seeing the classifier's output. A Sonnet agent does the lookups from `sample_to_label.csv`; an Opus agent audits a random third of its labels.
3. Report recall and precision against the labelled solar and not-solar BMUs, with Wilson intervals, and report the number of unknown labels, because unknowns are not dropped silently.
4. A second, independent miss check: take REPD's operational solar photovoltaic sites of 10 MW or more. For each, say whether the census has a BMU with a matching site name. A site with no BMU either has none (it is not BM-registered) or was missed. Report both counts without choosing between them, because the cause needs a hand check of a sample.

## Design-philosophy check

Research code: it runs only under `studies/` and never in production, so the inherent-stability rules (degrade, widen bands) do not apply, and a fetch that fails fails fast. It delivers none of the engineering hypotheses in `docs/design-philosophy/engineering-hypotheses.md` (cited by label `H1`, `T1.2`), so none is cited. It trades away no principle in `design-principles.md`. It touches no Patito contract, and `src/` and the other packages gain no import of `studies`.

**Anonymisation.** `CLAUDE.md` forbids publishing a metered generator's time series with its name or ID. A per-BMU table of capacity figures is not a time series, but the census table with identifiers stays in `data/studies/` (private) and the published page carries counts, sums, and recall only. Any plot of a BMU's daily profile, if drawn to show the method working, uses anonymised labels from a new label tuple and seed, never the `A` to `F` tuple, per the `study` skill.

## Tests

In `packages/studies/tests/test_bmu_behaviour.py`, one short test each, and for each the assertion that fails if the implementation is wrong:

- **Solar-shaped synthetic output** (a half-sine around solar noon, zero at night) classifies as solar. It fails if the night mask is inverted.
- **Flat output (a gas plant), noisy output with no diurnal pattern (wind), and output that is positive at night (a battery discharging at night)** each classify as not solar. Each fails if only the daytime features are used.
- **A series with the small negative night readings that real solar BMUs show** (auxiliary load, for example -0.18 MWh at night on a 112 MW unit) still classifies as solar. It fails if the night test demands exactly zero.
- **A series spanning the spring and autumn clock changes** keeps its night hours at the correct UTC times. It fails if the code reads settlement periods as UTC or as a fixed 48 per day.
- **Capacity normalisation:** two BMUs with the same shape and capacities of 10 MW and 200 MW give the same night ratio. It fails if the ratio is in raw MW.
- **A BMU with no data, or fewer than the minimum number of days, returns `no_data`** and does not raise. It fails if an empty frame raises.

Study scripts have no tests of their own, per the `study` skill; their check is the printed report. The mutation pass in `implement-issue` runs because `packages/studies/` changes.

## Docs to update

- **`docs/studies/`**: if the maintainer wants a published page (see open question 1), it follows the `study` skill's structure and carries the AI disclaimer. Otherwise a `README.md` in the study folder records what each file holds, which sources it came from, and when.
- **`studies/README.md`** and any index of study folders gain one line for the new folder.
- **`docs/design-philosophy/` and `docs/roadmap/`:** no change. The study delivers input to the capacity and metered-generator work; it does not complete a roadmap item, so no ship-time triage applies.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/solar_bmu_census/README.md
```

Plus, per the `data-validation` skill, on the first downloaded chunk and again at the end: a gap check on each BMU's half-hourly series, duplicate `(settlementDate, settlementPeriod, settlementRunType)` keys, and a check that rows from different settlement runs (`II`, `R2`, and so on) are not double counted.

## Risks and open questions

1. **Does the study need a published page, or only a report and README?** The issue asks for a table and a recall estimate. A page would let others cite the recall figure. **Recommendation:** a short page with counts, sums, and recall only, and the BMU-level table kept private. The page costs the two Opus science reviews the `study` skill requires; if the maintainer prefers a README only, that cost drops away.
2. **Is publishing the BMU-level capacity table acceptable?** Static capacities are not a time series, but the lookup would let a reader attach a generator's name to a size. **Recommendation:** keep it private until the maintainer decides.
3. **Study window.** B1610 is cheap (0.9 MB per BMU-year). **Recommendation:** 2023-01-01 to the latest complete month, which covers three summers and matches the IGCPU history; a BMU commissioned inside the window is judged on its commissioned months.
4. **Solar and battery hybrids.** A BMU that charges at night will look not solar. **Recommendation:** add a `solar_plus_storage` class when the night output is both negative and large and the daytime pattern is solar, and report the count separately.
5. **Fixed reference points for the night mask.** BMUs have no coordinates, so the night mask uses two reference points (southern and northern Great Britain). A BMU whose output starts or ends in twilight is not penalised. **Recommendation:** accept this, and test sensitivity to the zenith threshold (95, 100, and 105 degrees) in the report.
6. **Where the study fails.** If recall comes back low, the likely cause is hybrid-typed sites or BMUs with a fuel type of WIND. The plan does not widen the pool before seeing the sample, because the sample is drawn from the pool.
