# Plan: census of GB solar BMUs (issue #1069)

**Problem.** No Elexon field says which Balancing Mechanism Units (BMUs) are solar. The `fuelType` field of the BMU reference data has no solar value (null for 2,525 of 3,115 BMUs, `OTHER` for 103), and report B1420 (dataset IGCPU) types only 9 BMUs as "Solar". A throwaway prototype that selected solar BMUs from IGCPU's type found those 9 and nothing else. How many BM-registered solar BMUs exist, and how many the prototype missed, is unknown.

**Planned solution.** Classify every BMU with a generation capacity above zero by whether its settled output follows the sun, take the union with the IGCPU "Solar" type, and measure recall against the TEC register's list of built transmission-connected PV sites. A one-month live test (by the first plan review) found 9 BMUs whose half-hourly output correlates 0.84 to 0.91 with the cosine of the solar zenith, against at most 0.37 for every other BMU. The expected answer is therefore about 10 solar BMUs, not dozens. The plan extends the existing prototype script, which already fetches and joins the five sources, and adds the B1610 fetch, the classifier, and the recall check.

## Verdict, size and departures

**Verdict: worth doing, and the issue's scope stands.** The ceiling is the BM-registered solar fleet, not all GB solar, because most solar is small embedded capacity that has no BMU. The issue already says so.

**Size: complex, on trigger 4.** The five triggers:

1. **What gets stored:** a dataset and a result table under `data/studies/solar_bmu_census/`, and a report. No Patito model, no Delta table, no asset.
2. **Production serving path:** not touched.
3. **Degradation rule:** none touched.
4. **More than one defensible design:** yes. The designs are name-and-type matching only, behaviour only, or both with a measured recall. The plan takes the third.
5. **Callers not nameable without searching:** no. The code is new and has no callers.

Trigger 4 fired when the issue was sized, so all four reviews are planned. The first plan review found that the choice collapses to one cheap design and said the issue is closer to medium. The sizing stays complex, because the sizing rule says any trigger that fires makes the issue complex, but the second diff review (mutation pass) only runs if `packages/studies/` changes, which this plan avoids.

**Departures from the issue body:**

- **The candidate set is every BMU with generation capacity above zero, not only the null-fuel ones.** Reason: classification costs about 0.2 s per BMU-month, and the first plan review found no solar-shaped output among the 369 BMUs with a fuel type of WIND, CCGT, and so on, so the filter earns nothing.
- **The recall estimate comes from the TEC register, not a hand-labelled sample or REPD.** Reason: see "What the first plan review changed" below.
- **The deliverable is a README and `report.md`, not a docs page.** See open question 1.

## What changes, file by file

All new files are under `studies/solar_bmu_census/`. The folder starts from the untracked prototype `studies/solar_bmu_capacities/collate.py` (its fetching, caching, name matching and capacity-table code), renamed and split.

- **`fetch_sources.py`**: the prototype's cached fetches (BMU reference data, IGCPU in 700-day windows because the endpoint rejects more than 731 days, the TEC register CSV found from the NESO page, the REPD CSV found from the gov.uk page, and the Maximum Export Limit from `datasets/MELS/stream`), plus a B1610 fetch per BMU from `datasets/B1610/stream` with `bmUnit` (about 0.9 MB per BMU-year, measured on one BMU). One cache file per request, with the retrieval timestamp, resumable per the `data-download` skill.
- **`classify.py`**: for each BMU with generation capacity above zero and a `T_`, `E_`, or `M_` identifier, plus every BMU IGCPU types "Solar" or "Generation", compute one feature: the Pearson correlation between half-hourly output (B1610 `quantity` in MWh, converted to MW, indexed by `halfHourEndTime`, which is UTC) and `max(cos(zenith), 0)` at one central reference point (53°N, 1.5°W), taken from `studies.solar`. The window is the latest 12 complete months. A BMU is `solar` if the correlation exceeds a threshold of 0.6 (set from the gap in the one-month test, to be re-checked on the full window) or IGCPU types it Solar; `no_output` if it has fewer than 100 half-hours above zero in the window; otherwise `not_solar`. A BMU that IGCPU types Solar but whose correlation is below the threshold is listed for inspection, not silently reclassified.
- **`collate.py`** (the prototype's table builder): one row per solar BMU, one column per published capacity figure (Generation Capacity, IGCPU installed capacity, TEC, largest Maximum Export Limit, REPD installed capacity), each with its unit and basis in the README, never added together. The prototype's name evidence stays as a column. Writes `data/studies/solar_bmu_census/solar_bmus.csv` (private store).
- **`recall_check.py`**: matches each of the 18 TEC register rows with PV in `Plant Type` and a status of Built (12) or Under Construction/Commissioning (6) to a BMU or to none, by customer name, `MW Connected`, and lead party. Prints: how many of the 12 built rows have a solar BMU in the census, which rows do not, and the 13 hybrid rows (storage plus PV) separately, because a hybrid BMU's output may not follow the sun. The unmatched rows are listed in the private store only.
- **`report.py`**: prints every number the write-up quotes into `report.md`: the pool size, the correlation distribution (counts by band), the count of solar BMUs by connection type, the per-column capacity sums and missing counts, the BMUs in the gap band (correlation 0.3 to 0.8) and every BMU with no output, each listed for a hand check, and the TEC recall result.
- **`README.md`**: what each file holds, the unit and meaning of each capacity column, and what the census does not cover.

Nothing changes under `packages/` or `src/`. The `.gitignore` gains `cache/`.

## Design-philosophy check

Research code: it runs only under `studies/` and never in production, so the inherent-stability rules do not apply, and a failed fetch fails fast. It delivers none of the hypotheses in `docs/design-philosophy/engineering-hypotheses.md`, so none is cited. It trades away no principle in `design-principles.md`. It touches no Patito contract, and neither `src/` nor any package gains an import of `studies`. **Anonymisation:** `report.md` and any page carry counts, sums, and correlations by band, never a BMU name or identifier next to a time series. The BMU-level table stays in `data/studies/`, which is private.

## Tests

**No new unit tests, because no code goes into `packages/studies/`.** The study skill says a script's check is its own output, so each of these is checked in `report.md`, where a wrong result would be visible:

- **The correlation feature uses UTC half-hours correctly.** `report.md` prints the mean output by UTC hour of day for the IGCPU-typed solar BMUs; the peak must sit near 11:00 to 12:00 UTC in summer, and a clock-change error would show as a one-hour shift between the two halves of the year.
- **MWh is converted to MW.** `report.md` prints the largest output divided by Generation Capacity for each solar BMU; a value near 0.5 would mean the conversion is missing.
- **The IGCPU-typed BMUs separate from the rest.** `report.md` prints the correlation of each of the nine IGCPU-typed solar BMUs and the highest correlation among the BMUs that are not solar-typed; a gap is the evidence the threshold is meaningful.

If the diff review asks for a function to move into `packages/studies/` with tests, the mutation pass runs then.

## Docs to update

- **`studies/solar_bmu_census/README.md`**: new, described above.
- **`studies/README.md`**: gains one line for the new folder, if the file lists folders.
- **No `docs/` page** unless open question 1 is answered otherwise. No roadmap item ships, so no ship-time triage applies.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/solar_bmu_census/README.md
uv run python studies/solar_bmu_census/fetch_sources.py
uv run python studies/solar_bmu_census/report.py
```

Plus, per the `data-validation` skill, on the first downloaded chunk and again at the end: gaps in each BMU's half-hourly series, duplicate `(settlementDate, settlementPeriod, settlementRunType)` keys, and rows from different settlement runs (`II`, `R2`, and so on) not double counted.

## What the first plan review changed

The reviewer tested the plan's premise on the live API (one month of B1610 for all 230 pool BMUs and for 369 BMUs with other fuel types, 2025 and 2026) and found the plan over-built for a result of about 10 BMUs. Accepted, each verified or re-derived:

- **The night mask is empty for about two months of the year.** At 58.5°N the sun is only about 8° below the horizon at midsummer, so the zenith peaks near 98° and "zenith above 100° at both reference points" is never true from late May to late July. Re-derived: 58.5 + 23.44 − 90 = −8.06°. The correlation feature needs no mask.
- **A one-feature classifier replaces the four-feature one, and five classes become three.** The correlation separates 0.84 to 0.91 from at most 0.37; the extra features, the 95°, 100° and 105° zenith settings, and `solar_plus_storage` (no observed member) added nothing.
- **No `packages/studies/src/studies/bmu_behaviour.py`.** Only this study uses the function, and the `study` skill moves code into the package when a second study needs it.
- **The blind 60-BMU sample, the Sonnet labeller, the Opus audit, and the REPD cross-check are dropped.** About 10 of 230 pool BMUs are solar, so a 60-BMU sample holds 2 or 3 solar BMUs and any recall interval spans roughly 44% to 100%. The REPD check could not separate "no BMU" from "missed". The TEC register is a complete list: a transmission-connected solar site must hold TEC. Verified against the cached register: 12 rows with PV are Built and 6 are Under Construction/Commissioning.
- **The candidate pool filter is dropped** (see Departures).
- **The plan extends the existing prototype** instead of writing five new scripts.
- **No docs page by default.** A result of about 10 BMUs, with capacity sums over so few generators, comes close to publishing per-generator capacities.

Rejected or modified: none rejected outright. One caveat from the reviewer stands as a risk below, and the reviewer's suggestion that the issue be resized to medium is not taken (see Size).

## Risks and open questions

1. **Does the study need a published page, or only `report.md` and the README?** **Recommendation:** README and `report.md` only. A page costs the two Opus science reviews the `study` skill requires and would publish capacity sums over about 10 generators.
2. **Embedded recall is not measured.** The TEC register covers transmission-connected sites only, and all 9 behavioural positives found so far are transmission-connected. The census says so as a limitation. **Recommendation:** accept; no independent list of embedded solar BMUs exists.
3. **Hybrids.** About 3 of the 12 built TEC PV sites are expected to be missed, most likely solar-plus-storage sites whose BMU output nets a battery. The recall check names which. **Recommendation:** report them as a separate count rather than loosening the threshold until they pass.
4. **Study window.** **Recommendation:** the latest 12 complete months at run time, which covers one full seasonal cycle; a BMU commissioned inside the window is judged on its months of output.
5. **The prototype is uncommitted and unreviewed.** The diff review treats the code carried over from it as new code.
