# Plan: census of GB solar BMUs (issue #1069)

**Problem.** No Elexon field says which Balancing Mechanism Units (BMUs) are solar. The `fuelType` field of the BMU reference data has no solar value (null for 2,525 of 3,115 BMUs, `OTHER` for 103), and report B1420 (dataset IGCPU) types only 9 BMUs as "Solar". A throwaway prototype that selected solar BMUs from IGCPU's type found those 9 and nothing else. How many BM-registered solar BMUs exist, and how many the prototype missed, is unknown.

**Planned solution.** Classify every BMU that has settled output by whether that output follows the sun, take the union with the IGCPU "Solar" type, and measure recall against the TEC register's list of PV sites. Live tests by the two plan reviews found that, among the 230 candidate BMUs with a `T_`, `E_`, or `M_` identifier, 9 have half-hourly output that correlates 0.81 to 0.91 with the cosine of the solar zenith over 12 months, and no other BMU in that group scores above 0.39. Another 36 BMUs, mostly `2__` identifiers, also score above 0.5, and are probably aggregated portfolios of embedded generation. The expected answer is therefore about 10 single-site solar BMUs, plus a separately counted group of aggregates, not dozens. The plan extends the existing prototype script, which already fetches and joins the five sources, and adds the B1610 fetch, the classifier, and the recall check.

## Verdict, size and departures

**Verdict: worth doing, and the issue's scope stands.** The ceiling is the BM-registered solar fleet, not all GB solar, because most solar is small embedded capacity that has no BMU. The issue already says so.

**Size: complex, on trigger 4.** The five triggers:

1. **What gets stored:** a dataset and a result table under `data/studies/solar_bmu_census/`, and a report. No Patito model, no Delta table, no asset.
2. **Production serving path:** not touched.
3. **Degradation rule:** none touched.
4. **More than one defensible design:** yes. The designs are name-and-type matching only, behaviour only, or both with a measured recall. The plan takes the third.
5. **Callers not nameable without searching:** no. The code is new and has no callers.

Trigger 4 fired when the issue was sized, so all four reviews are planned. The first plan review found that the choice collapses to one cheap design and said the issue is closer to medium. The sizing stays complex, because the sizing rule says any trigger that fires makes the issue complex. The mutation pass runs on the classifier tests, because `packages/studies/` gains tests.

**Departures from the issue body:**

- **The candidate set is every BMU with B1610 rows, not only the null-fuel ones.** Reason: classification is cheap (1,705 BMUs for 12 months download in about 10 minutes), and the second plan review found 36 sun-following BMUs outside the `T_`, `E_`, `M_` prefixes that a narrower filter would drop.
- **The recall estimate comes from the TEC register, not a hand-labelled sample or REPD.** Reason: see "What the first plan review changed" below.
- **The deliverable is a README and `report.md`, not a docs page.** See open question 1.

## What changes, file by file

All new files are under `studies/solar_bmu_census/`. The folder starts from the untracked prototype `studies/solar_bmu_capacities/collate.py` (its fetching, caching, name matching and capacity-table code), which is renamed and split, and the prototype folder is deleted from the worktree it sits in so that two scripts never share the basename `collate.py`.

**Where data lives.** Results go under `PER_STUDY_DIR / "solar_bmu_census"`, downloads under `PER_STUDY_DIR / "solar_bmu_census" / "downloads"`, and BMU-level listings under `PRIVATE_DIR`, all constants already in `studies.sources`. The downloads are filed with the study rather than under `data/studies/downloads/`, because that folder is filed by weather product and its folder names are constants in `packages/studies/`; moving the downloads there later is a file move. The prototype's repository-wide `cache/` directory and its `.gitignore` entry are not carried over.

- **`fetch_sources.py`**: the prototype's cached fetches (BMU reference data, IGCPU in 700-day windows because the endpoint rejects more than 731 days, the TEC register CSV found from the NESO page, the REPD CSV found from the gov.uk page, and the Maximum Export Limit from `datasets/MELS/stream`), plus a B1610 fetch for **every BMU in the reference data except interconnectors** from `datasets/B1610/stream` with `bmUnit` (the second plan review fetched 1,705 BMUs for 12 months in about 10 minutes). One cache file per request, with the retrieval timestamp. The window is the 12 complete months ending a week before the run, because B1610 lags by about a week. The request window includes both ends, so a chunked fetch duplicates the boundary rows, and the fetch deduplicates on `(bmUnit, halfHourEndTime)`. Parse `halfHourEndTime` explicitly as UTC, because the string carries no zone marker. The script writes a lineage note (sources, URLs, retrieval timestamps) and a generated README beside the downloads, as the `data-download` skill requires, with its own short helper, because the existing `write_lineage_note` lives in another study folder that this script may not import. The script gets a fresh review before its first real run.
- **`classify.py`**: for every BMU with B1610 rows, compute one feature: the Pearson correlation between half-hourly output (`quantity`, with a UTC index) and `max(cos(zenith), 0)` at one central reference point (53°N, 1.5°W, from `studies.solar`), **starting at the BMU's first half-hour with positive output**, so that months before commissioning do not dilute it. The second plan review zero-filled the first months of the nine known solar BMUs and found the correlation falls to 0.49 to 0.53 after 9 empty months and to 0.18 to 0.20 after 11. The rules apply in this order:
    1. `no_output`: fewer than 100 positive half-hours in the window, or a constant output (correlation is NaN, which Polars compares as greater than 0.6, so NaN is tested for first).
    2. `solar`: correlation above 0.6 and a positive peak, or IGCPU types the BMU Solar. A BMU IGCPU types Solar but whose correlation is below 0.6 or whose class is `no_output` is listed for inspection and kept as `solar` by type, with a column that says so.
    3. `not_solar`: everything else.

    Output counts by identifier prefix: `T_`, `E_`, `M_` BMUs, and separately the `2__`, `V__`, and `C__` BMUs (the second plan review found 36 BMUs with correlation above 0.5 outside `T_`, `E_`, and `M_`, probably aggregated portfolios of embedded generation, and one IGCPU-Solar BMU is a `2__`). Whether those count as solar BMUs is open question 2.
- **`collate.py`** (the prototype's table builder): one row per solar BMU, one column per published capacity figure (Generation Capacity, IGCPU installed capacity, TEC, largest Maximum Export Limit, REPD installed capacity), each with its unit and basis in the README, never added together. The prototype's name evidence stays as a column.
- **`recall_check.py`**: reads the 18 TEC register rows with PV in `Plant Type` (12 Built, 6 Under Construction/Commissioning; 3 of the 6 are Embedded agreements, so the register is not transmission-only), plus the Built rows whose `Plant Type` lists only storage but whose name says solar. A TEC row and a BMU match many-to-many (a solar BMU and a battery BMU at one site, or two solar BMUs at one site), and a lead party is a trading party rather than the TEC customer, so name matching cannot be automated. The script reads a hand-made mapping table in `PRIVATE_DIR`, keyed by TEC Project ID and matching against every BMU, not only the census, and writes a first draft of the table from name and capacity matches for the hand check. It reports three outcomes for each row: no BMU found, a BMU found that the classifier called not solar, and a sun-following BMU found. Of the 12 Built rows, 10 list storage, so the report counts hybrid and pure-PV rows separately.
- **`report.py`**: prints into `report.md` every number a write-up would quote: the pool size, the correlation distribution by band (with NaN counted as its own band, so no BMU drops out of a count), the counts by class and by identifier prefix, the per-column capacity sums and missing counts, and the TEC outcome counts. **`report.md` holds counts only.** The BMUs in the gap band (correlation 0.3 to 0.8), the BMUs with no output, the IGCPU-typed BMUs that do not follow the sun, and each unmatched TEC row go to a listing in `PRIVATE_DIR`, never `report.md`.
- **`README.md`**: what each file holds, the unit and meaning of each capacity column, and what the census does not cover.
- **`studies/README.md`**: gains a row in "The studies" table and the per-study data table.

**Tests, in `packages/studies/tests/solar_bmu_census/`** (the `pythonpath` and `ty` `extra-paths` entries for the folder are added in `pyproject.toml`, as for the other study folders). The code under test is the classification rule, the correlation, the B1610 parsing, and the TEC filter, all kept in `classify.py` and `fetch_sources.py`:

- **A constant-output series classifies as `no_output` and never as `solar`.** It fails if NaN is compared before it is tested for.
- **A half-sine around solar noon with zero rows before commissioning classifies as solar whether the commissioning is 1 month or 10 months into the window.** It fails if the leading zeros are included.
- **A solar BMU with small negative night readings still classifies as solar.** It fails if the rule demands exactly zero at night.
- **A series that is positive only at night and a flat series do not classify as solar.** It fails if the sign is ignored.
- **A naive `halfHourEndTime` string parses as UTC, and two overlapping chunks deduplicate to one row per `(bmUnit, halfHourEndTime)`.** It fails if the code parses as local time or deduplicates on settlement date and period.
- **The TEC filter returns the Built, Under Construction, and storage-only-but-solar-named rows on a fixture register.** It fails if the filter reads only `Plant Type`.

Nothing changes under `src/`. The `packages/studies/` change is tests only, plus the `pyproject.toml` path entries, which takes the PR outside the `study` skill's self-merge conditions: it stops at a reviewed PR for the maintainer.

## Design-philosophy check

Research code: it runs only under `studies/` and never in production, so the inherent-stability rules do not apply, and a failed fetch fails fast. It delivers none of the hypotheses in `docs/design-philosophy/engineering-hypotheses.md`, so none is cited. It trades away no principle in `design-principles.md`. It touches no Patito contract, and neither `src/` nor any package gains an import of `studies`. **Anonymisation:** `report.md` and any page carry counts, sums, and correlations by band, never a BMU name or identifier. Every listing that names BMUs lives in `PRIVATE_DIR`.

## Checks printed in `report.md`

- **Mean output by UTC hour for the sun-following BMUs, for April to September and for November to February.** Both centres must sit within 12.0 ± 0.3 hours and within 0.25 hours of each other. A local-time parse would shift the summer centre by an hour.
- **The correlation band counts add up to the BMU count.** It fails if a NaN drops a BMU from a count.
- **The count of IGCPU-typed solar BMUs below the threshold.** It is expected to be 2, and a different count needs an explanation in the report.

## Docs to update

- **`studies/solar_bmu_census/README.md`** and the **`studies/README.md`** rows, described above.
- **No `docs/` page** unless open question 1 is answered otherwise. No roadmap item ships, so no ship-time triage applies.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/solar_bmu_census/README.md
uv run python studies/solar_bmu_census/fetch_sources.py
uv run python studies/solar_bmu_census/classify.py
uv run python studies/solar_bmu_census/report.py
```

Plus, per the `data-validation` skill, on the first downloaded chunk and again at the end: gaps in each BMU's half-hourly series, duplicate `(bmUnit, halfHourEndTime)` keys, and the lag of the latest row.

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

## What the second plan review changed

The second reviewer fetched 12 months of B1610 for 1,705 BMUs and ran the plan's classifier on it. Accepted, each checked:

- **The candidate pool was too narrow.** 36 BMUs outside `T_`, `E_`, `M_` have correlation above 0.5, and the first review's "at most 0.37 for every other BMU" covered only the BMUs it had fetched. The pool is now every BMU with rows, and the `2__`, `V__`, and `C__` hits are counted apart.
- **A NaN correlation classifies a constant-output BMU as solar.** Verified on Polars here: `pl.corr` of a constant series is `nan`, and `nan > 0.6` is `true`. The rule order now tests for no output first, with a unit test.
- **Zero rows before commissioning dilute the correlation.** The reviewer's zero-fill table (0.49 to 0.53 after 9 empty months) contradicted the plan's own risk. The correlation now starts at the first positive output.
- **The TEC recall check assumed a one-to-one match, and lead party and customer name cannot be matched automatically.** The check is now many-to-many against all BMUs, from a hand-made mapping table in the private store, with three outcomes for each TEC row.
- **The TEC counts were wrong in two places.** 10 of the 12 Built rows list storage (the plan said 13 hybrids across Built and Under Construction), 3 of the 6 Under Construction rows are Embedded, and two Built rows have a solar name but a storage-only plant type. All three are now stated.
- **The data layout broke the study conventions.** Results and downloads now sit under constants from `studies.sources`, and the repository-wide `cache/` ignore entry is dropped.
- **No lineage note or fetch-script review was planned.** Both are now in `fetch_sources.py`'s bullet.
- **`report.md` listed BMUs, contradicting the anonymisation rule.** `report.md` holds counts only, and the listings go to the private store.
- **The duplicate key was wrong.** Chunk boundaries duplicate rows, so the key is `(bmUnit, halfHourEndTime)`.
- **Four tests are added.** Two of the three `report.md` checks were weak (the MWh-to-MW check guards nothing, because a correlation ignores scale, and the "highest correlation" check would pass on NaN), so the checks are replaced by a numeric UTC criterion, a band-count closure, and a count of IGCPU-typed solar BMUs below the threshold, and real unit tests carry the rule.

Rejected or modified:

- **Modified: the reviewer's suggestion to move the cache under the downloads directory** is taken, but the downloads sit under the study's own folder rather than `data/studies/downloads/`, because the `NWP`, `reanalysis`, and `observations` folder names are constants in `packages/studies/` and adding one would change the package. Open question 5 asks whether to move them.
- **Not taken: the reviewer's option of moving `write_lineage_note` into `packages/studies/`.** A short helper in the study folder is smaller than a package change plus a mutation pass.

## Risks and open questions

1. **Does the study need a published page, or only `report.md` and the README?** **Recommendation:** README and `report.md` only. A page costs the two Opus science reviews the `study` skill requires and would publish capacity sums over about 10 generators.
2. **Do aggregate BMUs count as solar BMUs?** A `2__` (supplier) or `V__` (virtual lead party) BMU whose output follows the sun is probably a portfolio of embedded generation. One `2__` BMU is already typed Solar in IGCPU. **Recommendation:** count single-site `T_`, `E_`, and `M_` BMUs as the census, and report the aggregates as a separate count, because their capacities cannot be tied to one site.
3. **Do hybrid (solar plus storage) BMUs count?** 10 of the 12 Built TEC PV rows list storage, and a BMU that nets a battery into its output may not follow the sun. **Recommendation:** count a hybrid BMU as solar when its output follows the sun above the threshold, and report the hybrid TEC rows that map only to BMUs below the threshold as a separate count, rather than loosening the threshold until they pass.
4. **Embedded recall is not measured.** The TEC register lists mainly transmission-connected sites, and all 9 positives are transmission-connected. **Recommendation:** accept, and state it as a limitation; no independent list of embedded solar BMUs exists.
5. **Where do the downloads go?** **Recommendation:** under the study's own folder for now, and moved to `data/studies/downloads/observations/` if a later study reuses them.
6. **The PR stops at review.** The `pyproject.toml` path entries and the tests under `packages/studies/` take the diff outside the `study` skill's self-merge conditions, so the maintainer merges.
7. **The prototype is uncommitted and unreviewed.** The diff review treats the code carried over from it as new code.
