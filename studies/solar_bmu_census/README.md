# Solar-BMU census: which Balancing Mechanism Units in Great Britain are solar

**This folder holds the scripts behind [How many Balancing Mechanism Units in Great Britain are
solar?](https://openclimatefix.github.io/nged-substation-forecast/studies/solar-bmu-census/).** The
study finds the Balancing Mechanism Units (BMUs) whose settled output follows the sun, and sets the
five published capacity figures for them side by side. Every script's module docstring gives the
command that runs it.

**A script imports only from this folder, from `studies.*`, and from the other reviewed packages.**
The code is run in this order, and each script reads what the one before it wrote.

| Script | What it does |
|---|---|
| `fetch_sources.py` | Downloads the BMU register, the Installed Generation Capacity per Unit report (IGCPU, report B1420), the settled half-hourly output of each BMU (dataset B1610), NESO's Transmission Entry Capacity (TEC) register, and the Renewable Energy Planning Database (REPD), and writes a lineage note and a README beside them. |
| `classify.py` | Correlates each BMU's output with the sun and writes `classes.parquet`. The tests are in `packages/studies/tests/solar_bmu_census/`. |
| `collate.py` | Joins the five capacity figures onto each census BMU, with the site's technology and position, and writes the census table to the private store. |
| `recall_check.py` | Checks the census against the TEC register's PV projects. |
| `report.py` | Prints every number the page quotes into `report.md`, counts and sums only. |
| `census_charts.py` | Draws the page's four figures. |

## The capacity columns

**Each capacity column answers a different question, so the columns are never added together.**

- `generation_capacity_mw`: the Generation Capacity in the BMU register, in megawatts, declared by
  the BMU's lead party.
- `igcpu_installed_capacity_mw`: the installed capacity of the unit in IGCPU, in megawatts.
- `tec_mw`: the matched TEC project's connected capacity if built, or its agreed cumulative capacity
  if under construction, at its most advanced status, in megawatts. A hybrid project's figure
  includes its storage.
- `largest_mel_mw`: the largest Maximum Export Limit in the 30 days before the run, in megawatts.
- `repd_installed_capacity_mw`: the installed capacity of the matched REPD row, in megawatts.

## Hand-made tables

**Two hand-made tables sit beside the scripts, because the study matches projects to BMUs by
judgement where names differ.** `site_matches_reviewed.csv` gives, for each census BMU whose Elexon
name does not resemble its project's name, the matched TEC project, the matched REPD row, the
separately registered storage BMUs at the site, and the evidence for the match.
`tec_mapping_reviewed.csv` gives, for each TEC project that lists PV, the BMUs that belong to it, or
none. `collate.py` reads the first and `recall_check.py` the second.
