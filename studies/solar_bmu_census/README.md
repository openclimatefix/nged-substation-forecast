# Solar-BMU census: which Balancing Mechanism Units in Great Britain are solar

**This folder holds the scripts behind [How many Balancing Mechanism Units in Great Britain are
solar?](https://openclimatefix.github.io/nged-substation-forecast/studies/solar-bmu-census/).** The
study finds the Balancing Mechanism Units (BMUs) whose settled output follows the sun or that the
Installed Generation Capacity per Unit report (IGCPU) types as Solar, and sets the five published
capacity values for them side by side, with the 99th percentile of each BMU's own output. Every
script's module docstring gives the command that runs it.

**A script imports only from this folder, from `studies.*`, and from the other reviewed packages.**

**The scripts run in the order of the table below, and each script reads files that earlier scripts
wrote.**

| Script | What it does |
|---|---|
| `fetch_sources.py` | Downloads the BMU register, the IGCPU report (B1420), the settled half-hourly output of each BMU (dataset B1610), the National Energy System Operator's (NESO's) Transmission Entry Capacity (TEC) register, the Renewable Energy Planning Database (REPD), and NESO's map of the 14 distribution network operator (DNO) licence areas, then writes a lineage note and a README beside them. |
| `classify.py` | Correlates each BMU's output with the sun and writes `classes.parquet`. |
| `collate.py` | Fetches each census BMU's Maximum Export Limit (MEL), joins the five capacity values and the P99 of the BMU's output onto each census BMU with the site's technology and position, and writes the census table as `solar_bmus.csv` and `solar_bmus.parquet` in the study's data folder. |
| `recall_check.py` | Checks the census against the TEC register's photovoltaic (PV) projects. |
| `report.py` | Writes every number the page quotes to `report.md` and prints the report. |
| `census_charts.py` | Draws the page's five figures. |

The tests for `fetch_sources.py`, `classify.py`, and `collate.py` are in
`packages/studies/tests/solar_bmu_census/`.

## The capacity columns

**Each capacity column answers a different question, so the columns are never added together.**

- `generation_capacity_mw`: the Generation Capacity in the BMU register, in megawatts, declared by
  the BMU's lead party.
- `igcpu_installed_capacity_mw`: the installed capacity of the unit in IGCPU, in megawatts.
- `tec_mw`: the matched TEC project's connected capacity if built, or its agreed cumulative capacity
  at any other status, at its most advanced status, in megawatts. A hybrid project's value
  includes its storage.
- `largest_mel_mw`: the largest Maximum Export Limit in the 30 days before the run, in megawatts.
- `repd_installed_capacity_mw`: the installed capacity of the matched REPD row, in megawatts.

**A sixth column, `p99_output_mw`, measures the BMU's own output and is not a registered
capacity.** It is the 99th percentile (linear interpolation) of the BMU's half-hourly settled output
in megawatts (the megawatt-hours in each half-hour, times 2). The percentile is taken over the
series the classifier judges: after the 30 days that follow the BMU's first output (unless the BMU
was running in the window's first week), and without the half-hours of exactly zero output while the
sun is clearly up. Zeros at night and negative readings stay in. The column is empty for a BMU with
no judged output. It is never added to another column.

**`report.py` prints two more measures of observed output for each group, and neither is a
registered capacity.** `max of output (MW)` is the sum over the group's BMUs of each BMU's largest
half-hourly output (`classify.largest_output_mw`). Because the BMUs' largest outputs fall in
different half-hours, the sum overstates what the group delivered at once. `highest combined output
in one half-hour (MW)` is the maximum over time of the group's summed half-hourly output
(`report.coincident_peak_mw`). Neither is added to a registered capacity.

**`report.py` also estimates the solar part of each BMU's AC capacity (`solar_estimate.py`).** The
model is `min(r * a * c(t), a)`, with `a` the AC capacity, `r` the DC:AC ratio, and `c(t)` a shape.
The base case takes the cosine of the solar zenith as `c(t)`, fits `a` to the upper envelope of
output, and sets `r` to 1.4 (the median for fixed-tilt projects in Lawrence Berkeley National
Laboratory's Utility-Scale Solar report for 2023). The alternative shapes are the CAMS irradiance at
the BMU's own position and the mean CAMS irradiance of 18 grid points across Great Britain, both
read from `solar_estimate.CAMS_PUBLIC_POINTS_PATH`, and `report.md` compares the three on the
single-site BMUs that follow the sun.

## The location columns

**Six columns say where a BMU is, and none of them is a capacity.**

- `gsp_group`: the Elexon grid supply point (GSP) group identifier in the BMU register, such as
  `_B`. The register leaves the group empty for every transmission-connected (`T_`) BMU, so the
  column is empty for the nine transmission-connected single-site census BMUs.
- `dno_area`: the distribution network operator (DNO) whose licence area the GSP group names, from
  `collate.GSP_GROUP_AREAS`. The mapping is the attribute table of NESO's map of the 14 DNO licence
  areas. The column is empty when `gsp_group` is empty.
- `position_gsp_group` and `licence_area_by_position`: the GSP group identifier and the DNO of the
  licence area in NESO's map that contains the position of the matched REPD row. NESO calls the
  boundaries approximate. The columns are empty when the BMU has no REPD position.
- `km_to_nearest_other_area`: the distance in kilometres from that position to the nearest edge of
  any other licence area, which says how far inside its area the position sits.
- `repd_county` and `repd_region`: the county and region fields of the matched REPD row. These are
  REPD's own fields and are not licence areas.

## Hand-made tables

**Two hand-made tables sit beside the scripts, because the study matches projects to BMUs by
judgement where names differ.** `site_matches_reviewed.csv` gives, for each census BMU whose Elexon
name does not resemble its project's name, the matched TEC project, the matched REPD row, the
separately registered storage BMUs at the site, and the evidence for the match.
`tec_mapping_reviewed.csv` gives, for each TEC project that lists PV, the BMUs that belong to it, or
none. `collate.py` reads `site_matches_reviewed.csv`, and `recall_check.py` reads
`tec_mapping_reviewed.csv`.
