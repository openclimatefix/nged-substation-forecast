# INDDEM and INDGEN: do Elexon's indicated demand and generation line up with the GSP groups

**This folder holds the scripts behind [Do INDDEM and INDGEN line up with the GSP groups of NGED's
licence areas?](https://openclimatefix.github.io/nged-substation-forecast/studies/indgen-inddem-and-gsp-take/).**
The study describes Elexon's indicated demand (INDDEM) and indicated generation (INDGEN) series beside
the settled energy that each Grid Supply Point (GSP) group takes from the transmission system (AGV).
Every script's module docstring gives the command that runs it.

**A script imports only from this folder, from `studies.*`, and from the other reviewed packages.**

**The downloads live in `studies/market_downloads/` and run first.**

| Script | What it does |
|---|---|
| `market_downloads/fetch_system_series.py --sources elexon_inddem elexon_indgen --start 2025-08-31` | Downloads every INDDEM and INDGEN issue published from 31 August 2025 to 30 September 2026. |
| `market_downloads/fetch_system_series.py --sources elexon_demand_outturn` | Downloads the national demand outturn that the unit check uses. |
| `market_downloads/fetch_agv.py` | Downloads Elexon's AGV zips and keeps the `SF` settlement run. |
| `market_downloads/fetch_pv_live.py` | Downloads PV_Live's solar generation for NGED's four licence areas. |
| `market_downloads/fetch_pn_all_bmus_sample.py` | Downloads every BMU's Physical Notifications for four sample days. |
| `indgen_inddem_common.py` | Holds the paths, the study window, and the boundary-to-zone table from Elexon CVA Change Circular 235. |
| `build_tables.py` | Builds the two views of every target half-hour, recovers the zones, joins AGV, PV_Live, and the sampled Physical Notifications, writes the tables, and writes `report.md`, which holds every number the page quotes. |
| `make_charts.py` | Draws the page's 11 figures from the tables, as SVG under `docs/studies/assets/`. |

**The tests for the download scripts are in `packages/studies/tests/`.** They are
`test_market_system_series.py`, `test_market_downloads.py`, and `test_market_agv_pv_live.py`.
`build_tables.py` and `make_charts.py` have no tests, so the page's numbers are checked against
`report.md`.

**The tables and `report.md` are written to `data/studies/per_study/indgen_inddem/`.**
