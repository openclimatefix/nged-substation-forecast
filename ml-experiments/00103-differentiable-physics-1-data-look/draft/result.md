---
id: "00103"
slug: differentiable-physics-1-data-look
title: "Differentiable physics, part 1: which substations, rows, and dates the model can use"
part: 1
theme: Long history and drift
also_relevant_to: [Pooling across substations, "Demand: calendar and weather", Embedded generation]
status: planned
size: >-
  medium, descriptive (no model). Stored: only private files under
  data/ml-experiments/00103-differentiable-physics-1-data-look/. Serving path: not touched.
  Degradation rule: not touched. Designs: one, since each output is a count or a plot. Callers:
  only studies.*, nged_data.storage, and plotting, all known.
hypothesis: >-
  The trial area's power records hold enough clean, demand-only history for the series to go ahead
  as designed: 6 MW Primaries with at least 12 months before July 2025, and 8 MVA Primaries with
  over 3 site-years each of rows with no sunlight.
prediction: >-
  Timestamp: correlation peaks at zero shift before and after 2026-03-26 08:30 UTC. Sites: D5, D9,
  D10, D11, D14, D16 each hold at least 12 months of clean history before 2025-07-01 after cuts;
  D18 holds fewer. Early calibration: cuts D14 before about June 2020 and D15's first fortnight
  (February 2025), no other series. MVA dark rows: each MVA Primary holds 3.2 to 3.4 site-years,
  about 27 in all, with about 27% at 16:00 to 21:59 local time. Generator screen: flags D15, G-a,
  G-b, D1, and D6, and none of the other 15 Primaries; D15's night level has no temperature slope
  (within ±0.1 points per °C). Clip: yes at D5 only. Heating-slope drift: not detectable.
  Lockdown: the weekday daytime departure supports leaving out March 2020 to June 2021.
decision_rule: >-
  Descriptive; each decision has a fixed rule (see "Decisions handed on" in the plan below).
  Verdict `helps` means the data supports Parts 2 and 3 unchanged or with only the predicted cuts;
  `hurts` means a finding forces a redesign (a timestamp offset, fewer than 5 usable long-history
  MW sites, or under 2 dark site-years at more than 2 MVA sites); `inconclusive` means a decision
  could not be made from the plots.
benchmark: none (no model)
row_set_id: none (descriptive; defines dp_mw_v1 and dp_mva_dark_v1 for later parts)
metric: counts, shares, slopes, and correlations
verdict: <helps | hurts | inconclusive>
prod: no
take_home_message: <written after the results>
date: <date>
pr: <pr>
---

# Differentiable physics, part 1: which substations, rows, and dates the model can use

**Plan only; the write-up replaces this page.** The series brief and the Part 1 brief fix the
design; this page records what is committed before any figure is drawn.

## Plan

- **Data.** Cleaned power (`nged_data.storage.scan_cleaned_power`) and the raw Delta table (for
  the availability figure), the 22-series list with units from the metadata table, ERA5
  `temperature_2m`, CAMS `ghi_w_m2` (end-of-hour labels), and the 66 hand-filtered switching
  transfers (`~is_sentinel`, `kind == "real_operation"`, visibility `conserved` or `one_sided`),
  masked from `onset_utc` to `end_utc` at the donor, both sides, and every receiver. Period 2019-09-01
  to 2026-06-30. CPU only, 4 threads.
- **Figures 1 to 11**, in order: availability by month with switching ticks; raw weeks; monthly
  level by hour band; early-calibration check; weekday and weekend profile heatmaps; clip look;
  night power against temperature; correlation of monthly levels; generator screen; timestamp
  convention; MVA dark rows.
- **Printed checks.** Duplicate keys, NaN and null power, stuck runs of 12 or more half-hours,
  zeros outside the `substation_zero` flag, the per-month step check, MVA |S| below zero, and the
  scarce-data census (cold days, bank holidays, Christmas-run days per site and scoring year).
- **Decisions handed on**, each with a rule fixed now: site list and cut dates; early-calibration
  cuts (drop leading months while a band differs by more than 15 points from the same months a year
  later); MVA dark-row counts (a site under 2 site-years leaves the pool); generator screen (flag if
  negative dark share exceeds 1%, or a winter night slope within ±0.1 points per °C with an
  amplitude-to-mean ratio under 0.3; G-a, G-b, and D15 are positive controls); timestamp convention
  (zero shift confirms; any other peak stops the series); clip yes or no (in at least 2 summers the
  slope above 600 W/m² is above −1 point per 100 W/m² while the slope below 500 W/m² is below −4);
  heating-slope drift (first and last winter intervals disjoint in the same direction at 2 or more
  long-history sites, winter 2020-21 left out); lockdown window (months where at least 3 of D5, D10,
  D11, D14 depart from the same month of 2023-24 by over 5 points in the weekday 09:00 to 16:59
  band); switching-mask census.
- **Files handed on** (private): `sites.parquet`, `rows.parquet` (flags `in_pool`, `early_cut`,
  `lockdown`, `switching_mask`, `dark`, `scarce_kind`), `decisions.json`.
- **Review pipeline.** One code review and one numbers review, a prose sweep before any merge.
  The series brief asks Opus for the code and the reviews; this session writes the code itself and
  asks the manager to relay reviews.
