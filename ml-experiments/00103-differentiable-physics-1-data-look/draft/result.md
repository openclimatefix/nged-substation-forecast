---
id: "00103"
slug: differentiable-physics-1-data-look
title: "Differentiable physics, part 1: which substations, rows, and dates the model can use"
part: 1
theme: Long history and drift
also_relevant_to: [Pooling across substations, "Demand: calendar and weather", Embedded generation]
status: done
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
  D18 holds fewer. Early calibration: cuts D14 before about June 2020 and D15's first fortnight,
  no other series. MVA dark rows: each MVA Primary holds 3.2 to 3.4 site-years, about 27 in all,
  with about 27% at 16:00 to 21:59 local time. Generator screen: flags D15, G-a, G-b, D1, and D6,
  and none of the other 15 Primaries; D15's night level has no temperature slope (within ±0.1
  points per °C). Clip: yes at D5 only. Heating-slope drift: not detectable. Lockdown: the weekday
  daytime departure supports leaving out March 2020 to June 2021 (superseded: the series brief now
  fixes the window at 2020-03-23 to 2021-07-19).
decision_rule: >-
  Descriptive; each decision has a fixed rule, listed in the bullets below. Verdict `helps` means
  the data supports Parts 2 and 3 with no redesign; `hurts` means a finding forces a redesign (a
  timestamp offset, fewer than 5 eligible long-history MW sites, or under 0.5 dark site-years in a
  part's training window at more than 2 MVA sites); `inconclusive` means a decision could not be
  made from the plots. The cuts went beyond the predicted two series, and the dark-site-year
  threshold is the series brief's amended 0.5, so the wording differs from the plan's.
benchmark: none (no model)
row_set_id: none (descriptive; defines dp_mw_v1 and dp_mva_dark_v1 for later parts)
metric: counts, shares, slopes, and correlations
verdict: helps
prod: no
take_home_message: >-
  The data lets the series go ahead, with three changes. Cut leading months at 13 of the 16
  Primaries (19 months at D4). Fix the lockdown window in advance, because the data does not
  isolate it from drift. Count eligible long-history MW sites at each tuning year's first day, which
  gives 5 at 2024-07-01 because D9 holds only 9.0 months then (6 at 2025-07-01). The timestamp
  check is mixed per site, and whether that stops the series is the maintainer's decision.
date: 2026-10-19
pr: 1348
advice:
  - >-
    Start each Primary at the first usable date in the site list, which cuts 0.7 to 19.4 months at
    13 of the 16 Primaries, because their first months sit far from the same months a year later
    (figure 4).
  - >-
    Leave the fixed lockdown window (2020-03-23 to 2021-07-19) out of every pool except the
    COVID-flag arm, and do not try to detect it: weekday daytime levels sit 5 to 17 percentage
    points above a no-lockdown fit from August 2020 to September 2021 (figure 12).
  - >-
    Admit all 8 MVA Primaries to Part 3: each holds 0.90 to 0.96 dark site-years in 2023-07 to
    2025-06, above the 0.5 needed, and 2.36 to 2.68 over all history with the lockdown removed.
  - >-
    Count D9 as ineligible for tuning years that start before 2024-07-01 and eligible only
    from 2025-07-01 on (it holds 9.0 months before 2024-07-01 and 21.0 before 2025-07-01).
  - >-
    Screen generators on rows after the early-calibration cuts and the switching masks: D18's 1.7%
    negative dark share is 0% from its first usable date (figure 9).
  - >-
    Do not read the plan's summer-midday timestamp test as evidence for or against a stamp offset:
    its correlation is about -0.18 and flat across shifts, and the clear-sky-hours test peaks
    between 0 and +30 minutes (figure 10).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00103-differentiable-physics-1-data-look/
---

# Differentiable physics, part 1: which substations, rows, and dates the model can use

**The data lets the differentiable-physics series go ahead, with three changes.** Cut the leading
months at 13 of the 16 Primary substations, fix the lockdown window in advance, and count eligible
long-history MW sites at each tuning year's first day (5 at 2024-07-01, because D9 holds only 9.0
months then). Verdict: `helps`, exactly at the minimum of 5 eligible sites. The timestamp check is
mixed per site, and the manager has asked the maintainer whether that stops the series.

This experiment fits no model. It looks at the power records of 22 series (16 Primary substations,
four grid or bulk supply points, and two metered generators), at ERA5 temperature (the European
Centre for Medium-Range Weather Forecasts' fifth-generation reanalysis) and CAMS irradiance (the
Copernicus Atmosphere Monitoring Service), both estimates of past weather, and at 66
hand-filtered switching events. It decides which substations, rows, and dates the later parts can
use. A Primary substation is the transformer site that feeds a town or district. A MW Primary
reports net power in megawatts, and an MVA Primary reports apparent power in megavolt-amperes, so
only its dark rows (CAMS irradiance below 1 W/m²) show demand alone. A site-year is 17,532
half-hours. Power is shown as a percentage of each series' effective capacity.

## Figures

Figures 4 and 12 carry the findings that change Part 2. Figures 1 and 2 show the raw data. The two
metered generators appear only in figure 9, as G-a and G-b, as aggregate statistics with no dates.

![Months with clean data per series, 2019 to 2026, with switching events](figure_01_availability.svg)

![Net power in four weeks at the 8 MW Primaries](figure_02_raw_weeks.svg)

![Monthly level by hour band at the 8 MW Primaries](figure_03_monthly_levels.svg)

![First 12 months against the same months a year later, 16 Primaries](figure_04_early_calibration.svg)

![Mean daily profile by season-year at the 8 MW Primaries](figure_05_profiles.svg)

![Summer midday power against irradiance at the 8 MW Primaries](figure_06_clip.svg)

![Night power against temperature each winter](figure_07_heating.svg)

![Correlation of monthly levels across five long-history sites](figure_08_correlation.svg)

![Generator screen for all 22 series](figure_09_generator_screen.svg)

![Correlation with irradiance as the power stamps are shifted](figure_10_timestamp.svg)

![Dark site-years and dark-row profiles at the 8 MVA Primaries](figure_11_mva_dark.svg)

![Weekday daytime level against a no-lockdown fit](figure_12_lockdown.svg)

## Sites and dates

- **Roles** (`sites.parquet`): `mw_long` for D5, D9, D10, D11, D14, and D16; `mw_short` for D18;
  `generator_case` for D15; `mva_dark` for D2, D3, D4, D8, D12, D19, D21, and D22; `excluded` for
  D1, D6, D13, D17, G-a, and G-b.
- **First usable dates** come from the series design session's review of every series' start,
  which Part 1 may only move later: D4 2021-05-01, D14 2020-07-01, D12 2020-05-01, D5 2020-02-01, D3
  2020-01-01, D10 and D21 2019-12-01, D22 2019-11-01, D19 2019-10-15, D16 2021-02-01, D9 2023-10-01,
  D18 and D15 2025-03-01, and 2019-09-17 at D2, D8, and D11.
- **D12 keeps the rule's later date.** The review recommends no cut, and the 15-point rule cuts 7.4
  months (figure 4), so D12 starts 2020-05-01.
- **Months before a tuning year's first day** (2022-07-01, 2023-07-01, 2024-07-01, 2025-07-01): the
  six long-history sites hold 12 or more months before 2025-07-01 (D9 21.0). Five hold 12 or more
  before each of the other three days. D9 holds 9.0 months before 2024-07-01.
- **Dark site-years per MVA Primary:** 2.44 to 3.27 over all history, 2.36 to 2.68 with the lockdown
  window removed (20.3 in total), and 0.90 to 0.96 in 2023-07 to 2025-06, Part 3's training window.
  The share at 16:00 to 21:59 local time is 27.3%, 27.9%, and 27.2% in those three windows.
- **Row sets:** `dp_mw_v1` holds 513,379 half-hours and `dp_mva_dark_v1` holds 355,473 dark
  half-hours (`rows.parquet`; rows in the lockdown window or under a switching mask are flagged and
  left out of both).

## Decisions

- **Early-calibration rule.** The rule drops leading months while the night band (00:00 to 04:59) or
  the weekday daytime band (09:00 to 16:59) differs by more than 15 percentage points from the same
  month a year later. It cuts D14 9, D12 8, D15 8, D4 3, D18 2, and 1 month at D9, D16, D21, and D22
  (figure 4).
- **Generator screen** (negative dark share above 1%, run after cuts and masks on MW series only).
  It flags D1 (73%), D6 (26%), D15 (82%), G-a (62%), and G-b (65%). It leaves the other 15
  Primaries unflagged, among them D18 (0% after its cut, 1.7% before it with 201 of 202 negative
  rows in January 2025), and the bulk supply points D13 (0%) and D17 (0.09%). The MVA Primaries go
  unscreened.
- **Timestamp, clear-sky hours in April to September.** The mean correlation is most negative at
  zero shift, because demand falls as irradiance rises (-0.563 before the 2026-03-26 fix and -0.659
  after). Per site, the peak is at +30 minutes at 6 of 8 before the fix, and at -30 minutes at D5
  on both sides.
- **Timestamp, size of the effect.** The correlation at +30 minutes is weaker than at zero by
  0.007 before the fix and 0.013 after, and at -30 minutes by 0.042 and 0.037. The parabola vertex
  is at +10.6 minutes before the fix and +6.9 after.
- **Timestamp, the plan's test.** On summer middays (May to August, 10:00 to 14:00 UTC, clear-sky
  irradiance above 400 W/m²) the mean correlation is about -0.18 before and -0.06 after, and flat
  across shifts, so the test cannot locate a peak.
- **Timestamp, caveats.** The after side holds only April to 10 September 2026 (about 8,170 hourly
  rows), and no known-answer check was run.
- **Clip** (daytime power flattening above a ceiling): yes at D5 (4 of 7 summers) and D15 (2 of 2),
  and no at the other six MW Primaries; D14 meets the rule in 1 of 6 summers.
- **Heating-slope drift** is not detectable. No long-history site has a last-minus-first winter
  difference whose 95% interval excludes zero (D9's runs from -0.06 to +0.26 points per °C). The
  test uses the interval of the difference, which is stricter than the plan's non-overlap test.
- **Lockdown window.** The series brief fixes it at 2020-03-23 to 2021-07-19. A window detected
  from the data would not work: in April 2020 D11 sits 26 percentage points below a no-lockdown
  fit and D10 7, but from August 2020 to September 2021 the four sites with 2019 history sit
  mostly 5 to 17 percentage points above it (figure 12). The fit leaves similar excursions in
  months it was fitted on, so the departures show a poor fit as much as drift.
- **Correlation of monthly levels** across D5, D10, D11, D14, and D16: the mean pairwise correlation
  is 0.56 for the night band (60 months) and 0.47 for the evening band (17:00 to 20:59; 59 months),
  and the first principal component holds 66% and 58% of the variance (figure 8).
- **Switching masks.** 16,963 half-hours on the full grid are masked on the 16 Primaries, 15,522 of
  them in rows with data, the most at D5 (3,682) and the least at D2 (17). The 66 events start in
  2022-03, so no mask precedes then.

## Predictions

- **Held:** the six sites hold 12 months or more before 2025-07-01 and D18 fewer; about 27% of dark
  half-hours fall at 16:00 to 21:59; the screen flags D1, D6, D15, G-a, and G-b and none of the
  other 15 Primaries; heating-slope drift is not detectable; D14 and D15 are cut.
- **Failed:** the timestamp peak is at zero per site only after the fix at 5 of 8 sites; 13
  Primaries are cut, not 2; the MVA Primaries hold 2.44 to 3.27 dark site-years with the lockdown
  kept (24.4 in total), and 5 of 8 are below 3.2; D15's winter night slope is +0.48 points per °C,
  not within ±0.1; clip is at D5 and D15, not D5 only.
- **Superseded:** the lockdown prediction, now a fixed window.

## Data validation

- **Keys and values:** no duplicate `(series, time)` keys, no NaN power, no null irradiance or
  temperature to 2026-06-30, and no MVA value below zero.
- **Stuck runs of 12 or more identical half-hours, and zeros outside the `substation_zero` flag:**
  found only at the two metered generators.
- **Month-mean steps over 3 monthly standard deviations:** D11 (2020-04), D2, D4, D12, D18, D19,
  D21, and D6, a grid supply point whose monthly mean falls 218 MW in 2023-07.
- **Scarce days** (mean per long-history MW site): 3.8, 5.0, and 3.5 cold days (daily-mean
  temperature below 0 °C) in 2023-24, 2024-25, and 2025-26, and 12.0 in November 2022 to February
  2023. A full year holds 8 bank holidays, and the Christmas run (24 December to 1 January) holds
  9 days.

## Caveats

- ERA5 and CAMS estimates of past weather flatter the PV term (experiment 46).
- The switching masks and the early-calibration cuts are hindsight decisions: each cut compares
  months with a year later, and the masks come from labels written after the events.
- Four sites decide the lockdown figure, and D14 enters only from July 2020.
- The D9, D16, and D18 first-month cuts rest on very few readings.
- Every printed number is in `report.txt`.

## What the next experiment should answer

Part 2 should use the site list above and count eligible long-history sites at each tuning year's
first day, a proposal the series planner should confirm, because the series brief counts 6 sites
(4 of 6 needed) and this count gives 5 at 2024-07-01 (4 of 5). All 8 MVA Primaries clear the 0.5
dark site-year entry rule in Part 3's window. The shared slow factor (design item D2) is worth
keeping, given the correlations above.

## Review triage

- **Accepted and fixed:** the switching mask now includes the onset half-hour and ends at
  `end_utc`; first usable dates follow the design session's review; `default_rows` leaves out the
  lockdown and masked rows (`in_pool` stays the pool membership); D15 and D17 no longer use data
  or cuts that conflict with the review; the generator tables and report no longer print dates,
  and figure 9 sets `aria=False`; Part 3's window is 2023-07 to 2025-06; the July 2026 half-hour
  no longer enters the monthly levels; the correlations and census use rows after the cuts (and,
  for the correlations, the masks); `rows.parquet` carries `early_cut`; `sites.parquet` carries
  the clip and generator flags; `decisions.json` carries the cut dates, screen flags, dark counts,
  and masks.
- **Accepted as caveats:** the timestamp stop rule is left to the maintainer, the summer-midday test
  is reported beside the clear-sky one, and the eligibility count is a proposal.
- **Not changed:** the heating-drift test and the one-criterion generator screen differ from the
  plan, and both follow the series brief or are stricter.
