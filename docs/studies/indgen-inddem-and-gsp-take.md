# Elexon's INDDEM and INDGEN reach 15 to 42 hours ahead and do not line up with the GSP groups of NGED's licence areas

**This study describes two Elexon series, INDDEM and INDGEN, beside the settled energy that each of
Great Britain's 14 Grid Supply Point (GSP) groups takes from the transmission system.** INDDEM and
INDGEN sum the Physical Notifications (PNs) of every Balancing Mechanism Unit (BMU). A PN is the
level of generation or demand that a BMU's operator says it plans to deliver. INDDEM sums the BMUs
that plan to import, and INDGEN sums the BMUs that plan to export. Elexon publishes both for the
whole of Great Britain and for 17 overlapping regions, called boundaries. The study downloads 13
months of each series, and tests whether the series carry a regional signal that could help a
forecast of NGED's four licence areas. It makes no forecast, runs no significance test, and says
nothing about whether either series improves one.

- **Do INDDEM and INDGEN reach far enough ahead to help a forecast of the next 14 days?** No.
    - An INDDEM or INDGEN issue reaches 15 to 42 hours ahead of its publication time, depending on
      the hour of publication ([Reach](#how-far-ahead-an-issue-reaches)).
    - The two series are published every half-hour, 47 half-hour slots in a UTC day
      ([Reach](#how-far-ahead-an-issue-reaches)).
- **Can the regional series be recovered from the boundaries Elexon publishes?** Yes, to within
  rounding.
    - The 17 boundaries are sums of 17 non-overlapping study zones, defined in a 2012 Elexon
      circular. In 644,640 zone half-hours the recovered zones have the wrong sign in 26
      ([Recovering the zones](#recovering-the-zones)).
    - The recovered zones satisfy a second identity, between five boundaries, to within 2 MW
      ([Recovering the zones](#recovering-the-zones)).
- **Does the sum of the sampled PNs reproduce INDDEM and INDGEN?** Sometimes, and not as a rule.
    - On four sample days, the sampled PNs summed over every BMU reproduce INDDEM's national total
      to within 5 MW in 36 of 192 half-hours. They miss it by up to 1.7 GW in others, and the study
      did not find the cause ([Summing the
      PNs](#summing-the-pns-reproduces-inddem-in-some-half-hours)).
    - INDGEN's national total is a median of 628 MW below the sampled export PNs ([Summing the
      PNs](#summing-the-pns-reproduces-inddem-in-some-half-hours)).
- **Do the GSP groups line up with the zones?** Not cleanly.
    - A fit of each zone's INDDEM on the PNs summed by GSP group gives weights near 1 for a few GSP
      groups and fractions for most ([Mapping](#the-gsp-groups-do-not-map-cleanly-onto-the-zones)).
    - After the shared daily, weekly, and seasonal cycles are removed, the highest correlation
      between a GSP group's settled take and a zone's INDDEM is 0.72, and the median over all pairs
      is 0.07 ([Mapping](#the-gsp-groups-do-not-map-cleanly-onto-the-zones)).
- **Does INDDEM match the demand of NGED's four licence areas?** No.
    - The zone that correlates best with each NGED GSP group has 0.5 to 2.5 times the group's mean
      settled take ([NGED's four
      groups](#ngeds-four-gsp-groups-against-their-best-correlated-zones)).
    - PV_Live's estimate of solar generation embedded in the distribution network adds 8% to 21% to
      the settled take of the four NGED groups, so the take is far from gross demand ([Embedded
      solar](#embedded-solar-is-8-to-21-of-the-gross-demand-of-ngeds-groups)).
- **What the data and methods can support.** The page describes 13 months of public data, 4 sampled
  days of PNs, and one description of each series. It can support a choice of what to look at next.
  It cannot support a claim that either series helps or fails to help a forecast.

![Figure 1: Highest correlation of a GSP group's AGV with a zone's INDDEM, shared cycles removed](assets/indgen_inddem_correlations.svg)

- **To forecast NGED's licence areas, treat INDDEM and INDGEN as at most a 42-hour-ahead national
  and regional signal that no table yet maps to NGED's licence areas.** The 2012 circular defines
  the zones, and no source the study found maps them to GSP groups
  ([Discussion](#discussion-what-to-use)).
- **To describe regional demand, use the settled take of the GSP groups (AGV) and not INDDEM.** AGV
  is complete for 385 settlement dates and the INDDEM zones are not comparable with it
  ([Discussion](#discussion-what-to-use)).

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5). Several independent Claude reviewers have checked the
> method, the evidence, and the prose adversarially.

## Key findings

- **An AGV value, doubled, is 93% of the national demand outturn, so AGV is in megawatt-hours per
  half-hour** ([The unit of AGV](#the-unit-of-agv)).
- **The two series differ in size and shape across the 17 zones, and the national total follows the
  daily cycle** ([What the series look like](#what-the-series-look-like)).
- **The first issue of a UTC day differs from the latest issue by 543 MW for INDDEM and 1,188 MW for
  INDGEN on average, with the largest differences in the last 2.5 hours of the UTC day** ([The first
  issue and the latest issue](#the-first-issue-and-the-latest-issue)).
- **One issue of 18,605, published at 08:31 UTC on 3 November 2025, holds only half-hours that are
  already past** ([Reach](#how-far-ahead-an-issue-reaches)).
- **The settled take of the South West group goes negative in 362 half-hours of the study window,
  which means net export to the transmission system** ([Embedded
  solar](#embedded-solar-is-8-to-21-of-the-gross-demand-of-ngeds-groups)).

## Introduction

**INDDEM and INDGEN are published for each half-hour as a sequence of issues.** Each issue holds the
values for the half-hours from its publication time to the end of the following day or the day
after. Elexon publishes an issue every 30 minutes, and the Elexon Insights API serves the issues
without a key. This study calls the sequence of issues for one half-hour its vintages. A forecast
that used the series would have to use the issue published before the forecast, so each result below
says which issue it takes.

**A boundary is one of 17 regions that the Seven Year Statement of the National Electricity
Transmission System defines by transmission constraints, and a study zone is one of 17
non-overlapping regions that make up the boundaries.** [Elexon CVA Change Circular 235
(2012)](https://assets.elexon.co.uk/wp-content/uploads/2012/04/28170121/CVA_CC235.pdf), Appendix 1,
lists which zones make up each boundary. Boundary `N` is the national total. The circular also gives
the formulas that recover each zone from the boundaries, for example zone Z17 equals boundary B13
(South West), and zone Z11 equals boundary B17 (West Midlands). The zones follow the transmission
network, so a zone does not coincide with a distribution licence area.

**A GSP group is one of 14 regions that settlement uses, and each has a letter from `_A` to `_P`.**
NGED's four licence areas are the groups `_B` (East Midlands), `_E` (West Midlands), `_K` (South
Wales), and `_L` (South West). AGV is Elexon's Aggregated GSP Group Take Volume (data flow
CDCA-I029). AGV gives the energy each group takes from the transmission system in each half-hour, as
settled. Generation embedded in a group's distribution network reduces the group's take, so AGV is
net of embedded generation and is not gross demand.

| Series | What it is | Geography | Time coverage used | Source |
|---|---|---|---|---|
| INDDEM | Sum of the final PNs of the BMUs that plan to import, MW, negative | National total and 17 boundaries | Issues published 31 August 2025 to 30 September 2026 | [Elexon Insights API](https://data.elexon.co.uk/bmrs/api/v1/datasets/INDDEM) |
| INDGEN | Sum of the final PNs of the BMUs that plan to export, MW | The same | The same | [Elexon Insights API](https://data.elexon.co.uk/bmrs/api/v1/datasets/INDGEN) |
| AGV | Settled energy a GSP group takes from the transmission system, MWh per half-hour | 14 GSP groups | Settlement dates 1 September 2025 to 20 September 2026, `SF` settlement run | [Elexon Open Settlement Data](https://elexon.co.uk/data/open-settlement-data) |
| PV_Live | Estimated solar generation that does not take part in the Balancing Mechanism, MW | NGED's four distribution licence areas | 1 September 2025 to 30 September 2026 | [Sheffield Solar PV_Live](https://www.solar.sheffield.ac.uk/pvlive/) |
| Sampled PNs | Final PNs of every BMU, as MW segments | About 2,400 BMUs | Four Wednesdays, 192 half-hours | Elexon Insights API, dataset `PN` |

## Data and methods

**All numbers on this page are exploratory.** The study names no planned contrasts, because it tests
no hypothesis, and it runs no significance test, so no interval claims significance. A correlation
is a description of 18,000 half-hours of one year. It does not say how the correlation would differ
in another year.

**The study keeps two views of each target half-hour.** The latest view takes the latest issue
published at or before the half-hour starts. The first-of-day view takes the latest issue published
at or before 00:00 UTC on the target's UTC day. Both views are lookups as of a cut-off, so neither
uses a value published after its cut-off. Unless a figure says otherwise, a figure uses the latest
view.

**The study recovers the zones by least squares on the circular's table.** Zone Z12 belongs to no
boundary and appears only in the national total, so the 17 boundaries alone cannot recover it. The
study therefore solves the 18 equations (the national total and 17 boundaries) for the 17 zones. The
system has one equation more than unknowns, so a second identity checks the table: B16 minus B11
must equal B9 minus B17 minus B8, because both sides equal zone Z10.

**AGV is the `SF` settlement run, converted to megawatts.** Elexon publishes seven settlement runs
(II, SF, R1, R2, R3, RF, DF), and a settlement period has one row for every run published so far.
Taking the `SF` run puts the whole window on one vintage. The run appears about 20 days after the
settlement date, so AGV ends on 20 September 2026. Each AGV volume is in megawatt-hours, so
multiplying by 2 gives the mean megawatts of the half-hour. An import row is positive and an export
row is negative.

**An anomaly of a series is the series minus its mean at the same UK local half-hour, the same day
type (weekday or weekend), and the same month.** Every demand series shares the daily, weekly, and
seasonal cycles, so correlations of the raw series sit high between any two. On the 91 pairs of GSP
groups, the median correlation of AGV is 0.76 for the raw series and 0.47 for the anomalies.
Subtracting only a mean daily profile leaves the median at 0.77.

**The sum of the sampled PNs is a time-weighted mean of each BMU's notified level.** The PN of a BMU
is a sequence of segments that tile the settlement period, and the level changes linearly between
the two ends of a segment. The study averages each segment's two end levels, weights by the
segment's duration, and sums across BMUs, with imports and exports summed apart. Each BMU belongs to
one GSP group in Elexon's BMU register, and the fit in [the mapping
section](#the-gsp-groups-do-not-map-cleanly-onto-the-zones) uses that group.

**The study fits the zones' weights by non-negative least squares on 36 half-hours.** For each zone,
the study regresses the zone's INDDEM on the import PNs summed by GSP group, with non-negative
weights. The study uses only the half-hours in which the sampled PNs reproduce INDDEM's national
total to within 5 MW, because the sampled PNs and INDDEM differ elsewhere for a reason the study did
not find.

**PV_Live labels each half-hour by its end, so the study subtracts 30 minutes.** AGV, INDDEM, and
the national demand outturn label each half-hour by its start. After the shift, the mean profile of
the South West's solar generation in June and July centres on 12:15 UTC, the middle of the half-hour
that starts at 12:00 UTC, which is close to solar noon at 3.5° W.

## Results

### The unit of AGV

**AGV, doubled, is 93% of the national demand outturn, with a correlation of 0.996 over 18,480
half-hours, so AGV is in megawatt-hours per half-hour.** The national demand outturn (INDO) is
Elexon's initial estimate of national demand. The ratio has a 1st to 99th percentile range of 0.87
to 0.96. INDO is also net of embedded generation, and the ratio is flat through the day, so the page
does not attribute the 7% gap to embedded generation. The cause, which may be transmission losses or
demand connected directly to the transmission system, is not identified.

![Figure 2: AGV, doubled, is 93% of the national demand outturn](assets/indgen_inddem_unit_check.svg)

### Recovering the zones

**The zones recovered from the boundaries have the expected sign in all but 26 of 644,640 zone
half-hours.** Both series are whole megawatts, so the study allows 1 MW of rounding. The 26
exceptions are 9 half-hours of INDDEM and 17 of INDGEN, all in zone Z10. The second identity, B16
minus B11 equals B9 minus B17 minus B8, holds to within 2 MW in every half-hour.

![Figure 3: Zones recovered from the boundaries have the expected sign in nearly all half-hours](assets/indgen_inddem_zone_signs.svg)

### What the series look like

**National INDDEM averages 18.1 GW and INDGEN 29.4 GW over the year, and both follow the daily
cycle.** Both are highest in winter. The monthly mean of INDDEM peaks at 21.5 GW in January 2026 and
is lowest at 15.4 GW in May 2026, and the monthly mean of INDGEN peaks at 36.1 GW in January 2026
and is lowest at 24.5 GW in July 2026. INDGEN exceeds INDDEM by 11.3 GW on average, and the study
did not find what makes up the difference.

![Figure 4: National INDDEM and INDGEN over a year and two weeks](assets/indgen_inddem_national.svg)

**The 17 zones differ in size and in the shape of their day.** Zone Z12 holds the most INDDEM, 5.3
GW on average, and no boundary equals it. INDDEM exceeds INDGEN in five zones (Z5, Z11, Z12, Z14,
and Z17) and falls short of it in the other 12. Zone Z17, the South West boundary B13, holds 555 MW
of INDDEM and 501 MW of INDGEN on average.

![Figure 5: Mean daily profile of INDDEM and INDGEN in each of the 17 zones](assets/indgen_inddem_zone_profiles.svg)

### How far ahead an issue reaches

**An INDDEM issue reaches 15 to 42 hours ahead, far short of the 14 days of the forecast horizon.**
The reach of an issue is the time from publication to the end of the last half-hour in the issue. An
issue published in the first half of the UK day reaches about 28 hours at midnight and falls by one
hour for each hour of the day to about 18 hours. The long issue appears at about 12:00 UK local
time, reaches about 42 hours, and then falls again. A UTC day holds 47 half-hour slots with an
issue, because Elexon publishes none in the half-hour that starts at 08:30 UK local time. On 16 of
the 396 UTC days, INDDEM and INDGEN each have fewer than 47 slots with an issue, and the lowest
count is 42.

**One issue holds only half-hours that are already past.** The issue published at 08:31 UTC on 3
November 2025 holds 61 half-hours that end 3.5 hours before its publication. The study did not find
why. The issue does not affect either view, because a view only takes an issue that covers the
target half-hour.

![Figure 6: Hours ahead that each INDDEM issue reaches, by time of publication](assets/indgen_inddem_reach.svg)

### The first issue and the latest issue

**The first issue of the UTC day differs from the latest issue by 543 MW (INDDEM) and 1,188 MW
(INDGEN) on average, with the largest differences in the last 2.5 hours of the UTC day.** The mean
absolute difference is 3% of INDDEM's mean and 4% of INDGEN's. The 90th percentile of the absolute
difference is 1,274 MW for INDDEM and 2,421 MW for INDGEN. For targets between 22:00 and 24:00 UTC,
the first issue is 1.6 to 2.5 GW less negative than the latest issue for INDDEM, and 3.5 to 6.3 GW
lower for INDGEN. The study did not find the cause.

![Figure 7: First issue of the UTC day minus latest issue, by target half-hour](assets/indgen_inddem_first_versus_latest.svg)

### Summing the PNs reproduces INDDEM in some half-hours

**The sampled PNs reproduce INDDEM's national total to within 5 MW in 36 of 192 half-hours, and miss
it by up to 1.7 GW in others.** The 36 half-hours are all between 00:00 and 15:59 UK local time: 20
on 4 March 2026, 10 on 17 June 2026, and 6 on 16 September 2026. None is on 10 December
2025. Across the 192 half-hours, INDDEM is a median of 373 MW less negative than the sampled import
      PNs, and INDGEN is a median of 628 MW below the sampled export PNs, by up to 2.2 GW. The two
      gaps have opposite signs and a correlation of −0.98. INDDEM has a gap above 5 MW and INDGEN a
      gap below −5 MW in 81% of half-hours. The sampled PNs are the PNs that Elexon serves now,
      which are the final PNs. The study did not find what INDDEM and INDGEN leave out of the sum.

![Figure 8: INDDEM and INDGEN beside the sum of the sampled Physical Notifications](assets/indgen_inddem_pn_against_inddem.svg)

### The GSP groups do not map cleanly onto the zones

**Fitting INDDEM on the PNs summed by GSP group places a few groups in one zone and splits most
groups across several.** Zone Z14 (London) takes the group `_C` (London) with a weight of 1.14, and
zone Z17 (the South West boundary) takes `_L` (South West) with a weight of 0.64. Zone Z6 takes
`_F`, and zone Z8 takes `_M`. Zone Z12 takes `_A`, `_F`, and `_L` with weights of 1.2, 1.7, and 1.8.
Zone Z11, which equals the West Midlands boundary, takes `_E` with a weight of only 0.16. A group
that lies wholly in one zone would have a weight near 1 there, so weights of 1.7 and 1.8 show that
the fit does not describe how the groups' PNs enter INDDEM. The study fits 17 zones on 36 half-hours
of four days, so the fit is a description of one sample and not a table of the zones' membership.

![Figure 9: Fitted weight of each GSP group in each zone's INDDEM](assets/indgen_inddem_weights.svg)

**After the shared cycles are removed, the highest correlation between a GSP group's AGV and a
zone's INDDEM is 0.72, and the median over all pairs is 0.07.** The pair with the highest
correlation is the group `_N` (southern Scotland) and zone Z5. For NGED's groups, the highest
correlation is 0.50 for `_B` with zone Z12, 0.63 for `_E` with Z12, 0.31 for `_K` with Z17, and 0.49
for `_L` with Z17. Figure 1, at the top of the page, shows every pair.

### NGED's four GSP groups against their best-correlated zones

**The zone that correlates best with an NGED group has 0.5 to 2.5 times the group's mean AGV.** Zone
Z12, the best-correlated zone for the East Midlands (`_B`) and West Midlands (`_E`), has a mean
INDDEM of 5.3 GW, against a mean AGV of 2.1 GW and 2.2 GW. Zone Z17, the best-correlated zone for
South Wales (`_K`) and the South West (`_L`), has a mean INDDEM of 555 MW, against a mean AGV of 758
MW and 1,039 MW. The zones are therefore the wrong size for a comparison with a single GSP group,
and the correlations are too low for a scaled comparison. The page does not scale either series.

![Figure 10: NGED's GSP groups' AGV beside the INDDEM of the best-correlated zone](assets/indgen_inddem_nged.svg)

### Embedded solar is 8% to 21% of the gross demand of NGED's groups

**PV_Live's estimate of the solar generation in each NGED licence area is 8% to 21% of the area's
estimated gross demand over the year, where gross demand is AGV plus the solar generation.** The
share is 8% for the West Midlands (`_E`), 15% for the East Midlands (`_B`) and South Wales (`_K`),
and 21% for the South West (`_L`). On midsummer days, AGV falls while solar rises. The South West's
AGV is negative in 362 half-hours of the window, which means the group exported to the transmission
system. PV_Live estimates only the solar generation that does not take part in the Balancing
Mechanism, so the figure leaves out the solar that BMUs report.

![Figure 11: AGV with and without PV_Live's solar for NGED's four GSP groups](assets/indgen_inddem_pv_live.svg)

## Discussion: what to use

**To forecast NGED's licence areas, INDDEM and INDGEN are not a drop-in regional input.** The series
reach at most 42 hours, so they can inform at best the first two days of a 14-day forecast. No
source the study found maps the 17 zones to GSP groups or to NGED's supply points, and the fit here
does not recover a mapping. The sum of the PNs does not reproduce the series, so a user could not
rebuild a zone from the PNs. A study that wants to test the series needs a mapping first.

**To describe the demand of a GSP group, AGV is the series to start from, and it is not available
live.** AGV is settled, so the `SF` run arrives about 20 days after the settlement date. AGV is net
of embedded generation, and PV_Live's solar estimate turns AGV towards gross demand for the licence
areas whose solar PV_Live covers. A user who needs a live signal would need a different source.

**What would change these recommendations.** A published mapping from zones to supply points, an
explanation of the 5 MW to 1.7 GW differences between the sampled PNs and INDDEM, or a longer
history of regional PNs would each change what to try next. The study does not recommend running any
forecasting experiment, and leaves that choice to the issue tracker.

## Limitations

- **The window is 13 months of one period, and one year is not a sample of years.** Every
  correlation and mean describes that year.
- **The PN sample is four Wednesdays.** The fit of zone weights uses 36 half-hours, and the
  difference between the sampled PNs and INDDEM may differ on other days.
- **AGV uses the `SF` run only.** A later run would revise the values, and the study did not measure
  how much.
- **The unit of AGV is established by a ratio to INDO.** Elexon does not state the unit in the
  files.
- **The 7% gap between AGV and INDO is unexplained.**
- **The two views take a lookup as of a cut-off.** The first view uses a UTC midnight, while the
  settlement day follows UK clock time, so its largest differences fall where the UK settlement day
  changes. The study did not separate the effect of lead from the effect of that change.
- **PV_Live estimates and revises.** The study records each estimate's update time and does not
  correct for revisions.
- **A correlation of anomalies depends on how the anomaly is defined.** The definition removes a
  mean at the half-hour, day type, and month, and another definition would give other values.
- **The page corrects for no multiple comparisons.** It reports every group-and-zone correlation,
  and no significance test applies.

## Scope

The page says nothing about whether INDDEM or INDGEN improves a forecast of NGED's substations. It
does not cover the Insights API's other boundary datasets, the settlement runs other than `SF`,
GSP-level data finer than the 14 groups, the legacy BMRS archive, or any year before September
2025. It reads no NGED telemetry, so it names no NGED generator and shows no time series of one.

## Data and code availability

All inputs are public. INDDEM, INDGEN, the PNs, and the national demand outturn come from the Elexon
Insights API, which Elexon publishes under its [licence to use BMRS
data](https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/).
Contains BMRS data © Elexon Limited copyright and database right 2026. AGV comes from Elexon's Open
Settlement Data, which Elexon [announced as open
data](https://www.elexon.co.uk/bsc/article/making-electricity-market-data-more-openly-available-2/).
PV_Live comes from [Sheffield Solar](https://www.solar.sheffield.ac.uk/pvlive/). The downloads are
in
[`studies/market_downloads/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies/market_downloads),
and the analysis is in
[`studies/indgen_inddem/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies/indgen_inddem).
The tests of the download scripts are in `packages/studies/tests/`.

## Reproducing the figures

```bash
uv run python studies/market_downloads/fetch_system_series.py --sources elexon_inddem elexon_indgen --start 2025-08-31
uv run python studies/market_downloads/fetch_agv.py
uv run python studies/market_downloads/fetch_pv_live.py
uv run python studies/market_downloads/fetch_pn_all_bmus_sample.py
uv run python studies/market_downloads/fetch_system_series.py --sources elexon_demand_outturn
uv run python studies/indgen_inddem/build_tables.py
uv run python studies/indgen_inddem/make_charts.py
```

`build_tables.py` writes `report.md` to `data/studies/per_study/indgen_inddem/`. `report.md` holds
every number on this page, or the numbers from which a derived number is computed.
