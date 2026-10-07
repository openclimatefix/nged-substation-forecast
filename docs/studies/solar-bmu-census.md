# At least 10 Balancing Mechanism Units at 9 solar sites in Great Britain, and 8 of those sites have storage built or planned

**This study asks how many Balancing Mechanism Units (BMUs) in Great Britain are solar, and what
capacity each published figure gives those solar BMUs.** A BMU is the unit of trade in the Balancing
Mechanism, which the National Energy System Operator (NESO) uses to balance supply and demand.
Elexon's BMU register does not say which BMUs are solar. None of the register's 3,115 rows has a
fuel type of solar, and 2,525 rows have no fuel type at all. Of the 3,115 rows, 1,323 belong to
interconnectors between Great Britain and another country, 86 have no BMU identifier, and 1 repeats
an identifier, which leaves 1,705 BMUs. The study downloads 12 months of settled half-hourly output
(Elexon's report B1610, actual generation output per generation unit) for each of the 1,705 BMUs,
and asks whether each BMU's output follows the sun. [Why a single lookup is not
enough](#why-a-single-lookup-is-not-enough) explains why the register alone cannot answer the
question.

**Ten single-site BMUs are solar: nine follow the sun, and a tenth is typed as solar in Elexon's
Installed Generation Capacity per Unit (IGCPU) report but has no output yet.** All ten BMUs have a
`T_` or `E_` identifier, and the study treats each BMU with a `T_` or `E_` identifier as part of one
generating site. Cleve Hill's two BMUs share one site, so the ten BMUs sit at nine sites. Nine of
the ten BMUs sit at eight hybrid sites, which also hold storage, built or planned. Four of those
eight sites have a storage BMU of their own. The tenth BMU sits at a pure photovoltaic (PV) site,
where the study found no storage. The study reports totals for all ten BMUs, for the nine BMUs at
hybrid sites, and for the one BMU at the pure PV site.

**A further 28 aggregate BMUs also follow the sun or are typed as solar, but most of their
Generation Capacity is not solar capacity.** An aggregate BMU has a supplier, virtual, or other
identifier that does not name a single site. Five BMUs of TotalEnergies Gas & Power, which both
import and export heavily, hold 86% of the 28 BMUs' Generation Capacity, so the study reports the 28
aggregate BMUs apart.

**Each of the 12 built PV projects in NESO's Transmission Entry Capacity (TEC) register maps to a
BMU, but the census cannot say how many embedded solar BMUs it missed.** Seven of the 12 projects
map to a BMU in the census. An embedded BMU is connected to a distribution network.

**The study reads six public sources, and each gives a different kind of capacity.** The table says
who publishes each source, what it lists, and which capacity figure the study takes from it.

| Source | Published by | What it lists | Capacity figure the study takes |
|---|---|---|---|
| BMU register (`reference/bmunits/all`) | Elexon, which settles the Balancing Mechanism | Every registered BMU, with its identifier, name, lead party, and fuel type | Generation Capacity, which the lead party declares |
| B1610 | Elexon | Each BMU's settled output in each half-hour | None. The study uses the output itself |
| IGCPU (Installed Generation Capacity per Unit, report B1420) | Elexon, for NESO | Installed capacity of each generating unit, with a resource type such as "Solar" | Installed capacity, as NESO lists it |
| MELS | Elexon | The Maximum Export Limit (MEL) that each lead party submits for each BMU | The largest limit in the 30 days before the run |
| TEC register | NESO | Each project's export capacity agreed at the grid connection, by stage, with its plant types | Connected capacity if built, agreed capacity if not |
| REPD (Renewable Energy Planning Database) | Department for Energy Security and Net Zero | Each renewable project planned or built, with its technology, status, and grid position | Installed capacity of the project |

![Figure 1: Single-site BMUs' correlation with the sun, with a gap from 0.49 to 0.83](assets/solar_bmu_census_correlation.svg)

- **To find solar BMUs, read their output and not the register.** The fuel-type field never says
  solar, and a name search misses BMUs whose name does not mention solar.
- **To compare capacities, keep the five published capacity figures apart (B1610 gives output, not a
  capacity).** Generation Capacity and the largest Maximum Export Limit differ by up to 20.6 MW for
  one BMU, and the TEC register and REPD give a figure for a whole project, which can include
  storage.
- **At the four hybrid sites with a storage BMU, the solar BMU is metered apart from the storage on
  all but two days.** On 26 and 29 January 2026, Tebworth PV Power Park imported up to 35.3 MW, by
  night as well as by day.

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5). Several independent Claude reviewers have checked the
> method, the evidence, and the prose adversarially.

## Key findings

- **A wide gap in correlation with the sun, from 0.49 to 0.83, separates the nine single-site BMUs
  that follow the sun from the other single-site BMUs**, so the census does not depend on where the
  threshold sits in that gap ([Figure 1](#the-census)).
- **The ten single-site solar BMUs have 682.9 MW of Generation Capacity in all**, 50.0 MW of it at
  the pure PV site and 633.0 MW at the nine BMUs at hybrid sites ([Capacity](#capacity)).
- **Nine of the ten single-site solar BMUs lie south of 53°N in England, and the tenth lies in
  Scotland** ([Figure 2](#where-the-bmus-are)).
- **A solar BMU at a hybrid site follows the sun like the pure PV BMU, and at the four sites with a
  storage BMU the storage charges at midday and discharges in the early evening** ([Figures 3 and
  4](#what-a-solar-bmu-looks-like)).
- **All 12 built PV projects in the Transmission Entry Capacity register map to a BMU, and 7 of the
  12 projects map to a BMU in the census**, so the census missed no separately registered solar BMU
  that the hand mapping found at a built project ([Recall](#how-many-solar-bmus-were-missed)).
- **A further 28 aggregate BMUs, whose identifiers do not name a single site, also follow the sun or
  are typed as solar in IGCPU**, and their Generation Capacity is mostly not solar capacity
  ([Aggregates](#aggregate-bmus)).

## Introduction

**Several datasets publish a capacity for a BMU, and each figure measures a different quantity.**
The five figures are what the BMU's lead party declared the BMU could generate (Generation
Capacity), what NESO lists as installed (IGCPU), what the site may export to the grid (TEC), the
highest export level the lead party submitted to NESO (Maximum Export Limit), and what the Renewable
Energy Planning Database (REPD) holds for the project. Choosing among the five figures needs a list
of the solar BMUs first, and a single lookup cannot give that list.

### Why a single lookup is not enough

**A single lookup of the BMU register looks like enough, and it fails in four ways.** The lookup
would filter the BMU register for solar BMUs and add up their capacity. Each of the four problems
below breaks that plan, and together they are why the study reads several registers and then the
output itself.

**No register says which BMUs are solar.** The BMU register has a fuel-type field, but no row holds
a solar value: 2,525 of the 3,115 rows are blank, 103 say `OTHER`, and the rest name wind, gas,
hydro, nuclear, and similar fuels. IGCPU gives a resource type, but types only 9 BMUs as Solar and
79 as "Generation", which names no technology. Two of those 9 BMUs have no output in the window, so
the typed list is also not a list of BMUs that generate.

**A name search misses BMUs, and most aggregate BMUs have no name to search.** Three of the ten
single-site census BMUs have a register name that says neither solar nor PV: Beechgreen Energy Farm,
Kincraig, and Breach, whose register name is its own identifier. Of the 28 aggregate BMUs, 24 have
only their identifier as a name, so a name search cannot read them.

**The project registers do not name the BMU.** The TEC register and REPD each list projects and
carry no BMU identifier. Matching a BMU to its project took name similarity and, for 8 of the 10
single-site census BMUs, a hand match from the customer's name, the lead party, the capacity, and
the connection site.

**Capacity is five different numbers, and adding a column overcounts.** Generation Capacity is a
figure the lead party declares. IGCPU gives an installed capacity, TEC an export capacity agreed at
the grid connection, the Maximum Export Limit a level the lead party submits, and REPD the installed
capacity of the project. TEC and REPD describe a project, and one project can hold two BMUs: adding
the TEC column over the ten BMUs gives 1,083.8 MW, against 733.8 MW when each project is counted
once. A hybrid project's TEC figure can also include its storage. For the 28 aggregate BMUs that
follow the sun or are typed as solar, adding Generation Capacity gives 1,896.8 MW, of which 86%
belongs to five supplier BMUs that both import and export heavily, so most of that sum is not solar
capacity.

**Output identifies the technology where no register does, and it needs care.** A solar BMU's output
is zero at night and follows the sun's height by day, and the output of wind, gas, hydro, and
nuclear BMUs does not. The study therefore correlates each BMU's half-hourly output with the cosine
of the solar zenith angle over a year. A rule such as "zero at night" would fail in both directions:
it admits any BMU that is idle at night, and it rejects a solar BMU with a metering fault or a
battery that charges overnight. The correlation ranks every BMU on one scale, and the result is a
wide gap: 9 of the 689 single-site BMUs score 0.83 or more, and the highest of the other 490
single-site BMUs with enough output is 0.49, a hydro BMU. Two cleaning rules keep the correlation
honest. Output in the 30 days after a new site first generates is left out, because sites commission
in stages, and half-hours of exactly zero output while the sun is up are removed, because they are
faults and not weather.

### How this study differs from the other studies

**The study compares no weather products and fits no forecasting model.** The other studies in this
section work from weather data and from the output of generators connected to the electricity
network of National Grid Electricity Distribution (NGED). This study uses only public data and
covers all of Great Britain. Its numbers describe the solar BMUs registered in the Balancing
Mechanism, not the solar generation in NGED's trial area.

## Data and methods

**Planned and exploratory.** The study plan, written before any result, asked how many solar BMUs
exist, and what share of the solar BMUs the census finds (its recall). The study runs no
significance test, so no contrast is planned or exploratory. Every number is a count, a sum, or a
correlation.

**Windows and dates.** The B1610 window is 1 September 2025 to 31 August 2026: the 12 complete
months that end at least 2 weeks before the fetch, because Elexon publishes B1610 about a week in
arrears. The MELS window is the 30 days from 7 September to 7 October 2026, which falls after the
output window. The TEC register is the release dated 5 October 2026, and REPD is the Q2 2026
release. All six sources were fetched live on 7 October 2026.

**Classifying a BMU.** The classifier takes each BMU's output and drops two kinds of half-hour. The
classifier drops the 30 days after a BMU's first output, unless the BMU was already running in the
first week of the window, because a site often commissions in stages. The classifier also drops
half-hours of exactly zero output while the sun is clearly up (the cosine of the solar zenith angle
above 0.1). Zero output in daylight from a solar BMU is an outage, curtailment, or a metering fault,
not weather. Output of 0.01 megawatt-hours or less in a half-hour counts as meter noise, not
generation.

**The classifier scores each BMU by how closely its output follows the sun at one point in central
Great Britain.** The classifier computes the Pearson correlation between the output and the cosine
of the solar zenith angle, clipped at zero below the horizon, at the point 53°N, 1.5°W. The BMU
register gives no coordinates for a BMU, so the classifier uses one point. The census sites span
longitude 2.5°W to 1.1°E, so their solar noon differs by about 14 minutes, which is small beside the
half-hour resolution of the output. A BMU follows the sun if its correlation is above 0.6 and it has
at least 100 half-hours of output above 0.01 megawatt-hours. A BMU with fewer than 100 such
half-hours, or with constant output, has too little output to judge.

**The census holds every BMU that follows the sun or that IGCPU types as Solar.** IGCPU types 9 BMUs
as Solar.

**Single-site and aggregate BMUs.** Elexon's naming convention gives a BMU connected directly to the
transmission network a `T_` identifier, a BMU embedded in a distribution network an `E_` identifier,
and a miscellaneous BMU an `M_` identifier. The study treats a BMU with any of those three prefixes
as a single-site BMU, which is part of one site. No `M_` BMU is in the census. Under that
convention, a `2_` (supplier), `V_` (virtual), or `C_` identifier does not name a single site. The
study did not check each of those BMUs for a single site, so the page counts them apart and calls
them aggregate BMUs.

**Hybrid and pure PV.** A site is hybrid if it also holds storage. The study grades the evidence of
storage from strongest to weakest. The strongest evidence is a separately registered storage BMU
with output in the window. Next is an operational battery that REPD links to the site's solar row.
Weakest is storage that is planned or under construction, listed in the TEC plant type or in a REPD
battery row that is not yet operational. A site with none of the three kinds of storage evidence is
pure PV when its TEC plant type lists PV only, or when REPD has a solar row and no battery row.

**Matching a BMU to TEC and REPD.** TEC and REPD carry no BMU identifier, so the study matches each
BMU's name in the BMU register to a TEC or REPD project name, accepting a match with a
string-similarity score of at least 0.85. For 8 of the 10 BMUs, the Elexon name does not resemble
the project's name, so the study set the match by hand from capacity, customer name, lead party, and
connection site. The TEC register holds one row for each stage of a project, so the study takes each
project's most advanced row. From that row, the study uses the connected capacity if the project is
built and the agreed capacity if the project is still under construction.

**Recall.** The study checks the census against the 12 built and 6 under-construction TEC projects
that list PV, because a project that holds TEC and lists PV should have a BMU. A TEC plant type can
omit PV, so the list can miss a site. Each project is matched by hand to its BMUs, and the check
records whether any matched BMU is in the census.

**Examples.** Figures 3 and 4 show output in megawatts by day in UTC. The example BMUs were chosen
by a rule fixed before the plots were drawn: every pure PV BMU (there is one), and the solar BMUs at
hybrid sites with the highest, the median, and the lowest correlation with the sun. Where a hybrid
example's site has a storage BMU with output in both weeks, the figures add that storage BMU. The
weeks are those containing the solstices: 15 to 21 June 2026 and 15 to 21 December 2025.

**Every source is public, so the page names each BMU and shows its output.**

## Results

### The census

**Nine single-site BMUs follow the sun, with correlations from 0.83 to 0.88, and the next
single-site BMU scores 0.49 ([Figure 1](#the-census)).** Of the 1,705 BMUs with a B1610 file, 689
have a single-site identifier and 1,016 do not. Of the 1,705 B1610 files, 270 hold no rows: 46
belong to single-site BMUs and 224 to aggregate BMUs. Keeping the daytime zeros leaves the same nine
BMUs, with a lowest correlation of 0.82 and a highest among the other single-site BMUs of 0.38. Of
the 9 BMUs that follow the sun, 7 are also typed Solar in IGCPU, and 2 are found only by their
output. A tenth single-site BMU, Kincraig, is typed Solar in IGCPU and has no output in the window,
so Kincraig is in the census by type alone. Kincraig's TEC project is under construction with no
capacity connected. The commissioning rule left out 3,459 of 155,666 half-hours of the nine BMUs,
and the daytime-zero rule removed 2,112.

### Capacity

**Each register names the same ten BMUs differently.** A dash means the register has no match, and
"(identifier only)" means the BMU register gives the BMU its own identifier as its name. TEC and
REPD name a project, so Cleve Hill's two BMUs share one project name in each.

| BMU | Elexon BMU register | IGCPU | TEC register | REPD |
|---|---|---|---|---|
| `E_KINCS-1` | Kincraig | KINCS-1 | Kincraig Energy Centre | Kincraig - Solar Farm and battery storage system |
| `T_BLPFS-1` | Bulphan Fen Warley Green Solar | BLPFS-1 | Warley (tertiary) | Bulphan Fen Solar Farm & Battery Storage |
| `T_BRCHS-1` | (identifier only) | – | Breach Solar Farm | Breach Farm - Solar farm |
| `T_BURWS-1` | Beechgreen Energy Farm | BURWS-1 | Beechgreen Energyfarm | Burwell Solar Farm |
| `T_CLVHS-1` | Cleve Hill Solar 1 | CLVHS-1 | Cleve Hill Solar Park | Cleve Hill Solar Project |
| `T_CLVHS-2` | Cleve Hill Solar 2 | CLVHS-2 | Cleve Hill Solar Park | Cleve Hill Solar Project |
| `T_LARKS-1` | Larks Green Solar | LARKS-1 | Iron Acton | Larks Green Solar Farm |
| `T_SUTBS-1` | Sutton Bridge Solar Farm | SUTBS-1 | Walpole | Sutton Bridge Solar Farm |
| `T_TEBWS-1` | Tebworth PV Power Park | – | – | Tebworth - Solar Farm |
| `T_TYLNS-1` | Tye Lane Solar | TYLNS-1 | Bramford (Tertiary) | Tye Lane - Solar Farm |

**The table gives the five published capacity figures for each of the ten single-site solar BMUs.**
A dash means no figure was found. IGCPU does not list Breach or Tebworth, so two IGCPU figures are
missing. Tebworth has no TEC figure, because the only TEC project held by Tebworth's customer lists
storage only. One REPD figure is missing because the matched REPD row lists no capacity. Cleve
Hill's two BMUs share one TEC project and one REPD row, so both rows show the whole site's figure.
REPD does not say whether its figures are alternating-current (AC) or direct-current (DC) ratings.
At Breach and Larks Green, the REPD figures (67 MW and 70 MW) exceed each BMU's Generation Capacity
by about 17 MW and 20 MW, which a DC panel rating would explain. Breach's Maximum Export Limit is
also 67 MW against 49.9 MW in TEC.

| BMU | Name | Site | Generation Capacity (MW) | IGCPU installed (MW) | TEC (MW) | Largest MEL, 30 days (MW) | REPD installed (MW) |
|---|---|---|---|---|---|---|---|
| `E_KINCS-1` | Kincraig | Hybrid (planned), embedded | 20.6 | 20.0 | 20.62 | 0.0 | 21.0 |
| `T_BLPFS-1` | Bulphan Fen Warley Green Solar | Hybrid | 50.216 | 57.0 | 57.0 | 50.0 | 49.9 |
| `T_BRCHS-1` | Breach Solar Farm | Hybrid | 50.0 | – | 49.9 | 67.0 | 67.0 |
| `T_BURWS-1` | Beechgreen Energy Farm | Pure PV | 49.952 | 49.0 | 50.0 | 50.0 | 49.9 |
| `T_CLVHS-1` | Cleve Hill Solar 1 | Hybrid | 112.0 | 112.0 | 350.0 | 112.0 | 373.0 |
| `T_CLVHS-2` | Cleve Hill Solar 2 | Hybrid | 205.0 | 205.0 | 350.0 | 205.0 | 373.0 |
| `T_LARKS-1` | Larks Green Solar | Hybrid | 49.9 | 50.0 | 99.4 | 50.0 | 70.0 |
| `T_SUTBS-1` | Sutton Bridge Solar Farm | Hybrid (planned) | 49.419 | 49.0 | 49.9 | 43.0 | 49.9 |
| `T_TEBWS-1` | Tebworth PV Power Park | Hybrid | 45.928 | – | – | 40.0 | – |
| `T_TYLNS-1` | Tye Lane Solar | Hybrid | 49.9 | 49.0 | 57.0 | 50.0 | 49.9 |

**The table sums each column over the single-site census BMUs that have a value, and the number of
BMUs with a value is in brackets.** A project that two BMUs share is counted once in the TEC and
REPD columns. The TEC figure for a hybrid project can include its storage, so a TEC sum is not
comparable with a sum of Generation Capacity.

| Group (BMUs) | Generation Capacity (MW) | IGCPU installed (MW) | TEC (MW) | Largest MEL, 30 days (MW) | REPD installed (MW) |
|---|---|---|---|---|---|
| All solar BMUs, hybrids included (10) | 682.9 (10) | 591.0 (8) | 733.8 (9) | 667.0 (10) | 730.6 (9) |
| BMUs at hybrid sites (9) | 633.0 (9) | 542.0 (7) | 683.8 (8) | 617.0 (9) | 680.7 (8) |
| BMU at the pure PV site (1) | 50.0 (1) | 49.0 (1) | 50.0 (1) | 50.0 (1) | 49.9 (1) |

**Of the nine BMUs at hybrid sites (eight sites), four have a storage BMU with output at their site,
three (at two sites) have an operational battery in REPD, and two have storage planned or under
construction.** The operational batteries are 150 MW at Cleve Hill, which REPD links to both Cleve
Hill BMUs, and 0.66 MW at Breach. The study found no storage BMU at Cleve Hill.

**Generation Capacity and the largest Maximum Export Limit agree within 0.5 MW for 6 of the 10 BMUs,
and differ by up to 20.6 MW for the rest.** The two sums differ by only 2.3% because the per-BMU
differences partly cancel: the absolute differences add up to 7.4% of the Generation Capacity sum.
Kincraig alone accounts for 20.6 MW, because its Maximum Export Limit is zero. The Maximum Export
Limit is a level that each lead party submits, not a capacity.

### Where the BMUs are

![Figure 2: Maps of the ten single-site solar BMUs in Great Britain](assets/solar_bmu_census_map.svg)

**Nine of the ten single-site solar BMUs lie south of 53°N in England, seven of the nine east of the
Greenwich meridian, and the tenth lies north of 55.5°N in Scotland.** Each of the ten BMUs takes its
position from the REPD row matched to that BMU, because the BMU register gives no coordinates.

### What a solar BMU looks like

![Figure 3: A summer week of output for pure PV, hybrid-site solar, and storage BMUs](assets/solar_bmu_census_summer_week.svg)

![Figure 4: A winter week of output for the same BMUs](assets/solar_bmu_census_winter_week.svg)

**A solar BMU at a hybrid site follows the sun like the pure PV BMU, and at the four sites with a
storage BMU the storage is metered on that storage BMU, not on the solar BMU.** In the summer week
(15 to 21 June 2026) the pure PV BMU and the three solar BMUs at hybrid sites rise at dawn, peak
near midday, and fall to zero at dusk. Averaged over the year, the four storage BMUs at census sites
output between −6.0 MW and −4.0 MW at 13:00 UTC and between 9.0 MW and 11.8 MW at 18:00 UTC, so they
charge at midday and discharge in the early evening. Figures 3 and 4 show the storage BMUs at two of
the three hybrid examples, Larks Green and Bulphan Fen. The third hybrid example, Cleve Hill, has no
storage BMU that the study identified.

**On 26 and 29 January 2026, the meter of the solar BMU Tebworth PV Power Park carried flows that
look like the site's battery.** Across the ten single-site census BMUs, output falls below minus 5%
of Generation Capacity in 23 half-hours. Twenty-two of those half-hours belong to Tebworth PV Power
Park, which imported up to 35.3 MW (77% of its Generation Capacity) on 26 and 29 January 2026, by
day as well as by night. Tebworth PV Power Park also exported in 4 half-hours when the sun was more
than 3° below the horizon. Both patterns look like a battery, although the Tebworth site has a
storage BMU of its own. The twenty-third half-hour belongs to Cleve Hill Solar 1, at −22.4 MW. In
the same half-hour Cleve Hill Solar 2 exported 21.9 MW, so the pair nets to −0.5 MW. That netting
suggests an allocation of output between the two Cleve Hill BMUs, not storage. No other single-site
census BMU imports more than 5% of its Generation Capacity.

### How many solar BMUs were missed

**Every built PV project in the TEC register maps to a BMU, and 7 of the 12 map to a BMU in the
census.** Of the 10 built hybrid projects, 6 have a census BMU and 4 map only to BMUs outside the
census: 2 storage BMUs, 1 combined-heat-and-power BMU, and 1 wind-farm BMU. Of the 2 built pure PV
projects, 1 has a census BMU. The other, Sundon Pivoted Power, maps to a storage BMU: the project's
TEC plant type lists PV only, but its customer is a storage developer. Of the 6 under-construction
projects, 1 has a census BMU and 5 have no BMU in the census. Nine of the ten census BMUs match a
TEC project that lists PV. The tenth, Tebworth PV Power Park, has a customer that holds a TEC
project listing storage only.

**The hand mapping found no separately registered solar BMU at a built TEC project outside the
census.** None of the 5 built projects that map outside the census maps to a separately registered
solar BMU, so the census missed no such BMU at a built TEC project that lists PV. The PV at those
five projects may be metered inside the storage, combined-heat-and-power, or wind BMU, or may not be
built. The result depends on the hand mapping having found every BMU at each site.

### Aggregate BMUs

**A further 28 aggregate BMUs follow the sun or are typed Solar, and their Generation Capacity is
mostly not solar capacity.** The aggregate BMUs' correlations run from 0.62 to 0.90, with no gap in
which to set a threshold. Of the 28, 27 follow the sun by behaviour (24 if the daytime zeros are
kept). The 28 aggregate BMUs' Generation Capacity sums to 1,896.8 MW. The five BMUs of TotalEnergies
Gas & Power hold 1,622.3 MW of that sum, and the 14 BMUs of British Gas Trading hold 41.7 MW. The
five TotalEnergies BMUs both import and export heavily, so the Generation Capacity sum is not a
solar capacity. One `2_` BMU is typed Solar in IGCPU and has no output.

**The table gives the capacity figures that exist for each of the 28 aggregate BMUs.** None matches
a TEC project or an REPD row. The largest Maximum Export Limits sum to 68.0 MW, against a Generation
Capacity of 1,896.8 MW, because most supplier BMUs publish a limit of zero. A dash means no figure.

| BMU | Name in the BMU register | Lead party | Generation Capacity (MW) | IGCPU installed (MW) | Largest MEL, 30 days (MW) | Correlation with the sun |
|---|---|---|---|---|---|---|
| `2__ABGAS000` | (identifier only) | British Gas Trading Ltd | 5.428 | – | 0 | 0.87 |
| `2__ALOND001` | EDFE Aggregate Eastern 01 | EDF Energy Customers Limited | 0 | 16 | 0 | – |
| `2__ATGPL000` | (identifier only) | TotalEnergies Gas & Power Ltd | 240.735 | – | 0 | 0.81 |
| `2__BBGAS000` | (identifier only) | British Gas Trading Ltd | 4.84 | – | 0 | 0.88 |
| `2__BECOT003` | ECOT_VPP_11 | ECOTRICITY LIMITED | 16 | – | – | 0.63 |
| `2__BEDGE001` | (identifier only) | Edgware Energy Limited | 54.785 | – | 0 | 0.67 |
| `2__BTGPL000` | (identifier only) | TotalEnergies Gas & Power Ltd | 163.502 | – | 0 | 0.75 |
| `2__CBGAS000` | (identifier only) | British Gas Trading Ltd | 0.438 | – | 0 | 0.89 |
| `2__DBGAS000` | (identifier only) | British Gas Trading Ltd | 3.568 | – | 0 | 0.88 |
| `2__DRWED000` | (identifier only) | ENGIE Power Limited | 60 | – | 68 | 0.62 |
| `2__EBGAS000` | (identifier only) | British Gas Trading Ltd | 3.157 | – | 0 | 0.86 |
| `2__FBGAS000` | (identifier only) | British Gas Trading Ltd | 2.227 | – | 0 | 0.88 |
| `2__GBGAS000` | (identifier only) | British Gas Trading Ltd | 2.016 | – | 0 | 0.81 |
| `2__HBGAS000` | (identifier only) | British Gas Trading Ltd | 4.951 | – | 0 | 0.88 |
| `2__HTGPL000` | (identifier only) | TotalEnergies Gas & Power Ltd | 912.53 | – | 0 | 0.79 |
| `2__JAXPO000` | (identifier only) | AXPO UK LIMITED | 8 | – | – | 0.84 |
| `2__JBGAS000` | (identifier only) | British Gas Trading Ltd | 3.311 | – | 0 | 0.88 |
| `2__KBGAS000` | (identifier only) | British Gas Trading Ltd | 1.973 | – | 0 | 0.89 |
| `2__KTGPL000` | (identifier only) | TotalEnergies Gas & Power Ltd | 100 | – | 0 | 0.77 |
| `2__LBGAS000` | (identifier only) | British Gas Trading Ltd | 3.804 | – | 0 | 0.89 |
| `2__LTGPL000` | (identifier only) | TotalEnergies Gas & Power Ltd | 205.511 | – | 0 | 0.84 |
| `2__MBGAS000` | (identifier only) | British Gas Trading Ltd | 3.093 | – | 0 | 0.87 |
| `2__NBGAS000` | (identifier only) | British Gas Trading Ltd | 1.663 | – | 0 | 0.83 |
| `2__PBGAS000` | (identifier only) | British Gas Trading Ltd | 1.245 | – | 0 | 0.89 |
| `C__ESTAT019` | C__ESTAT019-AR4-BSP-700 | Statkraft Markets Gmbh | 32 | – | – | 0.87 |
| `C__LSTAT020` | C__LSTAT020-AR4-LSF-700 | Statkraft Markets Gmbh | 62 | – | – | 0.71 |
| `V__HFLEX002` | (identifier only) | Flexitricity Limited | 0 | – | 0 | 0.90 |
| `V__NFLEX003` | (identifier only) | Flexitricity Limited | 0 | – | 0 | 0.62 |

## Discussion: what to use

- **For a list of solar BMUs in Great Britain, use the ten single-site BMUs as a floor, not a
  total.** Add the 28 aggregate BMUs only where the application can accept BMUs that pool several
  sites.
- **For the capacity of a solar BMU, use Generation Capacity.** For 6 of the 9 BMUs with output, the
  largest output is within 1% of Generation Capacity. The largest Maximum Export Limit agrees with
  Generation Capacity within 0.5 MW for 6 of the 10 BMUs. The Maximum Export Limit is a level the
  lead party submits, not a capacity. The TEC and REPD figures describe whole projects.
- **To compare a solar BMU with a solar forecast, use the solar BMU on its own, and drop 26 and 29
  January 2026 for Tebworth PV Power Park.** At the four sites with a storage BMU the storage is a
  separate series. At Cleve Hill and Breach the study found no storage BMU and cannot say where the
  battery is metered.

## Limitations

- **The window is 12 months.** A BMU that started in the last month of the window has too little
  output to judge, and is in the census only if IGCPU types it Solar.
- **The classifier uses one reference point for the sun.** Solar noon at the census sites differs by
  about 14 minutes, which is small beside the half-hour resolution of the output.
- **The census's recall of embedded solar BMUs is unmeasured.** Only one embedded BMU is in the
  census, and the study found no list of embedded solar BMUs to check the census against.
- **The matches to TEC and REPD rest on names and judgement.** The study matched 8 BMUs and 18 TEC
  projects by hand, using lead party, customer, capacity, and connection site. A wrong match would
  change a technology label, a position, or a capacity sum.
- **A TEC plant type can omit PV or be wrong.** Tebworth's customer holds a project that lists
  storage only, and the Sundon Pivoted Power row lists PV only although its customer and its BMU are
  a storage developer and a storage BMU. The recall check can therefore miss a site. A TEC project
  with PV and storage could also hold a PV array that no BMU meters on its own.
- **The hybrid label rests on graded evidence.** Only four sites have a storage BMU with output.
  Breach's operational battery is 0.66 MW, and two sites have storage that is not yet built.
- **The pure PV site may have storage nearby.** Three embedded storage BMUs carry Burwell names
  (`E_BURWB-1`, `E_BURWB-2`, and `E_BURWB-3`), two of them with the same lead party as `T_BURWS-1`,
  and a 57 MW storage project awaiting consent sits at the same 400 kV substation. The study could
  tie none of the three Burwell storage BMUs, nor the 57 MW storage project, to the Beechgreen site.
- **Zero output at midday is read as an outage, curtailment, or a metering fault, and is removed.**
  Removing the zeros leaves the single-site classes unchanged and changes the count of aggregate
  BMUs that follow the sun by 3 (27 against 24).
- **REPD lists some sites that already generate as under construction**, so the study accepts both
  operational and under-construction REPD rows.
- **The Maximum Export Limit comes from a window after the output window.**

## Scope

**The study does not cover solar generation that is not a BMU.** Much of the solar capacity in Great
Britain is small generation embedded in distribution networks, with no BMU. The study also does not
cover wind or other fuels, a forecast of any BMU's output, or years other than September 2025 to
August 2026.

## Data and code availability

**All inputs are public.** The BMU register, B1610, IGCPU, and MELS come from the [Elexon Insights
API](https://bmrs.elexon.co.uk/api-documentation), the TEC register from [NESO's data
portal](https://www.neso.energy/data-portal/transmission-entry-capacity-tec-register), and REPD from
[the Department for Energy Security and Net
Zero](https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract).
The two hand-made match tables are committed beside the scripts. The code is in
[`studies/solar_bmu_census/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies/solar_bmu_census),
and its classifier tests are in `packages/studies/tests/solar_bmu_census/`.

## Reproducing the figures

```bash
uv run python studies/solar_bmu_census/fetch_sources.py
uv run python studies/solar_bmu_census/classify.py
uv run python studies/solar_bmu_census/collate.py
uv run python studies/solar_bmu_census/report.py
uv run python studies/solar_bmu_census/census_charts.py
```

`report.py` writes `report.md` to `data/studies/per_study/solar_bmu_census/`. `report.md` holds
every number on this page, or the figures from which a derived number is computed.
