# At least 10 Balancing Mechanism Units in Great Britain are single solar sites, and 8 of them share a site with storage

**This study asks how many Balancing Mechanism Units (BMUs) in Great Britain are solar, and what
capacity each published figure gives them.** A BMU is the unit of trade in the Balancing Mechanism,
which the National Energy System Operator (NESO) uses to balance supply and demand. No field in the
BMU register says which BMUs are solar: none of the 3,115 registered BMUs has a fuel type of solar,
and 2,525 have no fuel type at all. The study therefore downloads the settled half-hourly output of
the 1,705 BMUs that are not interconnectors, over 12 months, and asks whether each BMU's output
follows the sun.

**Ten single-site BMUs are solar: nine follow the sun, and a tenth is registered as solar but
produced no output in the window.** All ten have a `T_` or `E_` identifier, which means one
generating site. Eight of the ten sit at a site that also holds storage, and two sit at pure PV
sites. A further 28 BMUs with supplier, virtual, or other identifiers also follow the sun or are
registered as solar, and each can pool many sites, so the study reports them apart. The study cannot
say how many solar BMUs it missed: only 5 of the 12 built sites in NESO's Transmission Entry
Capacity register that list PV have a BMU in the census, and no list exists to check embedded BMUs
against.

![Figure 1: Single-site BMUs' correlation with the sun, with a gap from 0.50 to 0.83](assets/solar_bmu_census_correlation.svg)

- **To find solar BMUs, read their output and not the register.** The fuel-type field never says
  solar, and a name search misses units whose name does not say so.
- **To compare capacities, keep the five published figures apart.** The ten BMUs' Generation
  Capacity sums to 682.9 MW and their largest Maximum Export Limit to 667.0 MW, while two registers
  give a figure for a whole site, which can include storage.
- **A solar BMU at a hybrid site is metered apart from its storage.** None of the ten BMUs shows a
  battery in its own output.

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5). Several independent Claude reviewers have checked the
> method, the evidence, and the prose adversarially.

## Key findings

- **A wide gap separates the single-site BMUs that follow the sun from the rest**, so the threshold
  is not a close call ([Figure 1](#the-census)).
- **The ten single-site solar BMUs have 682.9 MW of Generation Capacity in all**, 99.4 MW of it at
  the two pure PV sites and 583.5 MW at the eight hybrid sites ([Capacity](#capacity)).
- **Nine of the ten lie south of 53°N in England, and one lies in Scotland** ([Figure
  2](#where-the-bmus-are)).
- **A solar BMU at a hybrid site looks like a pure PV BMU, and the storage is a separate BMU**
  ([Figures 3 and 4](#what-a-solar-bmu-looks-like)).
- **The census finds 5 of the 12 built PV sites in the Transmission Entry Capacity register**
  ([Recall](#how-many-solar-bmus-were-missed)).
- **28 aggregate BMUs also follow the sun or are registered as solar**, with 1,896.8 MW of
  Generation Capacity, and cannot be tied to single sites ([Aggregates](#aggregate-bmus)).

## Introduction

**Several datasets publish a capacity for a BMU, and the figures mean different things.** The
figures answer different questions: what the unit's lead party declared it could generate, what the
transmission operator lists as installed, what the site may export to the grid, what the unit was
actually allowed to export, and what the planning database holds for the project. Anyone who wants
"the capacity" of the solar BMUs first has to know which BMUs are solar, and the register does not
say.

**The study compares no weather products and fits no forecasting model.** Every other study in this
section scores weather products at Flexpectation's own generators. This study uses only public data
and covers all of Great Britain, so a number here describes the BM-registered solar fleet and
nothing about NGED's trial area.

## Data and methods

**Planned and exploratory.** The study plan, written before any result, named one question: how many
solar BMUs exist, and with what recall. The study runs no significance test and fits no forecasting
model, so no contrast is planned or exploratory, and every number is a count or a sum.

**Sources.** All six sources are public and were fetched live on 7 October 2026:

- the Elexon BMU register (`reference/bmunits/all`), which gives each BMU's identifier, name, lead
  party, and declared Generation Capacity;
- Elexon dataset B1610, the settled output of each BMU in each half-hour, in megawatt-hours, for 1
  September 2025 to 31 August 2026 (the 12 complete months that end at least two weeks before the
  run, because B1610 lags by about a week);
- Elexon report B1420 (dataset IGCPU), the installed capacity the transmission operator lists for a
  unit, with a resource type that is "Solar" for 9 BMUs;
- Elexon dataset MELS, the Maximum Export Limit of each solar BMU in the 30 days before the run;
- NESO's Transmission Entry Capacity (TEC) register, the export capacity agreed for each project at
  the grid connection; and
- the Renewable Energy Planning Database (REPD) of the Department for Energy Security and Net Zero,
  the installed capacity of each project, with an Ordnance Survey grid position.

**Classifying a BMU.** The classifier takes each BMU's output, drops the 30 days after its first
positive reading because a site often commissions in stages, and drops half-hours of exactly zero
output while the sun is clearly up (the cosine of the solar zenith angle above 0.1), which is almost
certainly a metering fault in a solar unit. It then computes the Pearson correlation between the
output and the cosine of the solar zenith angle, clipped at zero below the horizon, at one point in
central Great Britain (53°N, 1.5°W). A BMU has no coordinates, and across Great Britain the solar
noon moves by about 16 minutes with longitude, which is small beside half an hour. A BMU with a
correlation above 0.6 and at least 100 positive half-hours follows the sun; one with fewer than 100
positive half-hours, or a constant output, has no output to judge. The census holds every BMU that
follows the sun or has IGCPU resource type "Solar".

**Single-site and aggregate BMUs.** A `T_`, `E_`, or `M_` identifier names one generating site. A
`2_` (supplier), `V_`, or `C_` identifier can pool many sites, so the study counts those BMUs apart.

**Hybrid and pure PV.** A site is hybrid if it also holds storage. The study reads this from the TEC
register's plant type where the BMU matches a project, and otherwise from a battery row in REPD that
shares the site's name; a solar BMU matched to a solar REPD row with no battery row is called pure
PV, which is weaker evidence. TEC and REPD carry no BMU identifier, so the study matches site names
with a similarity score of at least 0.85, and 6 BMUs whose Elexon name does not resemble the site's
were matched by hand from capacity, connection site, and lead party. The two hand-made tables are
committed beside
the scripts.

**Recall.** The study checks the census against the 12 built and 6 under-construction TEC projects
that list PV, because a transmission-connected solar site must hold TEC. Each project is matched by
hand to its BMUs, and the check records whether any matched BMU is in the census.

**Examples.** Output is shown in megawatts on calendar dates. Each kind's examples are chosen by a
rule fixed before the plots were drawn: both pure PV BMUs, and for solar BMUs at hybrid sites the
first, middle, and last by correlation. A storage BMU is added where a hybrid example's site has one
with output in the window. The weeks are those containing the solstices: 15 to 21 June 2026 and 15
to 21 December 2025.

**Public data, so BMUs are named.** Every source is public, so the page names each BMU and shows its
output. The study uses no data from NGED's own generators.

## Results

### The census

**Of 1,705 BMUs with a B1610 file, 689 have a single-site identifier and 1,016 do not.** 270 files
hold no rows. Nine single-site BMUs follow the sun, with correlations from 0.83 to 0.89; the next
highest single-site BMU, classed as not solar, scores 0.50 ([Figure 1](#the-census)). Keeping the
daytime zeros changes no class: 9 single-site BMUs follow the sun either way. Of the nine, 7 are
also typed Solar in IGCPU, and 2 are found only by their output. A tenth single-site BMU is typed
Solar and has no output in the window, so it is in the census by type alone. The 30-day
commissioning rule left out 13,613 of 155,666 half-hours of the nine BMUs, and the daytime-zero rule
removed 1,950.

### Capacity

**The ten single-site solar BMUs, with the five published capacity figures for each.** A dash means
no figure was found. Cleve Hill's two BMUs share one TEC project and one REPD row, so both rows show
the whole site's figure.

| BMU | Name | Site | Generation Capacity (MW) | IGCPU installed (MW) | TEC (MW) | Largest MEL, 30 days (MW) | REPD installed (MW) |
|---|---|---|---|---|---|---|---|
| `E_KINCS-1` | Kincraig | Hybrid, embedded | 20.6 | 20.0 | 20.62 | 0.0 | 21.0 |
| `T_BLPFS-1` | Bulphan Fen Warley Green Solar | Hybrid | 50.216 | 57.0 | 57.0 | 50.0 | 49.9 |
| `T_BRCHS-1` | Breach Solar Farm | Hybrid | 50.0 | – | 49.9 | 67.0 | 67.0 |
| `T_BURWS-1` | Beechgreen Energy Farm | Pure PV | 49.952 | 49.0 | 50.0 | 50.0 | 49.9 |
| `T_CLVHS-1` | Cleve Hill Solar 1 | Hybrid | 112.0 | 112.0 | 550.0 | 112.0 | 373.0 |
| `T_CLVHS-2` | Cleve Hill Solar 2 | Hybrid | 205.0 | 205.0 | 550.0 | 205.0 | 373.0 |
| `T_LARKS-1` | Larks Green Solar | Hybrid | 49.9 | 50.0 | 99.4 | 50.0 | 70.0 |
| `T_SUTBS-1` | Sutton Bridge Solar Farm | Pure PV | 49.419 | 49.0 | – | 43.0 | 49.9 |
| `T_TEBWS-1` | Tebworth PV Power Park | Hybrid | 45.928 | – | – | 40.0 | – |
| `T_TYLNS-1` | Tye Lane Solar | Hybrid | 49.9 | 49.0 | – | 50.0 | 49.9 |

**The five published capacity figures differ for the same BMUs, and each answers a different
question.** The table sums each column over the single-site census BMUs that have a value. The TEC
and REPD figures describe a project, so a project that two BMUs share is counted once. The TEC
figure for a hybrid project includes its storage, so a TEC sum is not comparable with a sum of
Generation Capacity.

| Group (BMUs) | Generation Capacity (MW) | IGCPU installed (MW) | TEC (MW) | Largest MEL, 30 days (MW) | REPD installed (MW) |
|---|---|---|---|---|---|
| All solar BMUs, hybrids included (10) | 682.9 (10 BMUs) | 591.0 (8) | 826.9 (7) | 667.0 (10) | 730.6 (9) |
| Hybrid sites only (8) | 583.5 (8) | 493.0 (6) | 776.9 (6) | 574.0 (8) | 630.8 (7) |
| Pure PV sites only (2) | 99.4 (2) | 98.0 (2) | 50.0 (1) | 93.0 (2) | 99.8 (2) |

**Each cell gives the sum in megawatts, and the number of BMUs that have a value in brackets.** No
BMU has an unknown technology. Two IGCPU figures are missing for the two BMUs that IGCPU does not
list, and three TEC figures and one REPD figure are missing where no project matched.

### Where the BMUs are

![Figure 2: Maps of the ten single-site solar BMUs in Great Britain](assets/solar_bmu_census_map.svg)

**Nine of the ten lie south of 53°N in England, seven of the nine east of the Greenwich meridian,
and one lies north of 55.5°N in Scotland.** All ten have a position, taken from the REPD row matched
to each BMU.

### What a solar BMU looks like

![Figure 3: A summer week of output for pure PV, hybrid-site solar, and storage BMUs](assets/solar_bmu_census_summer_week.svg)

![Figure 4: A winter week of output for the same BMUs](assets/solar_bmu_census_winter_week.svg)

**A solar BMU at a hybrid site follows the sun like a pure PV BMU, and the storage is metered as its
own BMU.** In the summer week (15 to 21 June 2026) the two pure PV BMUs and the three solar BMUs at
hybrid sites rise at dawn, peak near midday, and fall to zero at dusk. The storage BMU at the
Bulphan Fen Warley Green site swings above and below zero with no daily pattern. Across the ten
census BMUs, output falls below minus 5% of Generation Capacity in only 23 half-hours of two BMUs,
and the lowest output is minus 77%. The census therefore sees no battery inside any solar BMU. Only
one hybrid example has a storage BMU in the figures: the storage BMU that the study mapped to Larks
Green Solar's site has no output rows in the window, and Cleve Hill has none identified.

### How many solar BMUs were missed

**The census has a BMU for 5 of the 12 built PV projects in the TEC register, and the other 7 split
into 3 whose BMUs are not solar and 4 with no BMU identified.** Of the 10 built hybrid projects, 4
have a census BMU, 3 map only to BMUs outside the census (a storage unit, a combined-heat-and-power
unit, and a wind-farm unit), and 3 have no BMU identified. Of the 2 built pure PV projects, 1 has a
census BMU and 1 has none. Of the 6 under-construction projects, 1 has a census BMU. Three census
BMUs have no TEC project that the study could match. A project with no BMU identified either has no
solar BMU or was missed by the classifier, and the study cannot tell which. At most 4 of the 12
built projects can be a solar BMU that the census missed, and the other 3 unaccounted-for projects
map to BMUs that are not solar.

### Aggregate BMUs

**28 BMUs with other identifiers follow the sun or are typed Solar, and their Generation Capacity
sums to 1,896.8 MW.** Their correlations form a continuum from 0.61 to 0.91, with no gap to set a
threshold in, so the classification of the aggregates is less certain than that of the single-site
BMUs. One `2_` BMU is typed Solar in IGCPU and has no output. The aggregates are probably portfolios
of embedded generation, and the study does not tie their capacity to sites.

## Discussion: what to use

- **For a list of GB solar BMUs, use the ten single-site BMUs as a floor, not a total.** Add the
  aggregates only if the use accepts pooled sites.
- **For a capacity of a solar BMU, use Generation Capacity or the largest Maximum Export Limit.**
  The two sums differ by 2.3% over the ten BMUs, and the TEC and REPD figures describe whole
  projects.
- **For a forecast of one solar farm, match the BMU to the project first.** A hybrid site's solar
  BMU and storage BMU are separate series, so the solar BMU alone is the one to compare with a solar
  forecast.

## Limitations

- **The window is 12 months, and the classifier uses one reference point.** A BMU that started in
  the last month of the window has too little output to judge, and is in the census only if IGCPU
  types it Solar.
- **Embedded recall is unmeasured.** Only one embedded BMU is in the census, and no list of embedded
  solar BMUs exists to check against.
- **The match to TEC and REPD rests on names, and 6 matches and the 19 TEC mappings are hand-made.**
  A wrong hand match would change a technology label or a capacity sum. Both tables are committed in
  `studies/solar_bmu_census/`.
- **A solar BMU is called pure PV from a weaker test than a hybrid is called hybrid.** One of the
  two pure PV BMUs rests on a REPD solar row with no battery row, which does not prove the site has
  no storage.
- **Zero output at midday is read as a metering fault.** It could instead be curtailment, and
  removing it only matters for the correlation, which does not change a class.
- **REPD lists some generating sites as under construction**, so the study accepts operational and
  under-construction rows.
- **The Generation Capacity used to normalise the plots is a declared figure.** A BMU's output can
  exceed it: the largest output of a BMU is 1.11 times its Generation Capacity.

## Scope

**The study does not cover solar generation that is not a BMU.** Most GB solar capacity is small
embedded generation with no BMU. The study also does not cover wind or other fuels, a forecast of
any BMU's output, or years other than September 2025 to August 2026.

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

`report.py` writes `report.md`, which holds every number on this page, to
`data/studies/per_study/solar_bmu_census/`.
