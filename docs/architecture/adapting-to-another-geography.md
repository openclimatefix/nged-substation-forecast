# Could this codebase forecast another country?

> **Status: Thought experiment — not planned work.** This page records an assessment made on
> 2026-08-05 (against `main` at `737fb86`) while OCF was preparing a pitch for an innovation
> project in India. There is no GitHub issue for any of it, no roadmap entry, and **no intention to
> refactor this codebase for portability**. We would not start any of this work unless we won that
> bid. Nothing here is a commitment, and no current design decision should be taken "so that India
> would be easier later" — see [Why we are not doing any of this now](#why-we-are-not-doing-any-of-this-now).

This page exists in the same spirit as
[Why Dagster, not Airflow?](why-dagster-not-airflow.md): a question was asked, we did the analysis,
and the reasoning is worth keeping auditable even though the answer is "not now".

## Part 1: for the pitch

This part is written for a reader who understands the energy system but does not read code. It
covers what the job would involve, how big it is, what we could honestly claim, and — most useful
in a Q&A — the **data sources worth asking the client about**.
[Part 2](#part-2-what-would-have-to-change-in-the-code) is the engineering detail behind it, and
can be skipped.

### The brief we were given

The assessment assumed the following brief:

- ~100,000 secondary substations, each reporting power flow every 15 minutes.
- Forecast **net demand** per substation.
- Additionally forecast **PV** per substation.
- **No PV metering anywhere**, and **no prior on installed capacity** at any site.

### The short answer

What it would take to point this system at that brief splits into three quite different pieces,
and they are worth keeping apart because only one of them is genuinely hard.

Several statements below rest on **assumptions we have not checked with the client** — mostly
borrowed from what NGED wanted, because that is the only comparable project we know. Each one is
flagged in place, and all of them are collected in [Questions we should ask them](#questions-we-should-ask-them) at the end of this part. If you read only
one other section before the pitch, read that one.

**Moving to a different country is easy.**  The weather data we already use (ECMWF ENS from
Dynamical) is a global product, so India needs no new weather source at all. The parts of the system
that handle spatial gridding, storage, and feature building are geography-agnostic. What *is*
British is a thin, well-contained layer: a map outline, a permitted latitude and longitude range,
the names of British substation categories, and the assumption that readings arrive every half hour
rather than every 15 minutes. Turning those from hard-coded facts into settings is a few weeks of
work, not a rewrite.

**Handling 100,000 substations is a real engineering project, but an ordinary one.** It is 40 times
as many substations as we are currently building towards for NGED, and — because the readings
arrive twice as often — around 80 times the volume of forecast data. The main consequence is that
we would have to stop training a separate model for each substation and instead train one model
that learns across all of them at once. That is already on our NGED roadmap for its own reasons,
and at 100,000 sites it becomes clearly the favourable choice, because each part of the model is
supported by far more data. Storage layouts would also need reworking, because a year of forecasts
at that scale runs to roughly 18 terabytes *stored* — that is, after the aggressive compression we
already apply, described in
[Forecast delivery](forecast-delivery.md#how-big-is-flexpectations-power-forecast-data). The same
forecasts in an uncompressed form, such as the JSON a REST API would send, would be a couple of
thousand terabytes. The distinction matters throughout this page, so every figure below says which
one it is.

**That 18 terabytes is a worst case, and the realistic number could be very much smaller.** It
assumes India wants exactly what NGED wants: a 14-day horizon, a full 51-member ensemble, and four
forecast runs a day. We have no basis for any of those three assumptions — they are simply the only
requirements we currently know. Each one multiplies the total, so each one we can relax shrinks it
sharply: a 2-day horizon alone cuts it sevenfold, delivering the agreed set of percentiles instead
of every raw ensemble member cuts it fourfold, and running once a day rather than four times cuts
it fourfold again. Plausible combinations land **between roughly 0.16 and 18 terabytes** — a 110×
spread, set entirely by answers we do not yet have. The per-answer breakdown is in
[Questions we should ask them](#questions-we-should-ask-them), and it is the reason those questions
matter more than any clever engineering: **asking about the forecast horizon is worth more than
anything we could do to the storage format.** If the answers do come back at the demanding end,
there are further options — see
[Radical options for shrinking what we store](#radical-options-for-shrinking-what-we-store).

**We would also expect to start with a trial of roughly 50 to 100 substations, not with all
100,000.** That is the same shape as our NGED programme, where we are running on a 32-substation
trial area before scaling to around 2,500, and it is worth saying out loud in the pitch, because it
changes the risk profile considerably. A trial at that size needs essentially none of the scale
engineering described above — it is comfortably within what the code handles today — so the first
year could be spent proving the hard part, which is whether we can actually recover unmetered
rooftop solar from Indian net-flow data, rather than on storage layouts. It also front-loads the
findings: we would have real results on a real Indian feeder within months, and the answers to the
questions below would by then be known rather than assumed, so the full-rollout design could be
sized against measured numbers instead of worst cases. What we should avoid promising is 100,000
substations in year one.

**Estimating unmetered solar is the research bet — and it is the same bet we are already making
for NGED.** Separating rooftop solar from underlying demand, with no generation meters and no
capacity register, is exactly the problem described in
[Net-demand disaggregation](../roadmap/disaggregation.md). The method there is designed for
precisely this: it treats a substation's reading as demand minus solar generation, models the solar
physically from sunlight, and infers the installed capacity as an unknown that only ever grows.
It does not need a capacity register.

Qualifications belong alongside that, and they cut both ways. In Britain we plan to use the
*metered* solar farms we can see to calibrate and sanity-check our estimates of the *unmetered*
ones. We *assume* the Indian brief offers no metered solar inside the dataset, so that anchor would
have to come from outside it — **but this is exactly the sort of thing to ask about** (see [Questions we should ask them](#questions-we-should-ask-them)),
because India does have a large, metered, utility-scale solar fleet with published output. Even
aggregated or partial metering would help. Using it would be new work rather than something we
already have, but it is much easier than working without any anchor at all. The physical
background is also harder: Indian rooftop panels lose a substantial fraction of their output to
dust between monsoons and recover when the rain washes them, and our method currently assumes
installed capacity only ever grows, so it has no way to express a loss that reverses.

Extending it to handle dust looks genuinely straightforward, though — and it is something **we
should probably add for Britain anyway**. The fix is to stop treating "how much is installed" and
"how well it is working" as one number: keep the existing installed-capacity term that only grows,
and multiply it by a separate cleanliness factor between zero and one that dust pushes down and
rain pushes back up. The physics is simple and well understood — soiling builds up roughly with
time since the last decent rainfall, and washes off above a few millimetres of rain — so it needs
only two or three new parameters per site, learned the same way as panel tilt and orientation
already are. Crucially, **the input it needs is rainfall, which we already download** as part of
our weather data. The honest caveat is that separating "the panels got dirty" from "someone
installed fewer panels than we thought" is a real statistical question that would need testing, not
just coding. Britain is normally rainy enough for soiling to be a small effect, but that is exactly
what a rain-driven model would predict, and in a long dry spell — London roofs under Saharan dust
after months without proper rain — it is not small at all. Work done here would pay off in both
countries.

Separately, high atmospheric dust also biases the satellite and forecast estimates of sunlight that
the whole method leans on.

Against that, 100,000 sites reporting every 15 minutes would be a far larger and finer-grained
dataset than the ~2,500 sites we have been designing towards all along for NGED's V2 (the V1 trial
area we are currently running on is 32 sites) — more sites means the shared parts of the model are
much better constrained, and 15-minute data separates a solar signal from a demand signal more
cleanly than half-hourly data does. There are also confounders India has and Britain
does not: load shedding and diesel backup generation both break the assumption that underlying
demand moves smoothly with the weather. Load shedding is the dangerous one, because it looks like
demand collapsing for no meteorological reason. Agricultural pumping is a special case worth
calling out as an opportunity rather than a problem: Indian agricultural feeders are largely
segregated and run to a published supply schedule, so a load that would otherwise be invisible is
partly *known in advance*. Whether we could actually obtain those schedules — and the load-shedding
schedules, which would help just as much — is another thing to ask (see [Questions we should ask them](#questions-we-should-ask-them)).

**For the bid**, the defensible claim is not "we can forecast substations" — many people can. It is
that OCF already has a designed, written-down method for recovering *unmetered* rooftop solar from
net substation flow without a capacity register, together with a published protocol for proving
whether it actually works ([Evaluating disaggregation](../techniques/disaggregation-evaluation.md)).
The Indian dataset would be a larger and finer-grained proving ground for that method than the one
it was built for, against a harder physical background.

**Rough size of the job:** about one engineer for four to five and a half months to have India
forecasting net demand at scale, with the solar-disaggregation research running alongside and
shared with the NGED work rather than duplicated. Only around three of those months are needed to
forecast a 50–100 substation trial; the rest is what it takes to go from a working trial to
100,000 substations, and can follow it. Two caveats, both worth turning into questions
(see [Questions we should ask them](#questions-we-should-ask-them)). It assumes the substation data
arrives in a sane bulk format; if it has to be polled per-substation across 100,000 sites, that is
a separate workstream we have not costed. And it assumes we are delivering *data* rather than a
user interface — if they want a dashboard or an HTTP API, that is additional work, though
**strictly additive** and not a rewrite of anything we would have built.

### Data sources that would materially help

The hard parts described above are mostly *data* gaps rather than method gaps, so it is worth being
concrete about what we would ask for or go and find. **This is the section to have in hand during a
Q&A**: each row is something we could reasonably ask the client, the utility, or a partner for.
Claims here were checked in August 2026 and are dated where they may drift.

| Source | What it gives us | Status for India |
|---|---|---|
| **Metered PV generation from India** | The missing anchor: known generation against which to calibrate the physics model before inverting it for unmetered fleets. | Exists, but utility-scale and spatially aggregated. India's [Central Electricity Authority (CEA)](https://cea.nic.in/) and [Grid Controller of India (Grid-India)](https://en.wikipedia.org/wiki/Power_System_Operation_Corporation) — the latter renamed in November 2022 from the Power System Operation Corporation (POSOCO) — publish national and regional generation. The five Regional Load Despatch Centres (RLDCs) and the State Load Despatch Centres (SLDCs, one per state) publish more granular real-time data. **Ask for this first.** |
| **SARAH-E** — Surface Solar Radiation Data Set – Heliosat, East, from EUMETSAT's Satellite Application Facility on Climate Monitoring (CM SAF) | The Indian Ocean Data Coverage (IODC) sibling of the SARAH product we plan to use for GB — and, critically, it carries **global, direct and direct-normal irradiance** (surface incoming solar radiation, SIS; direct irradiance, SID; and direct normal irradiance, DNI), so it has the beam/diffuse split the [DP solar model](../techniques/differentiable-physics.md) needs, at 0.05°. | Covers India. But Edition 1 runs **1999–2015** on Meteosat First Generation (Meteosat-5/7); we found no confirmed post-2015 extension. Useful for pre-training and for validating the physics, **not** for near-real-time. |
| **NSRDB** — the National Solar Radiation Database from the US National Renewable Energy Laboratory (NREL), Meteosat Indian Ocean Data Coverage (IODC) region, Physical Solar Model v3 | Global horizontal irradiance (GHI), direct normal irradiance (DNI) and diffuse horizontal irradiance (DHI) at 4 km on a **15-minute** grid — the same cadence as the substation metering in the brief, so no temporal interpolation is needed. | Covers the Indian Ocean Data Coverage (IODC) region including India. The most promising near-real-time-capable irradiance option we found; exact year coverage and licensing need confirming directly with the National Renewable Energy Laboratory (NREL). |
| **IMDAA** — the Indian Monsoon Data Assimilation and Analysis regional reanalysis | A **12 km** reanalysis for the Indian monsoon region (4D-Var, Met Office Unified Model), against ERA5's 31 km — built by India's National Centre for Medium Range Weather Forecasting (NCMRWF) with the UK Met Office and the India Meteorological Department (IMD). Domain 30–120°E, 15°S–45°N, hourly. | Covers 1979–2018, extended to 2020. Ends too early for live capacity estimation, but a strong **pre-training** reanalysis where ERA5 is weakest. |
| **Agricultural feeder supply schedules** | Turns the largest unmetered load into a known regressor, per the note above. | Published by DISCOMs (India's electricity distribution companies) where feeder segregation has been implemented. Worth asking for explicitly. |
| **Load-shedding / outage schedules** | Lets the regime detector be *supervised* rather than having to infer outages from the power signal alone. | Same logic, same ask. |
| **Rooftop net-metering registrations** | A *partial* capacity prior — not the register the brief says does not exist, but a lower bound on installed capacity in some areas. | Plausible via net-metering records held by the DISCOMs (India's electricity distribution companies) and rooftop subsidy schemes. Speculative until asked; would weaken the "no priors at all" framing in a good way. |

#### What we could get hold of today, without asking anyone

Worth separating from the table above, because "this data exists" and "we can use it" are different
claims, and for India the gap between them is unusually wide. Everything below was checked in
August 2026.

**Ground-station irradiance — the good news, then the catch.** India runs one of the world's
largest solar radiation networks: the **Solar Radiation Resource Assessment (SRRA)** network run by
India's **National Institute of Wind Energy (NIWE)**. It has 121 stations, each carrying a
pyrheliometer plus shaded and unshaded pyranometers, so they measure **direct normal irradiance
(DNI), diffuse horizontal irradiance (DHI) and global horizontal irradiance (GHI) separately** —
precisely the three-way split the differentiable-physics solar model consumes, and that neither the
ECMWF ensemble nor a GHI-only satellite product can give us.

The catch is access. This data sits behind a **non-disclosure agreement**, carries a **charge**,
and publication of it in any form requires written permission from India's Ministry of New and
Renewable Energy (MNRE) via the National Institute of Wind Energy. Raw data is released only for collaborative work under specific permission. So it is a
*negotiation*, not a download — and, if we bid, an early one, because a partner with existing
access to this network would be disproportionately valuable.

**The Baseline Surface Radiation Network (BSRN) is the free alternative, and it is genuinely
research-grade.** All of its data is free via the PANGAEA archive under FAIR (findable, accessible,
interoperable, reusable) terms. India has three stations: **Tiruvallur (TIR)**, station 59 at
13.09°N, 79.97°E, *operated by India's National Institute of Wind Energy (NIWE)*, with a compiled
record covering **2014-08 to 2019-01**; plus **Gandhinagar (GAN)** and **Gurgaon (GUR)**, both of
which the network itself flags as having periods of low data quality. Three sites is nothing like
national coverage, and the best record stopped in 2019, so this cannot validate a live service. What it *can* do is validate the physics: it is
enough to check a differentiable PV forward model, and to quantify satellite and reanalysis
irradiance bias over India, before committing to either. Note the pleasing detail that Tiruvallur
is run by the National Institute of Wind Energy — so this free, quality-controlled record is a
window onto the very same instrumentation programme whose main network sits behind a
non-disclosure agreement. The Global Energy Balance Archive (GEBA) holds Indian
records too, but monthly means only, which is too coarse to be useful here.

**Free PV power data exists, at two very different scales.**

- **Plant-level, high-resolution, tiny:** a widely-used public dataset covers **two Indian solar
  plants at 15-minute resolution over 34 days**, with **inverter-level** DC and AC power (22
  inverters per plant) alongside plant-level irradiance and module temperature. Thirty-four days is
  far too short for anything seasonal, but the shape is exactly right — 15-minute cadence, real
  Indian plants, and module temperature included — so it is a legitimate early test rig for the
  forward model's temperature derating and inverter clipping. A second public dataset from a
  350 kWp installation near Hassan, Karnataka pairs global, direct-normal and diffuse irradiance
  (GHI/DNI/DHI) with PV output.
- **Aggregate, long, modelled:** an openly-licensed [Zenodo dataset](https://doi.org/10.5281/zenodo.7824872)
  from the University of Bristol provides hourly wind and solar capacity factors for India from
  1979–2022 at state level to 1°×1°, plus *reported* production for 2012–2023. Read the two halves
  differently: the long capacity-factor series is **modelled**, so it is not independent evidence
  about irradiance, while the reported production is real but spatially aggregated.

**[PVOutput.org](https://pvoutput.org) has some Indian systems, but far too few to matter here.**
PVOutput is the obvious place to look — a global community platform where rooftop owners publish
live generation, and a genuinely useful source of per-site behind-the-meter data in other
countries. India *is* represented: filtering PVOutput's public system ladder to India returns a
full page of **at least 30 registered systems**, real installations with named inverters and panel
counts. But India does **not** appear in PVOutput's top-25 country table, whose 25th entry (New
Zealand) has 228 systems and 1.1 MW, against Australia's 18,089 systems and 131 MW at the top. So
Indian coverage sits somewhere between a few dozen and a couple of hundred systems.

We could not pin the exact figure down: PVOutput's public pages expose only the top 25 countries
and the first page of a filtered ladder, and an exact per-country count needs the PVOutput API,
which requires a registered account and API key. The number is not worth chasing, because the
conclusion does not change at any value in that range — a few dozen or even a few hundred
self-selected rooftops is not a usable sample against 100,000 substations.

Two further caveats even if coverage were better: bulk access (5-minute data in 365-day batches, or
whole-country daily output) sits behind PVOutput's paid **Data Services** tier, which is also where
the **commercial-use licence** lives, so the free tier is not usable for funded work; and
self-reported community data carries unverified capacity, orientation and shading metadata, which
is precisely the metadata a disaggregation anchor needs to be trustworthy.

The *reason* for the gap is itself informative for the bid: rooftop systems sold in India ship
with the manufacturer's own monitoring app, so the generation data exists but pools in **inverter
clouds run by the inverter manufacturers themselves** (the original equipment manufacturers, or
OEMs) rather than on community platforms. If per-site rooftop generation is wanted at any
scale, the realistic route is a commercial agreement with inverter manufacturers or installers,
not an open dataset. Worth raising as a question rather than assuming either way.

**What none of this gives us.** There is no free, per-site, long-record, near-real-time metered PV
generation for India — which is the thing that would actually anchor the unmetered inference. The
free sources are good enough to *build and sanity-check* the physics; the anchor still has to come
from the project partner, from Grid Controller of India (Grid-India) and the State Load Despatch
Centres (SLDCs), or from an agreement covering the Solar Radiation Resource Assessment (SRRA)
network. That is
worth saying plainly in a bid rather than implying the public data closes the gap.

#### ERA6 does not arrive in time

We checked, because it would change the reanalysis answer if it did.
[ERA6 production started on 6 March 2026](https://climate.copernicus.eu/copernicus-climates-era6-reanalysis-production-starts),
and the first data is scheduled for release **towards the end of 2027**, with four decades
available by early 2028. Helpfully, the rollout is *recent-first* — the two most recent decades
deploy before the backward extensions — so the release order is the one we would want. The timing
is still wrong: a roughly twelve-month project running through 2027 would be finishing as the first
ERA6 blocks appear, and no date has been announced for ERA6 in near-real time, which is what
[capacity estimation](../roadmap/capacity-estimation.md) actually needs. ERA5 is stated to continue
"as long as necessary".

**So we would plan on ERA5, and should be explicit about what that costs in India.**

#### Concerns about ERA5 over India

ERA5 is not bad over India in aggregate — one 2025 ground-station evaluation found it the most
balanced of the reanalyses tested, with a monthly mean bias of about **−3 W/m² (−0.8%)** against
observations from the India Meteorological Department (IMD). That headline number is reassuring for demand forecasting and misleading for
disaggregation, because disaggregation does not depend on the annual mean. It depends on
conditional accuracy in specific regimes, and on a quantity the mean hides entirely.

- **The error is seasonal, not uniform.** The same evaluation found ERA5 performs best under clear
  to moderately clear skies in the dry season (February–May) and **underestimates during the
  monsoon** (June–September), attributed to conservative cloud optical thickness. A bias that
  switches sign with the season maps directly onto a *seasonally varying* capacity estimate, which
  is exactly the artefact we would otherwise mistake for real fleet growth — and the
  [monotone capacity prior](../roadmap/disaggregation.md#unmetered-installed-capacity-grows-monotonically)
  cannot represent a correction that goes back down.
- **The direct/diffuse split is worse than the total.** Work on ERA5 over China found that
  misrepresented aerosols cause large deviations in the diffuse-to-direct ratio specifically,
  larger than the deviation in total irradiance. This matters more to us than to most users,
  because the differentiable-physics solar model consumes the beam/diffuse split rather than global
  horizontal irradiance (GHI) alone. A model can
  get total irradiance right and the split wrong, and we would absorb the difference as an error
  in panel tilt and azimuth.
- **31 km cannot resolve monsoon convection.** This is the motivation for the Indian Monsoon Data
  Assimilation and Analysis (IMDAA) reanalysis above, and it is a
  much bigger deal over India than over Britain, where the synoptic-scale argument for using a
  coarse reanalysis genuinely holds.
- **Aerosol loading over the Indo-Gangetic Plain is extreme and episodic** — crop-residue burning
  and dust events — and reanalysis aerosol treatments are generally climatological rather than
  event-resolving. We have not verified ERA5's specific aerosol configuration against this claim;
  it should be checked before it goes in a bid.

The practical conclusion is that ERA5 alone is not a sufficient irradiance basis for disaggregation
in India, and the mitigation is not a better reanalysis but a **satellite** product — the Surface
Solar Radiation Data Set – Heliosat, East (SARAH-E) for history, and the National Solar Radiation
Database (NSRDB) over the Indian Ocean Data Coverage region for anything near-real-time — with ERA5
or the Indian Monsoon Data Assimilation and Analysis (IMDAA) reanalysis carrying the non-radiative
fields. In GB the equivalent role is played by
[SARAH-3, from EUMETSAT's Satellite Application Facility on Climate Monitoring](../roadmap/data-sources.md),
which cannot be reused here: it covers ±65° longitude, and India begins at 68°E.

### Questions we should ask them

We have costed this against **NGED's** requirements, because those are the only ones we know. Every
volume figure on this page inherits three assumptions we have no basis for in India: a **14-day
horizon**, a **51-member ensemble**, and **four runs per day**. If any of them is wrong, the
storage numbers move enormously — and they mostly move *down*:

| If instead they want… | A year of forecasts, on disk |
|---|---|
| The assumed 14 days, 51 members, 4 runs/day | ~18 TB |
| 13 delivery quantiles rather than 51 raw members | ~4.6 TB |
| One run per day | ~4.5 TB |
| A 2-day horizon rather than 14 | ~2.6 TB |
| A 2-day horizon **and** quantiles | ~0.7 TB |
| A 2-day horizon, quantiles **and** one run per day | ~0.16 TB |

That is a **110× spread**, so the "worryingly large" number is really a statement about NGED's
requirements rather than about India's. Answering these questions is worth more than any
compression work.

**About scope and phasing:**

- **Are they expecting all 100,000 substations from the start, or would they accept a trial of 50 to
  100 first?** We should propose the trial. It removes essentially all of the scale engineering from
  the first phase (see
  [The real work is scale, not geography](#the-real-work-is-scale-not-geography)), which means the
  first year can be spent on the part that might not work — recovering unmetered rooftop solar —
  rather than on storage layouts. It also means the full rollout gets designed against measured
  numbers instead of the worst-case assumptions in the table above.
- **If a trial, which substations?** We would want the sample chosen for *variety* rather than
  convenience — a spread of rooftop-solar penetration, urban and rural, and at least a few feeders
  where someone independently knows roughly how much solar is installed, because those are what make
  the disaggregation results checkable rather than merely plausible.

**About the forecast itself:**

- **What is the forecast horizon?** The single biggest driver of both storage and method. Day-ahead
  operational scheduling, a few days for maintenance planning, and seasonal planning are three
  different products.
- **What decision does the forecast actually support?** Everything else follows from this, and it
  is the question most likely to change what we build. A forecast used to schedule a battery has
  different accuracy and latency requirements from one used for annual network planning.
- **Do they want probabilistic forecasts, or a single number?** We would strongly recommend
  probabilistic, and it is a genuine OCF differentiator — but it costs roughly 13–51× the storage,
  and it is only worth it if someone downstream will actually act on the uncertainty.
- **How often must it update, and how quickly after data arrives?** Update cadence multiplies
  storage directly; latency drives the whole deployment architecture.
- **Per substation, or aggregated?** The brief says per substation, but if most users consume a
  feeder- or region-level total, that changes both the model and the delivery format.

**About delivery:**

- **How do the forecast users want the data?** Our answer for NGED is Delta Lake on S3, for reasons
  set out in [Forecast delivery](forecast-delivery.md) — but that suits one power user with Python
  skills. Indian consumers might instead want an HTTP API, or a control-system (SCADA/EMS) might
  want a push feed. Worth asking early, because it is the assumption most likely to be silently wrong.
- **Do they want a user interface?** A control-room dashboard, a planner's map, or a web page an
  engineer can open are all plausible, and none of them is implied by the brief as written. Ask
  explicitly rather than inferring it, and ask *who* would use it — an operator making a decision in
  the next hour and an analyst reviewing last quarter want almost opposite things.
- **Who are the consumers, and how many?** A single utility analytics team and a hundred engineers
  spread across the DISCOMs (India's electricity distribution companies) imply different architectures.
- **Do they need the full forecast history, or only the latest run?** NGED wanting routine access to
  the entire backtest history is what drives our storage design. If India only needs recent
  forecasts, most of the volume problem disappears.

**A reassuring point to be able to make in the room:** if the answer to either of the first two
questions is "we want an API" or "we want a UI", that is **strictly additive** and costs us nothing
we have already built. As
[Forecast delivery](forecast-delivery.md#when-would-a-rest-api-earn-its-keep) puts it, adding a
REST API later is a thin, stateless service that reads the same Delta tables and serves slices of
them over HTTP — nothing has to be re-written, and the Delta tables remain the system of record
either way. The same is true of a UI. We would not be replacing Delta Lake; we would be adding some
code that *queries* Delta Lake.

This is not a theoretical claim. This project already ships two Marimo web apps —
`view_forecasts.py` and `map_and_timeseries.py` — that are exactly that: user interfaces which read
the same Delta tables directly with `pl.scan_delta`, added on top without changing the storage
layer at all. So "Delta Lake first, interfaces on top" is a pattern we have already exercised, not
one we would be trying for the first time on someone else's project.

**About data we would want:**

- **Can we get access to the Solar Radiation Resource Assessment (SRRA) network, or is there a
  partner who already has it?** See
  [above](#what-we-could-get-hold-of-today-without-asking-anyone) — this is the highest-value
  single ask, and it is a legal/commercial question rather than a technical one.
- **Is there any metered PV generation we can use as an anchor**, at any spatial resolution, even
  aggregated to feeder or state level? And is there a route to **monitoring data held by the inverter
  manufacturers**, which is where per-site rooftop generation actually pools in India?
- **Are agricultural feeder supply schedules and load-shedding schedules available?** Both convert
  a confounder into a regressor.
- **How much history comes with the substation data?** Two years is thin for seasonal effects, and
  the ECMWF ensemble archive only reaches back to 2024-04-01 regardless.
- **Do net-metering or rooftop-subsidy registrations exist** that would give partial installed
  capacity, even for a subset of sites?

## Part 2: what would have to change in the code

Everything below is the engineering detail supporting Part 1. It names specific files, functions
and line numbers, and assumes familiarity with the codebase.

### What is already geography-neutral

More is geography-neutral than we expected. None of the following would need to change for India:

- **The weather source.** `dynamical_data` reads the
  `ecmwf-ifs-ens-forecast-15-day-0-25-degree` catalogue
  ([`download.py:62`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/dynamical_data/src/dynamical_data/ecmwf_ens/download.py)),
  which is global. The spatial bounds are not hard-coded: they are derived at runtime from the
  minimum and maximum latitude/longitude of whatever H3 grid is passed in, so changing the boundary
  changes the download automatically. The one stated limitation — the slice fails across the
  anti-meridian — does not affect India. Two documented caveats do carry over and bite harder
  there, though (see [Data sources](../roadmap/data-sources.md)): the archive only extends back to
  2024-04-01, which is thin history for a 100,000-site training set, and its radiation is global
  short-wave only with no direct component — a bigger problem for PV disaggregation under heavy
  aerosol load than it is for Britain.
- **H3 gridding.** `geo.h3.compute_h3_grid_weights_for_boundary` takes any Shapely geometry. There
  is no Great Britain anywhere in `geo/h3.py`.
- **Storage.** `delta_store` is indifferent to geography; the NWP table is keyed by `h3_index`, so
  its size depends on the area covered, not on which country it is.
- **Feature engineering.** The tabular pipeline is vectorised across time series — `time_series_id`
  is only a join and grouping key — and, importantly, **lags and rolling windows are expressed as
  durations, not as counts of half-hour periods** (`pl.duration(hours=…)`,
  `rolling(period="…h")`). That single decision removes most of what would otherwise make a change
  of reporting interval painful.
- **The model interface.** `BaseForecaster` already documents that an implementation may hold "one
  sub-model per series, a single model spanning many series, or anything in between", so moving to a
  global model needs no base-class change.

### What is hard-wired to Great Britain

All of it sits in a thin layer, and most of it sits in `contracts`.

| Assumption | Where | Consequence for India |
|---|---|---|
| Latitude bounded to 49–61°N, longitude to −9–2°E | `contracts/power_schemas.py:154-162` | Validation **hard-fails** on any Indian coordinate. |
| `licence_area` is `Enum(["EMids"])` | `contracts/power_schemas.py:136` | The tightest single lock. |
| `substation_type` is the GB DNO voltage taxonomy (`BSP`, `GSP`, `Primary`, …) | `contracts/power_schemas.py:148` | Indian secondary substations do not map onto it. |
| `units` is `Enum(["MW", "MVA"])` | `contracts/power_schemas.py:131` | Probably fine, but should be checked against the Indian feed. |
| `LIST_OF_TIME_SERIES_TYPES` — 22 NGED categories, re-exported as the `AllFeatures` enum | `contracts/power_schemas.py` | Propagates into the ML schema. |
| Power bounded to ±1000 MW; `max_mw_threshold` / `min_mw_threshold` sized to GB primaries | `contracts/power_schemas.py`, `contracts/settings.py` | Secondary substations are far smaller; thresholds are meaningless as set. |
| The GB outline | `geo/great_britain/load.py` | Add a sibling region loader; swap one import in `defs/assets.py`. |
| `"Europe/London"` as a bare string literal in the feature engineer | `ml_core/features/tabular_feature_engineer.py:350` | Drives every local-time feature in the champion feature set. |
| `DISPLAY_TIME_ZONE = "Europe/London"`, asserted in the dashboard's axis titles | `dashboard/forecast_chart.py:40` | Display only, but it is a second hard-coded timezone. |
| H3 resolution 5 (~253 km² per cell) chosen for GB, and reached for via a **private** import from the ingest package | `defs/assets.py:40,141` | The NWP grid resolution currently lives inside `nged_data`; see the `PowerIngest` note [below](#how-we-would-structure-it). |
| `nged_s3_bucket_url` / `_access_key` / `_secret` are **required** settings with no defaults | `contracts/settings.py` | `Settings()` raises for any deployment with no NGED bucket. |

The exercise also surfaced a **latent correctness wart**, though a milder one than it first looks.
`local_utc_offset` is computed as
`(base_utc_offset + dst_offset).dt.total_seconds() // 3600` cast to `Int8`
([`tabular_feature_engineer.py:358`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/ml_core/src/ml_core/features/tabular_feature_engineer.py)),
so it can only ever represent whole-hour offsets. Note that `//` floors rather than truncates, so a
negative fractional offset moves *away* from zero.

In any single-timezone deployment this costs nothing: the feature is constant across the dataset,
so mapping UTC+5:30 to `5` discards no information a model could have used. The genuine failure
mode is **collision** in a mixed-offset deployment — India (+5:30) and Nepal (+5:45) both land on
`5`, silently merging two distinct zones — and, more immediately, legibility: neither the `// 3600`
nor the `Int8` states the whole-hour assumption it depends on. Tracked as
[issue #431](https://github.com/openclimatefix/nged-substation-forecast/issues/431).

### What is hard-wired to half-hourly

The list is narrower than it first appears, because of the duration-based lag design noted above:

| Assumption | Where |
|---|---|
| `validate()` **raises** unless every timestamp has `minute ∈ {0, 30}` | `contracts/power_schemas.py:48` |
| Field descriptions declaring a "30-minute observation period" | `contracts/power_schemas.py:18,25` |
| `stuck_window_periods = 48` (i.e. 24 hours at 30 minutes) | `contracts/settings.py` |
| NWP upsampled to `interval="30m"` | `ml_core/features/_nwp.py:121` |
| The live forecast spine, both its start offset and its step | `ml_core/_production_helpers.py:112,115` |
| A row-count guard assuming "51 members × 14 days × 48 half-hours" | `dashboard/forecast_chart.py` |

The one piece of real design work here is the **feature grammar**: lags are parsed from strings
like `power_lag_24h` into an integer number of hours, so there is currently no way to express a
15-, 30- or 90-minute lag. Generalising that from integer hours to durations is the substantive
change; everything else in the table is a constant.

### The real work is scale, not geography

Two different multipliers matter here, and it is worth keeping them apart. On **series count** —
the axis that governs how many models we train — 100,000 sites is **40×** the V2 design point of
~2,500, which is itself ~78× V1's 32. On **forecast-row volume**, the 15-minute sampling doubles it
again, so the storage and query pressure is around **80×**. Three things break.

**None of it breaks in a trial, though, and that is the important scheduling fact.** A first phase
of 50–100 substations — which is what we would expect to propose, mirroring NGED's own 32-site V1
trial area — sits comfortably inside what the code handles today: per-series XGBoost is fine at that
count, one run is a few million forecast rows rather than 6.9 billion, and `power_time_series`
partitioned by `time_series_id` gives 100 directories rather than 100,000. So everything in this
section is **rollout work, not entry cost**. It can be deferred until the disaggregation research —
the part that might not work — has been shown to work on real Indian data, and by then the answers
to [Questions we should ask them](#questions-we-should-ask-them) would be known, so it could be
sized against measured numbers rather than the worst-case assumptions used below. The one thing a
trial cannot defer is the 15-minute resolution work, because that is entry cost at any site count.

**One model per substation stops working.** This is the 40× axis. `XGBoostForecaster.train`
collects the whole population into memory and then loops over `group_by("time_series_id")` in
Python, holding every booster in RAM; `save()` then writes one `.ubj` file per series
([`forecaster.py:124`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py)).
That is fine for 32 series and already strained at 2,500. At 100,000 it is a non-starter, which
forces the **global model** — already planned as
[Global model per `time_series_type`](../roadmap/xgboost-improvements.md) (issue
[#104](https://github.com/openclimatefix/nged-substation-forecast/issues/104)), described there as
"the stepping stone to V2 scale". Its stated prerequisites — per-series target normalisation,
static per-series features, batched training — are exactly what an Indian deployment would need
anyway.

**The storage partitioning needs rework.** `power_time_series` partitions by `time_series_id`
([`assets.py:115`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/src/nged_substation_forecast/defs/assets.py)),
which would mean 100,000 Hive directories and a small-file explosion on every append; it would need
a date-based partition instead. `power_forecasts` partitions only by `(experiment_name, fold_id)`,
with no time or series axis. At the brief's scale a single full-ensemble run is roughly 100,000
series × 51 members × 14 days × 96 steps/day ≈ **6.9 billion rows per run**. At the current
6-hourly cadence (4 runs/day) that is of order 10 trillion rows per year.

Be careful which number that translates into, because the two differ by more than two orders of
magnitude:

| A year of forecasts at the brief's scale | Volume |
|---|---|
| **On disk**, in `delta_store`'s compressed Delta layout, at the measured ~1.8 bytes/row | **~18 TB** |
| **Uncompressed**, as JSON on the wire, at the measured ~356 bytes/row | **~3.6 PB** |

Both per-row figures are measurements from the existing `power_forecasts` table, recorded in
[Forecast delivery](forecast-delivery.md#how-big-is-flexpectations-power-forecast-data); only the
row count is extrapolated. **The ~18 TB is the storage bill; the petabyte figure is what you would
move if you ever serialised it naively**, which is the same argument that page makes for delivering
Delta rather than REST — it just gets 80× sharper here. In-memory footprint sits between the two
and depends on the frame's column set, so it is not quoted.

18 TB on disk is affordable on S3 but painful to read. The obvious mitigation is to persist the
thirteen agreed [delivery quantiles](../roadmap/delivery-tables.md#representation-2-percentiles)
rather than raw ensemble members: one row per `valid_time` instead of 51, so **51× fewer rows and
roughly 4× fewer stored values**. Note those two ratios are not the on-disk saving, and the on-disk
saving is the one that matters: compression already exploits much of the redundancy across members,
so the byte reduction would land nearer the 4× than the 51×. That needs measuring through the real
`delta_store` write path before anyone quotes a number for it.

**Polars' 32-bit row-index cap stops being an edge case — it becomes a write-path blocker.** As
documented in
[Architecture Overview](overview.md#the-other-hard-ceiling-polars-32-bit-row-index), row counts
silently wrap past 2³² rows, and materialising a single frame of ≥2³² rows is unsupported outright.
At V2 the cap affects one code path (the `metrics` asset's whole-fold collect). Here, **a single
run's 6.9 billion forecast rows exceed the 4.29-billion cap on their own**, so the output of one
inference run could not be materialised as one frame at all. Chunking the write is not an
optimisation at this scale; it is a precondition.

NWP volume, by contrast, scales with **area, not site count**. Great Britain's full ECMWF ensemble
archive measures **~40 GB per year on disk** in the same compressed Delta layout (the whole
development table is ~93 GB for 5.9 billion rows). India is roughly 15× the land area, so the
equivalent archive would be of order **600 GB per year on disk** — scaled by area, not measured,
and it says nothing about the in-memory cost of a query against it, which the
[input-pruning strategy](overview.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output)
exists to bound separately. The bigger loss is that the
`h3_index` pruning described in
[Architecture Overview](overview.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output)
stops helping: with 100,000 sites spread across the country, the cells the sites occupy *are* the
whole grid.

### Radical options for shrinking what we store

The volumes above assume we keep doing exactly what we do today, only more of it. If the
[open questions](#questions-we-should-ask-them) come back badly — long horizon, full ensemble,
frequent runs, full history retained — these are the levers worth reaching for. All four are
**unmeasured proposals**, and none should be quoted at anyone until it has been benchmarked through
the real `delta_store` write path on real data. The point of listing them is that the design space
is not exhausted, not that any specific number is available.

#### For `power_forecasts`

**Store residuals against a cheap deterministic baseline, not absolute power.** This is the one we
would try first, because it attacks a *documented* weakness. `delta_store.power_forecasts` uses
`BYTE_STREAM_SPLIT` on `power_fcst` precisely because
"[near-continuous ML output has no repeats for a dictionary to exploit](overview.md#core-components)"
— it is making the best of a column with no exploitable structure. Subtracting a cheap, exactly
reproducible baseline (a per-site time-of-week climatology, say) leaves a residual that is small,
centred on zero, and — crucially — *quantised*: after the existing 13-bit significand rounding, a
narrow-range residual collapses onto far fewer distinct values than a wide-range absolute power
does, which is exactly the repetition dictionary and RLE encoding need. The baseline is stored once
per site and added back on read. This does not change the schema's meaning, only its physical
representation, and it is reversible, so it costs no accuracy beyond the rounding we already
accept. Whether it beats `BYTE_STREAM_SPLIT` in practice is an empirical question and would need
the same head-to-head treatment as
[PR #268](https://github.com/openclimatefix/nged-substation-forecast/pull/268).

**Stop storing every init_time at full horizon.** At four runs a day against a 14-day horizon,
every `valid_time` is forecast **56 times**, and we keep all 56. That redundancy is the single
largest structural waste in the table, and most of it earns nothing: skill-versus-lead-time
analysis needs a *sample* of lead times, not the complete cross-product. Keeping the newest
forecast for each `valid_time` at full resolution, plus a deliberately sampled subset of older
init_times (say, one run per day at a handful of lead times), would cut the table by something like
an order of magnitude while preserving every analysis anyone has actually asked for. The cost is
that it is *lossy at the table level* — a backtest we did not anticipate becomes impossible rather
than merely slow — so it should follow the "do they need the full history?" question rather than
precede it.

#### For NWP

**Store weather anomalies against a climatology, not absolute values.** The same trick as
residualising forecasts, and it fits NWP even better: temperature at a given cell, day-of-year and
hour is highly predictable, so the anomaly is small and repetitive where the absolute value is
neither. `delta_store.nwp` currently gets nothing from dictionary encoding on the continuous
fields; anomalies would give it something to work with. There is a pleasing efficiency here — the
roadmap already plans to ingest a climatology for the
[weather-abnormality z-score features](../roadmap/xgboost-improvements.md), so the climatology
would be on hand for its own reasons and this becomes a storage-layout change rather than a new
data dependency.

**Delta-encode ensemble members against the control member.** Fifty-one members of the same run at
the same cell and valid_time are highly similar, so storing member 0 in full and the other 50 as
differences from it should compress far better after significand rounding. **But this collides
head-on with a decision we already measured**: `NWP_SORT_COLS` sorts member-early
(`init → member → valid → h3`) specifically so that single-member reads skip most row groups, which
was measured at ~5× faster and ~5× less peak memory than the alternative. Delta-encoding across
members needs members *adjacent* for the same cell and valid_time, which is the opposite ordering.
So this is a real trade — training reads (one member, fast) against storage — and it should only be
considered if storage genuinely becomes the binding constraint. Noting it mainly to record that the
option exists and why we would not take it lightly.

### PV disaggregation without capacity priors

This is the part the bid actually turns on, and it is the part that is already designed.

[`UniversalSolarFleetNode`](../techniques/differentiable-physics.md#scaling-to-aggregate-fleets-universalsolarfleetnode)
models exactly this object: an aggregate, unmetered solar fleet behind one substation, whose
installed capacity is unknown and is represented as a cumulative sum of non-negative weekly
increments (installations only ever add capacity) with a sparsity penalty, because installs happen
in bursts. It needs no capacity register. The
[convex dictionary baseline](../roadmap/disaggregation.md#the-convex-dictionary-baseline) — fit a
sparse, non-negative, monotonically-growing amount of each of a menu of candidate panel
orientations — needs no capacity prior either, and would be the right first deliverable.

What is genuinely **harder** in India:

- **No metered PV inside the dataset.** The NGED plan uses verified metered generators
  ([Capacity estimation](../roadmap/capacity-estimation.md)) to anchor the harder unmetered
  inference. The brief offers no such anchor. Note the careful wording: India certainly *has*
  metered solar — a large utility-scale fleet with published output — so an external anchor may
  well be obtainable. It would simply be new work, not something the method already assumes.
- **Reversible soiling, which the capacity prior cannot express.** Indian rooftop PV loses a
  substantial share of its output to dust between monsoons and recovers sharply once rain washes
  the panels. But
  [`UniversalSolarFleetNode`](../techniques/differentiable-physics.md#scaling-to-aggregate-fleets-universalsolarfleetnode)
  represents installed capacity as a cumulative sum of non-negative increments — **non-decreasing
  by construction** — precisely because installations only ever add. A loss that reverses is
  structurally inexpressible in that prior. In the GB design the mechanism that absorbs this kind
  of variation is *effective*-capacity tracking, which
  [scopes itself to metered generators](../roadmap/capacity-estimation.md) — the very anchor the
  previous bullet says is missing. The two gaps compound rather than being independent.

    The **fix is small and probably worth making for GB too**: factor the fleet's output into the
    existing monotone installed capacity multiplied by a *reversible* soiling ratio in (0, 1], and
    drive the soiling ratio with a two- or three-parameter differentiable state — accumulation
    proportional to time since rain, wash-off above a rainfall threshold, a learnable floor. It
    composes cleanly with `UniversalSolarFleetNode` rather than replacing it, keeps the monotone
    prior intact (which is doing real work identifying installs), and needs no new input:
    `precipitation_surface` is already in `_ECMWF_ENS_VARS_TO_DOWNLOAD`. The genuine cost is
    **identifiability testing**, not implementation — soiling and slower-than-assumed capacity
    growth both depress output, and they are separable only because soiling correlates with
    rainfall history and has a sawtooth shape where installs are steps. That separation needs
    demonstrating on synthetic data before it is trusted on real data.

- **Aerosol and monsoon bias in the irradiance itself.** The Indo-Gangetic Plain carries among the
  world's highest aerosol optical depth, which systematically biases satellite- and NWP-derived
  irradiance, and monsoon convection is poorly resolved at 0.25°. Because installed capacity is
  inferred *from* irradiance, a systematic irradiance bias becomes a systematic capacity bias.
  Worse, the high-resolution irradiance source the GB plan depends on — SARAH-3, from EUMETSAT's
  Satellite Application Facility on Climate Monitoring (CM SAF), see
  [Data sources](../roadmap/data-sources.md) — covers ±65° longitude, and India begins at 68°E, so
  it does not reach India at all. A replacement has to be sourced; the candidates are in
  [Data sources that would materially help](#data-sources-that-would-materially-help) in Part 1.
- **Confounders with no British analogue.** Load shedding and diesel gensets both violate the
  assumption that latent demand is smooth and weather-driven. Load shedding is the dangerous one:
  it resembles a demand collapse uncorrelated with weather, and an unguarded optimiser would
  explain it with phantom solar. Explicit regime detection would need to be budgeted, not bolted
  on. Unmetered agricultural pumping is the happier case — Indian agricultural feeders are largely
  segregated and supplied on a published schedule, which makes a large unmetered load partly
  *observable exogenous information* rather than a pure confounder.

What is genuinely **easier**:

- **100,000 sites instead of ~2,500.** The design's cross-site strength comes from hierarchical
  parameter sharing — universal basis shapes plus a small per-site style vector. The method has
  always been designed against V2's ~2,500 sites rather than against the 32 in the V1 trial area,
  so this is a 40× improvement on the design point, not a 3,000× one. That structure keeps
  improving with more sites, so the gain is real, but it is an easier win at the margin than the
  raw ratio to V1 suggests.
- **15-minute data instead of half-hourly.** Finer sampling separates the solar shape from the load
  shape more cleanly, particularly around sunrise and sunset ramps.

### How we would structure it

Recorded for completeness. This is what we would do *if* we won the bid; it is not what we are
doing.

**One monorepo, not a fork.** Forking would mean doing the disaggregation research twice, which is
the single most expensive mistake available here. Instead, promote geography to an explicit seam:

- A **`RegionProfile`** in `contracts` carrying the latitude/longitude bounds, the four enums, the
  sampling interval, the timezone, the H3 resolution, and the power thresholds — injected rather
  than hard-coded. This is the largest single edit, and it pays for itself by making the British
  assumptions *visible* rather than implicit.
- `nged_data` becomes one of several ingest packages behind a small **`PowerIngest`** protocol whose
  contract is "emit `PowerTimeSeries` and `TimeSeriesMetadata`". Only three modules import
  `nged_data` today, so the boundary is nearly clean already — with one wrinkle worth fixing on its
  own merits: `defs/assets.py:40` imports the **private** `_H3_RESOLUTION` out of the ingest package
  and feeds it to the H3 grid builder, so the NWP spatial resolution currently lives inside the
  DNO-specific ingest code. That constant belongs in the `RegionProfile`.
- `geo/great_britain/` becomes a small region registry.
- Two Dagster code locations over one set of shared packages.

Indicative sizing, which phase each workstream falls in, and how much is shared with NGED's own V2
work. "Trial" means needed to forecast 50–100 substations; "Rollout" means needed only to reach the
full 100,000:

| Workstream | Effort | Phase | Shared with NGED V2? |
|---|---|---|---|
| Region seam, 15-minute support, Indian ingest | 4–6 weeks | Trial | Seam yes; ingest no |
| Convex dictionary disaggregator | 8–12 weeks | Trial | **Yes — it is the V2 baseline** |
| Global model, replacing per-series XGBoost | 6–10 weeks | Rollout | **Yes — needed for V2 regardless** |
| Storage partitioning and metrics chunking at 80× | 6–8 weeks | Rollout | Mostly |
| Full differentiable-physics PV engine | 6–12 months | Either | **Yes** |

The phase column is the reason a 12-month project is plausible at all: only the first two rows are
entry cost, and they are the two that answer the question the bid actually turns on.

### Why we are not doing any of this now

Speculative generality is not free. A `RegionProfile` seam introduced today is a layer of
indirection that every NGED contributor pays for, on every change, in service of a project we may
not win. The correct move is to leave the British assumptions hard-coded and *legible* — this page
is a large part of what makes them legible — and to pay the refactoring cost only once there is a
second consumer to amortise it against.

The two exceptions are the `local_utc_offset` whole-hour assumption and the private
`_H3_RESOLUTION` import, both described above. Neither is really a portability concern — they are
ordinary code-quality items that happened to surface here — so both can be fixed on their own
merits whenever convenient, independently of anything on this page.

**What would change our mind:**

- **Winning the Indian bid.** The obvious trigger, and the only one that justifies the full seam.
- **Any second DNO or DSO engagement**, British or otherwise. A second British licence area would
  exercise most of the same seam — the enums and the required NGED settings — without the
  resolution or scale work.
- **The scale work becoming necessary on NGED's own merits.** The global model and the storage
  re-partitioning are already on the V2 path. If they land for NGED, the marginal cost of a
  geographic port drops sharply, and this assessment should be re-run rather than trusted.

## See also

- [Net-demand disaggregation](../roadmap/disaggregation.md) — the method that does most of the work
  in the Indian scenario, and the reason a fork would be the wrong structure.
- [Differentiable Physics](../techniques/differentiable-physics.md) — `UniversalSolarFleetNode` and
  the monotone capacity representation.
- [Evaluating disaggregation](../techniques/disaggregation-evaluation.md) — the protocol that turns
  the disaggregation claim into something testable.
- [Architecture Overview](overview.md) — the memory and row-index ceilings that set the scale limits
  quoted here.
- [Why Dagster, not Airflow?](why-dagster-not-airflow.md) — the same genre of assessment, reaching
  the same "not now, and here is what would change that" conclusion.
