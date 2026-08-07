# Could this codebase forecast another country?

> **Status: Thought experiment — not planned work.** This page records an assessment made on
> 2026-08-05 and revised on 2026-08-07 (against `main` at `737fb86`), while OCF was preparing a
> pitch for **ENTICE 3.0**, an innovation programme in India — see [The brief](#the-brief).
> There is no GitHub issue for any of it, no roadmap entry, and **no intention to
> refactor this codebase for portability**. We would not start any of this work unless we won that
> bid. Nothing here is a commitment, and no current design decision should be taken "so that India
> would be easier later" — see [Why we are not doing any of this now](#why-we-are-not-doing-any-of-this-now).

## If you read this page before 2026-08-07, read only this

Six things changed. Everything else on the page stands. We are targeting ENTICE 3.0's **problem
statement 2** (hyperlocal AI forecasting for DISCOMs); the storage problem statement is not ours.

1. **The money is much smaller than this page assumed** — up to USD 100,000, as *convertible*
   funding rather than a grant, and possibly split between winners. The page was scoped like an
   NGED-sized contract. → [What the award actually buys](#what-the-award-actually-buys)
2. **Rooftop PV *is* metered — monthly.** This falsifies the hardest constraint in the old brief.
   Biggest single improvement to our position — *if* the reads are generation rather than net
   export, and *if* they map to a substation. → [What the monthly PV totals buy
   us](#what-the-monthly-pv-totals-buy-us)
3. **Scale is one or two DISCOMs, not a country**, and the site count is open by a factor of
   several hundred — ~100,000 distribution transformers, or a few *hundred* substations. On the latter
   answer, the whole scale section of this page evaporates. → [How many sites, and at which
   voltage?](#how-many-sites-and-at-which-voltage)
4. **The pilot partners are named: BRPL (Delhi) and JVVNL (Jaipur).** Delhi has no *rostered* load
   shedding, which shrinks our most-feared confounder to a Rajasthan problem (it does still have
   unplanned outages). The two states also sit at opposite ends of the rooftop hosting-capacity
   range. → [The two pilot DISCOMs](#the-two-pilot-discoms-delhi-and-jaipur)
5. **They believe they know where most rooftop PV is installed**, as this page predicted from the
   regulations alone — though a register records *sanctioned* capacity, not working capacity.
   → [India does record domestic PV capacity](#india-does-record-domestic-pv-capacity)
6. **The "ask these three questions" list is now five**, led by the site-count question and by
   whether the monthly PV figure is *generation* or *net export* — a distinction that decides
   whether we have an anchor at all. → [Questions we should ask
   them](#questions-we-should-ask-them)

This page exists in the same spirit as
[Why Dagster, not Airflow?](why-dagster-not-airflow.md): a question was asked, we did the analysis,
and the reasoning is worth keeping auditable even though the answer is "not now".

## Part 1: for the pitch

This part is written for a reader who understands the energy system but does not read code. It
covers what the job would involve, how big it is, what we could honestly claim, and — most useful
in a Q&A — the **data sources worth asking the client about**.
[Part 2](#part-2-what-would-have-to-change-in-the-code) is the engineering detail behind it, and
can be skipped.

### The brief

The project is **ENTICE 3.0** — the third edition of the Energy Transitions Innovation Challenge,
run by the [Global Energy Alliance for People and Planet (GEAPP)](https://entice.energyalliance.org/)
in partnership with **Startup India** (part of India's Department for Promotion of Industry and
Internal Trade, DPIIT). It carries two problem statements; ours is the second:

> Develop a hyperlocal, AI-driven forecasting platform that enables DISCOMs to accurately predict
> net grid demand by simultaneously modeling distributed renewable energy (DRE) generation —
> including rooftop solar and decentralized solar assets.

(That is the official wording from the programme's own problem-statement page. Much of the press
coverage paraphrases it, and the paraphrase is weaker — quote the original.)

It is more specific than it first appears. It asks for **net** demand; it asks for that net demand
to be predicted by **simultaneously modelling the distributed generation**, naming rooftop solar
explicitly; and "hyperlocal" places the spatial unit below the feeder. So the statement asks for
decomposition, not merely for a net-flow forecast.
What it does not settle is *where the PV number comes from* — a register they hand us, or a quantity
we recover from the measurements — and that is the single largest fork in the project (see
[the scope question](#questions-we-should-ask-them)).

The other problem statement is non-lithium storage "deployable at the **distribution transformer
(DT) voltage level**", which tells us the programme as a whole is thinking at DT level rather than
at primary-substation level.

Four things about the programme's shape matter as much as the problem statement does:

- **It is a challenge with a pilot attached, not a procurement.** Applications have closed;
  shortlisted entrants attend a three-day bootcamp and pitch. GEAPP offers **up to USD 100,000 in
  milestone-based catalytic funding through a convertible instrument** — one that can convert to
  equity on a later financing round rather than being a grant. The wording is per winner and the
  number of winners is not published, so the actual award is uncertain; what it does and does not
  pay for is set out in [What the award actually buys](#what-the-award-actually-buys).
- **The deployment partners are named, so this is not generic India.** Pilots run with **BSES
  Rajdhani Power Limited (BRPL)** in South and West Delhi, and **Jaipur Vidyut Vitran Nigam Limited
  (JVVNL)** in Rajasthan. That pins down the climate, the regulator, the metering regime and the
  confounders — see [The two pilot DISCOMs](#the-two-pilot-discoms-delhi-and-jaipur).
- **The scale is one or two distribution companies, not a country.** That matters for every volume
  figure on this page, and it interacts with the unresolved question of which network asset we are
  forecasting — see [How many sites, and at which
  voltage?](#how-many-sites-and-at-which-voltage).
- **ENTICE 3.0 was launched as part of GEAPP's "India Grids of the Future" accelerator**, at Mumbai
  Climate Week in February 2026, anchored by up to **USD 25 million** for grid modernisation with a
  stated ambition of USD 100 million by 2030. ENTICE is explicitly that accelerator's innovation
  funnel, which makes "a route towards the larger pot" the programme's own framing rather than our
  hopeful reading of it.

#### What we know about the data

Checked August 2026. The metering and register facts came from OCF's co-founder ahead of the pitch;
the rest is from published sources.

- **The forecasting unit is unresolved.** The choice as put to us is roughly 100,000 distribution
  transformers, or a few *hundred* substations. They differ by a factor of several hundred, and almost
  everything in [Part 2](#part-2-what-would-have-to-change-in-the-code) hangs off the answer — see
  [How many sites, and at which voltage?](#how-many-sites-and-at-which-voltage).
- **Rooftop solar is mostly metered separately from demand, but only as monthly totals.** So the
  research problem is not blind disaggregation; it is recovering a *shape* whose integral we already
  know. That is a much better-posed problem, and it is the most valuable single fact we hold — see
  [What the monthly PV totals buy us](#what-the-monthly-pv-totals-buy-us).
- **They believe they know where most rooftop PV is installed**, with the caveat that a location on
  a register does not mean the system is connected, working, or clean. The registers behind that
  claim, and how far they can be trusted, are in
  [India does record domestic PV capacity](#india-does-record-domestic-pv-capacity). The caveat is
  precisely the gap our method fills.
- **Sites are assumed to report power flow every 15 minutes.** This one is *not* confirmed — it is
  the working assumption behind the resolution work in Part 2 and behind every volume figure on this
  page, and it is on the list of [things to ask](#questions-we-should-ask-them).

### What the award actually buys

**The award is smaller than a project of this shape normally attracts, and we do not know how much
smaller.** GEAPP offers "up to USD 100,000 in milestone-based catalytic funding through a
convertible instrument". The wording is per winner; the number of winners for 3.0 is not published,
and ENTICE 2.0 had four. So the realistic award is somewhere between roughly USD 25,000 and
USD 100,000 — a 4× uncertainty that we should close by asking rather than guessing, because it is
large enough to change what we can responsibly promise.

Set even the top of that range against the [effort table](#how-we-would-structure-it) at the end of
this page. The two **Trial**-phase rows — the region seam with 15-minute support and Indian ingest,
and the convex dictionary disaggregator — are the entry cost. Neither carries a week count, for the
reasons given at that table, so we cannot state a precise ratio; what we can say is that the entry
cost is the *floor* rather than the project, because it excludes the full differentiable-physics PV
engine at **six to twelve months** — the row the table classes as neither trial nor rollout because
it could fall in either. The convex baseline alone is a real deliverable and a real result, and
[What we can claim](#what-we-can-claim-and-what-we-should-not) presents it and the physics engine as
two forms that check each other. But the baseline is the cheaper of the two, so an award-sized
project delivers the half that benchmarks the method rather than the method itself.

So nothing on this page should be read as "here is what we would build for the prize money". Three
readings make sense of the award, and we should be clear with ourselves about which one we are
pitching:

- **The prize is not the point; the pilot access is.** The thing ENTICE actually hands over is a
  working relationship with two DISCOMs, their data, and their regulators — which is exactly what
  [Questions we should ask them](#questions-we-should-ask-them) says we cannot buy at any price.
  Getting a real Indian feeder's data in front of the disaggregation method is worth more to us than
  the cash.
- **The convertible funding is a route to the larger pot**, most obviously GEAPP's own USD 25
  million *India Grids of the Future* accelerator. ENTICE is structured as a funnel, and it is fair
  to treat a win as a qualifying round.
- **Most of what India needs, NGED is paying for anyway.** The "Shared with NGED V2?" column in the
  [effort table](#how-we-would-structure-it) is the whole argument: the convex dictionary
  disaggregator, the global model, and the physics engine are all NGED work regardless. The
  genuinely India-only cost is the first row — the region seam, the 15-minute support and the Indian
  ingest — and that row, rather than the whole table, is what should be set against the award.

One practical question we have not resolved, and should: **a convertible instrument is an awkward
fit for a non-profit.** Convertible funding presumes a future equity round to convert into. OCF has
no equity to issue, so either the instrument never converts — which is fine, and is how the
programme describes the downside — or the arrangement needs a variation. Worth settling with GEAPP
early rather than after a win. Note too that the "no lock-in, MIT-licensed, transferable capability"
argument in [What we can claim](#what-we-can-claim-and-what-we-should-not) was written for a
grant-making innovation funder. It is still true and still a genuine differentiator for a
philanthropic backer, but it is a *weaker* argument to an instrument whose upside is equity, and we
should not lean on it as hard in this room as we would with a British network operator.

### How many sites, and at which voltage?

This is the most consequential unresolved question in the brief: 100,000 distribution transformers,
or a few hundred substations. It is a factor of several hundred in site count, and essentially all of
[Part 2](#part-2-what-would-have-to-change-in-the-code) turns on it.

The ambiguity has a clean cause: **Indian distribution has three rungs where the British network
has two**, and "substation" is used for at least two of them.

| Rung | Typical count *per DISCOM* | The British reader's nearest analogue |
|---|---|---|
| 33/11 kV substation | hundreds to low thousands | Primary substation |
| 11 kV feeder | thousands | HV feeder |
| Distribution transformer (11 kV/LT) | tens of thousands to ~1 million | Secondary substation |

Both pilot DISCOMs bear that out. BRPL publishes **11,161 distribution transformers and 1,768
11 kV feeders** for FY 2024-25, serving 31.89 lakh (3.19 million) consumers across roughly 700 km²
of South and West Delhi. The Central Electricity Authority's *Report on Status of Metering as on
31 March 2025* gives JVVNL **905,646 distribution transformers and 10,457 11 kV feeders**, for a
consumer base above 5.5 million.

**Neither figure is 100,000**, and that is worth noticing rather than explaining away: 100,000 is
nine times BRPL's transformer fleet and about a ninth of JVVNL's, so it matches no asset class at
either partner. The most likely readings are that it is a *sample* of JVVNL's transformers, a
combined or rounded figure, or a national-scale ambition rather than the pilot's own population.
"A few hundred substations" is the better-behaved end: a 2016 trade profile of JVVNL implies roughly
1,800 33/11 kV substations, and BRPL's grid substations number in the dozens to low hundreds, so
"a few hundred" is plausible for BRPL and low by roughly an order of magnitude for JVVNL. **Both
halves of the question need a straight answer from the client**, because we cannot reconcile either
number against published asset counts.

Two consequences follow, and they point in opposite directions.

**National statistics about India are context, not scope.** Figures quoted later on this page — the
metered share of transformers against feeders, the urban/rural split, the agricultural-feeder
exemption — describe the country. They are useful for judging which reading of "substation" is
plausible and what a sample would look like, but the pilot draws on whatever BRPL and JVVNL actually
hold, not on a national sample.

**If the answer is "a few hundred substations", the entire scale section evaporates.** Everything in
[The real work is scale, not geography](#the-real-work-is-scale-not-geography) — the global model,
the storage re-partitioning, the Polars row-index ceiling — is triggered by site count, and a few
hundred sites is *smaller* than NGED's own V2 design point of ~2,500. Per-series XGBoost handles it
today. On that answer the project is entry cost only: the region seam, 15-minute support, Indian
ingest, and the disaggregation research. That is a very different and much more fundable shape, and
it is the shape that fits the award plus pilot access. **If the answer is 100,000 transformers**,
the scale section applies in full, but as *rollout* work that a first-year pilot can defer — which
is the argument that section already makes.

There is a third possibility worth raising rather than waiting to be asked: **the two are not
exclusive**. A sensible programme forecasts at 33/11 kV substation level first, because that is
where the metering is complete and reliable, and pushes down to transformer level where the DT
meters and the consumer-to-transformer mapping support it. Our method does not care which rung it
is pointed at; it cares about how many customers sit behind a measurement point, because that sets
how much PV is behind it and how much load diversity smooths the demand.

### The two pilot DISCOMs: Delhi and Jaipur

The two partners are close to opposites, and the split is a mixed blessing: **both of our hardest
physics concerns land squarely on the pilot areas, while our most-feared confounder largely does
not.**

**BSES Rajdhani Power Limited (BRPL), South and West Delhi.** BRPL serves about 3.2 million
consumers over roughly 700 km², so its network is dense and urban. Two things follow.

The bad news is that Delhi sits in the **Indo-Gangetic Plain**, which carries among the world's
highest and most episodic aerosol loading — crop-residue burning and dust — and which
[Concerns about ERA5 over India](#concerns-about-era5-over-india) identifies as a systematic source
of irradiance error. That applies at full strength here, so irradiance
sourcing is not a hypothetical risk but the central one. Note also that the one
closest research-grade irradiance record we found, the closed **Gurgaon (GUR)** station, sits about
25 km from central Delhi in the National Capital Region, with a record running 2014-07 to 2019-01.
It is a BSRN *Candidate* station rather than a full one, and a third-party catalogue flags it for
periods of low data quality, so "research-grade" should be checked rather than assumed. It is worth
chasing anyway because it is so well placed — and worth asking whether the Solar Radiation Resource
Assessment network described below has a station in the NCR or Haryana, which would be better
still.

The good news is substantial: **Delhi has essentially no load shedding.** Delhi
DISCOMs operate to a 24×7 supply norm and meet peak demand without deliberate curtailment,
including a record 8,748 MW on 29 June 2026. Their published outage data describes trippings and
planned maintenance rather than rostered shedding.

Be precise about what that does and does not buy, because the distinction matters to the method.
What disappears is the *scheduled* confounder — the clock-aligned, group-structured rotation that
[Modelling load shedding and diesel backup](#modelling-load-shedding-and-diesel-backup) devotes most
of its machinery to, including the rotation-learning problem. What does **not** disappear is
outages: trippings and planned maintenance produce exactly the same sustained, non-weather level
shift the detector has to handle, and Delhi has plenty of both. So Delhi shrinks this problem rather
than deleting it, and diesel backup correspondingly reduces to peak-shaving gensets rather than
outage cover. That section is otherwise scoped to Rajasthan.

**Delhi's transformer-level rooftop rule is genuinely unresolved, and we should not assert either
reading in the room.** Two incompatible accounts survive our research, and the disagreement is not
about a number but about the *direction* of the provision:

- Several secondary sources state that the Delhi Electricity Regulatory Commission (DERC) limits
  cumulative rooftop capacity to **15% of the local distribution transformer's capacity** — which
  would be the tightest cap we found anywhere.
- Against that, the 15% figure appears prominently in a **2013 DERC consultation paper** that was
  never notified, and the guidelines under the DERC (Net Metering for Renewable Energy) Regulations
  2014 appear to run the other way entirely, obliging the licensee to *offer* **not less than 20%**
  of each transformer's rated capacity. On that reading it is a floor on hosting capacity, not a
  ceiling.

We have not read the operative instrument end to end, and the two readings have opposite
consequences, so **this is the first thing to check before anyone repeats either version**. Under
the cap reading, Delhi's limit would be a genuinely informative upper bound on installed capacity
and reverse flow at a Delhi transformer would be rare, which would weaken the signal the
disaggregation most wants. Under the floor reading, headroom is mandated and reverse flow becomes
*more* likely over time, which helps us. Either way it sharpens
[the reverse-flow question](#questions-we-should-ask-them) — and note that a per-substation
technical-feasibility record has to exist inside the DISCOM under either reading, which is the part
we actually want.

There is a second, unambiguous, and more useful Delhi finding underneath this one. The DERC
Regulations 2014 define a **"renewable energy meter" as a unidirectional meter used solely to record
renewable generation**, require the licensee to install it, and make the licensee bear the cost. The
Delhi Solar Energy Policy 2023 then pays its generation-based incentive on *gross solar generation
per kWh*, disbursed monthly with the billing cycle. So in Delhi there is both a regulatory
requirement for a generation meter and a financial reason for it to be read every month — which is
the strongest evidence we have that the monthly figures described to us are genuine generation
rather than net export.

**Jaipur Vidyut Vitran Nigam Limited (JVVNL), Rajasthan.** JVVNL serves over 5.5 million consumers
across Jaipur and the surrounding districts, so it combines an urban core with a large rural
hinterland. It is the mirror image of BRPL.

Rajasthan is arid and dusty, on the margin of the Thar desert, so the **soiling** problem that
[Differentiable physics → Soiling](../techniques/differentiable-physics.md#soiling) exists to solve
is at its most severe here — and, helpfully, its most *observable*, because a sharp monsoon washing
signal is exactly what identifies a reversible cleanliness factor. Agricultural load is large and
supplied on a **rostered schedule**, which makes the agricultural-feeder point below a live
opportunity rather than a hypothetical one; JVVNL has gone further and sought *virtual* segregation
of agricultural load on existing 11 kV feeders using smart metering, which if it works would give us
the agricultural component as a labelled quantity. **PM-KUSUM Component C feeder-level solarisation
is active in Rajasthan**, with RERC tariff orders covering JVVNL, so agricultural feeders in the
pilot area may carry solar that no rooftop register knows about.

The Rajasthan Electricity Regulatory Commission (RERC) is at the opposite end. Its Grid Interactive
Distributed Renewable Energy Generating Systems (DREGS) Regulations 2021 capped cumulative capacity
at a distribution transformer at 50% of that transformer's rating, up from 30% under the 2015
regulations — and the First Amendment 2023 raised it again to **80%**, which is the figure in force.
Check this against the gazetted amendment rather than the widely-circulated *draft*, which amends
different regulations entirely and so appears to leave the 50% standing. The individual ceiling is
1 MW under net metering or net billing, and has been 1 MW since 2015; what the 2021 regulations
changed was the per-consumer cap tied to load, from 80% to 100% of sanctioned load or contract
demand. Net-metering *eligibility* is separately constrained by the central Electricity (Rights of
Consumers) Rules, which DREGS 2021 explicitly defers to.

So the Delhi and Jaipur halves of the pilot differ by more than a factor of five in permitted
penetration, on top of differing in climate, reliability and load mix. **That is a feature.** Two
DISCOMs this different are close to the "chosen for variety" trial population that
[Questions we should ask them](#questions-we-should-ask-them) asks for, and a method that works on
both is far more credible than one tuned to either.

### What the monthly PV totals buy us

Rooftop solar being separately metered, even only as monthly totals, is the most valuable single
fact we hold about this brief. It makes the task **recovering an unknown *shape* whose integral we
already know**, rather than recovering an unknown quantity outright. Four things follow, and the
first three are gifts.

**It supplies an anchor for the physics.** Dividing a site's monthly generation by the monthly
specific yield the physics model predicts gives a direct estimate of that site's *effective*
capacity — not sanctioned capacity from a register, but what the panels actually delivered. The
anchor is not free: it is monthly rather than half-hourly, it is per-consumer rather than
per-substation, and it may be export rather than generation. Those three caveats are the questions
at the end of this section.

**It gives soiling something to be fitted against.**
[Differentiable physics → Soiling](../techniques/differentiable-physics.md#soiling) says the real
cost of a reversible cleanliness factor is not the code but proving that "the panels got dirty" can
be separated from "fewer panels were installed than we thought". A monthly series separates them,
**but only after normalising for irradiance** — and that qualification is the whole argument, so it
is worth being exact about it. Raw monthly generation in India is dominated by the seasonal cycle of
sunlight itself, and it moves in the *opposite* direction to the soiling signal: output is highest
in the clear pre-monsoon months, when the panels are dirtiest, and falls during the cloudy monsoon,
when the rain is cleaning them. Read raw, the series would suggest the panels work best when
filthiest.

The quantity that carries the information is therefore the **ratio of measured generation to the
generation the physics model predicts from that month's irradiance** — a monthly performance ratio.
That ratio should be flat if the panels are clean and the assumed capacity is right. If instead it
declines through the dry season and steps back up after the first washing rain, soiling is the only
available explanation, because installed capacity is monotone and cannot fall. **It is the recovery,
not the decline, that does the work.**

Three cautions stop this being decisive, and they matter enough that we should not claim it
*resolves* the identifiability question — only that it gives it data to test against. The synthetic-
data demonstration that [Differentiable physics →
Soiling](../techniques/differentiable-physics.md#soiling) asks for is still required.

- **The normalisation is only as good as the modelled irradiance.**
  [Concerns about ERA5 over India](#concerns-about-era5-over-india) records a monsoon
  *underestimate*, which would inflate the post-monsoon performance ratio and manufacture exactly
  the recovery we are looking for. This test needs a satellite irradiance product, not ERA5.
- **Capacity and cleanliness are the same equation.** The bullet above uses the monthly ratio to
  estimate effective capacity assuming clean panels; this one uses the residual in that same ratio
  to estimate soiling assuming known capacity. Twelve numbers a year cannot pin down both without an
  extra assumption — most naturally that a chosen post-monsoon month is clean, which fixes capacity
  and lets the rest of the year express soiling. State that assumption and test it; do not smuggle
  it in.
- **It is a Delhi argument before it is a Rajasthan one.** Rajasthan is where soiling is worst, and
  also the state where we least expect *generation* metering rather than net export (see below).
  Under an export meter the signal is convolved with household load and the argument collapses.

**It makes the "second product" immediately deliverable.** [What we can
claim](#what-we-can-claim-and-what-we-should-not) pitches an observed, continuously-updated capacity
estimate as a product in its own right, against a register that records only what was sanctioned.
With monthly generation per site, the comparison is arithmetic rather than research: a registered
system generating near zero is never connected, disconnected, or almost completely shaded — which
is most of the caveat the client attaches to its own register. (Soiling is not on that list: it
costs roughly 5–25% of output, not all of it, which is why it needs the finer treatment above.) We
could report that in the first month of a pilot, before the disaggregation research has produced
anything — though it too assumes the reads are generation, since a fully self-consuming system reads
near zero on an export meter.

**What it does not give us is the shape, which is the whole job.** A monthly total says nothing
about whether the generation came on a clear morning or a hazy afternoon, and a forecast at
15-minute resolution needs the intra-day profile. So this constrains the disaggregation rather than
replacing it: the physics model still has to produce the half-hourly curve, and the monthly total
becomes a hard constraint it must integrate to.

Three questions come with it, and they belong in
[Questions we should ask them](#questions-we-should-ask-them):

- **Is it generation, or is it export?** These are very different quantities and the word "metered"
  covers both. A *net* meter records the surplus after on-site self-consumption, so it sees only the
  part of generation the household did not use — which is a function of the household's load, not
  just of the sun, so it is much weaker and not usable without deconvolving that load first. It is
  not worthless: export is a hard lower bound on generation, and paired with the import register it
  is a real constraint.
  A *gross* or generation meter records what the panels made. Delhi is the encouraging case: the
  Delhi Solar Energy Policy 2023 pays a generation-based incentive **on gross solar generation per
  kWh**, for five years, disbursed monthly with the billing cycle — ₹3/unit for residential systems
  up to 3 kW, ₹2/unit for residential above 3 kW and up to 10 kW, ₹2/unit for group housing and
  resident welfare associations up to 500 kW, and ₹1/unit for commercial and industrial systems
  capped at the first 200 MW deployed. That is a financial reason to meter and read generation every
  month, and it is the most plausible origin of the figures described to us. Rajasthan's
  net-metering regime may well give export only. Ask per state.
- **Can the generation be mapped to the measurement point?** The totals are per *consumer*; we need
  them summed to whatever substation or transformer we are forecasting. Consumer indexing to
  distribution transformer is a known weak spot in Indian DISCOM data generally, and an explicit
  workstream under the national distribution-reform programme. But we should expect it to be much
  better for *this* subset than for consumers at large, because the DISCOM has to run a
  technical-feasibility study against a named transformer for every rooftop connection it sanctions
  (see [India does record domestic PV
  capacity](#india-does-record-domestic-pv-capacity)) — solar consumers are precisely the ones whose
  transformer is on file. Where it is missing, consumer addresses allow approximate assignment, so
  the realistic fallback is division-level aggregation rather than losing the anchor entirely.
- **What are the reading dates, and is the cycle monthly or bi-monthly?** Indian billing cycles are
  staggered across consumers, so a "monthly" total is a per-consumer window with its own start and
  end dates rather than a calendar month. This is a bookkeeping requirement rather than a threat:
  given the dates, we integrate the model over each consumer's own window and nothing is smeared,
  and staggered windows behind one transformer actually carry *more* sub-monthly shape information
  than aligned ones would. Without the dates we are stuck summing mismatched windows. Worth checking
  the cadence too — many Indian DISCOMs bill domestic consumers bi-monthly, which would halve the
  number of constraints.

### The short answer

What it would take to point this system at that brief splits into three quite different pieces,
and they are worth keeping apart because only one of them is genuinely hard.

Several statements below rest on **assumptions we have not checked with the client** — mostly
borrowed from what NGED wanted, because that is the only comparable project we know. Each one is
flagged in place, and all of them are collected in [Questions we should ask
them](#questions-we-should-ask-them) at the end of this part. If you read only
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
forecasts in an uncompressed form, such as the JSON a REST API would send, would be over three
thousand terabytes. The distinction matters throughout this page, so every figure below says which
one it is.

**That 18 terabytes is a worst case, and the realistic number could be very much smaller.** It
assumes India wants exactly what NGED wants: a 14-day horizon, a full 51-member ensemble, and four
forecast runs a day. We have no basis for any of those three assumptions — they are simply the only
requirements we currently know. Each one multiplies the total, so each one we can relax shrinks it
sharply: a 2-day horizon alone cuts it sevenfold, and running once a day rather than four times
cuts it fourfold. Delivering the agreed set of percentiles instead of every raw ensemble member
should cut it around fourfold again, though that one is arithmetic we have not yet measured through
the real storage path. Plausible combinations land **between roughly 0.16 and 18 terabytes** — a
110× spread, set entirely by answers we do not yet have. (Strike the unmeasured quantile factor and
the spread is still 28×, so the argument does not rest on it.) The per-answer breakdown is in
[Questions we should ask them](#questions-we-should-ask-them), and it is the reason those questions
matter more than any clever engineering: **asking about the forecast horizon is worth more than
anything we could do to the storage format.** If the answers do come back at the demanding end,
there are further options — see
[Radical options for shrinking what we store](#radical-options-for-shrinking-what-we-store).

**We are proposing to start with a trial of roughly 50 to 100 substations, not with all 100,000.**
That is the same shape as our NGED programme, where we are running on a 32-substation trial area
before scaling to around 2,500, and it is worth saying out loud in the pitch, because it changes
the risk profile considerably. A trial at that size needs essentially none of the scale
engineering described above — it is comfortably within what the code handles today — so the first
year could be spent proving the hard part, which is whether we can actually recover unmetered
rooftop solar from Indian net-flow data, rather than on storage layouts. It also front-loads the
findings: we would have real results on a real Indian feeder within months, and the answers to the
questions below would by then be known rather than assumed, so the full-rollout design could be
sized against measured numbers instead of worst cases. What we should avoid promising is 100,000
substations in year one.

**Estimating unmetered solar is the research bet — and it is the same bet we are already making
for NGED.** This assumes they want the unmetered solar *recovered from the substation data* rather
than simply forecast from a capacity figure they already hold, which is a much lighter job and a
perfectly reasonable reading of the brief — **so check** (see
[Questions we should ask them](#questions-we-should-ask-them)); it is the single largest fork in
the project. Separating rooftop solar from underlying demand, with no generation meters and no
capacity register, is exactly the problem described in
[Net-demand disaggregation](../roadmap/disaggregation.md). The method there is designed for
precisely this: it treats a substation's reading as demand minus solar generation, models the solar
physically from sunlight, and infers the installed capacity as an unknown that only ever grows.
It does not need a capacity register.

Qualifications belong alongside that, and they cut both ways. In Britain we plan to use the
*metered* solar farms we can see to calibrate and sanity-check our estimates of the *unmetered*
ones. India gives us a different and in some ways better anchor: **rooftop solar in the pilot areas
is separately metered, as monthly totals**, so this is not blind disaggregation but the recovery of
a shape whose integral we already know. The caveats that come with that — monthly not half-hourly,
per-consumer not per-substation, and possibly export rather than generation — are set out in
[What the monthly PV totals buy us](#what-the-monthly-pv-totals-buy-us). India also has a
large, metered, utility-scale solar fleet with published output, which is a second, coarser
anchor. The physical
background is still harder than Britain's: Indian rooftop panels lose a substantial fraction of
their output to dust between monsoons and recover when the rain washes them, and our method
currently assumes installed capacity only ever grows, so it has no way to express a loss that
reverses.

Extending it to handle soiling looks genuinely straightforward, though — and it is something **we
should probably add for Britain anyway**. The fix is to stop treating "how much is installed" and
"how well it is working" as one number: keep the existing installed-capacity term that only grows,
and multiply it by a separate cleanliness factor between zero and one that dust pushes down and
rain pushes back up. That is three extra parameters, learned the same way as panel tilt and
orientation already are, and **the only input it needs is rainfall, which we already download**.
The honest caveat is that separating "the panels got dirty" from "someone installed fewer panels
than we thought" is a real statistical question that would need testing, not just coding — though
the **monthly generation totals** described above go a long way towards settling it, since installed
capacity cannot fall and a post-monsoon recovery in monthly yield therefore has only one available
explanation. Britain's
rainy average hides real dry-spell episodes, because the effect tracks time since the last washing
rain rather than the climate mean. Work done here would pay off in both countries. Separately,
high atmospheric dust also biases the satellite and forecast irradiance the whole method leans on.

Against that, 100,000 sites at 15 minutes would be a far larger and finer-grained dataset than the
~2,500 sites we have been designing towards all along for NGED's V2 (the V1 trial area we are
currently running on is 32 sites), and the shared parts of the model are much better constrained by
more sites. (Whether the pilot is at that scale at all is now itself an open question — see
[How many sites, and at which voltage?](#how-many-sites-and-at-which-voltage).) There are also
confounders India has and Britain does not: **load shedding** and **diesel backup generation** —
though these are much more a Rajasthan concern than a Delhi one, since Delhi runs to a 24×7 supply
norm with no rostered shedding
([The two pilot DISCOMs](#the-two-pilot-discoms-delhi-and-jaipur)).

Neither is new territory for us, though, and that is worth being able to say. A sustained step down
in metered power with no meteorological cause is precisely what NGED already presents us with when
the network is reconfigured, and we have a designed method for detecting those events unsupervised
([Switching events](../roadmap/switching-events.md)). The same detector applies here, and it
distinguishes the two cases for free: switched load reappears at a neighbour, whereas shed load
leaves the neighbourhood entirely, so conservation across the neighbourhood holds in one case and
fails in the other. Load shedding is in some ways the *easier* target, because it is a collapse to
near-zero rather than a partial transfer, and because it happens at night as well as by day —
night-time outages let us characterise an outage with no PV present to confuse matters, and that
transfers into daylight. Diesel separates from PV on shape and on the fact that it does not covary
with irradiance day to day. Modelling both is a few weeks on top of the disaggregation itself, and
much cheaper again if the shedding schedules are obtainable. Detail in
[Modelling load shedding and diesel backup](#modelling-load-shedding-and-diesel-backup).

Agricultural pumping is the happier case, and worth calling out as an opportunity: Indian
agricultural feeders are largely segregated and run to a published supply schedule, so a large
unmetered load is partly *known in advance*. Whether we can actually obtain those schedules — and
the load-shedding schedules — is one of the
[questions we should ask them](#questions-we-should-ask-them).

### What we can claim — and what we should not

Probably the most useful section here. Everything in it is defensible from work that already
exists; the second half is the part that stops us being caught out.

**The claim is not "we can forecast substations."** Many organisations can, and a procurement panel
has heard it before. Our claim is narrower and much harder to match: **OCF already has a designed,
written-down method for recovering *unmetered* rooftop solar from net substation flow with no
capacity register** — and a published protocol for proving whether it actually works. Six things
make that defensible rather than aspirational.

**1. The method exists on paper today, in two forms that check each other.** There is a **convex
dictionary baseline** — fitting a menu of known solar shapes, where the optimiser is guaranteed to
find the best fit available rather than getting stuck in a local one, and which is simple enough to
be reproducible and auditable — and a **differentiable-physics engine** that models each fleet's
tilt, orientation, temperature response and inverter clipping, then inverts that model to recover
what must have been generated ([Net-demand disaggregation](../roadmap/disaggregation.md),
[Differentiable physics](../techniques/differentiable-physics.md)). The baseline initialises the
engine and then permanently benchmarks it, so the engine only earns its complexity by beating it.
Worth being precise in the room: a convex solver finds the global best fit *of the objective it is
given*, which is not the same as being right — it cannot learn its way around an error in the
physics curves. That is exactly why the second, learnable route exists.

**2. We have written down, in advance, how we would be proved wrong.** This is rarer than the
method and worth dwelling on in the room. Disaggregation is hard to evaluate precisely because
nobody has ground truth for the thing being estimated — so it is easy to claim and almost
impossible to audit. Our
[evaluation protocol](../techniques/disaggregation-evaluation.md) sets out **six complementary
evaluations, each with different biases**, on the principle that agreement across them is the real
signal. Two of the six — physical-consistency and conservation residuals, and cross-source
corroboration — need **no labels at all**, which is precisely what lets the protocol survive a
dataset with no metered generation. The rest strengthen as metered anchors become available, which
is why the metering questions below matter so much. Anyone can assert a disaggregation result.
Committing in advance to how it will be
falsified is a much stronger signal, and it is exactly what an innovation funder should want.

**3. India would not be paying us to invent it.** The method is **designed in detail and funded**
through a British network operator's innovation programme — designed, note, not yet built, and
sitting on that programme's research track rather than being a committed build. An Indian project therefore inherits a
de-risked design rather than a blank sheet, and pays only for the extensions its own conditions
require: working without a metered anchor, without a clean irradiance product, and against physics
Britain does not have. Those extensions stay in the open codebase and are reusable by every Indian
distribution company that comes after. Very few bidders can offer a method that arrives part-funded
by another programme.

**4. The by-product may be worth as much as the forecast.** Recovering unmetered solar means
estimating **how much rooftop PV sits behind each substation, and how that grows over time** — an
*observed*, continuously-updated estimate — as against the DISCOM's own register, which records
*sanctioned* capacity for scheme and net-metered installations only and is never refreshed from
measurement. That distinction is the whole point, and it is worth making before someone in the room
says "we already have that register": theirs records what was approved, ours would record what is
actually there and working. That is not a footnote in India. As of March 2025 only about 42% of the
country's distribution transformers were metered at all, many state regulators cap
rooftop PV at a fraction of each transformer's rating, and DISCOMs are reported to be seeing
midday voltage rise and reverse flow on high-penetration feeders (reported in the trade press, not
something we have verified in data). An inferred, continuously updated
capacity estimate speaks directly to that: where the hosting headroom has gone, which transformers
are about to be stressed, and where the next reverse-flow problem appears. We should pitch it as a
second product, not as a side-effect.

**5. Physics, not just pattern-matching — and that has an operational payoff.** Because the solar
component is a physical model of the sun and the panels rather than a curve fitted to history, it
is constrained in ways a purely statistical model is not: it is identically zero at night, and
during the day it can only be as large as the physics and the fitted installed capacity allow. That
is what keeps Indian confounders like load shedding in check — a supply outage at midnight *cannot*
be explained away as solar under any parameter setting, and a daytime one is bounded rather than
free. The night-time half of that argument is airtight; the daytime half is only as tight as the
capacity prior, which is one more reason to want the capacity records above. The same property makes
the outputs explainable to an engineer,
which matters when a network planner has to act on them.

**6. It is open, and it is running.** The codebase is MIT-licensed and developed in the open,
so the client keeps everything, other Indian distribution companies can reuse it, and there is no
lock-in — an unusually good fit for innovation funding, which generally exists to create
transferable capability rather than a private asset. And the surrounding engineering is not
hypothetical. Orchestrated data pipelines and cloud storage run on AWS on a schedule today,
delivering a live forecast; experiment tracking and inspection dashboards are in daily use by the
team, though not yet themselves deployed as services. That is the unglamorous half of the work that
usually sinks projects like this, and it is largely done.

**What we should not claim.** Being straight about these protects the bid rather than weakening it,
and each one has a natural follow-up question we can turn back on the client.

- **Not a number for disaggregation accuracy, before the trial.** We can say precisely *how* we
  would measure it; we cannot say what it will come out at, and anyone who does is guessing.
- **Not per-site solar accuracy, until we know what the monthly PV reads actually are.** Monthly
  per-site totals would be a real anchor and would let us claim a great deal more — but only if they
  are *generation* rather than net export, and only if they can be mapped from consumer to
  substation. Until both are confirmed, promise the method, not the accuracy (see
  [What the monthly PV totals buy us](#what-the-monthly-pv-totals-buy-us)).
- **Not that public data closes the gap.** It does not. The free Indian sources are good enough to
  build and sanity-check the physics, not to anchor it.
- **Not that we have done this in India before.** We have not. What we have is a method, a
  protocol, a running system, and a British project that de-risks all three.

### Data sources that would materially help

The hard parts described above are mostly *data* gaps rather than method gaps, so it is worth being
concrete about what we would ask for or go and find. **This is the section to have in hand during a
Q&A**: each row is something we could reasonably ask the client, the utility, or a partner for.
Claims here were checked in August 2026 and are dated where they may drift.

| Source | What it gives us | Status for India |
|---|---|---|
| **The DISCOM's own monthly rooftop PV reads** | The anchor, and the client already holds it: per-site monthly generation totals, which pin down effective capacity and make soiling directly observable. | **Confirmed to exist** in the pilot areas (August 2026). Value depends entirely on three answers — generation or net export, mappable to a substation or not, and with what reading dates. See [What the monthly PV totals buy us](#what-the-monthly-pv-totals-buy-us). **Ask for this first.** |
| **Metered PV generation from India** | A second, coarser anchor: known generation against which to calibrate the physics model before inverting it for unmetered fleets. | Exists, but utility-scale and spatially aggregated. India's [Central Electricity Authority (CEA)](https://cea.nic.in/) and [Grid Controller of India (Grid-India)](https://en.wikipedia.org/wiki/Power_System_Operation_Corporation) — the latter renamed in November 2022 from the Power System Operation Corporation (POSOCO) — publish national and regional generation. The five Regional Load Despatch Centres (RLDCs) and the State Load Despatch Centres (SLDCs, one per state) publish more granular real-time data. |
| **SARAH-E** — Surface Solar Radiation Data Set – Heliosat, East, from EUMETSAT's Satellite Application Facility on Climate Monitoring (CM SAF) | The Indian Ocean Data Coverage (IODC) sibling of the SARAH product we plan to use for GB — and, critically, it carries **global, direct and direct-normal irradiance** (surface incoming solar radiation, SIS; direct irradiance, SID; and direct normal irradiance, DNI), so it has the beam/diffuse split the [DP solar model](../techniques/differentiable-physics.md) needs, at 0.05°. | Covers India. But Edition 1 runs **1999–2015 excluding 2006** on Meteosat First Generation (Meteosat-5/7); we found no confirmed post-2015 extension. Useful for pre-training and for validating the physics, **not** for near-real-time. |
| **NSRDB** — the National Solar Radiation Database from the US National Renewable Energy Laboratory (NREL), Meteosat Indian Ocean Data Coverage (IODC) region, Physical Solar Model v3 | Global horizontal irradiance (GHI), direct normal irradiance (DNI) and diffuse horizontal irradiance (DHI) at 4 km on a **15-minute** grid — matching the assumed substation metering cadence, so no temporal interpolation is needed. | Covers the Indian Ocean Data Coverage (IODC) region including India, but **only 2017–2019** — a three-year archive with no near-real-time extension we could find, which is the binding constraint. Excellent for building and validating the physics; not, on this coverage, a live input. Licensing and any extension need confirming with the National Renewable Energy Laboratory (NREL). |
| **IMDAA** — the Indian Monsoon Data Assimilation and Analysis regional reanalysis | A **12 km** reanalysis for the Indian monsoon region (4D-Var, Met Office Unified Model), against ERA5's 31 km — built by India's National Centre for Medium Range Weather Forecasting (NCMRWF) with the UK Met Office and the India Meteorological Department (IMD). Domain 30–120°E, 15°S–45°N; many products hourly, the rest 3-hourly. | Covers 1979–2018, extended to 2020. On that range it is a strong **pre-training** reanalysis where ERA5 is weakest but ends too early for live capacity estimation — though NCMRWF also publishes IMDAA-*like* products from its operational analysis over the same domain past 2020, so it is worth checking whether that continuation is close enough to real time to be a live input too. |
| **Agricultural feeder supply schedules** | Turns the largest unmetered load into a known regressor, per the note above. | Published by DISCOMs (India's electricity distribution companies) where feeder segregation has been implemented. Worth asking for explicitly. |
| **Load-shedding / outage schedules** | Lets the regime detector be *supervised* rather than having to infer outages from the power signal alone. | Same logic, same ask. |
| **Rooftop PV installed-capacity records** | A capacity prior: where the rooftop PV is, and how much of it. | Exists, and better than we expected. The client believes it knows where most rooftop PV is installed. See [India does record domestic PV capacity](#india-does-record-domestic-pv-capacity) below. Publicly it is state-level; per-substation records sit inside the DISCOMs (India's electricity distribution companies), which is where the pilot's would come from. **Ask for this.** |

#### India does record domestic PV capacity

How good a capacity prior can we actually get? Better than the phrase "unmetered rooftop solar"
suggests, because the installations themselves are counted carefully even where their output is not.
Checked August 2026.

India runs a large national residential rooftop scheme, **PM Surya Ghar: Muft Bijli Yojana**, and it
is a registry: every installation is a registered consumer of a named DISCOM (electricity
distribution company) with a sanctioned capacity in kW and an address. It reported **50.06 lakh
(5.01 million) beneficiary households and about 14.8 GW as of 4 August 2026** — the scheme's
official target is 10 million households by March 2027, with an interim government target of
7.5 million by December 2026. Attach the date to any of these: the scheme is adding roughly a lakh
of households every six days, so a figure quoted without one is wrong within a month. Note also
that it publishes both *installations* and *beneficiary households*, and the two differ — 39.38
lakh installations against 47.65 lakh households on the same day in July 2026 — so quote like for
like. Separately, the Ministry of New and Renewable Energy (MNRE) put India's grid-connected
rooftop solar at **23.16 GW as of 30 November 2025**, about 17% of the country's 132.85 GW solar
fleet. So the installations are counted, and counted well.

What is published is **state-level**: installation counts and capacity in MW per state, released
routinely through government press releases and the national open-data portal, with district-level
figures appearing occasionally as one-off answers to parliamentary questions rather than as a
maintained series. That is far too coarse to be a per-substation prior on its own.

**The more interesting finding is regulatory.** Many Indian *state* regulators cap the rooftop PV
connected to a single distribution transformer at a fraction of its rating, and require the DISCOM
to publish, in Telangana's wording, "the capacity available on each Distribution Transformer and
11 kV feeder of a substation and 33 kV feeder". A distribution transformer is essentially what a
British reader would call a secondary substation. The important consequence is that **a
per-substation installed-capacity record must exist inside the DISCOM**, because the distribution
company has to run a technical-feasibility study against that transformer for every new connection,
whether or not a cap binds — which is the most likely origin of the client's belief that it knows
where most rooftop PV sits. The publication duty generalises well across states, which is what makes
this worth asking about rather than a local curiosity.

Be careful with the framing, though. There is **no central Indian rule** imposing a
hosting-capacity fraction. The Electricity (Rights of Consumers) Rules delegate net-metering
arrangements to each State Commission — with a residual central default of 500 kW where a State
Commission has not legislated, which is a ceiling on *individual* system size rather than a
transformer fraction — so the fraction itself varies enormously between states. **The two states
that matter for ENTICE 3.0 sit near opposite ends of the range**: the Delhi Electricity Regulatory
Commission is reported at **15% of the local distribution transformer's rating** (with group and
virtual net metering exempt, and see the caveat about that figure in
[The two pilot DISCOMs](#the-two-pilot-discoms-delhi-and-jaipur)), while Rajasthan allows **80%**
since its First Amendment 2023, up from 50% in 2021 and 30% in 2015. For comparison, elsewhere it is
50% in Telangana's 2025 regulations, 70% under Maharashtra's 2019 regulations, 90% in Tamil Nadu,
and 100% of transformer
rating in Uttar Pradesh. Gujarat has relaxed its cap, though we could not confirm at source exactly
which limit was lifted. **Treat every one of these as unverified until read against the current
notified state regulation.** Only the Rajasthan figures above have been checked against the gazette
text; the Delhi entry is actively disputed, and the common failure mode is a secondary source
collapsing two distinct provisions — a transformer-level obligation on the licensee and a separate
per-consumer cap tied to sanctioned load — into a single spurious "hosting cap". So the cap is **not** a
reliable free upper bound on installed capacity in general — but in **Delhi specifically it is a
tight and therefore genuinely informative one**, with the corollary that reverse flow at a Delhi
distribution transformer should be correspondingly rare. And "shall be uploaded on the DISCOM
website" is an obligation, not evidence that it is
actually published, current, or machine-readable.

**Capacity is recorded; generation is the real gap — but that is changing on a useful timescale.**
MNRE mandated in December 2025 that PM Surya Ghar installations carry M2M SIM-based remote
monitoring, streaming real-time generation from the inverter's data logger to a national platform
on government-managed servers, with the inverter's data logger — rather than a separate solar
meter — as the primary measurement of generation. Odisha's regulator had recognised M2M-enabled
inverters as valid generation meters the month before, in November 2025, and MNRE has asked the
other state commissions to follow; it cannot bind them, so uptake will be uneven. Retrofitting the
existing fleet is not addressed, so much of the installed base likely stays dark. But a project
running through 2027 would coincide with a growing national stream of **per-site, real-time
rooftop generation** — precisely the anchor that does not exist today. Whether we could get access
is a question, not an assumption, and it is worth asking early because the answer could change the
shape of the disaggregation work substantially.

One more register is worth naming, because it sits exactly where our hardest confounder does:
**PM-KUSUM**, the national scheme solarising agricultural pumps and feeders. Those are precisely the
connections excluded from distribution-transformer metering, so a KUSUM register would cover PV that
is otherwise invisible to us twice over — unmetered as generation, and on unmetered transformers.
Worth asking about alongside the rooftop records.

Finally, if none of that is obtainable, a capacity prior can be **built rather than sourced**:
detecting rooftop panels from high-resolution satellite imagery is an active research area with
published Indian work. We would treat that as a fallback, not a plan — it is a project in itself,
and it yields panel *area*, not capacity or orientation.

#### What we could get hold of today, without asking anyone

Worth separating from the table above, because "this data exists" and "we can use it" are different
claims, and for India the gap between them is unusually wide. Everything below was checked in
August 2026.

**Ground-station irradiance — better news than we first thought.** India runs one of the world's
largest solar radiation networks: the **Solar Radiation Resource Assessment (SRRA)** network run by
India's **National Institute of Wind Energy (NIWE)**. NIWE describes it as **111 SRRA stations**
across two phases, with a consolidated table reaching 125 once stations sponsored by state agencies
are included. Each measures the full three-way split — **direct normal, diffuse horizontal and
global horizontal irradiance (DNI, DHI and GHI)** — which is what the differentiable-physics solar
model consumes, and what neither the ECMWF ensemble nor a GHI-only satellite product can give us.

We initially assumed this was locked up, and that appears to be **out of date**. NIWE states that a
May 2017 circular from India's Ministry of New and Renewable Energy (MNRE) put SRRA data into the
public domain **free of cost**, with access by registration and click-through terms rather than a
negotiated agreement. Three limits survive: **raw high-frequency time series** (one-minute,
ten-minute and hourly) are priced separately on NIWE's own data-sale page; stations sponsored by
bodies other than MNRE may still carry a charge set by the sponsor; and the **unprocessed** raw
data is released only for verification and collaborative work, by specific permission.
Redistribution to third parties is not permitted, and reproduction requires authorisation from NIWE.

**Treat this as promising but unconfirmed.** NIWE's site blocks automated access, so we read an
archived copy of its data policy which itself states a 2021 revision date. Before anyone repeats
the "free of cost" line in a pitch, someone should confirm it live — and it is still worth asking
whether the client or a partner already holds this data, because that would settle both the cost
and the format questions at once.

**The Baseline Surface Radiation Network (BSRN) is the free alternative, and it is genuinely
research-grade.** All of its data is free via the PANGAEA archive under FAIR (findable, accessible,
interoperable, reusable) terms. India has four stations, all now closed: **Tiruvallur (TIR)**,
station 59 at 13.09°N, 79.97°E, *operated by India's National Institute of Wind Energy (NIWE)*,
with a compiled record covering **2014-08 to 2019-01**; plus **Gandhinagar (GAN)**, **Gurgaon
(GUR)** and **Howrah (HOW)**. A third-party catalogue of radiation stations flags Gandhinagar and
Gurgaon as having periods of low data quality; the BSRN archive itself carries no such remark.
Four closed sites is nothing like national coverage, and the best record stopped in 2019, so this
cannot validate a live service. What it *can* do is validate the physics: it is enough to check a
differentiable PV forward model, and to quantify satellite and reanalysis irradiance bias over
India, before committing to either. Usefully, Tiruvallur is run by NIWE, so this free,
quality-controlled record is a window onto the same instrumentation programme as the SRRA network
above. The Global Energy Balance Archive (GEBA) holds Indian records too, but monthly means only,
which is too coarse to be useful here.

**Free PV power data exists, at two very different scales.**

- **Plant-level, high-resolution, tiny:** a widely-used public dataset covers **two Indian solar
  plants at 15-minute resolution over 34 days**, with **inverter-level** DC and AC power (22
  inverters per plant) alongside plant-level irradiance and module temperature. Thirty-four days is
  far too short for anything seasonal, but the shape is exactly right — 15-minute cadence, real
  Indian plants, and module temperature included — so it is a legitimate early test rig for the
  forward model's temperature derating and inverter clipping. A second public dataset from a
  350 kWp installation near Hassan, Karnataka pairs global, direct-normal and diffuse irradiance
  (GHI/DNI/DHI) with PV output — though that one is an undocumented upload whose provenance we
  could not confirm at source, so treat it as a lead rather than a dataset.
- **Aggregate, long, modelled:** an openly-licensed (CC-BY-4.0) [Zenodo dataset](https://doi.org/10.5281/zenodo.7824872)
  from the Universities of Reading and Bristol provides hourly wind and solar capacity factors for
  India from 1979 at state level and on a 1°×1° grid, plus *reported* wind, solar and hydro
  production for 2012–2023 sourced from India's national system operator. Read the two halves
  differently: the long capacity-factor series is **modelled**, so it is not independent evidence
  about irradiance, while the reported production is real but spatially aggregated.

**[PVOutput.org](https://pvoutput.org) has Indian systems, but almost none of them are alive.**
PVOutput is the obvious place to look — a global community platform where rooftop owners publish
live generation, and a genuinely useful source of per-site behind-the-meter data in other
countries. We queried its API directly (August 2026) rather than relying on the public pages, and
the finding is sharper than a headcount.

Sweeping 44 India-likely search terms, then keeping the results whose reported location was India,
turned up **42 distinct registered Indian systems**. Their upload histories look like this:

| Indian systems on PVOutput (found August 2026) | Count |
|---|---|
| Registered and found by this method | 42 |
| — of which never uploaded a single day | 23 |
| — of which uploaded 1 to 29 days | 9 |
| — of which uploaded 30 days or more | 10 |
| Of those 10: uploaded at least a year | 5 |
| Of those 10: **uploaded anything in the last month** | **2** |

The last row is the one that matters. **Two** Indian systems are actively reporting — a 3.9 kW and a
2.4 kW domestic rooftop, both with long histories (2,655 and 2,018 days) and both still uploading.
Everything else has either never produced data or stopped: the next most recent went quiet 16 weeks
ago, and the best-populated of the dead records (1,775 days) stopped 77 weeks ago.

**Treat 42 as a floor, not a count.** PVOutput's search cannot filter to India — its country filter
offers seven countries and India is not one — so we swept on names, and a system called "Home PV" is
invisible to that method. The count is bounded from the other side, though, and that is what makes
the conclusion safe: **India does not appear in PVOutput's 25-country table at all**, and that table
ranks by lifetime generation, with New Zealand 25th on 4.2 GWh (0.26% of the global total). So
whatever India's exact registration count is, its total contributed generation sits below New
Zealand's — and at the liveness rate we measured, that is a handful of live systems rather than a
fleet. Two live domestic rooftops cannot anchor anything at the scale of this brief.

An exact count *is* obtainable, just not free: PVOutput's paid Data Services tier will dump every
system in its "world" region — which includes India — with coordinates and install dates. That is
the same paid tier the bulk-data caveat below refers to, and it is not worth buying to confirm a
conclusion the free evidence already supports.
Two further caveats even if coverage were better: bulk access and the commercial-use licence both
sit behind PVOutput's paid tier, so the free tier is not usable for funded work; and self-reported
data carries unverified capacity, orientation and shading metadata, which is precisely what a
disaggregation anchor cannot afford to have unverified.

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
Centres (SLDCs), or from the PM Surya Ghar machine-to-machine generation feed as it builds out. The
Solar Radiation Resource Assessment (SRRA) network does not help here — it measures *irradiance*,
not generation, so it closes a different gap. All of that is worth saying plainly in a bid rather
than implying the public data closes the metered-generation gap.

#### ERA6 does not arrive in time

We checked, because it would change the reanalysis answer if it did.
[ERA6 production started on 6 March
2026](https://climate.copernicus.eu/copernicus-climates-era6-reanalysis-production-starts),
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
observations from the India Meteorological Department (IMD). That headline number is reassuring for
demand forecasting and misleading for
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
Database (NSRDB) over the Indian Ocean Data Coverage region, which covers 2017 to 2019 and so is
already years stale. Be clear about
what that leaves open, though: **neither is available in near-real time**, so on what we found there
is no confirmed high-quality irradiance input for a *live* capacity estimate over India — which is
one more reason the research-versus-live-service question below matters. ERA5
or the Indian Monsoon Data Assimilation and Analysis (IMDAA) reanalysis carry the non-radiative
fields. In GB the equivalent role is played by
[SARAH-3, from EUMETSAT's Satellite Application Facility on Climate
Monitoring](../roadmap/data-sources.md),
which cannot be reused here: it covers ±65° in both latitude and longitude, and India begins at
68.1°E — a miss of roughly 300 km, not a marginal clip.

### Questions we should ask them

We have costed this against **NGED's** requirements, because those are the only ones we know. Every
volume figure on this page inherits three assumptions we have no basis for in India: a **14-day
horizon**, a **51-member ensemble**, and **four runs per day**. If any of them is wrong, the
storage numbers move enormously — and they mostly move *down*:

| If instead they want… | A year of forecasts, on disk |
|---|---|
| The assumed 14 days, 51 members, 4 runs/day | ~18 TB |
| 13 delivery quantiles rather than 51 raw members | ~4.6 TB † |
| One run per day | ~4.5 TB |
| A 2-day horizon rather than 14 | ~2.6 TB |
| A 2-day horizon **and** one run per day | ~0.64 TB |
| A 2-day horizon **and** quantiles | ~0.7 TB † |
| A 2-day horizon, quantiles **and** one run per day | ~0.16 TB † |

† The horizon and cadence rows are simple arithmetic. The **quantile** rows assume the reduction in
stored *values* carries through to bytes on disk, which is not guaranteed: compression already
exploits much of the similarity between ensemble members, so the real saving could be smaller. It
needs measuring through the actual write path before we quote it to anyone — see
[Radical options for shrinking what we store](#radical-options-for-shrinking-what-we-store).

That is a **110× spread** — or 28× if you strike the three unmeasured quantile rows and keep only
the horizon and cadence arithmetic. Either way the "worryingly large" number is really a statement
about NGED's requirements rather than about India's. Answering these questions is worth more than
any compression work.

**About scope and phasing.** We are proposing to start with a trial of 50 to 100 substations before
scaling to the full population. **If you ask nothing else, ask these five, in this order:**

1. **Are the sites distribution transformers or substations** — the factor-of-a-thousand question
   ([detail](#how-many-sites-and-at-which-voltage)).
2. **Is the monthly rooftop PV data generation, or net export?** — the difference between having an
   anchor and not having one ([detail](#what-the-monthly-pv-totals-buy-us)).
3. **What is the forecast horizon?** The single biggest driver of storage and method.
4. **Do they want disaggregation at all**, or will they supply installed capacity as an input?
5. **Is this research or a live service?**

Everything else on this page moves less than those five do.

- **Do they actually want us to *disaggregate* PV from the substation data at all?** We have read
  the brief as requiring it, and that reading may simply be wrong. There is a much lighter
  alternative: forecast net substation demand as one product, and separately forecast PV generation
  from weather and an installed-capacity figure *they* supply. That alternative is genuinely on the
  table: the client believes it knows where most rooftop PV is installed, and Indian distribution
  companies hold per-transformer capacity records for their own interconnection purposes (see
  [India does record domestic PV capacity](#india-does-record-domestic-pv-capacity)). That raises
  the stakes on this question rather than settling it, because the register records *sanctioned*
  capacity and the client's own caveat is that not all of it is connected, working or clean.

    The gap between them is enormous, and it is almost the whole risk in the project. Forecasting
    PV from a known capacity is well-trodden work that we and many others can do. Recovering
    *unmetered* PV from net flow with no capacity register is the research bet — it is the thing
    that might not work, and it carries most of the cost, most of the schedule and essentially all
    of the uncertainty (see
    [PV disaggregation without capacity priors](#pv-disaggregation-without-capacity-priors)).

    The discriminating question is not about consistency — consistency can always be arranged by
    definition — but about direction: **where does the PV number come from, a register you give us,
    or must we recover it from the substation measurements?** If installed capacity is an *input*,
    this is ordinary PV forecasting and the project is dramatically cheaper, faster and lower-risk.
    If it is an *output*, that is the research bet, whatever anyone calls it.

    Two things follow that we should raise rather than wait to be asked. First, if they intend to
    supply a capacity figure, we need to know where it comes from and how much they trust it —
    which connects directly to
    [India does record domestic PV capacity](#india-does-record-domestic-pv-capacity). Second,
    there is a **middle path worth offering**: deliver the forecast from their capacity register,
    and run disaggregation alongside as the *second product* described in
    [What we can claim](#what-we-can-claim-and-what-we-should-not) — an independent, measured check
    on that register. Be clear what this does and does not buy: it removes the *deliverable's*
    dependence on the research, but it costs the same to build, so it is a risk decision rather than
    a saving. If they do not want disaggregation at all, we should say so plainly and price
    accordingly, rather than selling them research they did not ask for.

- **Is this a research project on historical data, or do they also want a live service?** The
  question most likely to be left implicit by both sides until it is expensive. A retrospective study on a historical export and a
  service that produces a forecast every morning are *different projects*: the second adds
  scheduled operation, monitoring, alerting, failure recovery and on-call, none of which the first
  needs at all. It also changes who we have to talk to on their side — a one-off bulk export is a
  conversation with a data team, whereas an operational feed is a conversation with whoever runs
  their control systems, and the second is usually much slower to arrange.

    It changes the *method* too, not just the engineering. Estimating installed solar capacity in
    near-real time needs weather inputs available in near-real time, and that is exactly where the
    Indian data landscape is weakest — the best irradiance products for India stop years short of
    the present (see [the sources table above](#data-sources-that-would-materially-help): SARAH-E
    ends in 2015, and the Indian Ocean coverage of the National Solar Radiation Database in 2019),
    and the reanalyses that do run near-real-time have
    [known weaknesses over India](#concerns-about-era5-over-india). For
    a purely retrospective study that constraint disappears entirely, and several options we had to
    rule out come back onto the table.

    Our recommendation, if they are undecided, is **research first with the live service as a
    defined second phase**. That matches the trial-first shape we are already proposing, it defers
    the operational cost until the method has been shown to work, and it means the live design gets
    made against measured requirements rather than guesses.

- **Which substations go in the trial?** We would want the sample chosen for *variety* rather than
  convenience — a spread of rooftop-solar penetration, urban and rural, and at least a few feeders
  where someone independently knows roughly how much solar is installed, because those are what make
  the disaggregation results checkable rather than merely plausible. The two pilot DISCOMs already
  give us a good deal of that variety for free — dense urban Delhi against urban-plus-rural Rajasthan,
  a 15% hosting cap against 50%, near-continuous supply against rostered agricultural supply (see
  [The two pilot DISCOMs](#the-two-pilot-discoms-delhi-and-jaipur)).

**About the rooftop PV metering.** Rooftop solar is separately metered in the pilot areas as monthly
totals, which is the most valuable thing we have learned and which
[What the monthly PV totals buy us](#what-the-monthly-pv-totals-buy-us) works through in full.
Four questions decide how much of that value we actually get, and all are cheap to ask: **is it
generation or net export?**; **can it be mapped from consumer to substation?**; **what are the
meter-reading dates**, given that Indian billing cycles are staggered rather than aligned to
calendar months; and **is the cycle monthly or bi-monthly**, since bi-monthly halves the number of
constraints.

**About the substation measurements themselves:**

- **What is the reporting interval?** We have assumed 15 minutes throughout, and that assumption
  drives the resolution work in Part 2, which is entry cost at any site count. It has never been
  confirmed, and Indian feeder and transformer metering commonly arrives on 15- or 30-minute blocks,
  so it is worth settling early rather than discovering late.

- **Is the metering directional (MW) or magnitude-only (MVA)?** This is the question we most wish we
  had asked NGED early. Apparent-power metering cannot see direction, so an exporting substation
  "bounces" off zero instead of going negative, and the reverse-flow periods — exactly the ones that
  reveal embedded PV — are the ones the reading destroys
  ([Data quality](../background/data-quality.md#apparent-power-mva-metering)). We handle it, and the
  [disaggregation design](../roadmap/disaggregation.md#apparent-power-mva-metering) reconstructs
  signed flow and compares its *magnitude* against the meter, but it is strictly harder than having
  the sign. We are **not confident either way for India**, and it is worth being clear why, because
  the standards question is easy to get wrong. The meter at a distribution transformer or an 11 kV
  feeder is a *transformer-operated* (CT/VT-connected) meter, which is a different device from the
  domestic smart meter: the relevant specifications are IS 14697 for conventional ones and
  IS 16444 **Part 2** for smart ones, under the Central Electricity Authority's metering
  regulations — and much of the installed base is likely to be the conventional kind. IS 16444
  Part 2 does cover meters "measuring energy in both directions", but it accommodates
  bidirectional measurement rather than requiring it, so direction is a procurement and
  configuration choice at each site rather than something the standard guarantees. Separate import
  and export registers *are* required for net-metered *consumer* connections, but that is a
  different metering point from the substation. **So the risk is twofold: the meter may not record
  export, and even where it does, the extract may discard it** — a pipeline built around billing
  can easily hand us a single net or apparent-energy figure. This is why the question is what
  fields we actually receive per site, rather than what the standards permit.
- **Does this population actually see reverse flow, and where?** Indian DISCOMs report midday
  voltage rise and reverse power flow on high-penetration feeders. We looked for a citable
  penetration threshold at which it begins and could not find a defensible one, so we should not
  quote a number: reverse flow starts, near-tautologically, when local generation exceeds coincident
  load, and where that threshold falls depends on the feeder's own daytime load shape — which we
  cannot characterise for India without their data. This matters twice: reverse flow is what makes
  magnitude-only metering lossy in the first place, and a substation that never exports gives the
  disaggregation much less to work with.
- **Are these distribution transformers, 11 kV feeders, or 33/11 kV substations?** *(Confirmed
  open. This is the live question — the choice as put to us is roughly 100,000 distribution
  transformers or a few hundred substations, and the two differ by a factor of several hundred. The
  DISCOM-scale reading, the three rungs of the Indian network, and what each answer does to the
  engineering are in [How many sites, and at which
  voltage?](#how-many-sites-and-at-which-voltage); the national arithmetic below is retained as
  background about India, not as this project's scope.)* "Secondary substation" is a European
  term and the Indian equivalent is ambiguous. It is worth resolving, because the arithmetic points
  one way: India has roughly **15.4 million distribution transformers but only ~259,000 11 kV
  feeders**, and as of March 2025 only **42% of distribution transformers were metered** (59% urban,
  39% rural) against **98% of 11 kV feeders**. Carry both through: there are about 6.5 million
  *metered* distribution transformers but only about 255,000 metered 11 kV feeders. So 100,000 sites
  is a routine ~1.5% slice of the metered transformer population, but would be **39% of every
  metered feeder in India** — which, nationally, would mean spanning essentially every distribution
  company in the country, and which for a two-DISCOM pilot simply rules the feeder reading out.
  On that arithmetic a transformer-level dataset is the far more plausible reading,
  though it is the client's answer that settles it. This matters well beyond terminology: it changes
  the number of customers behind each measurement point by an order of magnitude, and with it how
  much rooftop PV sits behind one and how much load diversity smooths it. Transformer-level also
  implies a sample **over-represented in urban areas without being mostly urban**, and the
  distinction matters. Urban transformers are metered at about 59% against 39% rural, but they are
  only ~15% of the fleet, so a metered-transformer sample still comes out roughly **four-fifths
  rural**. The reason for the gap is worth knowing before we design around agricultural load: under
  India's distribution-reform programme, transformers feeding **only agricultural consumers** are
  exempt from the smart-metering requirement, partly because those connections are earmarked for
  solarisation under the national PM-KUSUM scheme. Purely agricultural feeders are therefore
  under-represented — but with four-fifths of metered transformers rural, agricultural load stays
  very much in scope, and we should not design as though it had been filtered out.

**About the forecast itself:**

- **What is the forecast horizon?** The single biggest driver of both storage and method.
- **What decision does the forecast actually support?** The question most likely to change what we
  build, and the one everything else follows from.
- **Do they want probabilistic forecasts?** We would strongly recommend it, and it is a genuine OCF
  differentiator — but it multiplies the stored *values* by 13 if we deliver quantiles and by 51 if
  we deliver raw ensemble members (the effect on rows, and on bytes, differs from both — see the
  footnote above), so it is only worth it if someone downstream will act on the uncertainty.
- **How often must it update, and how quickly after data arrives?** Cadence multiplies storage
  directly; latency drives the deployment architecture.
- **Per substation, or aggregated?** The problem statement implies per substation, but if most users consume a
  feeder- or region-level total, that changes both the model and the delivery format.

**About delivery:**

- **How do the forecast users want the data?** For NGED we deliver bulk analytical tables in cloud
  storage, for reasons set out in [Forecast delivery](forecast-delivery.md) — which suits a single
  technical team pulling whole histories. Indian consumers might instead want an HTTP API, or a
  control system (SCADA/EMS) might want a push feed. (As the paragraph below these bullets notes,
  adding either is strictly additive.)
- **Do they want a graphical user interface?** And *who* would use it.
- **Who are the consumers, and how many?** A single utility analytics team and a hundred engineers
  spread across the DISCOMs (India's electricity distribution companies) imply different architectures.
- **Do they need the full forecast history, or only the latest run?** NGED wanting routine access to
  the entire backtest history is what drives our storage design. If India only needs recent
  forecasts, most of the volume problem disappears.

**A reassuring point to be able to make in the room:** if the answer to either of the first two
questions is "we want an API" or "we want a UI", that is **strictly additive** and costs us nothing
we have already built. As
[Forecast delivery](forecast-delivery.md#when-would-a-rest-api-earn-its-keep) sets out, an API
added later is a thin, stateless service that reads the same stored tables and serves slices of
them over HTTP. Nothing has to be re-written, and the stored tables remain the system of record
either way. The same is true of a user interface: we would not be replacing the storage layer, only
adding something that reads it.

This is not a theoretical claim. The project already ships two web dashboards that do exactly that
— they read the same stored tables directly and were added without touching the storage layer — so
"storage first, interfaces on top" is a pattern we have exercised, not one we would be trying for
the first time on someone else's project.

**About additional data sources.** The full survey is in
[Data sources that would materially help](#data-sources-that-would-materially-help); these are the
six worth actually raising in the room, in the order they are worth raising. Every one of them
makes the hard part of the project easier, and none of them is something we can obtain ourselves.

1. **The DISCOM's own per-substation rooftop PV capacity records.** Many Indian state regulators
   cap rooftop PV per distribution transformer and require the available headroom to be published,
   so a per-substation capacity record has to exist inside the distribution company — see
   [India does record domestic PV capacity](#india-does-record-domestic-pv-capacity). It leads this
   list because it is the input that could move the project off the research path entirely.
2. **Any metered PV generation, at any resolution.** This is the anchor the method wants, and the
   best route is now the one the client already holds: **the monthly per-site rooftop reads**
   described above — ask for them with their reading dates, and establish whether they are
   generation or net export. Three further routes are worth naming separately, because they are
   held by different people: the
   **PM Surya Ghar M2M real-time feed** now being built for new rooftop installations, the
   **inverter manufacturers' monitoring clouds** where per-site rooftop generation actually pools,
   and utility-scale output from **Grid Controller of India (Grid-India)** and the State Load
   Despatch Centres (SLDCs).
3. **Does anyone on their side already hold Solar Radiation Resource Assessment (SRRA) data**, the
   ground-station irradiance network run by the National Institute of Wind Energy (NIWE)? The
   processed data appears to be free to register for since a 2017 policy change, but the raw
   high-frequency series is still priced, and we have only read an archived copy of the access
   policy — so a partner who already holds it settles both the cost and the format question at
   once. See
   [What we could get hold of today](#what-we-could-get-hold-of-today-without-asking-anyone).
4. **Load-shedding schedules and agricultural feeder supply schedules.** Both turn a confounder we
   would otherwise have to infer into a known input.
5. **Network connectivity — which substations can exchange load with which.** It does not have to be
   a full network model; an adjacency list is enough. Without it we can still detect abrupt
   non-weather events, but we cannot attribute them, which is what separates a network
   reconfiguration from a shed feeder (see
   [Modelling load shedding and diesel backup](#modelling-load-shedding-and-diesel-backup)).
6. **How much history comes with the substation data?** Two years is thin for seasonal effects,
   and the ECMWF ensemble archive only reaches back to 2024-04-01 regardless.

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
  2024-04-01 (although Dynamical are back-filling now), which is thin history for a 100,000-site
  training set, and its radiation is global short-wave only with no direct component — a bigger
  problem for PV disaggregation under heavy aerosol load than it is for Britain.
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
| `"Europe/London"` as a bare string literal in the feature engineer | `ml_core/features/tabular_feature_engineer.py`, in `_apply_local_time_features` | Drives every local-time feature in the champion feature set. |
| `DISPLAY_TIME_ZONE = "Europe/London"`, asserted in the dashboard's axis titles | `dashboard/forecast_chart.py:40` | Display only, but it is a second hard-coded timezone. |
| H3 resolution 5 (~253 km² per cell) chosen for GB, and reached for via a **private** import from the ingest package | `defs/assets.py:40,141` | The NWP grid resolution currently lives inside `nged_data`; see the `PowerIngest` note [below](#how-we-would-structure-it). |
| `nged_s3_bucket_url` / `_access_key` / `_secret` are **required** settings with no defaults | `contracts/settings.py` | `Settings()` raises for any deployment with no NGED bucket. |

The UTC-offset feature is **not** one of these British assumptions.
`local_utc_offset_minutes` holds the local offset from UTC in minutes, in an `Int16`, so every
offset in scope is represented exactly and sub-hour zones stay distinct: India's +5:30 is `330` and
Nepal's +5:45 is `345`, and Australia's +9:30 (`570`) does not collide with +9:00 (`540`). A
mixed-offset deployment is representable, and the column name states its own units.

The one thing an adapted deployment would need to revisit is the era bound that
[issue #466](https://github.com/openclimatefix/nged-substation-forecast/issues/466) puts on
`PowerTimeSeries.time`, which is what keeps the offset a whole number of minutes. A handful of IANA
offsets are not. Each zone leaves mean solar time at its own date, so the era that needs excluding
differs by geography: `Europe/London` runs on UTC−0:01:15 until 1847, whereas `Asia/Kolkata` runs
on mean-time offsets of +5:53:20 and +5:21:10 until 1906. Nor is it only the deep past — Liberia
kept UTC−0:44:30 as legal time until 1972.

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

**Read this section conditionally.** Almost all of it is triggered by site count, and the site count
is the pilot's biggest open question: roughly 100,000 distribution transformers, or a few *hundred*
substations ([How many sites, and at which voltage?](#how-many-sites-and-at-which-voltage)). **On
the "few hundred" answer, everything down to the NWP paragraph falls away** — a few hundred series
is smaller than NGED's own V2 design point, and today's per-series XGBoost handles it unchanged. The
NWP paragraph is the exception, because it scales with the *area* we download weather for rather
than with site count, though a two-DISCOM pilot would need weather over Delhi and Rajasthan rather
than over all India. What follows otherwise assumes the demanding answer.

Two different multipliers matter here, and it is worth keeping them apart. On **series count** —
the axis that governs how many models we train — 100,000 sites is **40×** the V2 design point of
~2,500, which is itself ~78× V1's 32. On **forecast-row volume**, the 15-minute sampling doubles it
again, so the storage and query pressure is around **80×**. Three things break.

**None of it breaks in a trial, though, and that is the important scheduling fact.** A first phase
of 50–100 substations — which is what we are proposing, mirroring NGED's own 32-site V1
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
with no time or series axis. At 100,000 sites a single full-ensemble run is roughly 100,000
series × 51 members × 14 days × 96 steps/day ≈ **6.9 billion rows per run**. At the current
6-hourly cadence (4 runs/day) that is of order 10 trillion rows per year.

Be careful which number that translates into, because the two differ by more than two orders of
magnitude:

| A year of forecasts at 100,000 sites | Volume |
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
[Performance and Scale](performance.md#the-other-hard-ceiling-polars-32-bit-row-index), row counts
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
[input-pruning strategy](performance.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output)
exists to bound separately. The bigger loss is that the
`h3_index` pruning described in
[Performance and Scale](performance.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output)
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
"[near-continuous ML output has no repeats for a dictionary to exploit](performance.md#storage-formats-measured-not-assumed)"
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
neither. Note the mechanism carefully, because it is the opposite of the `power_forecasts` case:
NWP *already* compresses well through Parquet's default dictionary and run-length encoding, because
significand rounding collapses many H3 cells and ensemble members onto the same value — that is
precisely why `delta_store.nwp` does **not** use `BYTE_STREAM_SPLIT`, which measured *larger* on
this table. Anomalies would narrow the value range further and so should deepen a win we are
already getting, rather than unlocking a new one. The headroom is correspondingly smaller than for
`power_forecasts`, and entirely unmeasured. There is a pleasing efficiency here, though — the
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

- **No *half-hourly* metered PV inside the dataset.** The NGED plan uses verified metered generators
  ([Capacity estimation](../roadmap/capacity-estimation.md)) to anchor the harder unmetered
  inference. India's rooftop PV is separately
  metered as **monthly totals**, which is a genuine anchor on *energy* but gives nothing on shape
  ([What the monthly PV totals buy us](#what-the-monthly-pv-totals-buy-us)). Consuming a
  monthly constraint is new work either way — it means adding an integral constraint over the fitted
  generation, which the differentiable-physics route takes naturally and the convex baseline takes
  as an extra linear equality. India also *has* utility-scale metered solar with published output,
  so a second external anchor may be obtainable.
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
    existing monotone installed capacity multiplied by a *reversible* soiling ratio, composed with
    `UniversalSolarFleetNode` rather than replacing it, so the monotone prior stays intact. The
    parameterisation, the reason it needs no new input, and the identifiability question that is its
    real cost are all set out in
    [Differentiable physics → Soiling](../techniques/differentiable-physics.md#soiling), which is
    its durable home.

- **Aerosol and monsoon bias in the irradiance itself.** The Indo-Gangetic Plain carries among the
  world's highest aerosol optical depth, which systematically biases satellite- and NWP-derived
  irradiance, and monsoon convection is poorly resolved at 0.25°. Because installed capacity is
  inferred *from* irradiance, a systematic irradiance bias becomes a systematic capacity bias.
  Worse, the high-resolution irradiance source the GB plan depends on — SARAH-3, from EUMETSAT's
  Satellite Application Facility on Climate Monitoring (CM SAF), see
  [Data sources](../roadmap/data-sources.md) — covers ±65° in both latitude and longitude (per CM
  SAF's own product record, not the internal page linked above, which does not state the bounds),
  and India begins at 68.1°E, so it misses the country entirely by roughly 300 km. A replacement has to be sourced; the candidates are in
  [Data sources that would materially help](#data-sources-that-would-materially-help) in Part 1.
- **Confounders with no British analogue.** Load shedding and diesel gensets both violate the
  assumption that latent demand is smooth and weather-driven. Load shedding is the dangerous one:
  it resembles a demand collapse uncorrelated with weather, and an unguarded optimiser would
  explain it with phantom solar. Explicit regime detection would need to be budgeted, not bolted
  on — though much less of it is new than it first appears, because the machinery overlaps heavily
  with the switching-event detector already designed for NGED. See
  [Modelling load shedding and diesel backup](#modelling-load-shedding-and-diesel-backup) below.
  Unmetered agricultural pumping is the happier case — Indian agricultural feeders are largely
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
- **Directional metering is *possibly* more common than in NGED's trial area**, where NGED reports
  10 sites as non-directional — but we could not establish this, and should not assume it.
  Transformer-operated meters at feeders and distribution transformers (IS 14697, or IS 16444
  Part 2 for smart ones) can measure in both directions, but are not obliged to, so it is a
  per-site procurement question. What matters for us is also the *delivery* format: Indian feeder
  and transformer metering typically arrives as an interval load survey read from the meter's
  registers rather than as instantaneous telemetry, and whether that survey carries import and
  export separately is exactly the thing to ask. If signed flow
  survives the extract (a question, not an assumption — see
  [Questions we should ask them](#questions-we-should-ask-them)), the
  [MVA-bounce reconstruction](../roadmap/disaggregation.md#apparent-power-mva-metering) is a
  fallback we would not need. `TimeSeriesMetadata.units` is already a per-series
  `pl.Enum(["MW", "MVA"])`, so a mixed population needs no schema change either way.

#### Modelling load shedding and diesel backup

Sketched here because "explicit regime detection would need to be budgeted" is not a plan, and
because on inspection most of this is **reuse of the switching-event design rather than new
machinery**. Everything below is a proposal, not a tested method.

**Scope note: this section is about Rajasthan, not Delhi.** Load shedding is a JVVNL concern —
rural Rajasthan has rostered agricultural supply and curtailment — whereas Delhi runs to a 24×7
supply norm and its DISCOMs report peak demand met without cuts (see
[The two pilot DISCOMs](#the-two-pilot-discoms-delhi-and-jaipur)). If the pilot starts in Delhi,
none of what follows is entry cost. Read it as the cost of the Rajasthan half, and as the reason a
method proven on both is worth more than one proven on either.

**Load shedding is structurally the switching-event problem, in its easy regime.** Both produce a
*sustained level shift in metered power with no meteorological cause*, which is exactly what
[Stage 1 of the staged statistical
detector](../roadmap/switching-events.md#stage-1-changepoint-detection-on-the-baseline-residual)
is designed to find — changepoint detection on a baseline residual that has first been
**normalised** (per-substation,
per-time-of-day, so one threshold works fleet-wide) and **whitened** (a low-order autoregressive
fit, so slow NWP-error waves are not read as steps). That preparation is the hard-won part of the
design and it transfers unchanged. What differs is the operating point, and it differs favourably:
the [background on switching events](../background/switching-events.md) notes that the common,
difficult case is a *partial* transfer whose magnitude "shade[s] continuously down into the
measurement noise", so detection difficulty scales inversely with how much load moved. A shed feeder
is the opposite extreme — a near-total collapse toward
zero — which is the largest signal the detector will ever be asked to find.

**The neighbourhood-sum test separates shedding from reconfiguration — but it is not free, and it
has a data dependency worth naming.**
Stage 2's conservation fingerprint is that over a candidate set of {source + donor} substations the
*summed* residual should show no step, because a transfer moves load rather than destroying it.
Load shedding has the opposite signature: the load leaves the neighbourhood entirely, so the
summed residual steps down by the full amount. The same statistic, computed the same way,
separates a transfer (flat sum) from a genuine loss (stepped sum) without needing a second
detector. India presumably has both phenomena, so this matters. Do not overclaim it, though: a
**regional weather-model error** also steps every nearby series *and* their sum, which is the same
signature — over the Indian monsoon, exactly the failure mode to worry about. Telling a shed feeder
from an NWP bust needs the night-time and clock-alignment evidence below, not the sum test alone.

The dependency is **network topology**. Stage 2 scopes its subset search using a fixed lookup of
which substations can exchange load with which; without that adjacency the search runs over all
100,000 sites and is dead both computationally and statistically. We do not currently ask the
client for it anywhere, and we should — it is now in
[Questions we should ask them](#questions-we-should-ask-them). Note the asymmetry: detecting a
*shed* event needs only stage 1 and no topology at all, so the absence of a connectivity model
degrades this from "we can tell shedding from switching" to "we can detect events but not always
attribute them", rather than blocking the work.

**Learning the rotation schedule from the fleet, if we are not given it.** Indian load shedding is
typically *rotational*: groups of feeders are shed together, at repeating clock times, cycling
across groups. That is a strong and unusual signature — synchronous, calendar-aligned, and
group-structured — and at 100,000 substations it is recoverable by clustering substations on the
co-occurrence of their detected gate events. Cloud cover also correlates events spatially, but with
quite different structure: cloud is smooth, moves, and never aligns to the same clock time each
day. Recovering the groups would let us treat shedding as a *predictable* input for the bulk of
sites rather than a per-site latent nuisance, which is a much better place to be.

**How it enters the disaggregation model: a multiplicative gate, not an additive offset.** Shedding
does not subtract a load, it disconnects one, so the right form is

$$y_t = g_t \cdot \left(d_t - p_t\right) + \varepsilon_t$$

with $d_t$ latent demand, $p_t$ PV generation, and $g_t \in [0, 1]$ a **persistent, near-binary**
gate. The constraint doing the work is the word "persistent": a free per-timestep gate can absorb
any residual whatsoever and would demolish identifiability of everything else in the model. It must
be regularised into being step-shaped and rare — a sticky two-state transition prior (a relaxed
Markov switch, or a sigmoid over a latent driven by a total-variation penalty) so that turning the
gate on and off is expensive and holding it is cheap. If the schedules are obtainable, $g_t$ stops
being latent altogether and becomes a known regressor; this is the single largest reason
[the schedules question](#questions-we-should-ask-them) is worth asking.

**Night-time outages are free labelled data.** PV is zero at night, so any night-time shed episode
identifies the gate's depth, duration and start-time distributions with **no solar confound at
all**. Fit the gate's priors on night data, then carry them into daytime inference where the two
signals genuinely do compete. This is cheap, and it is the single most useful structural fact about
the problem.

**The guard against phantom solar is the clear-sky envelope, and the physics gives it to us.** The
feared failure is the optimiser explaining a shed interval with invented generation. The
differentiable-physics model cannot do this, because fitted PV is a *physical* function of solar
geometry and irradiance: it is identically zero at night, and bounded above by the clear-sky
maximum during the day. So a midnight collapse is inexplicable by PV under any parameter setting,
and a daytime one can be absorbed only up to a ceiling that scales with plausible installed
capacity. This is a genuine advantage of the physics route over a purely statistical
decomposition, and it is worth stating in the bid.

**Cold-load pickup should be modelled, not ignored.** Restoration is not a clean return to the
previous level: thermostatic and deferred load restarts together and overshoots, decaying back over
tens of minutes. One learnable time constant and one amplitude, applied on each gate release,
covers it. Leaving it out is worse than it sounds — a daily shed window produces a *repeating*
overshoot at the same clock time each day, which is precisely the shape a diurnal component will
happily absorb.

**One statistical warning: shedding is not missing-at-random.** Feeders are shed when the system is
stressed, which is to say during the hottest, highest-demand periods. So the cheap treatment —
detect shed intervals and drop them from the likelihood — is *safe for the PV parameters* but
**biased for the demand model**, because it systematically deletes the conditions the demand
forecast most needs to get right. The workable split is to mask for disaggregation and to model the
gate explicitly for forecasting.

**Diesel backup is a smaller problem than it first appears, for a structural reason.** A generator
serving premises on a shed feeder supplies that load *behind* the point of disconnection, so the
substation sees nothing of it — diesel mostly removes signal rather than adding a confusing one. It
bites in two narrower places. First, **staggered restoration**: load returns lower than expected
because some premises stay on generator for a while, which distorts the cold-load pickup shape
above rather than creating a separate phenomenon. Second, **peak-shaving gensets** running to a
tariff schedule, which appear as a rectangular demand notch with no weather cause.

Separating diesel from PV is easy on two independent grounds, and only one narrow case is
genuinely awkward. A genset produces a flat-topped block whose edges align to tariff or outage
boundaries, not to solar geometry, so it fails the shape test; and it fails the clear-sky-envelope
test above for the same reason PV cannot explain a midnight step. The awkward case is a peak-tariff
window that happens to sit across the solar peak — plausible in India, where the evening peak is
the binding one but afternoon industrial tariffs exist. There the discriminator is **day-to-day
covariation with irradiance**: PV varies with cloud from one day to the next, and diesel does not,
so the fitted component can be regressed against satellite irradiance and the invariant part
rejected. That test needs 100,000 sites and a decent irradiance record to have any power, which we
would have.

**What this costs, honestly.** The stage-1 *design* is shared with NGED, but whether NGED ever
builds it is [an open
decision](../roadmap/switching-events.md#the-decision-point-a-feature-based-mainline-vs-the-staged-detector):
the roadmap demotes the staged detector to a contingency behind a feature-based mainline, to be
justified by a measured gap rather than by anticipation. If that mainline wins for NGED, India pays
for the detector itself, and this is a materially larger number. Assuming the detector exists, the
gate, cold-load pickup, and the irradiance-covariation test are perhaps three to four weeks on top
of the disaggregator; assuming it does not, add the detector's own build. As with soiling, the real
cost is **identifiability testing on synthetic
data** — demonstrating that a sticky gate and a solar component do not trade off against each other
at realistic noise levels — and that should be done before either is trusted on real Indian data.

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
| Region seam, 15-minute support, Indian ingest | **Not estimated** | Trial | Seam yes; ingest no |
| Convex dictionary disaggregator | **Not estimated** | Trial † | **Yes — it is the V2 baseline** |
| Global model, replacing per-series XGBoost | 6–10 weeks | Rollout | **Yes — needed for V2 regardless** |
| Storage partitioning and metrics chunking at 80× | 6–8 weeks | Rollout | Mostly |
| Full differentiable-physics PV engine | 6–12 months | Either | **Yes** |

**The two Trial rows deliberately carry no week count.** Neither was ever estimated bottom-up, and
both depend on answers we do not have: the 15-minute feature-grammar work and the Indian ingest turn
on the metering format, the cadence and the shape of the extract, none of which are settled. We
would rather size them once the scoping questions are answered than defend a guess. The three
remaining rows are the same order of confidence — indicative, not costed.

The phase column is still the reason a 12-month project is plausible at all: only the first two rows
are entry cost, and they are the two that answer the question the bid actually turns on.

**Compare this table against the money before quoting any of it.** ENTICE 3.0 awards up to
USD 100,000, possibly split across winners. The "Shared with NGED V2?" column
is what makes the arithmetic work — three of the five workstreams are NGED work regardless, so the
India-only cost is the region seam, 15-minute support and Indian ingest. That, plus the fact that
the "few hundred substations" answer deletes the two Rollout rows entirely, is the shape that fits
the award. See [What the award actually buys](#what-the-award-actually-buys).

† The disaggregator is entry cost **only if they want disaggregation**. If they intend to supply an
installed-capacity register and want PV forecast from it, that row leaves the critical path
entirely and the trial shrinks to the first row — see
[the scope question](#questions-we-should-ask-them).

### Why we are not doing any of this now

Speculative generality is not free. A `RegionProfile` seam introduced today is a layer of
indirection that every NGED contributor pays for, on every change, in service of a project we may
not win. The correct move is to leave the British assumptions hard-coded and *legible* — this page
is a large part of what makes them legible — and to pay the refactoring cost only once there is a
second consumer to amortise it against.

The one exception is the private `_H3_RESOLUTION` import, described above. It is not really a
portability concern — it is an ordinary code-quality item that happened to surface here — so it can
be fixed on its own merits whenever convenient, independently of anything on this page.

**What would change our mind:**

- **Winning ENTICE 3.0.** The obvious trigger — though note that a win at ENTICE's funding level
  would justify the *region seam and 15-minute support*, not the full portability refactor. Only a
  follow-on engagement at the scale of GEAPP's *India Grids of the Future* accelerator would justify
  all of it.
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
