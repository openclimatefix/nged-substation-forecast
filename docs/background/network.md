# NGED's network and its data

**This page is the physical context the rest of the docs assume: what NGED operates, what generation
is connected to it, and the ways the trial-area telemetry misreports what it measures.** NGED is a
distribution network operator (DNO) in Great Britain, and its network (as of May 2026) consists of:

- 1,161 primary substations (33/11 kV and 66/11 kV)
- 271 bulk supply points / BSPs (132/33 kV and 132/66 kV)
- 52 grid supply points / GSPs (400/132 kV and 275/132 kV)
- ~1,500 industrial customer generators (not domestic); roughly 558 at 33 kV or 132 kV connected to
  GSP/BSP busbars, and ~1,000 on the 11 kV network downstream of primaries

![NGED's network](assets/NGED_network.png)

## Embedded generation on the network

**NGED's Embedded Capacity Register records what generation is connected.** The
[register](https://connecteddata.nationalgrid.co.uk/dataset/embedded-capacity-register) (August
2026) lists **5,958 MW of connected solar** and **1,456 MW of connected wind**. Hydro is a much
      smaller presence: **41 connected hydro sites totalling 25.7 MW** across all four licence areas
      — South Wales 13.7 MW over 10 sites, South West 6.2 MW over 18 sites, East Midlands 5.3 MW
      over 8 sites, West Midlands 0.5 MW over 5 sites — under half a percent of the connected solar
      capacity.

**The hydro fleet is overwhelmingly small run-of-river with no storage.** 39 of the 44 hydro entries
are `Hydro - Run of river`, and 29 of the 41 connected sites join at 0.4 kV, so output tracks
catchment flow almost directly. The largest connected schemes are Llyn Brianne (5.45 MW, Dyfed),
Elan Valley (4.0 MW, Powys), Chatsworth (3.7 MW, Derbyshire), Mary Tavy (2.6 MW, Devon), and
Ystradffin (1.99 MW, Dyfed). One entry is much larger — a 58.5 MW Cwm Rheidol scheme accepted to
connect in the South Wales area — but its target energisation date is 2037. The 41 connected sites
are spread across **32 distinct primary substations**, so no primary is hydro-dominated.

## Active network management caps what a generator may export

**A generator connected under active network management (ANM) may export only up to a cap the
network operator moves in real time, so its output is the smaller of what the weather allowed and
what the cap permitted.** A flexible connection of this kind is what lets a generator join a
network with no spare firm capacity: instead of waiting years for reinforcement, it accepts being
turned down when the local network fills. For anything that predicts generation from weather, the
capped hours are unpredictable by construction — no irradiance or wind field carries the state of
the network.

**NGED holds two records of it, and they are different quantities.** The
[`curtailment/` prefix](../roadmap/data-sources.md#what-the-curtailment-feed-holds) on the same S3
bucket as the telemetry publishes a derived half-hourly volume of megawatts lost. Separately, NGED
can export the raw setpoint history on request: one row each time a generator's export cap changed,
in megawatts, which is the signal the derived volume is computed from. The cap is published as a
negative number, because generation is negative in NGED's sign convention, and the largest
magnitude a generator's cap ever reaches is its connection limit rather than a curtailment. Reading
that limit as a curtailment volume inverts the signal.

**On the one metered generator whose setpoint history we hold, the cap is at the connection limit
80.5% of the time.** It sits at zero for 12.7% of the record, at 0.25 MW for 4.0%, and somewhere in
between for the remaining 2.8%, over 26 months. The zero periods run long — one span lasts 38 days
— and look like outage or works rather than the minute-by-minute trimming the rest of the record
shows. Where the cap never left the connection limit, that generator's output per unit of
irradiance matches the other five photovoltaic sites in the trial area to within half a percent,
which is the check that says the cap is being read the right way round. The
[beam/diffuse results](../results/beam-diffuse-split.md#one-site-is-curtailed-and-it-is-the-noisiest-of-the-six)
set out the measurement.

**Curtailment is not a capacity loss, which is why the two records matter beyond forecasting.** A
turned-down generator is still physically capable of its full output, so
[effective-capacity estimation](../roadmap/capacity-estimation.md) has to hold curtailment out
rather than absorb it. Neither record reaches back to the start of the telemetry, and only one
generator in the trial area has either, so an estimator cannot assume a label exists.

## Data quality in the trial area

**As is typical of distribution networks, NGED's distribution-level telemetry carries more gaps and
measurement artefacts than transmission-level telemetry.** The sections below are the issues
observed in the trial area, each a distinct phenomenon rather than one fault seen several ways. See
the ["Data sources" section of our Milestone 1
report](https://docs.google.com/document/d/1UF-mjfSdQfQxefAunDqEOr_GyYTjSlGk4EeuiNoXAxk/edit?tab=t.0#heading=h.etqoj9ahy92h)
for a more detailed discussion, and plenty of graphs.

### NGED Data Availability

Availability of the data for the 32 time series in the trial area:

![Availability of the data for the 32 time series in the trial
area](assets/NGED_data_availability_periods.png)

### Early ramp-up period

The first couple of months after a meter is installed tend to have poor data quality. Poor data
quality in that period is handled by simply dropping the first 2 months of each time series. ![Bad
data for the first few months for 3 substations](assets/bad_data_for_first_months.png)

### False zeros

Substation time series occasionally report zero when the true value is non-zero. These are
identifiable because they are isolated amongst non-zero values. ![Example plot of a BSP time series
in the trial area having some one-off falls to zero](assets/false_zeros_in_Boston_BSP.png).
![Distributions for a BSP and a primary in the trial area showing an excess of zeros, even though
the primary otherwise has no near-zero values](assets/histograms_showing_false_zeros.png)

### Stuck values

Some time series go "stuck" for hours or days (standard deviation near zero over a 24-hour window).

### Missing data

Gaps range from a few half-hours to months. Solar farms frequently have no data overnight
(expected), but also have unexplained daytime gaps. ![Examples of missing data from two solar sites
in the trial area, time series 23 and 29](assets/missing_data.png)

*Examples of missing data from two solar sites in the trial area, time series 23 and 29. Time series
29 has a known analogue metering issue.*

### Apparent power (MVA) metering

Some substations only have MVA meters, which report the *absolute value* of power flow — they cannot
detect direction. When generation exceeds demand and power flows "backwards", the MVA reading
increases rather than going negative. This "bouncing off zero" behaviour looks like a demand
increase but is actually reverse power flow. In the trial area, 10 sites are metered in apparent
power. Of those 10 sites, one site, and possibly two more, have shown reverse flow on sunny days.
The following figure shows power flow for Stickney primary (time series 14) and a nearby solar site
(time series 22); note the absence of peaks at Stickney primary on 3 and 4 May, when the solar site
generated less: ![Power flow for Stickney primary (time series 14) and a nearby solar site (time
series 22)](assets/MVA_metering_bounce_at_Stickney_primary.png)

### Switching events

Power is periodically diverted from one substation to another during maintenance or in response to
faults ("abnormal running arrangement"). Each substation spends roughly 10% of its operating time in
an abnormal arrangement. Switching events severely bias lagged-power features (the single most
informative feature for demand forecasting) if not detected and handled. Recovering the demand that
*would* have been metered under the normal running arrangement is described in [Switching
Events](switching-events.md); the staged solution plan is in the
[roadmap](../roadmap/switching-events.md) (v0.6 detector → v2 mixture models).

## Behavioural calendar effects on demand

**GB electricity demand depends on human behaviour as well as on weather, and a plain day-of-year
feature cannot represent the calendar's sharp cases.** Easter wanders across roughly 5 weeks of the
calendar, from late March to late April, so its behavioural signature smears into "normal spring"
unless an explicit holiday feature marks it. The bridge days around bank holidays and the
Christmas–New Year "run of Sundays" are milder versions of the same problem, and school half-terms
vary by county across NGED's licence areas. Demand on a bank holiday already looks like a Sunday,
and the Christmas–New Year fortnight is its own regime.

**Major broadcast events shift and synchronise demand too.** England playing in the later stages of
a Football World Cup shifts and synchronises evening demand across the country, including the
classic half-time TV-pickup surge. Unlike bank holidays, sporting fixtures of this kind are not
knowable years ahead.
