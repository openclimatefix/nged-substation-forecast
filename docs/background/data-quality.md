# Data Quality Challenges

NGED's distribution-level data is considerably messier than transmission-level data. Key issues observed in the trial area:

## NGED Data Availability

Availability of the data for the 32 time series in the trial area:

![Availability of the data for the 32 time series in the trial area](assets/NGED_data_availability_periods.png)

## Early ramp-up period

The first couple of months after a meter is installed tend to have poor data quality. Poor data quality in that period is handled by simply dropping the first 2 months of each time series. ![Bad data for the first few months for 3 substations](assets/bad_data_for_first_months.png)

## False zeros

Substation time series occasionally report zero when the true value is non-zero. These are identifiable because they are isolated amongst non-zero values. ![Example plot of a time series (Boston BSP) having some one-off falls to zero](assets/false_zeros_in_Boston_BSP.png). ![Distributions for Boston BSP and Sleaford primary showing abnormal amount of zeros, especially when Sleaford doesn’t experience any near-zero values](assets/histograms_showing_false_zeros.png)

## Stuck values

Some time series go "stuck" for hours or days (standard deviation near zero over a 24-hour window).

## Missing data

Gaps range from a few half-hours to months. Solar farms frequently have no data overnight (expected), but also have unexplained daytime gaps. ![Examples of missing data in streams from solar farms: Manor Farm and Canopus Solar Farm. Canopus has a known analogue issue.](assets/missing_data.png)

## Apparent power (MVA) metering

Some substations only have MVA meters, which report the *absolute value* of power flow — they cannot detect direction. When generation exceeds demand and power flows "backwards", the MVA reading increases rather than going negative. This "bouncing off zero" behaviour looks like a demand increase but is actually reverse power flow. NGED report that 10 sites in the trial area are non-directional, of which one (and potentially two others) has shown reverse power flow in the past, from studying sunny days. The following figure shows power flow for Stickney primary and Leverton Solar Park; note the absence of peaks at Stickney primary on May 3rd and 4th when Leverton experienced lower generation: ![Power flow for Stickney primary and Leverton Solar Park; note the absence of peaks at Stickney on May 3rd and 4th, when Leverton experienced lower generation](assets/MVA_metering_bounce_at_Stickney_primary.png)

## Behavioural calendar effects on demand

**GB electricity demand depends on human behaviour as well as on weather, and a plain
day-of-year feature cannot represent the calendar's sharp cases.** Easter wanders across
roughly 5 weeks of the calendar, from late March to late April, so its behavioural signature
smears into "normal spring" unless an explicit holiday feature marks it. The bridge days around
bank holidays and the Christmas–New Year "run of Sundays" are milder versions of the same
problem, and school half-terms vary by county across NGED's licence areas. Demand on a bank
holiday already looks like a Sunday, and the Christmas–New Year fortnight is its own regime.

**Major broadcast events shift and synchronise demand too.** England playing in the later
stages of a Football World Cup shifts and synchronises evening demand across the country,
including the classic half-time TV-pickup surge. Unlike bank holidays, sporting fixtures of
this kind are not knowable years ahead.

## Switching events

Power is periodically diverted from one substation to another during maintenance or in response to faults ("abnormal running arrangement"). Each substation spends roughly 10% of its operating time in an abnormal arrangement. Switching events severely bias lagged-power features (the single most informative feature for demand forecasting) if not detected and handled. Recovering the demand that *would* have been metered under the normal running arrangement is described in [Switching Events](switching-events.md); the staged solution plan is in the [roadmap](../roadmap/switching-events.md) (v0.6 detector → v2 mixture models).

See the ["Data sources" section of our Milestone 1 report](https://docs.google.com/document/d/1UF-mjfSdQfQxefAunDqEOr_GyYTjSlGk4EeuiNoXAxk/edit?tab=t.0#heading=h.etqoj9ahy92h) for a more detailed discussion, and plenty of graphs!
