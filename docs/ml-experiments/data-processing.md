# Data processing to apply in future experiments and production

**The experiments so far show that the power readings, the switching log, and the embedded-solar
capacity estimates all need processing before a model trains or scores on them.** This page lists
each step as an instruction. Every entry says where it applies (experiments, production, or both),
links the experiments that are the evidence, and states the scope the evidence covers. Where no
experiment has finished review yet, the step is in the [Provisional
entries](#provisional-entries-awaiting-review) section at the end, without links. A plan for an
experiment begins by reading this page. The `D` codes are the anonymised substation labels used in
the experiment write-ups, as in [experiment
51](00051-pvdis-3b-midday-loss-of-the-solar-term/result.md).

## Series and units

**Count 32 time series in version 1 of the trial area.** They are 16 Primary substations, 6 solar
farms, 3 wind farms, 2 grid supply points, 2 bulk supply points, and 3 other generators or storage
units, as the [home page](../index.md) lists. Applies to experiments and production.

**Read each series' unit from the `units` column of the metadata table, and treat the Primaries
metered in MVA as a different kind of series.** A series is metered in MW (signed active power) or
in MVA (apparent power, which is never negative). Eight of the 16 trial Primaries are metered only
in MVA. Applies to experiments and production. Evidence: [experiment
79](00079-pvdis-7-mw-at-mva-only-primaries/result.md) found no reverse flow at 7 of the 8 MVA-only
Primaries from April to September, and [experiment
56](00056-mva-only-primaries-skill-against-mw-primaries/result.md) found the best guess's skill at
MVA-only Primaries neither clearly worse nor clearly equal to the skill at MW Primaries
(inconclusive, 8 series against 8).

**Read power through the cleaned table, and read the raw table only when the question is about
availability.** The cleaned table holds only unflagged rows
(`nged_data.storage.scan_cleaned_power`). Applies to experiments and production. Evidence: the
[data-cleaning roadmap page](../roadmap/data-cleaning.md) and [experiment
6](00006-stage1a-demand-input-feature-sweep/result.md), which scores on cleaned power.

## Embedded solar capacity

**Hold embedded-solar capacity as one value per Primary substation per season-year, and do not
estimate it weekly.** In [experiment 45](00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md), no
pair of penalties on the grid was both quiet on noise and responsive to a real fall. The frozen
weekly fit had a path error of 0.040 against 0.027 for the season-year constant on the
semi-synthetic baseline, and gave 0.66 false steps per Primary-year against a limit of 0.5.
Refreshing the table once per season-year captured almost all of the gain from a current capacity in
[experiment 78](00078-pvdis-5-current-capacity-vintages/result.md) (0.018 points, 98.75% interval
-0.036 to -0.004). Applies to experiments and production. Scope: 8 Primaries metered in MW.

**Feed the forecast a capacity built only from data before the run, keyed by the run's date.** A
capacity fitted on all the data leaks the future. The latest week of a real-time capacity had 3.63
times the path error of the same week in the final fit ([experiment
45](00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md)), so do not read the newest week as a
measurement. Applies to experiments and production.

**Mask the weeks with no usable observation, and never read them as zero capacity.** One Primary
(Mount Bridge Boston) has 198 unobserved weeks of 354, and its season-year constant falls to about 0
MW from 2023-09-04 to 2025-08-25, all in unobserved weeks ([experiment
45](00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md)). Applies to any capacity table.

**Exclude the MVA-only Primaries from capacity fits.** The MVA reading folds the solar output into
positive demand, and [experiment 39](00039-pv-disaggregation-cams-injection/result.md) fitted a
negative capacity at Stickney in every year. A fit that unfolds the sign recovered capacity with a
median error of 0.09 against 0.29 for the fit that ignores the sign, but found direction with a
balanced accuracy of 0.78 (0.5 is a guess), and the 75% to 87% reverse flow at Stickney from 2021
cannot be checked without signed power ([experiment
79](00079-pvdis-7-mw-at-mva-only-primaries/result.md)). Applies to experiments. Confidence: medium.

**Drive the solar shape with CAMS irradiance, and report capacity as the 99th percentile of the
half-hourly AC output.** The Copernicus Atmosphere Monitoring Service (CAMS) irradiance recovered an
injected 99th-percentile capacity in [experiment
39](00039-pv-disaggregation-cams-injection/result.md), and [experiment
45](00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md) defines capacity the same way. Applies to
experiments and capacity tables.

**Treat Wrangle T1 and Wrangle T2 as one area for any outside solar count, and use the Primary's
polygon and not its point.** The two share one polygon, and one Primary's point lies 95 m outside
its own polygon ([experiment 45](00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md)). Applies to
experiments and production.

**Flag the Primaries whose capacity is confounded, and carry the error bound of each unflagged
one.** Kirton is flagged in 6 of 7 season-years, and a day-long cloud-driven load is detected in
only 0.35 of simulated trials, so the bound on the capacity at the seven unflagged Primaries is 0.18
to 0.71 and not 0.10 ([experiment 40](00040-pv-capacity-error-bars/result.md)). In [experiment
39](00039-pv-disaggregation-cams-injection/result.md), 6 of the 8 Primaries passed all four
real-data checks, and the placebo-sky check failed at Wrangle T1. Applies to experiments.

**Do not fit the panel orientation mix or the DC to AC ratio from a Primary's power, and use a fixed
menu instead.** The orientation mix is not identified on injected mixes ([experiment
73](00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md)). A fitted east-south-west shape
lowered the held-out fit error by 1.3% in experiment 73, but gave no forecast gain in [experiment
77](00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md), which keeps the all-south shape. Applies
to experiments.

**Shrink the forecast solar anomaly, and do not rely on it beyond about 2 days.** The residual of
the ensemble-based median has a slope of -0.42 to -0.69 on the oracle shift in every month and hour
bin ([experiment 51](00051-pvdis-3b-midday-loss-of-the-solar-term/result.md)), and -0.771 overall in
[experiment 46](00046-embedded-solar-share-of-primary-forecast-error/result.md), so a coefficient of
1 is too large. The ensemble forecast of the solar anomaly carries skill only to about 2 days ahead
(experiment 46). Shifting the ensemble mean scored 0.129 points (0.057 to 0.212) better than
shifting each member (experiment 51). Applies to experiments, in summer daylight hours.

**Remove the logged switching, padded by 24 hours, from every capacity fit.** A transfer moves the
donor substation's solar output and looks like a capacity step ([experiment
45](00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md) applies this mask). Applies to
experiments. Confidence: medium, because the write-up states the rule and does not test it.

## Fixed windows

**Use the fixed COVID window, 2020-03-23 to 2021-07-19, and do not detect it from the data.**
[Experiment 6](00006-stage1a-demand-input-feature-sweep/result.md) defines the indicator with these
dates, and the indicator lowered the error by 0.26 points on the selection fold. Applies to
experiments and production.

## Drift

**Do not correct drift with one level shift, because drift differs by series and by hour band.** In
12 of 13 series some band moves by at least 1 point of capacity a year. From April to September the
pooled slope is -1.30 points a year at midday (95% interval -2.32 to -0.92), -0.88 in the evening,
and +0.69 at night, and the all-hours slope is +0.04 because series drift in both directions
([experiment 92](00092-lv0-drift-diagnostics-on-stored-predictions-demand/result.md)). Subtracting a
trailing 365-day level removed 42% of the old training year's penalty with all 13 series and none of
it without the largest series ([experiment
93](00093-age1-old-training-year-after-level-normalisation-demand/result.md)). Applies to
experiments and production. Scope: 13 Primary series with at least 24 months of data.

**Do not expect an older training year to become usable by matching its weather product.** Changing
the weather product from 18 km to 9 km grid spacing does not explain why an old training year scores
worse than a recent one ([experiment
95](00095-age2-old-year-penalty-with-the-same-weather-product-demand/result.md)). Applies to
experiments and production.

## Timestamps and availability

**Treat ERA5 as available about 5 days after the event, so ERA5 at lag times is a ceiling and not an
input a live service can have.** Name the weather source used at lag times in every write-up.
Applies to experiments. Evidence: [experiment 37](00037-switching-first-look/result.md) and
[experiment 38](00038-switching-mask-value/result.md) both state the delay and call their numbers
retrospective.

## Weather products

**Train each model on the history of the weather product it will be served, and do not train on the
CEDA UKV archive and serve the live UKV.** The two are statistically different products, and the
CEDA archive is licensed for non-commercial use ([experiment
53](00053-ukv-vs-ens-solar-anomaly/result.md)). Applies to experiments and production.

## Switching and the switching log

**Mask switching with the 66 hand-filtered real transfers, and with nothing else.** The transfers
run from January 2022, and the raw log holds many events too small to matter ([experiment
101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md), filtering the labelled
set of [experiment 41](00041-sw2-0a-labeller-gate/result.md)). Applies to experiments, and to
production once a detector runs there.

**Do not read a logged closure that no detector flags as a closure that moved no load.** Of 381,830
logged half-hours on 16 series, 80.5% lie in closures over 60 days that no label covers, and only
2.6% lie inside labelled transfers of 25% or more of the receiving load ([experiment
101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)). Applies to experiments
and production.

**Treat a logged closure of 2 to 6 hours as unconfirmed, and treat unexplained alarms as an upper
bound on false alarms.** Nine one-sided confirmations of 2 to 6 hour closures appeared against about
6 on shifted logs, while longer closures gave 28 against 0.4 ([experiment
37](00037-switching-first-look/result.md)). The log misses real steps, so an alarm with no logged
cause is not necessarily false. Applies to experiments.

**Add the substation pair 14-25 to the interconnection list before attributing closures to it.** The
detector of [experiment 37](00037-switching-first-look/result.md) could not attribute those
closures. Applies to experiments and production.

**Treat the labelled set as reliable only for transfers of 25% or more of the receiving load that
last over 1 day.** Only that band reached labeller recall of 0.80 in [experiment
41](00041-sw2-0a-labeller-gate/result.md), so mark smaller, shorter, and unmetered-receiver cases as
not scored. Applies to experiments.

**Recompute `moved_share_of_receiver` from the events, and never read the stored column.** The
stored value is missing for 62 of the 81 visible events that touch a Primary series, because the
labeller wrote some series identifiers as strings and 20 events name no metered receiver
([experiment 44](00044-sw2-2-switching-share-of-best-guess-score/result.md)). Applies to
experiments.

**Report the 21 of 66 real transfers that have no metered receiver separately.** Their
receiving-load share is blank, and their megawatts are the labeller's estimate ([experiment
101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)). The recall gap on
injected transfers without a metered receiver comes mostly from small injected steps: 211 of 274
such transfers step by less than the alarm threshold ([experiment
58](00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md), [experiment
55](00055-sw2-8-neighbour-veto-single-series-detector/result.md)). Applies to experiments, and to
production once a detector runs there.

**Restore the missing residual readings at series 12 and 13, or count the affected events as
unobserved, before scoring on real transfers.** 6 of the 35 real transfers over 1 day (11.5 MW) have
no readings in the stored residual panel ([experiment
101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)), and series 12 and 13
have residuals only from 2026-02 ([experiment 37](00037-switching-first-look/result.md)). Applies to
experiments.

**Do not mask with the labels of the refit-offset neighbour-count detector (RO-NC).** RO-NC marks 9%
to 16% of half-hours per year from 2022, and masking on them raised the error ([experiment
89](00089-sw2-22-longest-history-nra-forecast-with-ara-lags-masked/result.md)). The detector labels
were also not shown to help on rows outside switching in [experiment
86](00086-sw2-12-detector-labels-for-nra-forecast/result.md). Applies to experiments and production.

**Do not set lagged power inside an abnormal running arrangement to missing.** Masking those lags
raised the error by 0.409 points of capacity (98.33% interval +0.281 to +0.517) on the rows with
such a lag ([experiment
89](00089-sw2-22-longest-history-nra-forecast-with-ara-lags-masked/result.md)). Applies to
experiments, for the Primary demand model.

**Do not remove the detected intervals from the history of the error correction.** The
threshold-weighted error rose 57% on half-hours outside logged switching ([experiment
38](00038-switching-mask-value/result.md)). Applies to production, with the caveat that the detector
of that experiment is an early one.

**Count overlapping detected intervals once when summing the megawatts, and end each interval when
the transfer ends.** Summing overlapping intervals made the detector's megawatts about 1.5 times the
true value from the second day, and the detector kept reporting megawatts for days after a transfer
ended ([experiment
88](00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/result.md)).
Applies to experiments and production.

**Expect the end of a found transfer to be known 19 to 24 hours late, and expect the detector's
score to separate poorly in the first 24 hours after an onset.** The 24-hour blocks delay the end
([experiment 43](00043-sw2-1-detector-scoreboard/result.md)), and the area under the curve is 0.60
in the first 24 hours ([experiment
91](00091-sw2-26-calibrated-probability-of-abnormal-arrangement/result.md)). Applies to production.

**Do not suppress detector alarms near the monthly refit of the residual model.** Suppression lost
about 5% of the moved megawatts, all in the 13% of headline-band transfers that start or end within
24 hours of a refit ([experiment 71](00071-sw2-18-refit-offset-and-gap-bridging/result.md)). Applies
to experiments and production.

## Tails and curtailment

**Count the half-hours beyond a limit before ranking forecasts on curtailment cost.** Two bulk
supply points have only 19 half-hours beyond a limit of 0.9 times their earlier maximum export,
which cannot rank forecasts ([experiment
84](00084-substation-limit-curtailment-cost-worked-example/result.md)). Applies to experiments.

**Label every curtailment cost "calibration to history, not a saving".** The experiment's headline
cost, GBP 571,823 at a scaled limit of 0.9 times the maximum, is a calibration to history and not a
saving ([experiment 84](00084-substation-limit-curtailment-cost-worked-example/result.md)). Applies
to experiments.

**Read tail coverage month by month, and expect it to swing.** The coverage of q95 of the stored
best-guess quantiles at Primary substations swung from 0.88 to 0.97 across months, and a
recalibration fitted on earlier months did not move coverage by more than its own interval
([experiment 97](00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)). Applies to
experiments, for the 12 to 36 hour lead band.

## Privacy and anonymisation

**Never publish the name, the identifier, or the time series of an NGED metered generator, and label
each one (for example "Solar farm A").** Substations are not covered by this rule. Public data, such
as the settled output of Balancing Mechanism Units in Elexon's published data, may be named. Applies
to experiments, studies, and production reports.

## Provisional entries awaiting review

**These entries come from experiments that have not finished review, so they link no write-up.**
Treat each as a proposal with the confidence stated, and replace it with a linked entry once the
write-up merges.

**Cut the first months of most demand series (provisional; high confidence for D4, D14, D9, D16, and
D18, medium to low for the rest).** The first 1 to 19 months of a series often sit far from the same
months in later years. A new meter can read near zero for days to weeks and then step up, and a new
site can commission in stages. Cut until the level, the daily shape, and the share of rows present
match the same months in later years, cut more when uncertain, and let a later look move a cut later
and never earlier. Applies to experiments, and to production for a series that has just started
reporting. The cuts found so far cover 15 of the 20 series examined:

| Series | Cut | Reason |
|---|---|---|
| D4 | 19 months | Wrong level, a stuck plateau, and a step on 2021-04-12 |
| D14 | 8 months | Half the level until a step of 2.2 MW on 2020-06-15 |
| D17 | 5.6 months | Not recorded |
| D5 | 4.5 months | Not recorded |
| D3, D10, D21 | 2.5 to 3.5 months | 15% to 70% of rows missing, then a sudden snap to normal |
| D18 | 2.6 months | Not recorded |
| D9, D16 | about 1.3 months | Near zero, then a staged switch-on |
| D13, D19, D22 | about 1 month | Low confidence |

**Count a series' history from its first unflagged row, which is later than its first row of any
kind (provisional; high confidence).** D9 starts in 2023-04 and its first unflagged row is in
2023-08. Only five Primaries hold long histories: D5, D10, and D11 start in 2019-09, D14 in 2019-10,
and D16 in 2020-12, and the late starters begin between 2023 and 2025.

**Use a Primary metered in MVA for demand only on dark rows (provisional; high confidence).** A dark
row is a half-hour where CAMS global horizontal irradiance is below 1 W/m². A winter filter of below
100 W/m² keeps rows where 43% have irradiance above 100 W/m², so solar leaks in. This leaves about
3.3 dark site-years per MVA Primary with the 2020 lockdown kept and about 2.65 without it.

**Do not treat grid supply points or bulk supply points as clean demand (provisional; high
confidence).** The grid supply points are dominated by generation (D1 is below 5 points of capacity
on 75% of nights), and the bulk supply points sum the trial Primaries (D17 reverses flow on 57% of
summer middays).

**Screen for embedded generation with the share of negative dark half-hours (provisional; high
confidence).** Series with 26% to 74% of their dark half-hours negative carry embedded generation or
an exporting grid connection. The screen cannot work on an MVA Primary. One Primary (D15) carries a
flat export of about 2 MW through one transformer, and the export changes over 2025 and 2026.

**Fix the COVID window at the dates above because a window found from the data picks up drift and
solar growth (provisional; medium confidence).** A drift-removed fit finds April 2020 more than 5
points low only at D11 (-26 points) and D10 (-7 points), and August 2020 to September 2021 sits 5 to
17 points above the fit at D5, D10, D11, and D14.

**Join power and CAMS timestamps on `(time - 1 µs).truncate("1h") + 1h` (provisional; medium to high
confidence).** A power timestamp marks the end of its half-hour and a CAMS timestamp marks the end
of its hour. A naive join on equal timestamps drops or misassigns half the rows, and a whole
half-hour error raised the error score by more than 0.04.

**Allow for a demand drop shared across sites in season-year 2022 (provisional; medium
confidence).** Evening and night demand fell by 5 to 10 points at D10, D11, and D5 and partly
recovered by 2025, probably after the rise in energy prices. A model trained across years can read
the drop as a trend.

**Give the series that changed regime an individual start date or a flag (provisional; medium
confidence).** D8 weekday daytime demand falls from about 9 MW to about 3 MW between 2023-10 and
2025, D12 switches between a high-night and a low-night regime, D18 steps up about 1.1 MW at night
around 2026-07-03, and D5 shows a flat midday floor since 2022. Month-mean steps of over three
monthly standard deviations occur at D6 (a drop of 218 MW in 2023-07) and D11 (2020-04).

**Check that heating slopes and monthly levels are not drifting before pooling sites (provisional;
medium confidence).** Winter heating slopes of night power show no detectable drift between the
first and last winter at the long-history sites, and monthly levels move together across sites (mean
pairwise correlation 0.53 at night and 0.43 in the evening after removing the seasonal cycle).
