# Plan: how well can the distribution of an embedded battery's next-day output be predicted?

**The problem.** NGED needs to know what the batteries on its distribution network will do
tomorrow, half-hour by half-hour. Some embedded batteries are Balancing Mechanism Units (BMUs) and
publish a Final Physical Notification (FPN) of their planned output; most are thought not to be,
so a forecaster of those sees only prices, the calendar, the weather, and the battery's own recent
output. Nobody has counted how many of the batteries in NGED's four licence areas are BMUs, and the
[battery and solar separation study](https://openclimatefix.github.io/nged-substation-forecast/studies/battery-pv-separation/)
used after-the-event inputs throughout, so the separation study bounds a forecast without
measuring one.

**The planned solution.** One page in three parts. A short census opens the page: how many of the
batteries connected in NGED's licence areas are BMUs, from NGED's Embedded Capacity Register (ECR)
matched to the Elexon BMU register, shown as a band with an honest unknown fraction. Part A
forecasts battery BMUs, using public output, prices, and FPNs across 35 embedded battery BMUs, and
asks what other batteries' FPNs add for a battery whose own FPN is unknown. Part B forecasts
non-BMU embedded batteries, which publish no FPN, with NGED battery A as the real case, and says
what the Part A results can and cannot transfer. Every forecast is probabilistic: each arm issues
a set of quantiles for every half-hour, scored by the continuous ranked probability score (CRPS)
and pinball loss, checked for calibration and sharpness, and compared with a probabilistic
climatology, which is the first rung of both ladders. Five contrasts are planned here, before any
result exists.

## Verdict, size, decisions, and departures

**Verdict: worth doing, with one caveat stated up front.** The census is cheap and answers a
question NGED asked. The forecast is worth running because the answer is not known, but the prior
study already found that, for NGED battery A, adding the day-ahead price rank to the time of day
lowered the held-out error by 0.00 points of the 99th percentile [-0.19, 0.16] (the prior study's
planned contrast B1, not met). A null result in Part B is therefore a likely outcome, and the plan
treats a null result as a finding, provided the positive control passes.

**Size: complex, by the five triggers.**

- Changes what gets stored: yes, a published page under `docs/studies/` and a results folder.
- Touches the production serving path: no. Nothing under `src/` or any non-study package changes.
- Touches a degradation rule: no.
- Admits more than one defensible design: yes. The issue times, the price-forecast input, the
  baseline definitions, the definition of a neighbouring battery, and the census's matching rule
  each have several defensible choices.
- Spans code whose callers cannot be named without searching: no. One move of shared loaders into
  `packages/studies` (below) has callers that a single `grep` over `studies/battery_pv_separation/`
  names.

**Reviews bought:** two plan reviews (simplicity, then correctness, run as one Opus pass with both
lenses in order), the code review of every script before its first run, two Opus
scientific-validity reviews, the diff review and mutation pass (because `packages/studies/`
changes), and the closing prose review with persona and evidence reviewers. Builder personas:
NESO's BMU registration team and NGED's connections team (on the census), and a battery optimiser
operator (on Parts A and B).

**Decisions the maintainer has made:**

- **The forecasts are probabilistic throughout, not point forecasts.** A probabilistic
  climatology, the empirical distribution of output by half-hour of day and day type, is rung one
  of the ladder and the reference for every skill score.
- **The page may say whether NGED battery A is a BMU.** The census states NGED battery A's BMU
  status. If NGED battery A turns out to be a BMU, Part B still uses NGED battery A with its FPN
  withheld, as the closest available stand-in for a non-BMU battery, and the page says so. The page
  never names the BMU, because naming it would tie NGED's private telemetry to a public identity.
- **The ECR on disk is the census's master list**, so no request to NGED for a storage list is
  needed.
- **The early day-ahead issue time is the production forecast's issue time.** The production
  service issues a forecast at 00:00, 06:00, 12:00, and 18:00 UTC
  (`docs/live_service/operations.md`, `live_forecasts_schedule`). The 06:00 UTC slot is the last
  slot before the N2EX day-ahead result, so the early issue time is 06:00 UTC. The late issue time
  is the 18:00 UTC slot, the first after both the N2EX result and the response auction result.
- **Domestic batteries are counted only if a dataset supports the count.** No dataset on disk does.
  The ECR starts at 50 kW. The Microgeneration Certification Scheme (MCS) publishes domestic
  battery installation statistics
  ([data.gov.uk](https://ckan.publishing.service.gov.uk/dataset/activity/mcs-certified-domestic-battery-installation-statistics));
  the research step checks the geography of those statistics. If the statistics resolve to local
  authorities, the census reports an approximate count of certified domestic batteries in NGED's
  licence areas, outside the BMU band and labelled as certified installations only. If the
  statistics do not resolve that finely, the page says that no dataset used here counts domestic
  batteries in NGED's licence areas.

**Departures from the brief.**

- **The day-ahead price is not an oracle after the auction.** The N2EX auction publishes tomorrow's
  hourly prices at about 10:00 UK time (unverified to the minute; step 1 checks), so a forecast
  issued later legitimately knows them. Perfect-foresight price is the upper bound only for a
  forecast issued before the auction.
- **The "no price" arm is a day-shuffled price, not an arm with the price columns removed.** The
  study skill requires equal column counts, and the prior study's wrong-week control (arm A4) gained
  4.9 of the 8.5 points that real prices gained, so price columns carry a generic daily shape that
  a fair contrast has to cancel.
- **The planned contrasts are labelled D1 to D5**, so that a contrast's label never collides with a
  rung's label (rungs are C1 to C3, A0 to A5, and B1 to B2).

## When a Physical Notification is known

**A Physical Notification is first submitted on the day before, revised until gate closure, and
fixed at gate closure; the free Elexon API keeps only the final version.** The sources:

| Claim | Source | Status |
|---|---|---|
| Physical Notifications for each settlement period of the next day are required by 11:00 on the day before | Grid Code BC1.4.2, read in an [Ofgem-hosted drafting of BC1](https://www.ofgem.gov.uk/sites/default/files/docs/2005/01/9449-bc1-gc-drafting-for-eelps.pdf) | Verified in that document; the current Grid Code's wording not checked |
| Initial Physical Notifications are "usually submitted a day at a time, at 10am the day before" | [Elexon glossary: Initial Physical Notification](https://www.elexon.co.uk/bsc/glossary/initial-physical-notification/) | Verified |
| A notification may be revised until gate closure, 1 hour before the settlement period, and the Final Physical Notification is the notification prevailing at gate closure | [Elexon glossary: Final Physical Notification Data](https://www.elexon.co.uk/bsc/glossary/final-physical-notification-data/) | Verified from a search summary of that page, not a full read |
| The API's PN records carry `dataset`, `settlementDate`, `settlementPeriod`, `timeFrom`, `timeTo`, `levelFrom`, `levelTo`, `nationalGridBmUnit`, and `bmUnit`, with no publish time, notification time, or revision number | The API's [OpenAPI specification](https://data.elexon.co.uk/swagger/v1/swagger.json), schema `PhysicalNotificationData`, and a live call to `/datasets/PN/stream` | Verified |
| The Maximum Export and Import Limits (MELS, MILS) do carry `notificationTime` and `notificationSequence` | The same specification, schema `DeliveryLimitMaxData` | Verified |
| Elexon publishes a PN close to real time after it is received, including versions before gate closure | No source found | Unverified |
| Elexon's push service (IRIS) delivers each PN message as received, so a consumer archiving the feed would keep earlier versions | [Elexon's IRIS announcement](https://www.elexon.co.uk/bsc/article/new-features-coming-to-the-elexon-insights-real-time-information-service/) describes the service, not its PN messages | Unverified |

**What the plan can honestly claim for each issue time:**

- **06:00 UTC on the day before:** no Physical Notification for the target day is known, even in
  principle, because notifications are due at about 10:00 to 11:00. No FPN arm is possible.
- **18:00 UTC on the day before:** a Physical Notification for every half-hour of the target day
  exists by then, but the API holds only the version prevailing at gate closure, which was
  submitted after 18:00 whenever the battery revised it. Using the API's values at 18:00 would leak.
  A day-ahead FPN arm is possible only from versions archived at the time: polling the API at
  18:00 each day, or recording the IRIS feed, from now on. Neither gives history, so this study has
  no day-ahead FPN arm, and the page names forward archiving as the way to get one.
- **1 hour ahead:** the issue time is gate closure for the target half-hour, so the final version
  is exactly the version prevailing at the issue time. The FPN for the target half-hour and every
  earlier half-hour is known; the FPN for any later half-hour is not, and no arm uses it. How many
  minutes after gate closure Elexon publishes the FPN is unverified, and the page states the
  assumption that a forecaster has the FPN at gate closure.

## Census: how many of NGED's batteries are BMUs

**The census counts the batteries connected in NGED's four licence areas, and the share of them,
by count and by megawatts, that are BMUs submitting FPNs, as a band.** NGED's licence areas map
one-to-one to four Elexon Grid Supply Point (GSP) groups: East Midlands (`_B`), West Midlands
(`_E`, which Elexon names "Midlands"), South Wales (`_K`), and South West (`_L`, which the BMU
register names "South Western").

**The ECR is on disk and holds what the census needs except a BMU identifier.** The file is
`/home/jack/dev/nged-substation-forecast/literature/NGED/NGED ECR AUG 2026.xlsx` (template version
3.0, last updated 6 August 2026), outside the repository and not committed. Its generation and
storage rows sit in two sheets, "Register Part 1 50kW - <1MW" and "Register Part 1 - ≥1MW" (7,211
rows between them), with the header on each sheet's second row. Each row carries the export and
import meter point numbers; the customer name, site name, address, and postcode; OSGB eastings and
northings; the Grid Supply Point, bulk supply point, and primary substation by name; the connection
voltage and licence area; up to three technologies, each with a storage capacity in MWh, a duration
in hours, and a registered capacity in MW; the connection status ("Connected" or "Accepted to
connect"); the maximum export and import capacities; the dates; two service-provider flags; and
NGED's reference.

**A first count of the ECR (exploratory, aggregate only):** 560 rows list storage as one of their
technologies, of which 176 are connected and 384 accepted to connect. Of the 176 connected rows,
121 list storage as the first technology and the rest are hybrid sites, and 136 are below 1 MW.
The storage MWh, the duration, and both service-provider flags read "data not available" in every
connected storage row, so the census classifies size by export capacity alone.

**The match to the BMU register uses three keys, because the ECR has no BMU identifier and no
public register maps a meter point number to a BMU.** The licence area gives the GSP group. The
ECR's customer and site names are compared with the BMU register's `bmu_name` and
`lead_party_name` at the solar census's similarity threshold of 0.85
(`studies/solar_bmu_census/collate.py`, `MATCH_THRESHOLD`). The maximum export capacity must lie
within 10% of the BMU's `generation_capacity_mw`. The NESO Enduring Auction Capability (EAC)
unit-to-BMU table (`downloads/market/neso_eac_unit_to_bmu/`) flags a battery that trades frequency
response as a non-BM unit. Every proposed match goes into a hand-reviewed table beside the script,
as the solar census did.

**Three rungs:**

1. **Rung C1: list.** The ECR's connected storage rows, split into storage-only and hybrid sites,
   and the reviewed storage BMUs (`per_study/battery_pv_separation/bmu_list.csv`) whose GSP group
   is one of the four. Transmission-connected battery BMUs are a separate row, since a
   transmission-connected battery is not embedded.
2. **Rung C2: match**, as above.
3. **Rung C3: classify each connected ECR battery** as own BMU submitting FPNs, own BMU without
   FPNs, response provider without a BMU, or no BMU found; and **measure the recall** by running
   the match backwards from the reviewed storage BMUs in the four GSP groups.

**The BMU share is reported as a band.** The lower bound counts only matched batteries. The upper
bound adds the "no BMU found" batteries times the miss rate the recall check measures. A battery
in "no BMU found" may sit inside a supplier's BMU (an S-type BMU reports one total for many sites),
may be a BMU the match missed, or may have no BMU; the page says that the census cannot tell these
apart. Size classes follow the thresholds that decide BMU obligation (below 1 MW, 1 to 10 MW, 10 to
50 MW, 50 to 100 MW, and 100 MW or more; step 1 checks the 50 MW and 100 MW thresholds). A class
holding fewer than three batteries in a figure is merged upwards. The 384 accepted but unconnected
rows are a separate count. NGED battery A's BMU status is stated.

**A first look supports the guess that most embedded batteries are not BMUs (exploratory).** Of
the 113 reviewed storage BMUs, 47 are embedded (`bmu_type` E), and 5 of those with at least 95% of
the window's settled output (B1610) sit in NGED's four GSP groups, against 176 connected storage
rows in the ECR.

**Census figures:** Figures 3 to 5 in the figure list below.

## Shared forecast design (Parts A and B)

**Three issue times, each defined by what is known at that moment.** All times are UTC. The target
day is the N2EX delivery day, read from the `delivery_date` column of
`downloads/market/neso_n2ex_day_ahead/`.

| Issue time | When | Price input | Own output known up to | FPN |
|---|---|---|---|---|
| DA-early | 06:00 on the day before, the production slot before the auction | A forecast of the day-ahead price | 06:00 on the day before | None exists yet |
| DA-late | 18:00 on the day before, the production slot after the N2EX and response auction results | The actual N2EX price | 18:00 on the day before | Exists, but not retrievable as known at 18:00 |
| ID-1h | Gate closure, 60 minutes before the target half-hour | The actual N2EX price | The half-hour that ended at issue | Own and neighbours' FPNs for the target and earlier half-hours (BMUs only) |

**The battery's own recent output is assumed to be available in real time.** NGED reads its own
telemetry in near real time. Elexon's settled output (B1610) arrives about 7 days late, so Part A
assumes the distribution network operator sees each BMU's output in its own telemetry. The page
states the assumption.

**The persistence value is the latest observed output at the target's half-hour of day that is
known at the issue time.** At DA-early that is usually two days before the target day; at DA-late
it is the day before for half-hours that ended by 18:00 and two days before for the rest; at ID-1h
it is the output in the half-hour that ended at issue.

**The price inputs bracket what a price forecast could deliver:**

- `actual`: the N2EX price. Legitimate at DA-late and ID-1h; the upper bound at DA-early.
- `naive`: the same hour seven days earlier. The lower bound at DA-early.
- `model`: an XGBoost model of the N2EX price, given the NESO wind forecast
  (`downloads/market/elexon_windfor/`) as published before 06:00, the calendar, and the `naive`
  price, fitted out of fold on the same four folds. The wind forecast file keeps every publication
  with its `publish_time`, so an as-of join reproduces what was known; the last publication before
  06:00 reaches 68 hours past the start of its day, so it covers the whole target day. The NESO
  day-ahead demand forecast (`downloads/market/elexon_ndf/`, and `elexon_tsdf/` likewise) is not
  used: its last publication before 06:00 reaches only 27 to 28 hours past the start of its day,
  about 03:00 on the target day, so the demand forecast for the target day does not exist at
  DA-early. The page reports the price model's own error relative to `naive`; the studies the
  prior page cites reach 0.40 to 0.61 of a same-hour-last-week rule, so a relative error above
  about 0.8 marks the price model as weak.
- `shuffled`: the price of a different day of the same calendar month and day type, drawn by a
  fixed seed. The `shuffled` arm is the no-information arm and the negative control. The same
  shuffled day replaces the price on training and test rows alike, and every column derived from
  price (rank, daily mean, daily range, the rank-rule value) is computed from the shuffled price.

The EPEX half-hourly auction is not free (from EUR 665 a month), so every price is hourly and both
half-hours of an hour share a rank.

### Why the forecast is a set of quantiles, never a Gaussian

**A battery's half-hourly output is bounded and usually has three modes, so every arm states its
distribution as quantiles.** Output cannot exceed the export limit or the import limit. In a given
half-hour a battery is mostly charging hard, idle at exactly zero, or discharging hard, so the
distribution is a point mass at zero between two humps near the limits. A Gaussian centred on the
mean puts most of its probability on the half-charged middle the battery rarely occupies, and some
of it beyond the limits. Every arm therefore issues quantiles, and the quantiles can represent a
point mass (several levels equal to zero) and a gap between modes (a jump between neighbouring
levels). Figure 12 shows the Gaussian's misfit on the data; no Gaussian arm is fitted.

**The quantile set is the 13 delivery levels** (`contracts.common.DELIVERY_QUANTILES`: p1, p2, p5,
p10, p20, p35, p50, p65, p80, p90, p95, p98, and p99), the levels the production service issues.
Every arm emits all 13, sorted within each row to repair quantile crossing (as
`studies.cross_validation.crps` does), and clipped to the 0.1st and 99.9th percentiles of the
battery's output on the training folds.

### How each arm produces a distribution

- **`clim` (rung 1, the reference): probabilistic climatology.** The empirical quantiles of the
  battery's output over the 56 complete days before the issue time, at the target's half-hour of
  day and the half-hours either side, on days of the same type (working day, or weekend and bank
  holiday). The trailing window carries the season, which a fold-based climatology cannot, because
  each held-out fold is a whole season. A working-day sample holds about 120 values and a
  weekend sample about 48, so the empirical p1 and p99 rest on one or two values; the page says so.
  The window cannot reach before 1 September 2025, so `clim` uses every complete day available up
  to 56, and **no arm is scored before 1 October 2025**, by which date at least 28 days are
  available. Scored months are therefore October 2025 to August 2026, 11 months, with fold 0
  holding two of them.
- **`persistence_conformal`:** the persistence value plus the empirical quantiles of persistence's
  own errors on the training folds, by half-hour of day.
- **`rank_conformal`:** the scaled schedule from `rank_rule_schedule`, plus the empirical quantiles
  of the schedule's residuals on the training folds, computed separately for half-hours the
  schedule marks as charging, idle, and discharging. The separate residual sets are what let the
  distribution keep its three modes.
- **`xgb_quantile`:** an XGBoost model with the `reg:quantileerror` objective at the 13 levels.
  `studies.cross_validation.fit_one_fold` fits the same kind of multi-quantile model but
  hard-codes its nine `QUANTILE_LEVELS`, so `battery_forecast.py` follows its pattern with the 13
  levels rather than changing a function other studies' published numbers rest on.

**The XGBoost model has one fixed column tuple, built by one function:** half-hour of day, day of
week, price, within-day price rank, the day's mean price, the day's price range, the rank-rule
value, the persistence value, the median of `clim`, one own-FPN slot, and three neighbour slots. A
slot not in use holds its filler: the slot's own value from the same half-hour seven days earlier,
or, for a battery with no FPN, the median of `clim` lagged one week. One model per battery and
issue time, refitted per fold, `colsample_bytree` 1, three seeds, and both hyperparameter settings
on the planned contrasts' arms. At DA-early the `model` price fed to the training rows is the price
model's out-of-fold forecast, so the quantile model learns the spread a forecast price brings.

**The rank-rule schedule is scaled by one coefficient fitted on the training folds**, and its
duration (2, 4, 6, or 8 half-hours) is fitted per battery on the training folds.

### Scores, calibration, and sharpness

**The headline score is CRPS, as a percentage of the battery's own 99th-percentile absolute output
(p99), the normalisation the prior study used.** CRPS is computed from the 13 quantiles as twice
the pinball loss summed over the levels, each level weighted by the probability gap it represents
(half the distance to each neighbouring level, with the end levels extended to 0 and 1). The
repository's `studies.cross_validation.crps` does the same with nine equally spaced levels and
equal weights; the new weighted function is tested to give the same answer as `crps` on those nine
levels. Every arm is scored on the same 13 levels, so the coarse middle of the set (p35 to p65)
coarsens every arm alike. Over all half-hours of the target day (DA) or the target half-hour
(ID-1h).

**Every arm also reports, as the [evaluation metrics
page](https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/#probabilistic-metrics)
defines them:**

- the pinball loss at each of the 13 levels, and their mean;
- coverage (PICP) and mean width of the six symmetric delivery bands, p35–p65 out to p1–p99. The
  reference for coverage is the nominal rate, because these quantiles come from fitted models, not
  a finite ensemble; for `clim` the page also gives the finite-sample reference;
- a reliability diagram: the share of half-hours whose output fell below each forecast quantile,
  against the quantile's level;
- the CRPS skill score against `clim`, one minus the ratio of mean CRPS values, printed beside each
  contrast as a reading aid. The contrasts themselves are differences in CRPS, because a ratio of
  means does not resample cleanly by month;
- the median's mean absolute error, so the page connects with the prior study's point-forecast
  numbers.

**Coverage alone can be bought with width, so coverage is always shown beside width**, and the
proper scores (CRPS and pinball loss) penalise a band that is too wide or too narrow.

**Rows, folds, and intervals.** A half-hour is scored only if every arm of its setting has its
inputs; drop decisions read the target and the shared inputs, never one arm's input. Folds are the
prior study's four blocks of three whole months (`FOLD_MONTHS`). Intervals come from
`studies.bootstrap.bootstrap_difference` and `bootstrap_absolute` on the per-row CRPS, resampling
whole calendar months and a seed, 2,000 resamples, with months shared across batteries because
every battery faces the same market days. Empirical and rule-based arms carry their one result
under all three seed labels. The second hyperparameter setting runs on every planned contrast and
any result near the 5% line.

**The window is 1 September 2025 to 31 August 2026**, the window of the B1610 download, the market
downloads, and the prior study; scoring starts on 1 October 2025 (above).

**Compute budget.** Before each rung, time one fit at each setting and project the rung's total.
At 35 batteries, four folds, and three seeds, an arm is 420 fits per setting. If a rung projects
over about two hours on the workstation, run fits in parallel processes (four threads each, as
`THREADS_PER_FIT` sets) rather than cutting seeds or batteries; check the CPU load first.

## Part A: forecasting battery BMUs

**The testbed is every embedded battery BMU with enough public data, fixed now:** every BMU in the
reviewed `bmu_list.csv` with `bmu_type` E and at least 95% of the window's half-hours present in
both B1610 and the half-hourly PN file. A first look counts 35 such BMUs with 12 lead parties; the
largest lead party holds 14 of the 35, and 3 of the 35 sit in NGED's GSP groups (an exploratory
subgroup, too small to summarise alone). The report lists the lead parties. Transmission-connected
battery BMUs are excluded, because the question is about embedded batteries.

### Rungs

- **Rung A0: the instrument works on a known answer.** 20 synthetic batteries driven by
  `lp_schedule` on the actual N2EX price, with energy durations of 1, 2, and 4 hours, plus noise of
  10% of p99; and 20 "price-blind" batteries whose output is the trailing time-of-day profile of a
  real embedded BMU plus the same noise. On the price-driven batteries, `xgb_quantile` given
  `actual` must have a lower CRPS than the same model given `shuffled`, statistically significant
  at the 5% level (the positive control), and must not on more than 2 of the 20 price-blind
  batteries (the false-alarm control; with 20 tests at a one-sided 2.5% rate, two or more chance
  hits happen about one run in eleven, and three or more about one run in eighty). The known
  answer also checks calibration: on the price-driven batteries the p10–p90 coverage of
  `xgb_quantile` given `actual` should lie within 5 points of 80%.
- **Rung A1: the probabilistic climatology `clim`**, at all three issue times. Every later rung is
  scored against `clim`.
- **Rung A2: the other baselines and the price schedule**: `persistence_conformal` and
  `rank_conformal`, with each price source.
- **Rung A3: `xgb_quantile`** at DA-early and DA-late, with each price source.
- **Rung A4 (exploratory): the battery's own FPN at ID-1h.** The own-FPN slot holds the BMU's FPN
  for the target half-hour (`own_fpn`); the control holds the same BMU's FPN from the same
  half-hour seven days earlier. The FPN at gate closure is expected to track output closely, so
  this rung is not a planned contrast; it supplies the denominator of the share recovered in rung
  A5.
- **Rung A5: other batteries' FPNs at ID-1h, with the target's own FPN withheld.** This rung asks
  whether a forecaster who cannot see a battery's FPN gains from the FPNs the rest of the fleet
  publishes, and is the test that Part B depends on.

### What another battery's FPN can carry that prices cannot

**A neighbour's FPN states, at gate closure, what another battery's optimiser decided, and three
channels could make that decision informative about the target.** First, a few optimisers schedule
many batteries, and batteries run by one optimiser, or by optimisers with a shared method, respond
to the same signals at the same half-hour. Second, the FPN reflects information no free price
carries: intraday continuous trades, the optimiser's own price forecast, and its view of the
imbalance price. Third, the FPN reflects the power held back for frequency response, which the
daily EAC auction sets for many batteries at once.

**The planned neighbour set is every other embedded battery BMU in the testbed with a different
lead party from the target:**

- **A different lead party** removes the target's own optimiser. A forecaster of a non-BMU battery
  does not know which optimiser runs the battery, and a same-party neighbour, often a second unit
  at the same site, can submit an FPN nearly identical to the target's.
- **The whole GB testbed, not the same GSP group or Grid Supply Point.** GB has one wholesale price
  zone; the BMU register gives only the GSP group; and the 3 testbed batteries in NGED's GSP groups
  are too few to summarise.
- **Not by size or duration.** The BMU register gives power but not energy, the ECR's duration
  column is empty, and size within the testbed is narrow (median 49 MW).

One alternative neighbour set is exploratory: the same lead party (the optimiser channel at full
strength), for targets whose lead party has another testbed unit.

**The three neighbour slots hold:** the neighbours' mean FPN for the target half-hour, each FPN as
a fraction of its own battery's p99; the same mean for the half-hour before; and the share of
neighbours whose FPN is discharging at the target half-hour. A neighbour missing its FPN for a
half-hour drops out of that half-hour's statistics rather than counting as zero. Every target's
neighbour list is printed into the report. The arms of rung A5 are all `xgb_quantile`, with the
`actual` price and the own-FPN slot holding its filler:

- `no_neighbour`: every slot holds its filler. The no-neighbour baseline, beside `clim` and
  `persistence_conformal`.
- `neighbour_fpn`: the neighbours' FPN statistics.
- `neighbour_fpn_shuffled`: the same statistics from a different day of the same month and day
  type, drawn by a fixed seed (the negative control). The shuffle keeps the fleet's typical daily
  shape and removes the target day's decisions.

**Leakage in rung A5:** at ID-1h the issue time is gate closure for the target half-hour, so each
neighbour's FPN for the target half-hour and every earlier half-hour is the version prevailing at
the issue time (see "When a Physical Notification is known"). No arm uses an FPN for a later
half-hour, and rung A5 has no day-ahead setting.

**What the page reports beside the contrast:** the share of the own-FPN gain in CRPS that
neighbours recover, `(no_neighbour − neighbour_fpn) / (no_neighbour − own_fpn)`, the p10–p90
coverage and width of each arm, and the contrast repeated with the largest lead party removed from
the neighbour set (exploratory).

### Planned contrasts for Part A

All are differences in mean CRPS, in points of p99, treatment minus reference, pooled over the
testbed, so a negative value favours the treatment. Each is printed with both arms' CRPS skill
score against `clim` and both arms' p10–p90 coverage and width.

- **D1 (DA-late):** `xgb_quantile` given `actual` minus `clim`. Does a probabilistic forecast
  issued after the auctions beat the probabilistic climatology at all?
- **D2 (DA-late):** `xgb_quantile` given `actual` minus `xgb_quantile` given `shuffled`. Is the gain
  the target day's price, or the month's typical shape?
- **D3 (DA-early):** `xgb_quantile` given `model` minus `xgb_quantile` given `shuffled`. Does a
  price forecast available at the production issue time sharpen or shift the distribution
  usefully? The `actual` and `naive` arms bracket D3 and are exploratory.
- **D4 (ID-1h, own FPN withheld):** `neighbour_fpn` minus `neighbour_fpn_shuffled`. Do other
  batteries' FPNs improve the distribution forecast for a battery whose own FPN is unknown?

## Part B: forecasting non-BMU embedded batteries

**A forecaster of a non-BMU embedded battery has five inputs:** the day-ahead price (forecast at
06:00, actual from about 10:00); the calendar; the weather and the NESO demand and wind forecasts,
which act mainly through the price; the battery's own telemetry up to the issue time; and, from
Part A, the fleet's FPNs at gate closure. The forecaster does not have the battery's planned
output, its response contracts, its optimiser, or its state of charge, which can only be estimated
by integrating the telemetry.

**NGED battery A is the one real case.** The prior study's private series, loaded by the existing
private loader, shown only as "NGED battery A", with days numbered 1 to 7 and output as a fraction
of its own 99th-percentile absolute output. The prior study's data-quality check stands: 98.0% of
half-hours present and 2.1% exact zeros.

### Rungs

- **Rung B1: the Part A ladder on NGED battery A.** `clim` first, then the other baselines, the
  price schedule, and `xgb_quantile` at DA-early and DA-late, with each price source, and at ID-1h
  with its own telemetry. NGED battery A's output histogram is shown first, because the earlier
  page found its price response weaker than the public batteries', and its distribution may have a
  different shape.
- **Rung B2 (exploratory): the fleet's FPNs at ID-1h.** Rung A5's arms with the neighbour set
  every testbed BMU (NGED battery A has no lead party a forecaster can see), and without the
  `own_fpn` arm.

### Planned contrast for Part B

- **D5 (DA-late, NGED battery A):** `xgb_quantile` given `actual` minus `clim`, in mean CRPS.

**What Part A can transfer to Part B, and what it cannot.** Part A can transfer the size of the
price signal at each issue time, the calibration a given method reaches, and the value of the
fleet's FPNs at gate closure. Part A cannot transfer the behaviour that comes from being a BMU: a
testbed battery takes balancing acceptances, which explain much of a BMU battery's output in the
prior primer, and submits FPNs that its optimiser plans around. The testbed batteries are mostly
about 50 MW; a non-BMU battery is usually smaller, and may sit behind a meter with site load. So a
Part A result is evidence about non-BMU batteries only where rung B1 or B2 agrees on NGED battery
A, and one battery cannot confirm a general claim.

## Rules for every planned contrast

**A planned contrast is met when the difference in mean CRPS is negative and statistically
significant at the 5% level under both hyperparameter settings.** CRPS is a proper score, so a
lower CRPS cannot be bought by widening or narrowing the distribution dishonestly. A met contrast
whose treatment's p10–p90 coverage misses 80% by more than 10 points is still met, and the page
flags the miscalibration beside it. A contrast that fails under the second setting is
reported as not met, with both results. No correction is made for multiple comparisons across the
five, and the page says so. Every other number is exploratory, and anything added after the first
run is post hoc.

**False-alarm rules.**

- D2 and D3 count only if rung A0's false-alarm control passes.
- D4 counts only if `neighbour_fpn_shuffled` is not better than `no_neighbour` by more than D4's own
  effect, which would mean the neighbour columns carry fleet shape rather than the day's decisions.
- A null D5 is read as "no effect" only if rung A0's positive control passed, and otherwise as "an
  effect as large as the interval's bound is not excluded".

**What counts as failure.** D1 not met means a day-ahead probabilistic forecast of an embedded
battery BMU from public inputs describes tomorrow no better than the trailing 56-day climatology.
D2 not met with D1 met means the gain is calendar shape, not price. D4 not met, or met with
neighbours recovering less than a tenth of the own-FPN gain in CRPS, means the fleet's FPNs say
little about another battery's next half-hour; a null D4 beside a strong same-lead-party arm would
place the information in the optimiser, not the fleet.
Each outcome is published as the finding.

## What the data can and cannot show

- **Can show:** the out-of-sample CRPS, pinball loss, calibration, and sharpness of probabilistic
  forecasts from public inputs for 35 embedded battery BMUs over 11 scored months; how much of the
  climatology's CRPS the target day's price removes at two issue times; how much the own FPN and
  other batteries' FPNs remove at gate closure; and the same, less the own FPN, for one NGED
  battery.
- **Cannot show why a battery did what it did.** Response contracts and the optimiser are not
  observed.
- **The neighbour testbed is BMUs predicting BMUs**, so D4 may overstate what the fleet's FPNs say
  about a non-BMU battery. Rung B2 is the one direct check.
- **One NGED battery is a case study, not a sample.** D5's interval covers month-to-month variation
  of that one battery only. NGED battery A's whole-year state-of-charge drift of 20.8 hours (prior
  page) suggests its meter may include site consumption, which nothing here tests.
- **No day-ahead FPN arm exists**, because the free API keeps only the final version.

## Figure list

**The page is mostly figures, in three runs: what the data look like, each technique working on a
simple case before a hard one, and the results.** Every technique figure draws `clim` beside the
technique, so a reader sees the gain or its absence without reading a number. Where a technique
fails, its figure says so in the title and shows the failure as plainly as a success: the fan that
misses the truth, the reliability line that leaves the diagonal, or the interval that straddles
zero. Testbed BMUs are public and are named with output in MW on calendar dates. NGED battery A
appears only as a fraction of its own p99 on days numbered 1 to 7. The week shown for any battery
is fixed by a stated rule (the prior study's `WEEK_START`, plus the week of the largest daily price
spread and the week of the smallest), never chosen by eye.

| No. | What it shows | Rung | Message |
|---|---|---|---|
| 1 | Headline: CRPS skill against `clim` by lead time (1 hour, then 6-hour bins of lead from the DA-late and DA-early issue times), with 95% intervals, for `xgb_quantile` and `rank_conformal`, testbed pooled and NGED battery A as its own line | A1–A5, B1 | Skill is highest at 1 hour ahead with an FPN and falls with lead; the page states the actual ordering and the lead at which skill reaches zero, if any. |
| 2 | The planned contrasts D1 to D5 with intervals (`interval_panel`), and the CRPS leaderboard per issue time (`leaderboard_panel`) | All | Which planned contrasts are met, and by how much, at a glance. |
| 3 | Census: connected batteries by size class, coloured by BMU status, unknown band hatched | C3 | Most of the batteries connected in NGED's licence areas by count are not BMUs; the band shows how sure that is. |
| 4 | Census: the same in megawatts of export capacity | C3 | The megawatt share held by BMUs differs from the count share, because BMU batteries are the large ones. |
| 5 | Census: the matching funnel and its recall | C2–C3 | How many register rows matched, and how many BMUs the match would miss. |
| 6 | A timeline of one day: the N2EX result, the response auction, the Physical Notification deadlines, gate closure, and the three issue times | — | What a forecaster knows at 06:00, at 18:00, and at gate closure. |
| 7 | The data, per battery: a typical day (mean and p10–p90 of output by half-hour of day), one small panel per testbed battery and NGED battery A | Data | Batteries differ in when they charge and discharge, and how spread their days are. |
| 8 | The data: one week of output for four testbed batteries side by side on shared dates, with the day-ahead price below | Data | Batteries run by different parties move together on some days and apart on others. |
| 9 | The data: price curves, a week of N2EX prices, and the spread of the within-day price range across the year | Data | Some days have a deep price trough and peak worth cycling for, and some barely any. |
| 10 | The data: FPN against metered output, one week for two testbed batteries, and a scatter for all | Data | The FPN tracks metered output closely but not exactly; the gap is the balancing actions and response. |
| 11 | The data: neighbours side by side, one target's FPN and metered output over a week, with the mean FPN of its planned neighbour set and of its own lead party's other units | Data, A5 | Whether other batteries' plans visibly line up with the target's before any model is fitted. |
| 12 | The data: histograms of half-hourly output for four testbed batteries and NGED battery A, with the Gaussian of the same mean and spread over one | Data | Output piles up at zero and near the limits, so a Gaussian puts its probability where the battery rarely is. |
| 13 | Known answer: one week of a price-driven synthetic battery (left) and a price-blind one (right), truth over the fans of `clim`, `xgb_quantile` given `shuffled`, and `xgb_quantile` given `actual` | A0 | With a battery that follows price, the price arm's fan hugs the truth; with one that ignores price, the price arm is no better than the climatology. |
| 14 | Known answer: positive and false-alarm control results with intervals, all 40 synthetic batteries | A0 | The instrument detects a real price effect and stays silent on 18 or more of 20 null batteries. |
| 15 | `clim` on one real testbed week: p10–p90 and p1–p99 fans with the truth | A1 | What the baseline every later figure is compared with looks like, and where it is wide. |
| 16 | The price schedule, simple case beside hard case: the weeks of the largest and the smallest daily price spread, truth over the fans of `clim` and `rank_conformal` | A2 | On a day with a clear price trough and peak, the schedule places its modes at the right half-hours; on flat-price days its fan is no better than the climatology's. |
| 17 | `xgb_quantile` against `clim` and `rank_conformal` on the fixed real week at DA-late | A3 | How much sharper the learned distribution is on real data, and where it still misses. |
| 18 | The price bracket at DA-early: CRPS of `xgb_quantile` given `naive`, `model`, `shuffled`, and `actual`, intervals | A3 | How much of the perfect-price gain a price forecast available at 06:00 keeps. |
| 19 | Reliability diagrams, one panel per issue time: observed share below each quantile against its level, one line per arm, with the diagonal; p10–p90 width printed in each line's label | A1–A5 | Which arms are calibrated, which are overconfident or too wide, and how wide the calibrated ones are. |
| 20 | Per-battery CRPS skill against `clim`, one dot per testbed battery, arms in rows | A1–A5 | Whether the pooled result holds for most batteries or rests on a few. |
| 21 | Gate closure: one week as fans of `no_neighbour`, `own_fpn`, and `neighbour_fpn`, then rung A5's CRPS with the D4 interval, `own_fpn` beside it, and the share of the own-FPN gain recovered | A4–A5 | How much of what the battery's own FPN tells, the fleet's FPNs tell without it. |
| 22 | NGED battery A: typical day and output histogram beside the testbed's spread | B1 | How NGED battery A's behaviour differs from the BMUs before any forecast. |
| 23 | NGED battery A: days 1 to 7 as fans of `clim` and `xgb_quantile` given `actual` | B1 | Whether a price-aware forecast sharpens the distribution for a non-BMU battery at all. |
| 24 | NGED battery A: CRPS skill of every arm, including the fleet-FPN arm of rung B2, with intervals, beside the testbed distribution of the same skill | B1–B2 | Where the one non-BMU battery sits relative to the BMUs, and what transfers. |

## What changes, file by file

**Shared code moves into `packages/studies`, because a second study folder now needs the
loaders.** The import rules forbid one study folder importing another.

- New `packages/studies/src/studies/battery_market.py`: `market_frame`, `battery_output`,
  `p99_output_mw`, `FOLD_MONTHS`, and `WEEK_START`, moved verbatim from
  `studies/battery_pv_separation/battery_inputs.py`, which re-imports them. The NGED battery A
  loader (`studies/battery_pv_separation/nged_battery_a.py`) moves the same way. The diff review
  confirms the moved bodies are unchanged; re-run only the prior study's cheapest script
  (`battery_rung1.py`) and confirm its saved output is bit-identical, as a check that the imports
  resolve to the same code.
- New `packages/studies/src/studies/battery_forecast.py`, with tests: the as-of join of a vintaged
  forecast at an issue time, the three issue-time frames, `clim` and the persistence value, the
  conformal residual quantiles by schedule state, the gap-weighted CRPS, coverage and width per
  band, the reliability table, the clipping of quantiles to bounds, the day-shuffle draw used by
  `shuffled` and `neighbour_fpn_shuffled`, and the neighbour statistics. The scoring tests assert
  that the gap-weighted CRPS equals `studies.cross_validation.crps` on its nine levels, that a
  forecast whose every quantile equals the truth scores zero, and that coverage counts a value on
  a band's edge as inside. The `clim` tests assert that the trailing window never includes the
  target day, and that at DA-late it never includes a half-hour ending after 18:00 on the day
  before. The persistence tests assert the same of the persistence value at each issue time.
  The tests assert that no row published after the issue time is used, that a row published
  exactly at the issue time is, that no FPN for a half-hour after the target enters, that the
  target and its same-lead-party units never appear in its own neighbour set, that a missing
  neighbour drops out rather than counting as zero, and that the shuffle never draws the target
  day itself.
- Reused unchanged: `studies.battery_dispatch.rank_rule_schedule` (for the arms) and
  `lp_schedule` (for rung A0's synthetic batteries only); `studies.bootstrap.bootstrap_difference`,
  `bootstrap_absolute`; `studies.cross_validation.booster_parameters`, `crps` (as the test
  reference), `SEEDS`, `PRIMARY_HYPER_PARAMETERS`, `SENSITIVITY_HYPER_PARAMETERS`;
  `contracts.common.DELIVERY_QUANTILES`;
  `studies.charts.leaderboard_panel`, `interval_panel`, `figure`, `planning`, `report_contrasts`,
  `assert_matches_printed`; `studies.sources.study_dir_for`; the solar census's
  `MATCH_THRESHOLD` and `normalise`.
- New folder `studies/embedded_battery_forecast/`: `ecr.py` (reads the two Part 1 sheets),
  `census.py`, `census_charts.py`, `forecast_inputs.py`, `forecast_synthetic.py`,
  `forecast_price_model.py`, `forecast_bmus.py` (Part A), `forecast_nged_battery_a.py` (Part B),
  `forecast_report.py`, `forecast_charts.py`, and a README listing every output file.
- Results in `data/studies/per_study/embedded_battery_forecast/` (via `study_dir_for`), with
  per-row out-of-fold predictions and losses for every arm, battery, seed, and setting.
- Page `docs/studies/embedded-battery-forecast.md`, under Studies > Forecasts, with its charts in
  `docs/studies/assets/`.

## Work breakdown for the Sonnet agents

Each step names its data and where the data comes from. Every script gets a fresh code review
before its first run. Every data source named below is on disk under `data/studies/` except the
ECR (in `literature/NGED/`) and, conditionally, the MCS statistics.

1. **Research (one Opus agent, read-only, web allowed; brief in a file, as in the prior study).**
   Settle the rows marked unverified in "When a Physical Notification is known"; the current Grid
   Code BC1 wording; the minute the N2EX and EAC results are published; the ECR's licence and
   whether the August 2026 file is the latest; the current England and Wales thresholds for BMU
   obligation; and the geography and licence of the MCS domestic battery statistics. Mark each
   answer verified or not.
2. **Download, only if step 1 finds the MCS statistics resolve to local authorities and are free
   under an open licence (one Sonnet agent, `data-download` and `data-validation` skills)**, into
   `data/studies/downloads/registers/`, with a README and lineage file. Skip the download if the
   source needs an account, a payment, or a licence accepted in a person's name, and the page says
   that no dataset used here counts domestic batteries.
3. **Census (rungs C1 to C3).** Inputs: the ECR workbook, the BMU register
   (`downloads/market/elexon_bmu_reference/`), the reviewed BMU list, the EAC unit table, the
   private roster for NGED battery A's status, and the MCS statistics if downloaded. Run the
   `data-validation` checks on the ECR read first: rows per sheet, the storage-row count, and
   duplicate references. Output: `census_report.md` and the reviewed match table.
4. **Package moves and new module.** Move the loaders, write `battery_forecast.py` and its tests,
   re-run `battery_rung1.py` and compare, and run `uv run pytest --run-studies -n auto
   packages/studies`.
5. **Input frames.** `forecast_inputs.py` builds, per battery and issue time, the target, the price
   sources, the as-of wind forecast, the own FPN
   (`downloads/market/elexon_pn/elexon_pn_half_hourly.parquet`), the neighbour statistics, and
   own-output lags. Inputs: B1610 under `per_study/solar_bmu_census/inputs/b1610/` (column `half_hour_end_time`, so shift to the
   period start before joining), the market downloads, the lead parties from the BMU register, and
   NGED battery A from the private store. Print each arm's column list, each target's neighbour
   list, and the testbed count (expected 35) into the report.
6. **Rung A0**, then stop if the positive control fails.
7. **Price model** (`forecast_price_model.py`), with its error relative to `naive` in the report.
8. **Part A, rungs A1 to A5** (`forecast_bmus.py`).
9. **Part B, rungs B1 and B2** (`forecast_nged_battery_a.py`).
10. **Report** (`forecast_report.py`): every arm's absolute error, the five planned contrasts at
    both settings, the controls, and the exploratory rows, each labelled.
11. **First Opus science review**, triage, re-run what the reviewer asks.
12. **Charts and draft page**, then the **second Opus science review**, triage, re-run.
13. **Diff review and mutation pass**, then the **prose review** with persona and evidence
    reviewers, then merge if the study skill's three conditions hold.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest --run-studies -n auto packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict   # then read the rendered page and its links
```

Plus the CI steps the verification set omits: `pydoclint` and the docs-link checker.

## Plan review: what was cut and corrected

**One Opus reviewer ran both plan lenses in order (simplicity, then correctness).** The changes:

- **Cut for simplicity, because none of them decides a planned contrast:** the linear-programme
  schedule arm and its column (the rank rule stays as the price schedule; `lp_schedule` stays only
  to drive rung A0's synthetic batteries), the 20-scenario price mixture, the Gaussian arm (Figure
  12 makes the same point from the data), the conformal recalibration of XGBoost, the fold-based
  climatology, the FPN-on-its-own arm, the metered-neighbour arm, two of the three alternative
  neighbour sets, the pooled transfer rung (formerly B2), the TEC key in the census match, and the
  re-run of every prior-study script after the package move (a verbatim move needs one re-run).
- **Contrasts cut from six to five.** The own-FPN contrast at gate closure moved to exploratory,
  because an FPN fixed at gate closure is expected to track output and the contrast would decide
  nothing; the own-FPN arm stays as the denominator of the share recovered.
- **The quantile set is the 13 delivery levels, not 23.** Every arm is scored on the same levels,
  so the contrasts are unaffected, and an XGBoost multi-quantile fit grows one tree per level per
  round, so 13 levels take about 40% less compute.
- **Figures cut from 29 to 24:** the seasonal typical day (no conclusion rests on it), and the
  sharpness-against-coverage panel (width now sits in the reliability diagram's labels), with two
  simple-then-hard pairs merged into single figures.
- **Corrected: the price model drops the NESO demand forecast.** The last demand forecast
  published before 06:00 on the day before reaches only about 03:00 on the target day, so the
  input does not exist at DA-early; the wind forecast reaches 68 hours and stays.
- **Corrected: scoring starts on 1 October 2025.** The trailing 56-day climatology has no history
  before the window starts on 1 September 2025, which would have left the first fold almost empty.
- **Corrected: the testbed is 35 BMUs with 12 lead parties**, counted on disk with the 95% rule
  applied to B1610 and to the PN file alike (the draft's "about 38" counted B1610 only), and 3 of
  them sit in NGED's GSP groups, not 5.
- **Corrected: the false-alarm control allows 2 of 20 price-blind batteries, not 1.** At a
  one-sided 2.5% rate, two or more chance hits in 20 occur about one run in eleven, which would
  have failed a sound instrument too often.
- **Added:** a definition of the persistence value at each issue time, the shuffle's application to
  training rows and to every price-derived column, and a compute-budget check before each rung.

## Risks

- **The ECR's names may not match the BMU register's names**, because the ECR names the customer
  and the site while the BMU register names the unit and its trading party. The census would then
  rest on capacity and GSP group alone, and the unknown band would widen. The recall check
  measures how much.
- **One lead party holds 14 of the 35 testbed BMUs**, so the testbed behaves like fewer batteries
  than its count. D4's neighbour rule excludes the target's own party, and the page repeats D4
  without the largest party.
- **The price model without a demand forecast may be weak.** If its error is above about 0.8 of
  `naive`, D3 measures a weak forecast, and the page says so beside the `actual` bracket.
- **Eleven scored months give few months per fold** (two in fold 0), so every interval rests on 11
  months. The B1610 download could be extended back a year for the testbed; the plan does not do
  so unless the first science review asks.
- **Compute.** About 13 XGBoost arms at 420 fits each, plus the second setting on the planned
  contrasts' arms, is several thousand multi-quantile fits; the per-rung timing check in "Compute
  budget" decides how they are spread across processes.

## Open questions for the maintainer

None. The four questions from the previous draft are recorded above as decisions.
