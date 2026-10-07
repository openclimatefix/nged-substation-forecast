# Studies

This section holds the **findings of experiments we have run**, written up so the conclusion
survives the code that produced it. It complements the other sections:
[background](../background/index.md) describes the *problem*, [techniques](../techniques/index.md)
explains the *methods*, the [roadmap](../roadmap/index.md) says what we plan to build, and
[architecture](../architecture/overview.md) documents what is already built.

**A page lands here when a study answers a question that outlives its code.** The code behind a page
here is deliberately lighter than production code — no Dagster asset, no data contract, and no
maintenance promise — and lives in
[`studies/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies),
which merges so that the measurement can be audited and re-run. What the project keeps is the
answer, which is why each page states its numbers, the checks the result survived, and the limits on
how far it generalises.

**Every study here uses only Flexpectation's own data, and its purpose is to inform Flexpectation's
choices.** Each study scores its weather products or methods only at the generators in
Flexpectation's trial area in Lincolnshire. No study here compares results across many regions or
climates, so a result may not hold elsewhere.

- [Does a weather product's beam/diffuse split help a PV forecast?](beam-diffuse-split.md) — on a 5
  km satellite retrieval the published direct-beam field cuts photovoltaic (PV) power error by 1.8%
  beyond what a separation model recovers from the total irradiance; on a 31 km reanalysis no effect
  is detected at the setting named before the run; and choosing the better irradiance product
  matters about 34 times more than the split does. The page also measures the two irradiance
  products against each other, a gradient-boosted tree against a fitted five-parameter physical PV
  model, what calibrating that physical model with a tree recovers, and what the per-site error
  drift says about estimating a generator's effective capacity. It reads NGED's two records of
  active network management against each other and against the telemetry, and finds the setpoint
  history the one to build on.

- [How many Balancing Mechanism Units in Great Britain are solar?](solar-bmu-census.md) — a census
  of the BMUs whose settled output follows the sun, found because no field in the BMU register says
  which units are solar. The study finds 10 single-site solar BMUs, 8 of them at sites that also
  hold storage, reports 28 aggregate BMUs apart, and sets the five published capacity figures
  side by side.

## Past weather

**Three studies score how well each weather product describes weather that has already happened.**
The [Past-weather overview](past-weather/index.md) summarises them, and the [Methods
page](past-weather/methods.md) states the methods they share.

- [Which weather product best describes past sunshine?](past-weather/solar.md) — a
  satellite retrieval describes past sunshine far better than any weather model tested; among the
  models, ICON-D2 is best as served but its advantage shrinks within hours of each run, and ICON-EU
  no longer beats UKV once UKV's hour is rebuilt from its own snapshots. The page says which product
  each offline consumer — capacity estimation, training history, historical features, and
  disaggregation — should read.
- [Which weather product best describes past wind?](past-weather/wind.md) — at the
  three metered wind farms, the Met Office's UKV and the German weather service's ICON-D2 describe
  past hub-height wind best of the five products tested and both beat the ERA5 reanalysis, by
  more from April to September in intervals estimated separately for each half of the year; ICON-EU
  also beats ERA5; and ICON global has the largest error of the
  five, about half of its gap to ICON-EU a pair of steps in the wind Open-Meteo's archive serves for
  it at one generator.
- [Does blending weather products beat the best single weather
  product?](past-weather/blending.md) — at the six solar farms and three wind farms, an XGBoost
  model given several weather products at once beats one given the best single product with its
  neighbouring hours: by 0.13 points of capacity for solar and 0.48 for wind. The gain comes from
the
  other products' weather rather than from the extra columns, a blend of UKV and ICON-EU that a live
  service could read beats UKV alone, and an XGBoost blend beats a linear stack of single-product
  predictions. An XGBoost model given CAMS's split plus SARAH-3's global irradiance, the two
  satellite retrievals the past-solar study compared, beats CAMS's split with its neighbouring hours
  by 0.18 points and plain CAMS's split by 0.20 points, on a longer row set from January 2021.
- [Does CEDA's UKV or ERA5 describe past wind and temperature
  better?](past-weather/ukv-ceda-vs-era5.md) — at three wind farms, an XGBoost model given ERA5's
  wind has a power error 0.125 percentage points of capacity lower than an XGBoost model given the
  6-hourly UKV archive from CEDA. The difference is statistically significant and below the planned
  0.16-point margin, so the planned rule gives ERA5. The gap grows with the archive's lead, and the
  gap does not shrink when both XGBoost models read 10 m wind alone. At four Met Office stations the
  archived UKV's temperature is 0.124 K closer than ERA5's, so the planned rule gives UKV, but the
  advantage falls from 0.250 K at lead 0 to 0.035 K at lead 5. At six solar farms the choice of
  temperature moves power error by no more than 0.009 points (95% interval, whole row set).
- [Do CEDA's and Open-Meteo's archives of UKV give the same power forecasts?](past-weather/ukv-ceda-vs-openmeteo.md)
  — at the nine metered farms over 23 months, the two archives agree closely at the start of each run
  (a mean absolute temperature difference of 0.098 K, and Open-Meteo's wind speed about 3% lower).
  An XGBoost model of wind power trained on CEDA's archive loses 0.441 points of capacity [+0.286,
  +0.600] when given Open-Meteo's wind, a level bias that rescaling the speed mostly removes. CEDA's
  larger power error over all hours (wind +0.270 points once two spans of Open-Meteo's wind are
  dropped) is mostly its longer leads: at lead 0 the wind gap is +0.030 [-0.043, +0.122]. By the
  plan's rule the two archives are not mixed. The page does not compare CEDA's archive with the Met
  Office's live feed.

## Forecasts

**Five studies score power forecasts driven by weather forecasts.**

- [How accurate is a power forecast driven by ECMWF ENS at each
  horizon?](forecasts/ens-horizons.md) — at the 6 solar farms and 3 wind farms, an XGBoost model
  given the ENS ensemble mean beats every forecast that reads no weather forecast to day 5 for solar
  and day 7 for wind; from day 7 for solar and day 10 for wind it no longer beats climatology, and
by
  day 14 climatology is ahead, with the ensemble mean adding no statistically significant skill, in
a
  post hoc check, over the same model given no weather at all. The ensemble mean beats the control
  member and beats training on every member, and rebuilding solar radiation through the clear-sky
  index beats the straight-line resample the live service uses today at days 0 to 3.
- [Do other weather forecasts beat ECMWF's ensemble at day-ahead lead?](forecasts/matched-lead.md)
  — at the 6 solar farms and 3 wind farms, and reading each product as the study does, ECMWF's
  ensemble mean beats UKV, ICON-EU, and GEFS at matched lead for solar; for wind it beats UKV and
  GEFS, and ICON-EU cannot be separated from it. A blend of ENS, ICON-EU, and IFS 0.25° lowers the
  wind error but shows no detectable solar gain at a lead a 09:00 UTC service could use.
- [Which single weather forecast should be added to ECMWF's ensemble mean
  first?](forecasts/blends-with-ens.md) — at the 6 solar farms and 3 wind farms, adding ICON-EU at the
  optimistic lead (exploratory; an upper bound on a live service's gain) lowers the wind error at
  day 1 by 0.58 points of capacity [0.48, 0.69]; AIFS Single
  is the one product the page can name for solar, at day 2, by 0.33 [0.18, 0.48]. Elsewhere AIFS
  Single and ICON-EU are not separable.
- [Does adding UKV from CEDA to ECMWF's ensemble mean lower the error at lead days 1 to
  4?](forecasts/ukv-ceda-blends.md) — at the 3 wind farms, adding UKV-CEDA lowers the error at days
  1 and 2 by 0.24 to 0.27 and 0.19 to 0.21 points of capacity, at both hyperparameter settings,
  against both shuffled controls, after the Bonferroni correction, and with any one month dropped.
  At wind day 3 the gain rests on February 2026. At the 6 solar farms the gain is about 0.1 points
  at days 1 to 3, unresolved under the planned rule at days 1 and 2. Day 4 is inconclusive.
- [How do Open-Meteo's ensemble-mean products compare for solar and wind
  power?](forecasts/ensemble-means.md) — over 88 summer days at the 6 solar farms and 3 wind farms,
  ICON-D2-EPS's ensemble mean gave the lowest power error of the four Open-Meteo ensemble means,
  7.4% of capacity for solar against 7.7% for ECMWF ENS's mean, and 5.75% for wind at 10 m, level
  with ECMWF ENS's mean. The comparison is descriptive and uses stitched series that carry no run
  time.
