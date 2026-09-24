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
- [Which weather product best describes past sunshine?](weather-products-for-past-solar.md) — a
  satellite retrieval describes past sunshine far better than any weather model tested; among the
  models, ICON-D2 is best as served but its advantage shrinks within hours of each run, and ICON-EU
  no longer beats UKV once UKV's hour is rebuilt from its own snapshots. The page says which product
  each offline consumer — capacity estimation, training history, historical features, and
  disaggregation — should read.
- [Which weather product best describes past wind?](weather-products-for-past-wind.md) — at the
  three metered wind farms, the Met Office's UKV and the German weather service's ICON-D2 describe
  past hub-height wind best of the five products tested and both beat the ERA5 reanalysis, mostly
  from April to September; ICON-EU also beats ERA5; and ICON global has the largest error of the
  five, about half of its gap to ICON-EU a pair of steps in the wind Open-Meteo's archive serves for
  it at one generator.
- [Does blending weather products beat the best single weather
  product?](blending-weather-products.md) — at the six solar farms and three wind farms, an XGBoost
  model given several weather products at once beats one given the best single product with its
  neighbouring hours: by 0.13 points of capacity for solar and 0.48 for wind. The gain comes from the
  other products' weather rather than from the extra columns, a blend of UKV and ICON-EU that a live
  service could read beats UKV alone, and an XGBoost blend beats a linear stack of single-product
  predictions. Blending the two satellite retrievals the past-solar study compared, CAMS and
  SARAH-3, beats CAMS's split with its own neighbouring hours by 0.18 points and plain CAMS's split
  by 0.20 points, even though both products take their clouds from satellite images rather than
  from a weather model.
- [How accurate is a power forecast driven by ECMWF ENS at each
  horizon?](ens-forecast-horizons.md) — at the 6 solar farms and 3 wind farms, an XGBoost model
  given the ENS ensemble mean beats every forecast that reads no weather forecast to day 5 for solar
  and day 7 for wind; from day 7 for solar and day 10 for wind it no longer beats climatology, and by
  day 14 climatology is ahead, with the ensemble mean adding no statistically significant skill, in a
  post hoc check, over the same model given no weather at all. The ensemble mean beats the control
  member and beats training on every member, and rebuilding solar radiation through the clear-sky
  index beats the straight-line resample the live service uses today.
