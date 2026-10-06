# Past-weather studies

**The studies below score how well each weather product describes weather that has already happened,
at the metered generators in Flexpectation's trial area in Lincolnshire.** Each study fits one
XGBoost model per generator and scores it by mean absolute error as a percentage of the generator's
capacity. The parts of this project that read past weather are capacity estimation, training
history, historical features, and disaggregation, and each reads one weather product.

- [Which weather product best describes past sunshine?](solar.md) — a
  satellite retrieval describes past sunshine far better than any weather model tested; among the
  models, ICON-D2 is best as served but its advantage shrinks within hours of each run, and ICON-EU
  no longer beats UKV once UKV's hour is rebuilt from its own snapshots. The page says which product
  each offline consumer — capacity estimation, training history, historical features, and
  disaggregation — should read.
- [Which weather product best describes past wind?](wind.md) — at the
  three metered wind farms, the Met Office's UKV and the German weather service's ICON-D2 describe
  past hub-height wind best of the five products tested and both beat the ERA5 reanalysis, by
  more from April to September in intervals estimated separately for each half of the year; ICON-EU
  also beats ERA5; and ICON global has the largest error of the
  five, about half of its gap to ICON-EU a pair of steps in the wind Open-Meteo's archive serves for
  it at one generator.
- [CERRA's and NORA3's wind against ERA5's](reanalysis-wind.md) — at the three wind farms, an
  XGBoost model given CERRA's 100 m wind has a higher error than an XGBoost model given ERA5's,
  mostly at one farm, and an XGBoost model given NORA3's 100 m wind does not differ clearly from
  ERA5's. The page does not rank CERRA against NORA3, and ICON-D2 has the lowest estimated error in
  both blocks.
- [Does blending CERRA's wind heights beat one height?](cerra-wind-levels.md) — at three wind
  farms, an XGBoost model given CERRA's 10 m and 100 m wind speeds has lower error than one given
  100 m alone, and the pair gives most of the gain of using four or five heights. CERRA is a
  reanalysis, so the page says nothing about forecast skill.
- [Does CERRA's wind direction add to its wind speed?](cerra-wind-direction.md) — at three wind
  farms, an XGBoost model given CERRA's wind direction as well as its speed has about 5% lower error
  than one given the speed alone, at 100 m and at 10 m, and the two heights do equally well. Direction
  at all five heights adds no more than about 0.005 points beyond direction at 100 m, but the study's
  check of its own sensitivity to veer was weak. CERRA is a reanalysis, so the page says nothing about
  forecast skill.
- [Does blending weather products beat the best single weather product?](blending.md) — at the six
  solar farms and three wind farms, an XGBoost model given several weather products at once beats an
  XGBoost model given the best single product with its neighbouring hours, by 0.13 points of
  capacity for solar and 0.48 for wind. The gain comes from the other products' weather rather than
  from the extra columns. A blend of UKV and ICON-EU that a live service could read beats UKV alone,
  and an XGBoost blend beats a linear stack of single-product predictions. An XGBoost model given
  CAMS's split plus SARAH-3's global irradiance, the two satellite retrievals the past-solar study
  compared, beats CAMS's split with its neighbouring hours by 0.18 points and plain CAMS's split by
  0.20 points, on a longer row set from January 2021.
- [Does CEDA's UKV or ERA5 describe past wind and temperature better?](ukv-ceda-vs-era5.md) — at
  three wind farms, an XGBoost model given ERA5's wind has a power error 0.125 percentage points of
  capacity lower than an XGBoost model given the 6-hourly UKV archive from CEDA. The difference is
  statistically significant and below the planned 0.16-point margin, so the planned rule gives ERA5.
  The gap grows with the archive's lead, and the gap does not shrink when both XGBoost models read
  10 m wind alone. At four Met Office stations the archived UKV's temperature is 0.124 K closer than
  ERA5's, so the planned rule gives UKV, but the advantage falls from 0.250 K at lead 0 to 0.035 K
  at lead 5. At six solar farms the choice of temperature moves power error by no more than 0.009
  points (95% interval, whole row set).

**The [Methods page](methods.md) states what the studies share.** The Methods page holds the
row sets and their site-hours, the capacity normalisation, the month-block folds, the bootstrap
intervals, the rule for planned and exploratory comparisons with the list of planned contrasts, the
second hyperparameter setting, and the limits every study shares.
