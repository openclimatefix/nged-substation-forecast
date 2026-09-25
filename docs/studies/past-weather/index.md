# Past-weather studies

**Three studies score how well each weather product describes weather that has already happened, at
the metered generators in Flexpectation's trial area in Lincolnshire.** Each study fits one XGBoost
model per generator and scores it by mean absolute error as a percentage of the generator's
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
  past hub-height wind best of the five products tested and both beat the ERA5 reanalysis, mostly
  from April to September; ICON-EU also beats ERA5; and ICON global has the largest error of the
  five, about half of its gap to ICON-EU a pair of steps in the wind Open-Meteo's archive serves for
  it at one generator.
- [Does blending weather products beat the best single weather product?](blending.md) — at the six
  solar farms and three wind farms, an XGBoost model given several weather products at once beats an
  XGBoost model given the best single product with its neighbouring hours, by 0.13 points of
  capacity for solar and 0.48 for wind. The gain comes from the other products' weather rather than
  from the extra columns. A blend of UKV and ICON-EU that a live service could read beats UKV alone,
  and an XGBoost blend beats a linear stack of single-product predictions. An XGBoost model given
  CAMS's split plus SARAH-3's global irradiance, the two satellite retrievals the past-solar study
  compared, beats CAMS's split with its neighbouring hours by 0.18 points and plain CAMS's split by
  0.20 points, on a longer row set from January 2021.

**The [Methods page](methods.md) states what the three studies share.** The Methods page holds the
row sets and their site-hours, the capacity normalisation, the month-block folds, the bootstrap
intervals, the rule for planned and exploratory comparisons with the list of planned contrasts, the
second hyperparameter setting, and the limits every study shares.
