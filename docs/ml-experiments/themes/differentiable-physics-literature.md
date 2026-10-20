# Differentiable physics: literature scan

**This page holds bare-bones notes from a small literature scan, made on 2026-10-20 to inform the
design of the differentiable-physics experiments.** A Sonnet agent read 16 papers in full. Every
number below comes from the full text of the paper named, and all the papers are held in the
project's private library folder `literature/papers/`. The notes are a working record, not a
reviewed survey. The project's main review is [The current state of the art in energy
forecasting](../../background/energy-forecasting-review.md). "nRMSE", "nRPS" are the papers' own
normalised metrics.
Scope caveat for all GAM results: they are regional or national load (GB GSP Groups, France, US
cities), day-ahead, not Primary substations.

## 1. Tricks we're missing (ranked)

1. **Put the Kalman state on the coefficients of the frozen GAM effects, with learned diagonal
   noise,
   not only on a level plus anomaly.**
    - Paper: [de Vilmarest et al. (2024)](https://doi.org/10.1109/TPWRS.2023.3310280), Sec. II-B,
      Table I. Freeze the fitted effects, standardise them, and let the weight vector theta_t of
      (effects, intercept) follow a random walk. Q is diagonal, found by greedy likelihood (R
      package
      `viking`).
    - Gain (14 GB GSP Groups, 2021, nRMSE): Kalman-dynamic 0.332 vs offline GAM 0.635, GAM refitted
      yearly 0.387, GAM refitted daily 0.377, Kalman with Q=0 (static) 0.380.
      So: Q=0 only matches daily refits; the learned Q is what wins (about 12% better than daily
      refitting).
    - Second source: [de Vilmarest and Goude (2022)](https://arxiv.org/abs/2110.00334), Table III
      (MAE in MW, competition month): linear model offline 22.8, dynamic 14.9, dynamic with Q set
      big and proportional to I 11.2, VIKING 11.5. Q proportional to I was best there but not in the
      pre-competition data, so the Q choice depends on how much the data drifts.
    - Slot-in: after the convex fit, freeze the fitted columns (daily profile, heating, solar,
      holiday). Run a Kalman filter whose state is one multiplier per block of hours on each
      column. [Experiment
      106](../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md) filtered
      only level and anomaly; this lets heating sensitivity and PV scale
      adapt.
    - Cost: CPU seconds per site per year for the filter; minutes to fit Q on a training year.
2. **Let the Kalman state learn capacities, so the explicit capacity input matters less.**
    - Paper: de Vilmarest et al. (2024), Sec. IV-D. Remove the embedded solar and wind capacity
      inputs: offline nRMSE rises by more than 10%; Kalman-adapted nRMSE falls by 0.4%. Scope: 2019,
      GB GSP Groups.
    - Slot-in: our convex PV capacity is already non-decreasing; a PV-scale multiplier in trick 1
      gives a cheap second estimate, and the gap between the two flags capacity steps.
    - Cost: free once trick 1 exists.
3. **Fit the quantile regression on the Kalman residuals (not the offline residuals), with the GAM
   prediction, its square, and hour-of-day as covariates; optionally adapt it online.**
    - Paper: de Vilmarest et al. (2024), Table II (nRPS, 2019 / 2020 / 2021): offline GAM + offline
      quantile regression 0.231 / 0.338 / 0.454; Kalman Gaussian quantiles 0.212 / 0.217 / 0.222;
      Kalman + offline quantile regression 0.206 / 0.214 / 0.217; Kalman + online gradient descent
      with step size chosen by Bernstein online aggregation (BOA) 0.202 / 0.201 / 0.209.
    - Reading: nearly all the gain comes from filtering the mean (0.231 to 0.206 in 2019). The extra
      from the online quantile step is 2 to 6%.
    - Slot-in: replace our residual-quantile spread by the covariate quantile regression. Online
      gradient descent on the pinball loss is five lines. Cost: CPU seconds.
4. **Covariate-dependent Generalised Pareto tails for the extreme quantiles.**
    - Paper: [Browell and Fasiolo (2021)](https://arxiv.org/abs/2103.10335), Sec. V-B and Table IV.
      Quantile regression on GAM residuals for the body; GPD beyond the 1st and 99th percentile with
      a scale that depends on covariates. Plain quantile regression is not calibrated beyond 1% and
      99%. The conditional GPD cut upward reserve by up to 24.6% and downward by up to 10.8% against
      a static GPD (probability levels 0.01% to 0.25%).
    - Fit in `mgcv` (R); in Python it needs a small custom maximum-likelihood fit. Fits our
      curtailment-tail goal. Cost: CPU minutes per site.
5. **Learn the observation variance online (VIKING) rather than a fixed sigma.**
    - Paper: [de Vilmarest (2022)](https://arxiv.org/abs/2104.10777); used in de Vilmarest and
      Goude (2022) Table III (VIKING 11.5 vs dynamic 14.9 on the linear model; no better than
      hand-set big Q). Gain is modest and mainly saves tuning. Cost: CPU seconds.
6. **Exponentially smoothed temperatures with two time constants, plus smoothed daily min and max.**
    - [Obst et al. (2021)](https://arxiv.org/abs/2009.06527), Sec. 3: GAM terms f(Temp95),
      f(Temp99),
      f(TempMin99, TempMax99), f(t, Temp), factors per step on half-hourly data. Also the
      GEFCom2014 winner's smoothed temperature, alpha = 0.95
      ([Fasiolo et al. (2021)](https://arxiv.org/abs/1707.03307), Sec. 7). No paper isolates the
      gain.
    - Slot-in: add two decayed columns (our 6 h to 1 week means are the same idea) plus smoothed
      daily extremes. Cost: minutes.
7. **Detrend the target before the tree model.**
    - [Rubattu et al. (2023)](https://arxiv.org/abs/2305.05575), Sec. 3: fit a linear trend in the
      training fold, subtract it, fit LightGBM, add it back. Shown in Fig. 3 as zero-mean residuals;
      no number. Same paper: temporal-hierarchy combination improved CRPS about 5% and interval
      score about 10% (BigDEAL 2022 data, not public). For the XGBoost side of the blend only.
8. **Intraday autoregressive correction of the residuals, one model per horizon.**
    - de Vilmarest and Goude (2022), Sec. III-B and Table II: offline GAM MAE 20.7 to 19.3 MW in a
      stable period; MLP 57% lower in the unstable year. Likely already covered by our anomaly
      state; check before building.

## 2. Our current choices against the literature

- **Fixed 15.5 C hinge:** no paper found uses a hinge. All the GAM papers above fit a free smooth
  of temperature (cubic spline) and let the data find the threshold
  ([Fasiolo et al. (2021)](https://arxiv.org/abs/1707.03307) f(T) with rank 15; Obst et al. (2021)).
  So the
  literature neither supports nor contradicts a fixed threshold; a spline would be the check, and
  the learned-threshold plan in Part 6 is the nearer middle road.
- **Smoothed temperatures (6 h, 1 d, 3 d, 1 week):** supported. Every EDF-line paper uses
  exponential smoothing at two time constants (0.95 and 0.99 per step); see trick 6. With
  half-hourly steps those are roughly 10 h and 50 h (my arithmetic, not stated in the paper).
- **Monthly drift knots:** weakly supported. The literature's answer to level shifts is the Kalman
  coefficient state, which beat yearly and daily refits (trick 1). The finding of [experiment
  106](../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md) (drift adds
  nothing once the filter runs) agrees. No paper tested monthly spline knots.
- **Kalman level and anomaly states:** supported, but narrower than the literature's state (trick
  1).
  Adapting only the level discards the slow-changing heating and PV sensitivities that de Vilmarest
  et al. found drifted.
- **Residual-quantile spread (per block of hours and quarter):** supported as a first step
  (Table II: offline quantile regression on filtered residuals already gives most of the gain),
  but poorly calibrated in the tails (Browell and Fasiolo: beyond 1% and 99%).
- **Convex, non-negative, non-decreasing PV capacity:** no paper found tests monotonic capacity.
  de Vilmarest et al. let the capacity coefficient move freely and it learned embedded capacity
  (trick 2). [Tang et al. (2024)](https://doi.org/10.1016/j.segan.2024.101396) estimate capacity
  with
  a neural network from 24 h net-load curves, so a convex physical fit is a different route.

## 3. Hierarchy

- **Reconciliation barely helps the bottom level.** [Nespoli et
  al.](https://arxiv.org/abs/1910.03976)
  (Swiss low-voltage grid, 24 bottom series): minT with graphical-lasso covariance cut aggregate
  RMSE
  by up to 10% but bottom-level RMSE by under 1%; minT with Ledoit-Wolf shrinkage made the bottom
  level worse. Their suggested fix: reconcile conditionally on the previous steps' forecast errors.
- **Projection helps little when every node uses the same model.**
  [Brégère and Huard (2022)](https://arxiv.org/abs/2003.00585) (GB household populations, 3-month
  test,
  errors summed over all nodes): projection alone cut error about 1% for GAM, 5% for random forest;
  adding an online expert-aggregation step cut it a further 10 to 15%. Read: the gain came from
  mixing in other nodes' forecasts as experts, not from the constraint.
- **Probabilistic reconciliation:** [Jeon et al. (2019)](https://arxiv.org/abs/1808.02635), temporal
  hierarchy, two Crete wind farms: reconciling ranked (not stacked) samples improved CRPS; best
  method cut 24 h CRPS by 26.6% and 21.1% at the two farms. Temporal, not across sites, so only
  indirect support for our sums.
- **No paper found that tested ensemble copula coupling, the Schaake shuffle, or the variogram score
  on substation demand; none found for soft hierarchical losses or hierarchical Bayesian load
  models with measured bottom-level gains.** Searches: arXiv API titles and our library. Not
  exhaustive.
- **Practical reading:** for Primaries (bottom), expect under 1 to 5% from reconciliation. The
  finding of [experiment 36](../00036-hierarchy-aware-bsp-gsp-forecasts/result.md)
  finding (child forecast error is the limit) agrees. The larger payoff the literature shows is for
  BSP and GSP aggregates, and from online expert mixing.

## 4. Solar disaggregation

- **Identification by regression on an irradiance proxy plus a load proxy.**
  [Kara et al. (2018)](https://www.sciencedirect.com/science/article/pii/S2352467717301169): net
  active power regressed on measured reactive power (load proxy) and a nearby irradiance reading,
  daytime only, with a correction for capacitor-bank switches (threshold 90 kvar per phase).
  Recovered a 7.5 MWp site with RMSE about 450 kW (about 6%); validated against the site's own
  meter.
  Our analogue: MVAr, where NGED telemetry has it.
- **Unsupervised, validated on metered PV.**
  [Sossan et al. (2019)](https://arxiv.org/abs/1706.04821): PV modelled as capacity-weighted sum
  over
  21 tilt/azimuth planes from transposed GHI; four estimators; best (frequency-band method) nRMSE
  4.6 to 5.6% for 4 Swiss houses (35.3 kWp) with ground-truth PV meters. Also tested at 50% and 25%
  of capacity.
- **Neighbour controls:** [Cheung et al. (2023)](https://doi.org/10.1109/TSUSC.2022.3192456) use
  neighbouring customers without PV as the load model; MAE 28.37% below a prior disaggregation
  model (Ausgrid, 300 customers). Needs known no-PV customers; at Primary level the equivalent is
  nearby Primaries with little PV.
- **Capacity from net-load shape.** Tang et al. (2024): a model trained on summer data only had
  18.1%
  lower RMSE than an all-year model with a third of the training data; Monday to Thursday only
  gave 26.9% lower RMSE; error about 10% once more than 20% of households have PV. Supports fitting
  PV capacity on the summer or shoulder windows only, and averaging curves (doubling averaged
  curves cut RMSE 25.1%, 17.5 to 13.1 kW).
- **Validation pattern across the papers:** a metered subset gives ground truth; sensitivity
  to sampling time and penetration; none validates at Primary scale against known capacity. A
  synthetic known-answer injection (as in the gate of [experiment
  109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md)) is stricter than
  what they did.
- **Not read in full (so no claims):** Kanchana et al. (2026) GAN data curation (paywalled page
  only).

## 5. Papers saved (all in `literature/papers/`)

Newly downloaded:

- `fasiolo-2021-fast-calibrated-additive-quantile-regression.pdf`: smoothed pinball loss and
  calibrated learning rate; beats boosting and the GEFCom2014 winner on EDF data.
- `de-vilmarest-2022-viking-variational-bayesian-variance-tracking.pdf`: online noise-variance
  learning for the Kalman filter.
- `bregere-2022-online-hierarchical-forecasting-power-consumption.pdf`: aggregation plus projection
  on a hierarchy of GB households.
- `jeon-2019-reconciliation-probabilistic-forecasts-wind-power-arxiv.pdf`: probabilistic temporal
  reconciliation.

Already held and read in full for this scan:

- `de-vilmarest-2024-adaptive-probabilistic-net-load.pdf`: the main source (tricks 1 to 3).
- `de-vilmarest-2021-state-space-models-win-dataport-competition.pdf`: Q choices and intraday
  correction.
- `obst-2020-adaptive-methods-covid-lockdown-france.pdf`: GAM temperature terms; Kalman variants.
- `browell-fasiolo-2021-regional-net-load.pdf`: GAM against linear benchmark (pinball skill over 10%
  better for GAM-Point than a Vanilla linear benchmark, all 14 GSP Groups); conditional GPD tails.
- `nespoli-2022-hierarchical-demand-forecasting-benchmark.pdf`,
  `rubattu-2023-probabilistic-lightgbm-temporal-hierarchies.pdf`: reconciliation and temporal
  hierarchies.
- `kara-2018-...`, `sossan-2018-...`, `cheung-2023-...`,
  `sensitivity-2024-distributed-pv-capacity-estimation-ann.pdf`: solar disaggregation.
- `hertel-2026-load-forecasting-across-grid-levels-v2.pdf`: monthly retraining helps; no hierarchy
  content.

Not found open (searched arXiv titles only): Wickramasuriya et al. minT paper, Ben Taieb et al.
smart
meter hierarchy, Rangapuram et al. coherent probabilistic forecasts.
