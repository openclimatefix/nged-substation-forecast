# Methods shared by the past-weather studies

**The past-weather studies score every weather product on the same rows, with the same folds, the
same error measure, and the same intervals, and this page states those methods once.** The studies
are [Which weather product best describes past sunshine?](solar.md), [Which weather product best
describes past wind?](wind.md), and [Does blending weather products beat the best single weather
product?](blending.md). Each study page states only what is specific to that study. Several
sentences below were written for the solar page, so "this page" in a sentence below means the study
page that repeats the sentence.

## Row sets

**Five row sets are used across the studies, and every product in a study is scored on the study's
own row set.** A site-hour is one hour of one generator's metered output. The counts below are the
solar page's.

| Row set | Products scored | First to last hour | Site-hours |
|---|---|---|---|
| Main | The eight products of the ranking | December 2022 to August 2026 (2022-12-01 to 2026-08-31) | 76,727 |
| Extra Open-Meteo models | All 12 products, including four more weather models | November 2024 to August 2026 | 40,243 |
| ECMWF ENS | ENS, with ERA5 and CAMS refit on the same hours | 1 April 2024 to 10 September 2026 | 54,447 |
| Weather stations | The nearest Met Office weather station, with CAMS and ERA5 | 2022-12-01 to 2025-12-31 | 60,033 |
| Record | ERA5, CAMS, SARAH-3, and ICON-DREAM-EU | January 2021 to August 2026 | 115,594 |

**A longer record for two questions.** The year-by-year comparison with ERA5 and SARAH-3's
comparison by satellite use a second row set, built the same way from the four products whose
records reach back to January 2021: ERA5, CAMS, SARAH-3, and ICON-DREAM-EU. That row set holds
115,594 site-hours from January 2021 to August 2026.

## Capacity normalisation

**Every error on this page is a mean absolute error as a percentage of each generator's own 99th
percentile of output, which the page calls its capacity.** That capacity is a statistic of the
metered output, not the generator's registered or export capacity. Each error is that of an XGBoost
model fitted per generator to predict hourly output from one product, so the figures rank how much
each product's sunshine says about the output once that XGBoost model has been fitted.

**One normalisation.** Each hour's error is divided by its own generator's capacity before any mean
or difference.

## Month-block folds

**Each XGBoost model is trained on some blocks of whole months and scored on the others, which it
has never seen; each held-out block is called a fold.**

## Bootstrap intervals

**Intervals from whole months.** The six generators share their weather, so each 95% interval comes
from resampling whole calendar months 2,000 times, each time also drawing one of three XGBoost fits
that differ only in their random seed. This page calls a difference statistically significant at the
5% level when its 95% interval from resampling whole months lies wholly on one side of zero, and not
statistically significant at the 5% level when the interval includes zero. The test covers
month-to-month variation in the weather and the fitting seed only, not variation between generators.
The intervals are not corrected for the number of comparisons, so among the many exploratory rows
some will reach significance by chance.

**The folds are cut by `studies.cross_validation` and the intervals computed by `studies.bootstrap`,
both covered by tests.**

## Planned and exploratory comparisons

A contrast is the difference between two products' errors on the same hours. A comparison is planned
when it was written down before any result existed; every other figure is exploratory, chosen or
added after results were seen. The distinction matters because with many comparisons, about 1 in 20
exploratory rows reaches significance at the 5% level by chance, so an exploratory result is a lead
to follow up rather than a finding. A chart holding both kinds marks each planned row "(planned)". A
chart whose rows are all one kind says so once, in its subtitle.

The ranking rests on four planned contrasts: CAMS against ICON-D2, ICON-EU against ICON-D2, ICON-EU
against UKV, and ICON global against ICON-EU. Two more were written before SARAH-3 and ICON-DREAM-EU
were scored: SARAH-3 against CAMS, and ICON-DREAM-EU against ERA5. Three more planned contrasts,
including ECMWF's 9 km global model against ICON-EU, are scored on the four extra Open-Meteo models'
own, shorter row set: see [HARMONIE-AROME as Open-Meteo serves it trails the ICON model of similar
grid
spacing](solar.md#harmonie-arome-as-open-meteo-serves-it-trails-the-icon-model-of-similar-grid-spacing)
and [ECMWF-IFS-HRES against ICON-EU is not resolved, and IFS-HRES beats
ERA5](solar.md#ecmwf-ifs-hres-against-icon-eu-is-not-resolved-and-ifs-hres-beats-era5). Two further
planned contrasts, ECMWF ENS against ERA5 and against CAMS, are scored on ENS's own row set: see
[ECMWF ENS beats ERA5 and trails CAMS](solar.md#ecmwf-ens-beats-era5-and-trails-cams). Three further
planned contrasts are scored on the weather-station section's own row set: the nearest weather
station against CAMS and against ERA5, and CAMS with the station against CAMS with a shuffled copy
of the station's irradiance. See [The nearest station is a worse input than CAMS and a better input
than ERA5](solar.md#the-nearest-station-is-a-worse-input-than-cams-and-a-better-input-than-era5).

### The 14 planned contrasts

| Row set | Planned contrast |
|---|---|
| Main | CAMS against ICON-D2 |
| Main | ICON-EU against ICON-D2 |
| Main | ICON-EU against UKV |
| Main | ICON global against ICON-EU |
| Main | SARAH-3 against CAMS |
| Main | ICON-DREAM-EU against ERA5 |
| Extra Open-Meteo models | KNMI HARMONIE-AROME against ICON-EU |
| Extra Open-Meteo models | DMI HARMONIE-AROME against ICON-D2 |
| Extra Open-Meteo models | ECMWF-IFS-HRES against ICON-EU |
| ECMWF ENS | ENS's mean of members against ERA5 |
| ECMWF ENS | ENS's mean of members against CAMS |
| Weather stations | The nearest weather station's irradiance and temperature against CAMS |
| Weather stations | The same against ERA5 |
| Weather stations | CAMS with the nearest station against CAMS with a shuffled copy of the station's irradiance |

## The second hyperparameter setting

**A second XGBoost setting is fitted for every planned contrast, every deciding contrast, and every
result near the 5% line.** The second setting is
`studies.cross_validation.SENSITIVITY_HYPER_PARAMETERS`. A second setting shows whether an ordering
belongs to the features or to the settings. A result is near the 5% line when one bound of its 95%
interval lies within 20% of the interval's width from zero. The second setting is dropped for
exploratory arms that are not near the line. Where an arm has no saved fit at the second setting,
the page says so and prints no second-setting number. A page shows the second setting as one table
or as a marker on the chart, never as a doubled chart. A verdict needs both settings to agree, and
where a contrast changes sign or significance under the second setting, the page says so.

## Limits shared by the past-weather studies

**Every accuracy figure is recalibrated per generator.** A product with a large but stable bias
scores well here. The implied-capacity measure is the only evidence on this page about each
product's uncorrected bias.

**The intervals describe these six generators only.** The intervals resample months, not generators,
so they say nothing about how a generator elsewhere would rank the products.

**The comparison is not lead-equal.** The served lead is part of what a consumer receives, so the
as-served ranking answers the consumer's question. A comparison at a held-equal lead is a forecast
comparison, and belongs to
[#810](https://github.com/openclimatefix/nged-substation-forecast/issues/810), the study comparing
weather models for UK power forecasting.
