Please implement a simple XGBoost model to forecast the power flowing through NGED's primary
electricity substations. Please put your implementation in @packages/xgboost_forecaster/.

ECMWF ensemble weather forecasts can be found in @packages/dynamical_data/data/. These are in tabular
format, using H3 resolution 5 discrete spatial index.

The substation power data can be found in @data/NGED/parquet/live_primary_flows/. This data follows
the schema described in @packages/contracts/src/contracts/data_schemas.py.

You can find how to get the the lat/lngs of each substation by looking at the dashboard in
@packages/dashboard/main.py.

Please train a very simple model on a handful of substations. Please check that the substation data
looks sane. Please write the code to match up the substation lat/lns with the H3 indicies in the
ECMWF data.

Please note that the power data only goes back a few weeks, even though the weather data goes back
2 years.
