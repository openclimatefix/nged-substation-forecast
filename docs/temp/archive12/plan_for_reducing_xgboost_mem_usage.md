Our task is to figure out why tests/test_xgboost_dagster_integration.py is taking up so much RAM
that it crashes my laptop (which has 32 GB of RAM). This same test ran fine at the end of yesterday.
It seems that something we've done today (see git history) has increased RAM usage.

I've captured the log whilst running the test, and whilst watching RAM usage. The logs show that
training XGBoost doesn't use much RAM. And, for the first minute or so, the evaluation step also
doesn't use that much RAM. But then the RAM spikes about 1 minute after starting the evaluation
step. I'm wondering if the output from running inference is being pickled to pass it to the plotting
script?

## Tasks

- Please add lots of logging to tests/test_xgboost_dagster_integration.py. I want to know when each
step (training, inference, plotting, etc.) starts and ends.
- Look at whether any large dataframes are going to be pickled by Dagster.
- Check that we're training on just the control NWP member, and performing inference on all 51 NWP
  ensemble members, for all 4 `time_series_ids`.
- Make sure that we're not doing anything insane like trying to run inference on all 33
  `time_series_ids` in our dataset. We should only be running inference on 4.
- Are we running inference on all 4 `time_series_id`s at once?


## The tail of the logs during `uv run pytest -s tests/test_xgboost_dagster_integration.py`

2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - cleaned_actuals - Cleaned data shape after cleaning: (51737, 3)
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - cleaned_actuals - Validated data shape: (51737, 3)
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - cleaned_actuals - Filtered cleaned actuals to current partition. Partition range: [2026-03-26 00:00:00+00:00, 2026-03-27 00:00:00+00:00). Data shape: (13556, 3)
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - cleaned_actuals - Saved cleaned actuals to Delta table at data/NGED/delta/cleaned_actuals
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - cleaned_actuals - STEP_OUTPUT - Yielded output "result" of type "DataFrame". (Type check passed).
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - cleaned_actuals - Writing file at: /tmp/pytest-of-jack/pytest-61/test_xgboost_dagster_integrati0/cleaned_actuals/2026-03-26 using CompositeIOManager...
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - cleaned_actuals - ASSET_MATERIALIZATION - Materialized value cleaned_actuals.
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - cleaned_actuals - HANDLED_OUTPUT - Handled output "result" using IO manager "io_manager"
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - cleaned_actuals - STEP_SUCCESS - Finished execution of step "cleaned_actuals" in 121ms.
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - processed_nwp_data - STEP_START - Started execution of step "processed_nwp_data".
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - processed_nwp_data - Loading file from: /tmp/pytest-of-jack/pytest-61/test_xgboost_dagster_integrati0/all_nwp_data using CompositeIOManager...
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - processed_nwp_data - LOADED_INPUT - Loaded input "all_nwp_data" using input manager "io_manager", from output "result" of step "all_nwp_data"
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - processed_nwp_data - STEP_INPUT - Got input "all_nwp_data" of type "LazyFrame". (Type check passed).
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - processed_nwp_data - processed_nwp_data config: time_series_ids=[16, 27, 31, 6] start_date='2026-01-28' end_date='2026-03-26'
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - processed_nwp_data - all_nwp_data schema: ['h3_index', 'ensemble_member', 'pressure_surface', 'geopotential_height_500hpa', 'wind_u_10m', 'precipitation_surface', 'wind_u_100m', 'wind_v_10m', 'temperature_2m', 'dew_point_temperature_2m', 'downward_long_wave_radiation_flux_surface', 'downward_short_wave_radiation_flux_surface', 'wind_v_100m', 'pressure_reduced_to_mean_sea_level', 'categorical_precipitation_type_surface', 'init_time', 'valid_time']
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - processed_nwp_data - STEP_OUTPUT - Yielded output "result" of type "LazyFrame". (Type check passed).
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - processed_nwp_data - Writing file at: /tmp/pytest-of-jack/pytest-61/test_xgboost_dagster_integrati0/processed_nwp_data using CompositeIOManager...
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - processed_nwp_data - ASSET_MATERIALIZATION - Materialized value processed_nwp_data.
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - processed_nwp_data - HANDLED_OUTPUT - Handled output "result" using IO manager "io_manager"
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - processed_nwp_data - STEP_SUCCESS - Finished execution of step "processed_nwp_data" in 28ms.
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - train_xgboost - STEP_START - Started execution of step "train_xgboost".
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - train_xgboost - Loading file from: /tmp/pytest-of-jack/pytest-61/test_xgboost_dagster_integrati0/processed_nwp_data using CompositeIOManager...
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - train_xgboost - LOADED_INPUT - Loaded input "nwp" using input manager "io_manager", from output "result" of step "processed_nwp_data"
2026-04-11 14:42:01 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - train_xgboost - STEP_INPUT - Got input "nwp" of type "LazyFrame". (Type check passed).
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - train_xgboost - Reading cleaned actuals from data/NGED/delta/cleaned_actuals
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - train_xgboost - Filtered requested substations to 4 healthy ones.
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - train_xgboost - time_series_metadata_filtered shape: (4, 14)
2026-04-11 14:42:01 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - train_xgboost - sub_ids: [16, 27, 31, 6]
2026/04/11 14:45:45 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/11 14:45:45 INFO mlflow.utils.uv_utils: Detected uv project: found uv.lock and pyproject.toml in /home/jack/dev/python/nged-substation-forecast
2026/04/11 14:45:46 INFO mlflow.utils.uv_utils: Detected uv project: found uv.lock and pyproject.toml in /home/jack/dev/python/nged-substation-forecast
2026/04/11 14:45:46 INFO mlflow.utils.environment: Detected uv project at /home/jack/dev/python/nged-substation-forecast. Attempting to export requirements via 'uv export'.
2026/04/11 14:45:46 INFO mlflow.utils.uv_utils: Exported 172 dependencies via uv
2026/04/11 14:45:46 INFO mlflow.utils.environment: Successfully exported 172 requirements from uv project. Skipping package capture based inference.
2026/04/11 14:45:46 WARNING mlflow.utils.environment: Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - train_xgboost - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - train_xgboost - Writing file at: /tmp/pytest-of-jack/pytest-61/test_xgboost_dagster_integrati0/train_xgboost using CompositeIOManager...
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - train_xgboost - ASSET_MATERIALIZATION - Materialized value train_xgboost.
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - train_xgboost - HANDLED_OUTPUT - Handled output "result" using IO manager "io_manager"
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - train_xgboost - STEP_SUCCESS - Finished execution of step "train_xgboost" in 3m45s.
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - evaluate_xgboost - STEP_START - Started execution of step "evaluate_xgboost".
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - evaluate_xgboost - Loading file from: /tmp/pytest-of-jack/pytest-61/test_xgboost_dagster_integrati0/train_xgboost using CompositeIOManager...
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - evaluate_xgboost - LOADED_INPUT - Loaded input "model" using input manager "io_manager", from output "result" of step "train_xgboost"
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - evaluate_xgboost - Loading file from: /tmp/pytest-of-jack/pytest-61/test_xgboost_dagster_integrati0/processed_nwp_data using CompositeIOManager...
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - evaluate_xgboost - LOADED_INPUT - Loaded input "nwp" using input manager "io_manager", from output "result" of step "processed_nwp_data"
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - evaluate_xgboost - STEP_INPUT - Got input "model" of type "XGBoostForecaster". (Type check passed).
2026-04-11 14:45:46 +0100 - dagster - DEBUG - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - 107646 - evaluate_xgboost - STEP_INPUT - Got input "nwp" of type "LazyFrame". (Type check passed).
2026-04-11 14:45:46 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - evaluate_xgboost - Reading cleaned actuals from data/NGED/delta/cleaned_actuals
2026-04-11 14:45:46 +0100 - dagster - INFO - xgboost_integration_job - 0a39fbd8-324c-41c5-88a0-553f813d030f - evaluate_xgboost - Filtered requested substations to 4 healthy ones.
