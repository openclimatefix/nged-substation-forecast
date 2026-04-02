# In `conf/model/xgboost.yaml`
- Rename `latest_available_weekly_lag` to `latest_available_weekly_power_lag`

# In `packages/dynamical_data/src/dynamical_data/processing.py`
- Please use a Patito data contract for the return type of `process_ecmwf_dataset`, and validate the
  DataFrame before returning it. In some ways, this is the single most important place in the code
  to validate the data. (i.e. I want to validate the data at the boundaries of our system). I
appreciate that a new complexity is that the `Nwp` data contract uses float32, but we're saving as
`uint8`. Maybe the simplest answer that `process_ecmwf_dataset` does everything up to (but not
including) the 8-bit rescaling? So it can validate and return a `pt.DataFrame[Nwp]` object? And we
call `scale_to_uint8` from `download_and_scale_ecmwf`?


# In `contracts`
- Rename `SubstationFlows` to `SubstationPowerFlows`.

# Testing
- What is `tests/repro_plot.html`? The plot is blank! Where does this plot come from? And why is it
  blank?
