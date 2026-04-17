## Files in this directory

- `nwp_metadata.csv`: Hand-written. Conforms to the `NwpMetaData` data contract.
- `scaling_params_for_ecmwf_ens_0_25_degree.csv`: Created by
  `packages/dynamical_data/scaling/compute_scaling_params.py`. Conforms to the `NwpScalingParams` data
  contract. Defines the min/max bounds and ranges for scaling NWP variables to 12-bit integers for storage on disk in the `NwpOnDisk` data contract. These scaling parameters are derived from historical ECMWF ENS data to ensure that the full range of physical values can be represented within the integer range.
