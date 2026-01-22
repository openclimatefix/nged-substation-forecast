# NGED substation forecast

TODO(Jack): Adapt the OCF template README for this project :)

## Directory structure

This repo is a `uv` [workspace](https://docs.astral.sh/uv/concepts/projects/workspaces): A single
repo which contains multiple Python packages. We also have a `Dagster` workspace for orchestrating
data & ML pipelines.

- `dg.toml`: Dagster workspace configuration.
- `src/nged_substation_forecast/`: The main Dagster project with a nested src structure.
- `deployments/local/`: Local Dagster deployment configuration managed by the dg tool.
- `packages/`: Python packages within this `uv` workspace. 

## Development

1. Install `uv`
1. `uv sync`
1. `uv run pre-commit install`

To run Dagster:
1. `uv run dg dev`

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
