# NGED substation forecast

TODO(Jack): Adapt the OCF template README for this project :)

## Development

This repo is a `uv` [workspace](https://docs.astral.sh/uv/concepts/projects/workspaces): A single
repo which contains multiple Python packages.

1. Ensure [`uv`](https://docs.astral.sh/uv/) is installed following their [official documentation](https://docs.astral.sh/uv/getting-started/installation/).
1. `uv sync`
1. `uv run pre-commit install`

To run Dagster:
1. `uv run dg dev`
1. Open http://localhost:3000 in your browser to see the project.

NGED CKAN API token:
1. Log in to NGED's Connected Data platform.
1. Go to "User Profile" -> API Tokens -> Create API token -> Copy your API token
1. Paste your API token into `.env` after `NGED_CKAN_TOKEN=`.

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
