# NGED substation forecast

[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

To external contributors: Please note that this repo holds the very-early-stage research code for a new
project, and there will be a lot of code churn over the next few months. As such, this repo isn't
suitable for external contributions at the moment, sorry.

---

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

Optional: To allow Dagster to remember its state after you shut it down:
1. `mkdir ~/dagster_home/`
2. Put the following into `~/dagster_home/dagster.yaml`:
    ```yaml
    storage:
      sqlite:
        base_dir: "history"
    ```
3. Add `export DAGSTER_HOME=<dagster_home_path>` to your `.bashrc` file, and restart your terminal.

## Environment variables

This code expects the following environment variables to be set. For example, you could put these
into a `.env` file:

```env
# A token for NGED's Connected Data portal: https://connecteddata.nationalgrid.co.uk
NGED_CKAN_TOKEN=

# NGED's AWS S3 bucket containing their data
NGED_S3_BUCKET_URL=https://
NGED_S3_BUCKET_ACCESS_KEY=
NGED_S3_BUCKET_SECRET=
```

### NGED CKAN API token:
1. Log in to [NGED's Connected Data](https://connecteddata.nationalgrid.co.uk) platform.
1. Click on your username (top right), and go to "User Profile" -> API Tokens -> Create API token
   -> Copy your API token. If you need more help then see [NGED's docs for getting an API
   token](https://connecteddata.nationalgrid.co.uk/api-guidance#api-tokens).
1. Paste your API token into `.env` after `NGED_CKAN_TOKEN=`.

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
