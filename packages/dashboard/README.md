# Dashboard

Marimo web apps for visualising power forecasts, telemetry, and evaluation metrics. The apps live
at the package root; their shared, unit-testable logic (the data-source toggle, the forecast
chart builder) lives in the importable `dashboard` package under `src/`.

- **`view_forecasts.py`** — inspect a single forecast run: pick a time series, a fold (`live` or
  a CV fold), and a forecast init time, then see every forecast ensemble member (thin grey lines)
  against the observed power (thick blue line), from 24 hours before the init time to 14 days
  after it. The x-axis is labelled at Europe/London midnight with the day of week and date.
- **`map_and_timeseries.py`** — a map of every time series in the trial area; click a dot to see
  its observed power.

Run an app with:

```bash
uv run marimo edit packages/dashboard/view_forecasts.py
```

## Switching between local and S3 data sources

Each app has a **Data source** toggle (`local` / `s3`) that switches which data tables it
reads without restarting marimo, so you can compare a fully local pipeline against production
data in one session.

- **local** reads only the root `.env` — the same local-pipeline config the rest of the app uses.
- **s3** layers a git-ignored `packages/dashboard/.env.s3` on top of the root `.env`, overriding
  the data-path roots (`DATA_PATH_INTERNAL`, `DATA_PATH_DELIVERY`) and the `DATA_STORE_*`
  credentials to point at the real S3 buckets.

To enable S3 mode, copy the committed example and fill in the real values:

```bash
cp packages/dashboard/.env.s3.example packages/dashboard/.env.s3
```

Only the data tables follow the toggle. Local artifact paths (model cache, production model) are
never overridden by `.env.s3`, so they stay laptop-local in both modes. If `.env.s3` is missing,
the `s3` selection falls back to local paths and the UI flags it.
