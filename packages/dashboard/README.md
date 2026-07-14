# Dashboard

Marimo web app for visualising power forecasts, telemetry, and evaluation metrics.

## Switching between local and S3 data sources

The dashboard has a **Data source** toggle (`local` / `s3`) that switches which data tables it
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
