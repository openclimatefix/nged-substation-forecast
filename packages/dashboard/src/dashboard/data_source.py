"""The dashboard apps' local/S3 data-source toggle.

Both marimo apps (``map_and_timeseries.py``, ``view_forecasts.py``) show a "Data source" radio
and re-instantiate :class:`contracts.settings.Settings` from the selected source via
:func:`settings_for_source`, so production S3 data can be viewed without restarting marimo.
See the dashboard README for how to set up ``packages/dashboard/.env.s3``.
"""

from pathlib import Path
from typing import Final, Literal

from contracts.settings import PROJECT_ROOT, Settings

DataSourceType = Literal["local", "s3"]
"""The two data sources the dashboard apps can read from."""

ROOT_ENV: Final[Path] = PROJECT_ROOT / ".env"
"""Root .env — the local-pipeline config shared with the rest of the app."""

DASHBOARD_S3_ENV: Final[Path] = PROJECT_ROOT / "packages" / "dashboard" / ".env.s3"
"""Git-ignored S3-mode overrides, layered on top of ROOT_ENV when the toggle is 's3'."""


def settings_for_source(source: DataSourceType) -> Settings:
    """Instantiate Settings for the dashboard's selected data source.

    "local" reads only the root .env (the local pipeline, same as the rest of the app).
    "s3" layers packages/dashboard/.env.s3 on top of the root .env, overriding the
    data-path roots and object-store credentials to point at the real S3 buckets, so
    production data can be viewed without restarting marimo.

    Only the data tables follow the toggle: .env.s3 sets DATA_PATH_INTERNAL,
    DATA_PATH_DELIVERY and the DATA_STORE_* credentials. It deliberately does not set
    LOCAL_ARTIFACTS_PATH, so the model cache and production model stay laptop-local in
    both modes. A missing .env.s3 is silently skipped by pydantic-settings, so "s3"
    then falls back to the root .env's local paths (see :func:`source_status_message`).
    """
    if source == "s3":
        # _env_file is a pydantic-settings builtin kwarg not modelled by ty's synthesised
        # BaseModel __init__; the list layers .env.s3 over the root .env (later file wins).
        return Settings(_env_file=[ROOT_ENV, DASHBOARD_S3_ENV])  # ty: ignore[unknown-argument]
    return Settings()


def source_status_message(source: DataSourceType, settings: Settings) -> tuple[str, bool]:
    """Build the status line shown under the "Data source" radio.

    Returns ``(markdown_message, is_warning)``. The warning case is a selected "s3" source with
    no ``.env.s3`` file to read credentials from, in which case the app silently fell back to
    the root .env's local paths and the user should be told.
    """
    if source == "s3" and not DASHBOARD_S3_ENV.exists():
        return (
            f"No `{DASHBOARD_S3_ENV.name}` found next to this app. Copy "
            f"`{DASHBOARD_S3_ENV.name}.example` to `{DASHBOARD_S3_ENV.name}` and fill in "
            "the S3 buckets and credentials. Falling back to local data.",
            True,
        )
    return f"Reading **{source}** data from `{settings.nged_data_path}`.", False
