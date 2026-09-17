"""The dashboard apps' local/S3 data-source toggle.

Both marimo apps (``map_and_timeseries.py``, ``view_forecasts.py``) show a "Data source" radio
and re-instantiate `contracts.settings.Settings` from the selected source via
`settings_for_source`, so production S3 data can be viewed without restarting marimo. Swapping
the whole `Settings` object is what keeps the toggle to two functions: no reader downstream of
`Settings` has to know, or ask, which source the paths and credentials came from. See the
dashboard README for how to set up ``packages/dashboard/.env.s3``, and which read-only credentials
belong in that file.
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

    "local" reads only the root .env (the local pipeline, same as the rest of the app). "s3"
    layers packages/dashboard/.env.s3 on top of the root .env, overriding the data-path roots and
    object-store credentials to point at the real S3 buckets, so production data can be viewed
    without restarting marimo.

    Only the data tables follow the toggle: .env.s3 sets DATA_PATH_INTERNAL, DATA_PATH_DELIVERY,
    and the DATA_STORE_* credentials. The file deliberately does not set LOCAL_ARTIFACTS_PATH, so
    the production model stays laptop-local in both modes. A missing .env.s3 is silently skipped by
    pydantic-settings, so "s3" then falls back to the root .env's local paths — which is the case
    `source_status_message` warns about.

    Args:
        source: Which data source the app's "Data source" radio currently selects.

    Returns:
        A `Settings` built from the root .env alone for "local", or from the root .env with
        packages/dashboard/.env.s3 layered on top for "s3".
    """
    if source == "s3":
        # _env_file is a pydantic-settings builtin kwarg; the list layers .env.s3 over the
        # root .env (later file wins).
        return Settings(_env_file=[ROOT_ENV, DASHBOARD_S3_ENV])
    return Settings()


def source_status_message(source: DataSourceType, settings: Settings) -> tuple[str, bool]:
    """Build the status line shown under the "Data source" radio.

    The warning case is a selected "s3" source with no ``.env.s3`` file to read credentials from.
    The app has then silently fallen back to the root .env's local paths, so the reader has to be
    told before mistaking local output for production output.

    Args:
        source: Which data source the app's "Data source" radio currently selects.
        settings: The `Settings` `settings_for_source` built for that source, whose
            ``nged_data_path`` the non-warning message quotes back to the reader.

    Returns:
        ``(markdown_message, is_warning)`` — the markdown to render, and whether to render that
        markdown as a warning callout rather than as plain text.
    """
    if source == "s3" and not DASHBOARD_S3_ENV.exists():
        return (
            (
                f"No `{DASHBOARD_S3_ENV.name}` found next to this app. Copy "
                f"`{DASHBOARD_S3_ENV.name}.example` to `{DASHBOARD_S3_ENV.name}` and fill in "
                "the S3 buckets and credentials. Falling back to local data."
            ),
            True,
        )
    return f"Reading **{source}** data from `{settings.nged_data_path}`.", False
