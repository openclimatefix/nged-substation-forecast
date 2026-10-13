"""Compare production wind interpolation with independent synthetic linear-vector answers.

Run with `uv run python scripts/diagnostics/plot_wind_interpolation.py`. Writes the docs figure
and a CSV/report under `data/studies/wind_storage/interpolation-proof/`. No weather download or
trained forecasting model is needed.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from ml_core.features.tabular_feature_engineer import _engineer_features
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, DATA_GREEN

BEFORE: Final[str] = "Before (speed/bearing)"
EXPECTED: Final[str] = "Expected"
AFTER: Final[str] = "After (u/v)"
OUTPUT: Final[Path] = Path("data/studies/wind_storage/interpolation-proof")
FIGURE: Final[Path] = Path("docs/studies/assets/wind-interpolation")


def _production_values(*, u: list[float], v: list[float]) -> pl.DataFrame:
    """Run a synthetic interval through the production feature orchestrator."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    times = [start + timedelta(minutes=30 * step) for step in range(7)]
    nwp = pl.LazyFrame(
        {
            "init_time": [start, start],
            "valid_time": [times[0], times[-1]],
            "time_series_id": ["synthetic"] * 2,
            "ensemble_member": [0] * 2,
            "wind_u_10m": u,
            "wind_v_10m": v,
        }
    ).with_columns(pl.col("wind_u_10m", "wind_v_10m").cast(pl.Float32))
    power = pt.LazyFrame(
        {"time_series_id": ["synthetic"] * 7, "time": times, "power": [1.0] * 7}
    ).set_model(PowerTimeSeries)
    metadata = pt.DataFrame(
        {"time_series_id": ["synthetic"], "time_series_type": ["substation"]}
    ).set_model(TimeSeriesMetadata)
    return (
        _engineer_features(
            selected_features={"wind_speed_10m", "wind_direction_10m"},
            power_time_series=power,
            time_series_metadata=metadata,
            nwp=nwp,
            power_fcst_init_time=start,
            nwp_init_time=start,
        )
        .collect()
        .sort("valid_time")
    )


def _comparison() -> pl.DataFrame:
    """Return dense analytic curves, old arithmetic curves, and production half-hourly dots."""
    angle = np.deg2rad(10.0)
    east, north = float(10 * np.sin(angle)), float(-10 * np.cos(angle))
    crossing = _production_values(u=[east, -east], v=[north, north])
    opposing = _production_values(u=[10.0, -10.0], v=[0.0, 0.0])
    hours = np.linspace(0, 3, 181)
    fraction = hours / 3
    expected_bearing = 360 + np.rad2deg(np.arctan((2 * fraction - 1) * np.tan(angle)))
    expected_speed = np.sqrt(north**2 + east**2 * (1 - 2 * fraction) ** 2)
    actual_bearing = np.rad2deg(
        np.unwrap(np.deg2rad(crossing["wind_direction_10m"].to_numpy().astype(float)))
    )
    panels = (
        ("bearing", 350 - 340 * fraction, expected_bearing, actual_bearing),
        (
            "crossing_speed",
            np.full(181, 10.0),
            expected_speed,
            crossing["wind_speed_10m"].to_numpy(),
        ),
        (
            "opposing_speed",
            np.full(181, 10.0),
            10 * np.abs(1 - 2 * fraction),
            opposing["wind_speed_10m"].to_numpy(),
        ),
    )
    frames = []
    report = ["# Synthetic interpolation proof", "", "Expected u/v change linearly.", ""]
    for panel, before, expected, actual in panels:
        np.testing.assert_allclose(actual, expected[::30], atol=5e-5, rtol=0)
        for series, values, times in (
            (BEFORE, before, hours),
            (EXPECTED, expected, hours),
            (AFTER, actual, hours[::30]),
        ):
            frames.append(
                pl.DataFrame(
                    {
                        "panel": panel,
                        "series": series,
                        "hours": times,
                        "value": np.asarray(values, dtype=float),
                    }
                )
            )
        report.append(
            f"- {panel}: midpoint before={before[90]:.6f}, "
            f"expected={expected[90]:.6f}, after={actual[3]:.6f}; "
            f"maximum after error={np.max(np.abs(actual - expected[::30])):.8f}."
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    return pl.concat(frames)


def _panel(
    *,
    frame: pl.DataFrame,
    name: str,
    title: str,
    ylabel: str,
    domain: list[float],
    note: str = "",
) -> alt.LayerChart:
    base = alt.Chart(frame.filter(pl.col("panel") == name)).encode(
        x=alt.X(
            "hours:Q",
            title="Hours since first native forecast step",
            scale=alt.Scale(domain=[0, 3]),
            axis=alt.Axis(values=[0, 0.5, 1, 1.5, 2, 2.5, 3]),
        ),
        y=alt.Y("value:Q", title=ylabel, scale=alt.Scale(domain=domain, zero=False)),
        color=alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(
                domain=[BEFORE, EXPECTED, AFTER], range=[BRAND_ORANGE, DATA_GREEN, DATA_BLUE]
            ),
            legend=alt.Legend(orient="top", direction="horizontal"),
        ),
    )
    lines = (
        base.transform_filter(alt.datum.series != AFTER)
        .mark_line(strokeWidth=3)
        .encode(  # ty: ignore[unresolved-attribute]
            strokeDash=alt.StrokeDash(
                "series:N",
                legend=None,
                scale=alt.Scale(domain=[BEFORE, EXPECTED], range=[[7, 4], [1, 0]]),
            )
        )
    )
    dots = base.transform_filter(alt.datum.series == AFTER).mark_point(
        filled=True, size=75, stroke="white", strokeWidth=1
    )
    return (lines + dots).properties(
        width=590,
        height=155,
        title=alt.TitleParams(text=title, subtitle=[note] if note else []),
    )


def main() -> None:
    """Assert the known answers and export the self-contained figure, report, and CSV."""
    frame = _comparison()
    frame.write_csv(OUTPUT / "comparison.csv")
    chart = (
        alt.vconcat(
            _panel(
                frame=frame,
                name="bearing",
                title="North crossing: 10 m/s winds from 350° to 10°",
                ylabel="Wind from (degrees; 360° = North)",
                domain=[0, 380],
                note="After/expected bearings are unwrapped through 360°; 370° means 10°.",
            ),
            _panel(
                frame=frame,
                name="crossing_speed",
                title="North crossing: speed dips as components cancel",
                ylabel="Wind speed (m/s)",
                domain=[9.8, 10.05],
                note="Speed axis spans 9.8–10.05 m/s.",
            ),
            _panel(
                frame=frame,
                name="opposing_speed",
                title="Opposing 10 m/s winds: expected midpoint is calm",
                ylabel="Wind speed (m/s)",
                domain=[0, 10.3],
            ),
            alt.Chart(
                pl.DataFrame(
                    {
                        "text": [
                            (
                                "Agreement means the implementation follows the stated "
                                "linear-vector assumption,"
                            ),
                            "not observed weather truth.",
                        ],
                        "line": [0, 1],
                    }
                )
            )
            .mark_text(align="left", fontSize=12)
            .encode(  # ty: ignore[unresolved-attribute]
                text="text:N", y=alt.Y("line:O", axis=None), x=alt.value(0)
            )
            .properties(width=590, height=35),
            spacing=18,
        )
        .properties(
            title=alt.TitleParams(
                text="Interpolate wind components before deriving speed and bearing",
                subtitle=[
                    "Two synthetic 3-hour intervals; expected u (eastward) and v (northward)",
                    "change linearly. Dashed line: before. Solid line: expected. Dots: after.",
                ],
                anchor="start",
                fontSize=17,
            )
        )
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_view(stroke=None)
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE.with_suffix(".svg"))
    chart.save(FIGURE.with_suffix(".png"), scale_factor=1.5)
    print((OUTPUT / "report.md").read_text())


if __name__ == "__main__":
    main()
