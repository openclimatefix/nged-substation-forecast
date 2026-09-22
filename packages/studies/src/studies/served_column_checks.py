"""Two assertions about what a downloaded irradiance column actually holds.

**Both checks are absolute rather than relative:** each needs no reference product, and cannot be
satisfied by two sources being wrong the same way. Both raise rather than warn. A study that trains
on a misread column produces a number nobody can tell apart from a real one, which is the R&D half
of
<https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/>:
production degrades, research fails fast.
"""

import logging
from typing import Final

import numpy as np
import polars as pl

_LOG: Final[logging.Logger] = logging.getLogger("studies.served_column_checks")

MIN_ELEVATION_FOR_RECONSTRUCTION_DEGREES: Final[float] = 10.0
"""Rows at a lower sun than this are left out of the reconstruction test.

The reconstruction multiplies the default column by the ratio of the instantaneous to the hour-mean
cosine of the solar zenith angle, and that ratio grows without bound as the sun sets — so near the
horizon it amplifies the default column's 1 W m⁻² rounding into a residual of many W m⁻². That
residual is a fact about rounding rather than about which hour the label names, and leaving it in
would force a threshold loose enough to stop discriminating.
"""

MAX_INSTANT_RECONSTRUCTION_RMS_W_M2: Final[float] = 5.0
"""How far the reconstructed snapshot may sit from the served one.

Measured at one meter's coordinates over June 2026, the reconstruction lands at a root-mean-square
error of 0.65 W m⁻² for the global flux and 0.14 for the direct flux — the scale of the 1 W m⁻²
rounding the default columns carry. The threshold leaves room for other sites and seasons while
staying well inside the failures it exists to catch: assuming the hour *beginning* at the label
gives 68 W m⁻² and assuming an hour centred on it gives 26, so a half-hour error in either
direction is at least five times the threshold rather than a marginal call.
"""


def check_hourly_value_is_a_backward_mean(*, frame: pl.DataFrame) -> None:
    """Assert the hourly column is a backward mean over the hour ending at its label.

    **The check is absolute rather than relative:** it needs no reference product, and cannot be
    satisfied by two sources being wrong the same way. A half-hour error in an hourly label is the
    fault it exists to catch.

    Open-Meteo's own downloader states the mechanism. UKV publishes radiation as an instantaneous
    snapshot, and Open-Meteo divides that snapshot by the ratio of the instantaneous cosine of the
    solar zenith angle to its mean over the preceding hour before storing it, so the stored value is
    a backward-looking hourly mean holding the clear-sky index fixed across the hour. Asking for the
    `_instant` column multiplies the same ratio back. Reconstructing one column from the other and
    the sun's geometry therefore pins both the conversion and which hour the label names.

    What the check settles is which of the two served columns the arms should read, which
    the study's `sources.PointTemporalType` records and explains.

    Args:
        frame: The downloaded rows, carrying the geometry `_solar_geometry` adds.

    Raises:
        ValueError: If either flux fails to reconstruct.
    """
    daylight = frame.filter(
        (pl.col("ghi_w_m2") > 0.0)
        & (pl.col("solar_zenith_deg") < 90.0 - MIN_ELEVATION_FOR_RECONSTRUCTION_DEGREES)
    )
    factor = daylight["cos_zenith_instant"].to_numpy() / np.maximum(
        daylight["cos_zenith_hour_mean"].to_numpy(), 1e-9
    )
    for default_column, instant_column in (
        ("ghi_w_m2", "ghi_instant_w_m2"),
        ("bhi_w_m2", "bhi_instant_w_m2"),
    ):
        error = daylight[default_column].to_numpy() * factor - daylight[instant_column].to_numpy()
        root_mean_square = float(np.sqrt(np.mean(error**2)))
        _LOG.info(
            "backward-mean reconstruction of %s: RMS %.3f W m-2 on %d daylight rows",
            default_column,
            root_mean_square,
            daylight.height,
        )
        if root_mean_square > MAX_INSTANT_RECONSTRUCTION_RMS_W_M2:
            msg = (
                f"{default_column} does not reconstruct {instant_column} from the hour ending at "
                f"its label: RMS {root_mean_square:.2f} W m-2 against a threshold of "
                f"{MAX_INSTANT_RECONSTRUCTION_RMS_W_M2}. Either the label names a different hour "
                "or Open-Meteo has changed the conversion; settle which before training on it."
            )
            raise ValueError(msg)


CLEARNESS_BIN_WIDTH: Final[float] = 0.05
ZENITH_BIN_WIDTH_DEGREES: Final[float] = 5.0
MIN_ROWS_PER_BIN: Final[int] = 30
MIN_DIRECT_FRACTION_SPREAD: Final[float] = 0.05
"""How much the published direct fraction must vary inside one `(clearness, zenith)` bin.

A separation model's direct fraction is by construction a function of the clearness index and the
solar zenith angle, so inside a fine bin on those two it is very nearly constant whatever formula it
uses, and the spread it leaves comes only from the bin's own width. Measured over 2025 at one
meter's coordinates on these bin widths, the median within-bin spread is 0.118 for UKV's published
fraction and
0.016 for an Erbs fraction derived from the same global irradiance — so the threshold sits three
times above the separation-model floor and well below what UKV's own field gave.
"""

MIN_ZENITH_FOR_SPREAD_DEGREES: Final[float] = 80.0
"""Rows at a lower sun than this are left out of the spread test.

The clearness index is a ratio taken against a small extraterrestrial flux there, so it is mostly
noise, and binning on a noisy axis would inflate the within-bin spread of anything.
"""


def check_direct_is_not_a_separation_model(*, frame: pl.DataFrame) -> None:
    """Assert the published direct fraction carries information beyond clearness and geometry.

    **This is the one check that reaches the era no sampling against the Met Office's own files
    can**, because it needs no reference product: Open-Meteo's UKV archive runs from 2022-03-01 and
    the Met Office's AWS bucket holds a rolling two years, so the earlier half can only be checked
    from the inside.

    What it would catch is the failure that would void arm C outright — a mirror that reconstructed
    the beam from global irradiance with a separation model rather than serving the model's own
    field. Arm C would then be a copy of arm B, and the headline contrast would be a measurement of
    floating-point noise.

    Args:
        frame: The downloaded rows, carrying the geometry `_solar_geometry` adds.

    Raises:
        ValueError: If the within-bin spread sits at the separation-model floor.
    """
    binned = (
        frame.filter(
            (pl.col("solar_zenith_deg") < MIN_ZENITH_FOR_SPREAD_DEGREES)
            & (pl.col("ghi_w_m2") > 20.0)
            & (pl.col("extraterrestrial_horizontal_w_m2") > 50.0)
        )
        .with_columns(
            direct_fraction=(pl.col("bhi_w_m2") / pl.col("ghi_w_m2")).clip(0.0, 1.0),
            clearness_bin=(pl.col("clearness_index") / CLEARNESS_BIN_WIDTH).floor(),
            zenith_bin=(pl.col("solar_zenith_deg") / ZENITH_BIN_WIDTH_DEGREES).floor(),
        )
        .group_by("clearness_bin", "zenith_bin")
        .agg(spread=pl.col("direct_fraction").std(ddof=0), rows=pl.len())
        .filter(pl.col("rows") >= MIN_ROWS_PER_BIN)
    )
    if binned.height == 0:
        msg = (
            f"no (clearness, zenith) bin holds {MIN_ROWS_PER_BIN} rows, so the spread test "
            "cannot run"
        )
        raise ValueError(msg)
    median_spread = binned.select(pl.col("spread").median()).item()
    _LOG.info(
        "direct-fraction spread: median %.4f within %d populated bins",
        median_spread,
        binned.height,
    )
    if median_spread < MIN_DIRECT_FRACTION_SPREAD:
        msg = (
            f"the published direct fraction varies by only {median_spread:.4f} inside a "
            f"(clearness, zenith) bin, against a threshold of {MIN_DIRECT_FRACTION_SPREAD}. That "
            "is what a separation model applied to global irradiance looks like, and it would make "
            "arm C a copy of arm B."
        )
        raise ValueError(msg)
