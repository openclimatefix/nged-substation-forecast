# ruff: noqa: E501
r"""Stage 0 (zero-fit): where does experiment 323's F1 harm sit?

Splits the F1-minus-default quantile-score difference of experiment 323's stored scored-year rows
by the clear-sky index of the ensemble-mean shortwave flux (bands 0 to 0.3, 0.3 to 0.6, 0.6 to 0.9,
above 0.9), by sun-elevation band, by hour of the day, by lead band, and by Primary, and compares
the distribution of the clear-sky index in the training rows (analysis-proxy weather) with that of
the ensemble mean at forecast time. The ensemble-mean shortwave is the raw (uncalibrated) mean of
the 51 members from experiment 111's store, which the network rows of experiment 201 carry as
`drv_sw_w_m2`; the response itself sees the calibrated flux, so this is an approximation.

    source /tmp/env325.sh && uv run --no-sync python ml-experiments/00325-*/e00325_stage0.py
"""

import e00323_basis as basis323  # ty: ignore[unresolved-import]
import e00323_common as common323  # ty: ignore[unresolved-import]
import e00325_common as common
import numpy as np
import polars as pl

CSI_EDGES = (0.0, 0.3, 0.6, 0.9, 10.0)
CSI_LABELS = ("0-0.3", "0.3-0.6", "0.6-0.9", ">0.9")
ELEVATION_EDGES = (0.0, 15.0, 30.0, 45.0, 90.0)
ELEVATION_LABELS = ("0-15", "15-30", "30-45", ">45")
LEAD_EDGES = (0.0, 24.0, 72.0, 168.0, 240.0)
LEAD_LABELS = ("0-24 h", "24-72 h", "72-168 h", "168-240 h")
PLACEBO = ("D10", "D16")


def band(*, values: np.ndarray, edges: tuple[float, ...], labels: tuple[str, ...]) -> np.ndarray:
    """Return each value's band label."""
    position = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(labels) - 1)
    return np.array(labels, dtype=object)[position]


def load_rows() -> pl.DataFrame:
    """Return the scored-year rows with F1's and the default's loss, the index, and the elevation."""
    losses = pl.read_parquet(
        common323.REPORT_DIR / "scored_rows.parquet",
        columns=[
            "site",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "daylight",
            "sunny_daylight",
            "loss_default",
            "loss_f1",
        ],
    )
    feature_rows = pl.concat(
        [
            pl.read_parquet(
                common.NET_DIR / "v0r" / "features" / f"{site}.parquet",
                columns=[
                    "power_fcst_init_time",
                    "valid_time",
                    "era",
                    "drv_sw_w_m2",
                    "elevation_deg",
                ],
            ).with_columns(site=pl.lit(site))
            for site in common.SITES
        ]
    )
    scored = feature_rows.filter(pl.col("era") == "ens")
    joined = losses.join(
        scored.select("site", "power_fcst_init_time", "valid_time", "drv_sw_w_m2", "elevation_deg"),
        on=["site", "power_fcst_init_time", "valid_time"],
        how="inner",
        validate="1:1",
    )
    index = basis323.clear_sky_index(
        ghi_w_m2=joined["drv_sw_w_m2"].to_numpy().astype(np.float64),
        elevation_deg=joined["elevation_deg"].to_numpy().astype(np.float64),
    )
    return joined.with_columns(
        csi=pl.Series(index),
        difference=pl.col("loss_f1") - pl.col("loss_default"),
        hour=pl.col("valid_time").dt.hour(),
    ).filter(pl.col("daylight").fill_null(False))


def summarise(*, rows: pl.DataFrame, by: np.ndarray, labels: tuple[str, ...]) -> dict:
    """Return, per band, the row count, mean difference, and share of the net total difference."""
    difference = rows["difference"].to_numpy()
    net = difference.sum()
    out = {}
    for label in labels:
        mask = by == label
        out[label] = {
            "n": int(mask.sum()),
            "mean_difference": float(difference[mask].mean()) if mask.any() else float("nan"),
            "share_of_net": float(difference[mask].sum() / net) if net != 0 else float("nan"),
        }
    return out


def training_distribution() -> dict:
    """Return quantiles of the clear-sky index of daylight rows in training (proxy) and forecast."""
    pieces = [
        pl.read_parquet(
            common.NET_DIR / "v0r" / "features" / f"{site}.parquet",
            columns=["era", "drv_sw_w_m2", "elevation_deg"],
        ).filter(pl.col("elevation_deg") > 15.0)
        for site in common.SITES
    ]
    frame = pl.concat(pieces)
    out = {}
    for era in ("past_weather", "ens"):
        part = frame.filter(pl.col("era") == era)
        index = basis323.clear_sky_index(
            ghi_w_m2=part["drv_sw_w_m2"].to_numpy().astype(np.float64),
            elevation_deg=part["elevation_deg"].to_numpy().astype(np.float64),
        )
        out[era] = {f"p{p}": float(np.quantile(index, p / 100)) for p in (5, 25, 50, 75, 95)} | {
            "share_above_0.9": float((index > 0.9).mean()),
            "share_0.3_to_0.9": float(((index > 0.3) & (index <= 0.9)).mean()),
            "n": len(index),
        }
    return out


def main() -> None:
    """Write the stage 0 report."""
    rows = load_rows()
    csi = band(values=rows["csi"].to_numpy(), edges=CSI_EDGES, labels=CSI_LABELS)
    elevation = band(
        values=rows["elevation_deg"].to_numpy(), edges=ELEVATION_EDGES, labels=ELEVATION_LABELS
    )
    lead = band(values=rows["lead_hours"].to_numpy(), edges=LEAD_EDGES, labels=LEAD_LABELS)
    sunny = rows["sunny_daylight"].to_numpy()
    placebo = rows["site"].is_in(list(PLACEBO)).to_numpy()
    report: dict = {
        "n_daylight_rows_0_168h": rows.height,
        "net_mean_difference_all_daylight": float(rows["difference"].mean()),  # ty: ignore[invalid-argument-type]
        "by_csi_all_daylight": summarise(rows=rows, by=csi, labels=CSI_LABELS),
        "by_csi_sunny_daylight": summarise(
            rows=rows.filter(pl.Series(sunny)), by=csi[sunny], labels=CSI_LABELS
        ),
        "by_csi_placebo_D10_D16": summarise(
            rows=rows.filter(pl.Series(placebo)), by=csi[placebo], labels=CSI_LABELS
        ),
        "by_elevation_all_daylight": summarise(rows=rows, by=elevation, labels=ELEVATION_LABELS),
        "by_lead_all_daylight": summarise(rows=rows, by=lead, labels=LEAD_LABELS),
        "by_hour_all_daylight": {
            int(h): {
                "n": int((rows["hour"] == h).sum()),
                "mean_difference": float(rows.filter(pl.col("hour") == h)["difference"].mean()),  # ty: ignore[invalid-argument-type]
            }
            for h in sorted(rows["hour"].unique().to_list())
        },
        "by_site_and_csi": {
            site: summarise(
                rows=rows.filter(pl.col("site") == site),
                by=csi[(rows["site"] == site).to_numpy()],
                labels=CSI_LABELS,
            )
            for site in common.SITES
        },
        "csi_distribution_training_vs_forecast_above_15deg": training_distribution(),
    }
    harm = report["by_csi_all_daylight"]
    share_mid = harm["0.3-0.6"]["share_of_net"] + harm["0.6-0.9"]["share_of_net"]
    report["prediction"] = {
        "share_of_all_daylight_net_harm_at_csi_0.3_to_0.9": share_mid,
        "needs_at_least": 0.6,
        "sunny_gain_share_above_0.9": report["by_csi_sunny_daylight"][">0.9"]["share_of_net"],
        "passes": bool(share_mid >= 0.6),
    }
    common.write_json_atomic(path=common.REPORT_DIR / "stage0.json", content=report)
    print(report["prediction"], flush=True)
    for key in (
        "by_csi_all_daylight",
        "by_csi_sunny_daylight",
        "by_csi_placebo_D10_D16",
        "by_elevation_all_daylight",
        "by_lead_all_daylight",
    ):
        print(
            key,
            {
                k: (v["n"], round(v["mean_difference"], 3), round(v["share_of_net"], 2))
                for k, v in report[key].items()
            },
        )
    print(report["csi_distribution_training_vs_forecast_above_15deg"])


if __name__ == "__main__":
    main()
