# ruff: noqa: E501
"""Write a short fitted-properties card for each farm (labels A to F only) into the private data folder.

A card holds the plant's fitted scale and output ceiling (as fractions of capacity), the P2 spread
and forecast bin edges, the median and 5-95% residual offsets in the middle spread bin, and the
blend's mean 5-95% band width and coverage by clear-sky band. Cards use the last refit (August 2026,
own history).

    uv run --no-sync python ml-experiments/00300-*/e00300_cards.py
"""

import e00300_common as common
import e00300_fit as fit
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

Q05, Q50, Q95 = (level_column(level=lv) for lv in (0.05, 0.5, 0.95))
CSI_EDGES = (0.3, 0.6, 0.9)
CSI_LABELS = ("<0.3", "0.3-0.6", "0.6-0.9", ">=0.9")


def main() -> None:
    """Print each farm's card and write the tables."""
    rows, bands = [], []
    for site in common.FARMS:
        st = fit.Setting(
            setting="own_history",
            site=site,
            start=common.REFIT_MONTHS[-1],
            n_spread_bins=fit.N_SPREAD_BINS,
        )
        mid = fit.N_SPREAD_BINS // 2
        levels = list(common.LEVELS)
        i05, i50, i95 = levels.index(0.05), levels.index(0.5), levels.index(0.95)
        rows.append(
            {
                "farm": site,
                "plant_scale": st.scale,
                "output_ceiling": st.ceiling,
                "forecast_edges": np.round(st.mu_edges, 3).tolist(),
                "spread_edges": np.round(st.spread_edges, 4).tolist(),
                "mid_spread_median_offset_by_forecast_bin": np.round(
                    st.cells[:, mid, i50], 3
                ).tolist(),
                "mid_spread_band_width_by_forecast_bin": np.round(
                    st.cells[:, mid, i95] - st.cells[:, mid, i05], 3
                ).tolist(),
            }
        )
        pred = pl.concat(
            [
                pl.read_parquet(f)
                for f in sorted((common.PRED_DIR / "own_history").glob(f"blend_s0_{site}.parquet"))
            ]
        )
        csi = common.load_farm(site).frame.select("time", "csi")
        pred = pred.join(csi, on="time").with_columns(
            band=pl.when(pl.col("csi") < CSI_EDGES[0])
            .then(pl.lit(CSI_LABELS[0]))
            .when(pl.col("csi") < CSI_EDGES[1])
            .then(pl.lit(CSI_LABELS[1]))
            .when(pl.col("csi") < CSI_EDGES[2])
            .then(pl.lit(CSI_LABELS[2]))
            .otherwise(pl.lit(CSI_LABELS[3])),
            width=pl.col(Q95) - pl.col(Q05),
            inside=((pl.col("actual") >= pl.col(Q05)) & (pl.col("actual") <= pl.col(Q95))).cast(
                pl.Float64
            ),
        )
        bands.append(
            pred.group_by("band")
            .agg(
                mean_width_points=100 * pl.col("width").mean(),
                coverage_90=pl.col("inside").mean(),
                rows=pl.len(),
            )
            .with_columns(farm=pl.lit(site))
        )
    cards = pl.DataFrame(rows)
    cards.write_json(common.DATA_DIR / "farm_cards.json")
    pl.concat(bands).sort("farm", "band").write_csv(common.DATA_DIR / "farm_bands.csv")
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(120)
    print(cards)
    print(pl.concat(bands).sort("farm", "band"))


if __name__ == "__main__":
    main()
