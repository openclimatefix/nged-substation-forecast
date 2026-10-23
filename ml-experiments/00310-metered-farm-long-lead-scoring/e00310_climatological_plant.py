# ruff: noqa: E501
"""Post hoc check: does the plant's margin over climatology at days 8 to 10 need forecast weather?

Refit P2 (the spread-binned plant) with each row's plant forecast and spread replaced by their
average over the training rows in the same month and half-hour of day, so the forecast carries the
season and sun geometry but none of the ensemble's day-specific weather. Compare its quantile score
against P2 with the real ensemble weather and against `climatology`, own history, in days 1 and
8 to 10. Descriptive point estimates, farms equal-weighted.

    uv run --no-sync python ml-experiments/00310-*/e00310_climatological_plant.py
"""

import e00310_common as common
import e00310_fit as fit
import numpy as np
import polars as pl


def score(*, q: np.ndarray, y: np.ndarray) -> float:
    """Return twice the mean pinball loss over the 11 levels, in points of capacity."""
    q = np.sort(q, axis=1)
    error = y[:, None] - q
    pin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    return float(200.0 * pin.mean())


def main() -> None:
    """Print the three scores per band."""
    floors = {
        b: pl.read_parquet(common.DATA_DIR / f"floors_{b}.parquet")
        .filter(pl.col("arm") == "floor_climatology")
        .filter(pl.all_horizontal([pl.col(c).is_finite() for c in fit.level_columns()]))
        for b in ("d1", "d8_10")
    }
    for band in ("d1", "d8_10"):
        real, clim = [], []
        for site in common.FARMS:
            real_f, clim_f = [], []
            for start in fit.REFIT_STARTS:
                st = fit.Setting(
                    band=band, setting="own_history", site=site, start=start, n_spread_bins=3
                )
                y = st.test["output"].to_numpy().astype(float)
                real_f.append((score(q=st.clamp(st.test_base), y=y), len(y)))
                key = ["month", "half"]

                def keyed(frame: pl.DataFrame) -> pl.DataFrame:
                    return frame.with_columns(
                        month=pl.col("time").dt.month(),
                        half=pl.col("time").dt.hour() * 2 + pl.col("time").dt.minute() // 30,
                    )

                train, test = keyed(st.train), keyed(st.test)
                mean = train.group_by(key).agg(
                    mu_c=pl.col("mu").mean(), sp_c=pl.col("plant_spread").mean()
                )
                train = train.join(mean, on=key, how="left")
                test = test.join(mean, on=key, how="left").with_columns(
                    mu_c=pl.col("mu_c").fill_null(pl.col("mu")),
                    sp_c=pl.col("sp_c").fill_null(pl.col("plant_spread")),
                )
                st.train = train.with_columns(mu=pl.col("mu_c"), plant_spread=pl.col("sp_c"))
                st.test = test.with_columns(mu=pl.col("mu_c"), plant_spread=pl.col("sp_c"))
                st.fit_p2()
                clim_f.append((score(q=st.clamp(st.p2_quantiles(frame=st.test)), y=y), len(y)))
            real.append(np.average([s for s, _ in real_f], weights=[n for _, n in real_f]))
            clim.append(np.average([s for s, _ in clim_f], weights=[n for _, n in clim_f]))
        fl = floors[band]
        floor_score = np.mean(
            [
                score(
                    q=fl.filter(pl.col("time_series_id") == s)
                    .select(fit.level_columns())
                    .to_numpy(),
                    y=fl.filter(pl.col("time_series_id") == s)["actual"].to_numpy(),
                )
                for s in common.FARMS
            ]
        )
        print(
            f"{band}: P2 real weather {np.mean(real):.3f}, P2 seasonal-average weather {np.mean(clim):.3f}, climatology {floor_score:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
