"""The band quantile head on the real record: logged transfers, ordinary rows, and tail coverage.

The real record is the stream with no injected megawatts. For each of the six substations this
script scores K0, GAM_ONLY, and the cross-fitted BASE and PIT heads (fitted on injected worlds and
the other month parity, saved by `e00312_heads.py`) against measured power. Rows are the replicate
and scored periods. Groups: DURING a logged transfer touching the substation's series, AFTER (48
hours after its end), and ORDINARY (outside every logged transfer by 7 days, every fourth run,
weight 4). The rows overlap the injected worlds' training rows, because the real record is the
background of every world; the head scoring a row was fitted on months of the other parity.

    PYTHONPATH=... uv run --no-sync python e00312_real.py
"""

from __future__ import annotations

import e00297_fed as fed
import e00297_score as score
import numpy as np
import polars as pl
import xgboost as xgb
from e00297_common import LOGGED_PATH
from e00312_common import REAL_DIR
from e00312_degraded import normal_scores, window_means
from e00312_heads import BOOSTERS, matrix

AFTER_STAMPS = 96
PAD = 336


def site_frame(site: str) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Rows of one site on the real record with the head inputs, their group, and K0's mean."""
    data = fed.get_site(site)
    n = len(data.stream)
    offset = fed.stamp_offset(data)
    series = str(fed.SERIES_OF_SITE[site])
    logged = pl.read_parquet(LOGGED_PATH).filter(
        (pl.col("side_a") == series) | (pl.col("side_b") == series)
    )
    during = np.zeros(n, dtype=bool)
    after = np.zeros(n, dtype=bool)
    near = np.zeros(n, dtype=bool)
    event = np.full(n, -1, dtype=np.int64)
    for i, r in enumerate(logged.iter_rows(named=True)):
        lo, hi = r["first_stamp"] + offset, r["last_stamp_exclusive"] + offset
        during[lo:hi] = True
        after[hi : hi + AFTER_STAMPS] = True
        near[max(lo - PAD, 0) : hi + PAD] = True
        event[lo : hi + AFTER_STAMPS] = i
    k0 = fed.correction_for(data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0)
    cutoffs_run = data.cutoffs(delay_hours=fed.DELAY_HOURS)
    scales = score.site_scales(site)
    ratio_run = score.asof_ratio_per_run(
        data=data, stream=data.stream, cutoffs_run=cutoffs_run, rms=scales["stream_rms"]
    )
    pit_grid = normal_scores(site=site, stream=data.stream)
    pit_run = window_means(x=pit_grid, cutoffs=cutoffs_run)
    ext_run = window_means(x=np.abs(pit_grid), cutoffs=cutoffs_run)
    period_ok = data.period_rows(period="replicate") | data.period_rows(period="scored")
    rows = np.flatnonzero(period_ok)
    pos = data.position[rows]
    ok = np.isfinite(data.actual[rows])
    rows, pos = rows[ok], pos[ok]
    group = np.where(
        during[pos],
        "DURING",
        np.where(after[pos], "AFTER", np.where(~near[pos], "ORDINARY", "MARGIN")),
    )
    stride = data.run_of_row[rows] % score.ORDINARY_RUN_STRIDE == 0
    keep = (group != "ORDINARY") | stride
    keep &= group != "MARGIN"
    rows, pos, group = rows[keep], pos[keep], group[keep]
    run_index = data.run_of_row[rows]
    c0_run = k0[data.offsets[:-1]]
    month = (data.month_code[pos] % 100).astype(np.int8)
    frame = pl.DataFrame(
        {
            "row": rows,
            "month": month,
            "hour_band": data.hour_band[rows],
            "quarter": data.quarter[rows],
            "gam_mean": data.member_mean[rows],
            "k0_mean": data.member_mean[rows] + k0[rows],
            "k0_correction": k0[rows],
            "c0": c0_run[run_index],
            "u": c0_run[run_index] / scales["c0_scale"],
            "asof_ratio": ratio_run[run_index],
            "lead_hours": data.lead_hours[rows],
            "pit24": pit_run[run_index],
            "ext24": ext_run[run_index],
            "group": group,
            "event": event[pos],
            "actual": data.actual[rows],
        }
    )
    return frame, k0, data.member_mean[rows] + k0[rows]


def main() -> None:
    """Score the real record and write one table per site."""
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    for site in fed.SERIES_OF_SITE:
        data = fed.get_site(site)
        frame, _, k0_mean = site_frame(site)
        rows = frame["row"].to_numpy().astype(np.int64)
        parity = (frame["month"].to_numpy() % 2).astype(np.int64)
        arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        k0_full = np.zeros(data.n_rows)
        k0_full[rows] = frame["k0_correction"].to_numpy()
        arms["K0"] = (
            fed.quantiles_at(data=data, correction=k0_full, spread=fed.k0_spread(site), rows=rows),
            k0_mean,
        )
        arms["GAM_ONLY"] = (
            score.gam_only_quantiles(data=data, site=site, rows=rows),
            data.member_mean[rows],
        )
        for variant in ("BASE", "PIT"):
            x = matrix(frame=frame, variant=variant)
            q = np.empty((len(frame), 11))
            for p in (0, 1):
                idx = np.flatnonzero(parity == p)
                booster = xgb.Booster()
                booster.load_model(BOOSTERS / f"{site}_{variant}_{p}.ubj")
                q[idx] = k0_mean[idx][:, None] + booster.predict(xgb.DMatrix(x[idx]))
            arms[f"HEAD_{variant}"] = (np.sort(q, axis=1), k0_mean)
        delta = np.zeros(len(data.cache.end_times))
        scored = score.score_arms(
            arms=arms, rows=rows, data=data, delta_points=delta, edges=score.edges_of(data=data)
        )
        out = scored.join(
            frame.select("row", "group", "event", "month", "lead_hours", "actual"),
            on="row",
            suffix="_f",
        )
        out = out.with_columns(site=pl.lit(site))
        out.with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(
            REAL_DIR / f"{site}.parquet"
        )
        print(site, frame["group"].value_counts().to_dict(as_series=False), flush=True)


if __name__ == "__main__":
    main()
