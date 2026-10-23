# ruff: noqa: E501
"""Score experiment 318: per-row losses, farm-equal means, month-block intervals at 99.17%.

Per band and setting, rows are the scored-year runs that every arm compared holds with finite
quantiles, including experiment 310's `climatology` and `manual_heuristic` rows, experiment 310's
mixture arm (`ref_mdn310`, three seeds), and experiment 313's pure network (`ref_pure_mdn313`,
five seeds). The seeds of each arm are averaged quantile by quantile before sorting. The bootstrap
resamples the scored months with farm-equal weights; the level is the Bonferroni level for six
planned contrasts. `h1` and `h2` are the first and second half of the scored year (the replicate).

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
        uv run --no-sync python ml-experiments/00318-*/e00318_score.py
"""

from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_score as s310  # ty: ignore[unresolved-import]
import e00318_common as common
import numpy as np
import polars as pl

LO_PCT, HI_PCT = 0.415, 99.585
FLOORS: Final[tuple[str, ...]] = ("floor_climatology", "floor_heuristic")
PLANNED: Final[tuple[tuple[str, str], ...]] = (
    ("lead", "default"),
    ("full", "default"),
    ("full_pooled", "default"),
    ("full_pooled", "full"),
    ("pure_lead_pooled", "pure_pooled"),
    ("pure_plant", "ref_pure_mdn313"),
)
EXTRA: Final[tuple[tuple[str, str], ...]] = (
    ("pure_plant", "default"),
    ("pure_lead_pooled", "ref_pure_mdn313"),
    ("full", "lead"),
    ("pure_pooled", "ref_pure_mdn313"),
    ("pure_lead_pooled", "full_pooled"),
    ("pure_lead_pooled", "default"),
    ("default_pooled", "default"),
    ("full_pooled", "default_pooled"),
    ("full_shuffled", "default"),
    ("full_nolag", "full"),
    ("full_nolag", "default"),
    ("default", "ref_mdn310"),
    ("default", "floor_climatology"),
    ("default", "floor_heuristic"),
    ("full", "floor_climatology"),
    ("full_pooled", "floor_climatology"),
    ("ref_pure_mdn313", "default"),
)
METRICS: Final[tuple[str, ...]] = (
    "score",
    "pin01",
    "pin99",
    "cover01",
    "cover99",
    "ae_mean",
    "median_minus_mean",
)
HALVES: Final[dict[str, tuple[str, ...]]] = {
    "h1": ("2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"),
    "h2": ("2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"),
}


def seed_average(*, files: list, arm_name: str | None = None) -> pl.DataFrame:
    """Return one row per (farm, time, init_time) with quantiles averaged over seeds."""
    frame = pl.concat(
        [
            pl.read_parquet(f).select(
                *s310.KEYS, "lead_hours", "arm", "setting", "band", "actual", *s310.QCOLS
            )
            for f in files
        ]
    ).with_columns(pl.col(s310.QCOLS).cast(pl.Float64), pl.col("lead_hours").cast(pl.Float64))
    if arm_name is not None:
        frame = frame.with_columns(arm=pl.lit(arm_name))
    return frame.group_by(*s310.KEYS, "arm", "setting", "band").agg(
        pl.col("lead_hours").first(),
        pl.col("actual").first(),
        *[pl.col(c).mean() for c in s310.QCOLS],
    )


def load_band(*, band: str) -> pl.DataFrame:
    """Return per-row losses of every compared arm and setting on the band's common rows."""
    floors = pl.read_parquet(c310.DATA_DIR / f"floors_{band}.parquet")
    pieces = []
    for setting in c310.SETTINGS:
        mine = sorted((common.PRED_DIR / band / setting).glob("*.parquet"))
        per_arm = {}
        for f in mine:
            arm = f.name.split("_s")[0] if "_s" in f.name else f.name
            per_arm[arm] = per_arm.get(arm, 0) + 1
        for arm, n_files in per_arm.items():
            seeds = common.OWN_SEEDS if setting == "own_history" else common.ZERO_SEEDS
            if n_files != len(seeds) * len(c310.FARMS) * 4:
                msg = f"{band} {setting} {arm}: {n_files} files, expected {len(seeds) * 24}"
                raise ValueError(msg)
        parts = [seed_average(files=mine)]
        parts.append(
            seed_average(
                files=sorted((c310.PRED_DIR / band / setting).glob("mdn_s*.parquet")),
                arm_name="ref_mdn310",
            )
        )
        parts.append(
            seed_average(
                files=sorted((common.E313_PRED_DIR / band / setting).glob("pure_mdn_s*.parquet")),
                arm_name="ref_pure_mdn313",
            )
        )
        columns = parts[0].columns
        parts.append(
            floors.with_columns(
                setting=pl.lit(setting), lead_hours=pl.col("lead_hours").cast(pl.Float64)
            ).select(columns)
        )
        both = pl.concat([p.select(columns) for p in parts])
        finite = both.filter(pl.all_horizontal([pl.col(c).is_finite() for c in s310.QCOLS]))
        arms = finite["arm"].unique().to_list()
        keep = (
            finite.group_by(s310.KEYS)
            .agg(n=pl.col("arm").n_unique())
            .filter(pl.col("n") == len(arms))
            .select(s310.KEYS)
        )
        pieces.append(s310.row_losses(frame=finite.join(keep, on=s310.KEYS, how="inner")))
    return pl.concat(pieces).with_columns(
        month=pl.col("init_time").dt.strftime("%Y-%m"), farm=pl.col("time_series_id")
    )


def main() -> None:
    """Print and write the score and contrast tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    scores, contrasts, all_rows = [], [], []
    for band in c310.BAND_ORDER:
        rows = load_band(band=band)
        all_rows.append(rows)
        for setting in c310.SETTINGS:
            sub = rows.filter(pl.col("setting") == setting)
            n = sub.group_by("arm").agg(n=pl.len())
            if n["n"].min() != n["n"].max():
                msg = f"arms hold different row counts in {band} {setting}:\n{n.sort('n')}"
                raise ValueError(msg)
            for half, months in [("all", None), *HALVES.items()]:
                part = sub if months is None else sub.filter(pl.col("month").is_in(months))
                for metric in METRICS:
                    sums, counts, arms = s310.farm_sums(rows=part, value=metric)
                    point, draws = s310.boot(sums=sums, counts=counts, seed=318)
                    idx = {a: i for i, a in enumerate(arms)}
                    for a in arms:
                        lo, hi = np.nanpercentile(draws[idx[a]], [LO_PCT, HI_PCT])
                        scores.append(
                            dict(band=band, setting=setting, half=half, metric=metric, arm=a, value=point[idx[a]], lo=lo, hi=hi)
                        )
                    if metric not in ("score", "pin01", "pin99", "ae_mean"):
                        continue
                    for arm, ref in (*PLANNED, *EXTRA):
                        if arm not in idx or ref not in idx:
                            continue
                        change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                        diff = draws[idx[arm]] - draws[idx[ref]]
                        lo, hi = np.nanpercentile(change, [LO_PCT, HI_PCT])
                        dlo, dhi = np.nanpercentile(diff, [LO_PCT, HI_PCT])
                        contrasts.append(
                            dict(
                                band=band, setting=setting, half=half, metric=metric, arm=arm, reference=ref,
                                planned=(arm, ref) in PLANNED,
                                points=point[idx[arm]] - point[idx[ref]], points_lo=dlo, points_hi=dhi,
                                percent=100.0 * (point[idx[arm]] - point[idx[ref]]) / point[idx[ref]],
                                lo=lo, hi=hi,
                            )
                        )
        print("scored", band, flush=True)
    pl.concat(all_rows).write_parquet(common.DATA_DIR / "row_losses.parquet")
    pl.DataFrame(scores).write_csv(common.DATA_DIR / "scores.csv")
    pl.DataFrame(contrasts).write_csv(common.DATA_DIR / "contrasts.csv")
    table = pl.DataFrame(scores).filter(
        (pl.col("metric") == "score") & (pl.col("half") == "all")
    )
    for setting in c310.SETTINGS:
        print(f"--- {setting}: quantile score by band")
        print(
            table.filter(pl.col("setting") == setting)
            .pivot(on="arm", index="band", values="value")
            .sort("band")
        )


if __name__ == "__main__":
    main()
