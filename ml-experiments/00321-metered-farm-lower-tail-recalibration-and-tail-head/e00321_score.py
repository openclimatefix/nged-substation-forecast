# ruff: noqa: E501
"""Score experiment 321: per-row losses, farm-equal means, month-block intervals at 99.83%.

Per band and history setting, the rows are the scored-year runs that every compared arm holds with
finite quantiles. Research default 1 is experiment 310's stored `blend`; its three families, the
paired five-seed comparator, and the floors sit beside the new arms. The quantile score is twice the
mean pinball loss over the 11 levels in points of capacity. Coverage is reported strictly below the
quantile (headline) and at or below it (the rule of experiments 310 and 318), with the exact-zero
share of the actual values. Intervals resample the 12 scored months with farm-equal weights at the
Bonferroni level for six planned contrasts per band (99.83%); `h1` and `h2` are the halves of the
scored year.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00321-... \
        uv run --no-sync python ml-experiments/00321-*/e00321_score.py
"""

from typing import Final

import e00310_blend as b310  # ty: ignore[unresolved-import]
import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_score as s310  # ty: ignore[unresolved-import]
import e00321_assemble as asm
import e00321_common as common
import e00321_recal as recal
import numpy as np
import polars as pl

s310.DRAWS = 40_000
LO_PCT, HI_PCT = 0.0833, 99.9167
QCOLS: Final[list[str]] = recal.QCOLS
NEW_ARMS: Final[tuple[str, ...]] = (
    "recal",
    "recal_shrunk",
    "recal_stale7",
    "recal_swapped",
    "zi_alone",
    "zi_blend",
    "mdn5_318",
    "rd1_5seed",
    "tail",
    "tail_shuffled",
    "pure_lead_pooled",
    "combo",
)
STORED: Final[dict[str, str]] = {
    "rd1": "blend",
    "rd1_gam": "gam",
    "rd1_mdn": "mdn_ens",
    "rd1_xgb": "xgb",
}
PLANNED: Final[tuple[tuple[str, str], ...]] = (
    ("recal", "rd1"),
    ("recal_shrunk", "rd1"),
    ("zi_blend", "rd1"),
    ("tail", "rd1"),
    ("pure_lead_pooled", "rd1"),
    ("combo", "rd1"),
)
EXTRA: Final[tuple[tuple[str, str], ...]] = (
    ("zi_blend", "rd1_5seed"),
    ("zi_alone", "mdn5_318"),
    ("zi_alone", "rd1_mdn"),
    ("rd1_5seed", "rd1"),
    ("recal_stale7", "rd1"),
    ("recal_swapped", "rd1"),
    ("tail_shuffled", "rd1"),
    ("tail", "tail_shuffled"),
    ("pure_lead_pooled", "rd1_mdn"),
    ("rd1", "floor_climatology"),
    ("rd1", "floor_heuristic"),
    ("recal", "floor_climatology"),
    ("tail", "floor_climatology"),
    ("rd1_mdn", "floor_climatology"),
    ("rd1_gam", "floor_climatology"),
    ("rd1_xgb", "floor_climatology"),
    ("rd1_mdn", "rd1_gam"),
    ("rd1_mdn", "rd1_xgb"),
    ("rd1", "rd1_mdn"),
    ("combo", "tail"),
    ("combo", "recal"),
)
METRICS: Final[tuple[str, ...]] = (
    "score",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "strict01",
    "strict05",
    "strict95",
    "strict99",
    "cover01",
    "cover99",
    "zero_actual",
    "ae_mean",
    "median_minus_mean",
)
CONTRAST_METRICS: Final[tuple[str, ...]] = ("score", "pin01", "pin05", "pin95", "pin99", "ae_mean")
HALVES: Final[dict[str, tuple[str, ...]]] = {
    "h1": ("2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"),
    "h2": ("2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"),
}


def row_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses on sorted quantiles, with strict coverage and the zero indicator."""
    out = s310.row_losses(frame=frame)
    q = np.sort(frame.select(QCOLS).to_numpy(), axis=1)
    actual = frame["actual"].to_numpy()
    extra = {
        f"strict{name}": (actual < q[:, i]).astype(float) for name, i in s310.TAIL_INDEX.items()
    }
    return out.with_columns(
        *[pl.Series(k, v) for k, v in extra.items()],
        pl.Series("zero_actual", (actual <= common.ZERO_OUTPUT).astype(float)),
    )


def load_arm_frames(*, band: str, setting: str) -> dict[str, pl.DataFrame]:
    """Return every compared arm's quantile frame for a band and setting."""
    columns = ["time_series_id", "time", "init_time", "lead_hours", "actual", *QCOLS]
    frames: dict[str, pl.DataFrame] = {}
    for name, stored in STORED.items():
        frames[name] = (
            recal.load_stored(band=band, setting=setting, arm=stored)
            if stored in ("blend", "mdn_ens")
            else b310.load_component(band=band, setting=setting, arm=stored)
        )
    for name in NEW_ARMS:
        files = sorted((asm.ASSEMBLED / band / setting).glob(f"{name}_s0_?.parquet"))
        if files:
            frames[name] = asm.read_assembled(band=band, setting=setting, arm=name)
    return {
        k: v.select(columns).with_columns(pl.col(["lead_hours", "actual", *QCOLS]).cast(pl.Float64))
        for k, v in frames.items()
    }


def load_band(*, band: str) -> pl.DataFrame:
    """Return per-row losses of every compared arm and setting on the band's common rows."""
    pieces = []
    floors = pl.read_parquet(c310.DATA_DIR / f"floors_{band}.parquet")
    for setting in c310.SETTINGS:
        frames = load_arm_frames(band=band, setting=setting)
        # The floors file holds both floors under one arm name; split by its arm column.
        arms_in_floors = floors["arm"].unique().to_list()
        for key in ("floor_climatology", "floor_heuristic"):
            if key in arms_in_floors:
                frames[key] = (
                    floors.filter(pl.col("arm") == key)
                    .select("time_series_id", "time", "init_time", "lead_hours", "actual", *QCOLS)
                    .with_columns(pl.col(["lead_hours", "actual", *QCOLS]).cast(pl.Float64))
                )
        tagged = []
        for name, frame in frames.items():
            tagged.append(
                frame.with_columns(
                    arm=pl.lit(name), setting=pl.lit(setting), band=pl.lit(band)
                ).select(
                    "time_series_id",
                    "time",
                    "init_time",
                    "lead_hours",
                    "actual",
                    "arm",
                    "setting",
                    "band",
                    *QCOLS,
                )
            )
        both = pl.concat(tagged)
        finite = both.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
        required = [a for a in finite["arm"].unique().to_list() if a != "combo"]
        keep = (
            finite.filter(pl.col("arm").is_in(required))
            .group_by(s310.KEYS)
            .agg(n=pl.col("arm").n_unique())
            .filter(pl.col("n") == len(required))
            .select(s310.KEYS)
        )
        pieces.append(row_losses(frame=finite.join(keep, on=s310.KEYS, how="inner")))
    return pl.concat(pieces).with_columns(
        month=pl.col("init_time").dt.strftime("%Y-%m"), farm=pl.col("time_series_id")
    )


def main() -> None:
    """Print and write the score and contrast tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    scores, contrasts, all_rows = [], [], []
    for band in c310.BAND_ORDER:
        rows = load_band(band=band)
        all_rows.append(rows)
        for setting in c310.SETTINGS:
            sub = rows.filter(pl.col("setting") == setting)
            counts_by_arm = sub.group_by("arm").agg(n=pl.len())
            if counts_by_arm["n"].min() != counts_by_arm["n"].max():
                msg = f"arms hold different row counts in {band} {setting}:\n{counts_by_arm.sort('n')}"
                raise ValueError(msg)
            for half, months in [("all", None), *HALVES.items()]:
                part = sub if months is None else sub.filter(pl.col("month").is_in(months))
                for metric in METRICS:
                    sums, counts, arms = s310.farm_sums(rows=part, value=metric)
                    point, draws = s310.boot(sums=sums, counts=counts, seed=321)
                    idx = {a: i for i, a in enumerate(arms)}
                    for a in arms:
                        lo, hi = np.nanpercentile(draws[idx[a]], [LO_PCT, HI_PCT])
                        scores.append(
                            {
                                "band": band,
                                "setting": setting,
                                "half": half,
                                "metric": metric,
                                "arm": a,
                                "value": point[idx[a]],
                                "lo": lo,
                                "hi": hi,
                            }
                        )
                    if metric not in CONTRAST_METRICS:
                        continue
                    for arm, ref in (*PLANNED, *EXTRA):
                        if arm not in idx or ref not in idx:
                            continue
                        diff = draws[idx[arm]] - draws[idx[ref]]
                        change = 100.0 * diff / draws[idx[ref]]
                        lo, hi = np.nanpercentile(change, [LO_PCT, HI_PCT])
                        dlo, dhi = np.nanpercentile(diff, [LO_PCT, HI_PCT])
                        contrasts.append(
                            {
                                "band": band,
                                "setting": setting,
                                "half": half,
                                "metric": metric,
                                "arm": arm,
                                "reference": ref,
                                "planned": (arm, ref) in PLANNED,
                                "points": point[idx[arm]] - point[idx[ref]],
                                "points_lo": dlo,
                                "points_hi": dhi,
                                "percent": 100.0
                                * (point[idx[arm]] - point[idx[ref]])
                                / point[idx[ref]],
                                "lo": lo,
                                "hi": hi,
                            }
                        )
        print("scored", band, flush=True)
    pl.concat(all_rows).write_parquet(common.DATA_DIR / "row_losses.parquet")
    pl.DataFrame(scores).write_csv(common.DATA_DIR / "scores.csv")
    pl.DataFrame(contrasts).write_csv(common.DATA_DIR / "contrasts.csv")
    table = pl.DataFrame(scores).filter((pl.col("metric") == "score") & (pl.col("half") == "all"))
    for setting in c310.SETTINGS:
        print(f"--- {setting}: quantile score by band")
        print(
            table.filter(pl.col("setting") == setting)
            .pivot(on="arm", index="band", values="value")
            .sort("band")
        )


if __name__ == "__main__":
    main()
