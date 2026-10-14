"""Post hoc diagnostic: how much of each arm's error is a constant level offset per series.

Every arm's median under-forecasts the validation year (the signed error of the median is
negative, see `bias.parquet`). This script subtracts, for each arm and series, the mean signed
error over the whole validation year from that arm's median, then scores the mean absolute error
again. The offset is estimated on the validation year itself, so the result is an oracle bound on
what a level correction could recover, never a forecast.

Output: `work/results/debias.parquet` with the Primary substations' mean absolute error of the
median, before and after, by arm and lead scope; and a table on standard output.

    uv run python ml-experiments/00013-primary-demand-against-stored-baselines/e00013_debias.py
"""

import e00013_common as common
import e00013_score as score
import polars as pl

ARMS: tuple[str, ...] = ("manual_heuristic", "climatology", "production", "s1", "n1", "n1_nolag")


def main() -> None:
    """Write and print the before-and-after errors."""
    series = common.series_table()
    primary = series.filter(pl.col("substation_type") == "Primary")["time_series_id"].to_list()
    cases = score._actual_table(series=series)
    raw = score.collect_arms(cases=cases)
    frames = {}
    for name in ARMS:
        frames[name] = (
            raw[name]
            .filter(
                (pl.col("seed") == 0)
                & pl.col("time_series_id").is_in(primary)
                & pl.col("actual").is_finite()
                & (pl.col("valid_time") < score.HOLDOUT)
            )
            .select(*score.KEYS, "q50", "actual")
        )
    shared = frames[ARMS[0]].select(score.KEYS)
    for frame in frames.values():
        shared = shared.join(frame.select(score.KEYS), on=score.KEYS, how="inner")
    rows = []
    for name, frame in frames.items():
        kept = (
            frame.join(shared, on=score.KEYS, how="semi")
            .with_columns(
                error=(pl.col("q50") - pl.col("actual")) * 100.0,
                lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                / 60.0,
            )
            .with_columns(offset=pl.col("error").mean().over("time_series_id"))
            .with_columns(corrected=pl.col("error") - pl.col("offset"))
        )
        for scope, condition in {
            "all leads": pl.lit(True),
            "up to 48 h": pl.col("lead") < 48,
            "3 d or more": pl.col("lead") >= 72,
        }.items():
            part = kept.filter(condition)
            rows.append(
                {
                    "arm": name,
                    "scope": scope,
                    "mae": part.select(pl.col("error").abs().mean()).item(),
                    "mae_after_level_correction": part.select(
                        pl.col("corrected").abs().mean()
                    ).item(),
                    "mean_offset": part.select(pl.col("error").mean()).item(),
                    "rows": part.height,
                }
            )
    table = pl.DataFrame(rows)
    table.write_parquet(score.RESULTS_DIR / "debias.parquet")
    with pl.Config(tbl_rows=30, tbl_cols=10, tbl_width_chars=160):
        print(
            table.with_columns(pl.col("mae", "mae_after_level_correction", "mean_offset").round(2))
        )


if __name__ == "__main__":
    main()
