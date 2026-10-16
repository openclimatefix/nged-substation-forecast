"""Print every number experiment 00044's write-up quotes, from `e00044_score.py`'s results.

uv run python ml-experiments/00044-sw2-2-switching-share-of-best-guess-score/e00044_report.py
"""

import e00044_score as score
import polars as pl


def _line(*, row: dict) -> str:
    return f"{row['point']:9.3f}  ({row['lower_95']:.3f} to {row['upper_95']:.3f})"


def main() -> None:
    """Print the events, the shares, the bounds, and the per-series table."""
    cases = score.load_cases()
    events = score.load_events(series=cases.series)
    first, last = cases.frame["valid_time"].min(), cases.frame["valid_time"].max()
    print(f"cases {cases.frame.height} from {first} to {last}; series {cases.series}")
    labels = (
        events.unique("event_id")
        .select(
            n_visible=pl.len(),
            n_with_share=pl.col("share").is_not_null().sum(),
            n_stored_share=pl.col("stored_share").is_not_nan().sum(),
            n_trustworthy=pl.col("trustworthy").sum(),
            n_trustworthy_complete=pl.col("trustworthy_complete").sum(),
        )
        .row(0, named=True)
    )
    print("visible real events, all years:", labels)
    for event_set in score.EVENT_SETS:
        chosen = events.filter(pl.col(event_set))
        in_year = chosen.filter((pl.col("end") >= first) & (pl.col("onset") <= last))
        months = sorted({f"{t:%Y-%m}" for t in in_year["onset"]})
        print(
            f"\n== {event_set}: {chosen['event_id'].n_unique()} events in all years; "
            f"{in_year['event_id'].n_unique()} overlap the scored year on "
            f"{in_year['time_series_id'].n_unique()} series; onset months {months}"
        )
        results = pl.read_parquet(score.WORK_DIR / "results.parquet").filter(
            pl.col("event_set") == event_set
        )
        for row in results.iter_rows(named=True):
            print(f"{row['statistic']:48s} {_line(row=row)}")
        print(
            pl.read_parquet(score.WORK_DIR / "per_series.parquet")
            .filter(pl.col("event_set") == event_set)
            .drop("event_set")
        )


if __name__ == "__main__":
    main()
