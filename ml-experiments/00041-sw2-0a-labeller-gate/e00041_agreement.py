"""Agreement between the two Sonnet runs on the SW2-0b cards, and the Opus adjudication counts.

uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_agreement.py
"""

from __future__ import annotations

import collections
import json

import polars as pl
from e00041_common import WORK_DIR
from e00041_kappa import agree, kappa

OUT = WORK_DIR / "real"
VIS = ("conserved", "one_sided")


def main() -> None:
    """Print kappa, the disagreement counts, and the visible real events' size summaries."""
    meta = json.loads((OUT / "meta.json").read_text())
    one = json.loads((OUT / "labels_sonnet1_merged.json").read_text())
    two = json.loads((OUT / "labels_sonnet2_merged.json").read_text())
    ids = sorted(two)
    four = kappa([one[c]["visibility"] for c in ids], [two[c]["visibility"] for c in ids])
    binary = kappa(
        ["v" if one[c]["visibility"] in VIS else "n" for c in ids],
        ["v" if two[c]["visibility"] in VIS else "n" for c in ids],
    )
    print(f"{len(ids)} cards labelled twice; kappa {four:.3f} (four labels), {binary:.3f} (binary)")
    real = [c for c in ids if not meta[c]["kind"].startswith("sentinel")]
    disagree = [c for c in real if not agree(one[c], two[c])]
    low = [c for c in real if one[c]["visibility"] in VIS and one[c].get("confidence") == "low"]
    print(
        f"real cards: {len(disagree)} disagreements, "
        f"{len(set(low) - set(disagree))} more low-confidence"
    )
    adjudicated = json.loads((OUT / "adjudicate_cards.json").read_text())
    print("sent to Opus:", len(adjudicated))
    frame = pl.read_parquet(OUT / "labelled_set.parquet")
    ops = frame.filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(list(VIS)),
    ).with_columns(
        hours=(pl.col("logged_open_utc") - pl.col("logged_close_utc")).dt.total_minutes() / 60,
        share=pl.col("moved_share_of_receiver").fill_nan(None),
        donor_share=pl.col("moved_share_of_donor_range").fill_nan(None),
    )
    print(
        ops.select(
            pl.len().alias("visible_operations"),
            pl.col("moved_mw_p50").median().alias("median_mw"),
            pl.col("moved_mw_p50").quantile(0.1).alias("p10_mw"),
            pl.col("moved_mw_p50").quantile(0.9).alias("p90_mw"),
            pl.col("donor_share").median().alias("median_share_of_donor_range"),
            pl.col("share").is_not_null().sum().alias("with_receiver_share"),
        )
    )
    bands = ops.group_by(
        pl.when(pl.col("hours") < 6)
        .then(pl.lit("2 to 6 h"))
        .when(pl.col("hours") < 24)
        .then(pl.lit("6 to 24 h"))
        .otherwise(pl.lit("over 1 d"))
        .alias("duration")
    ).len()
    print(bands.sort("duration"))
    sizes = (
        ops.filter(pl.col("share").is_not_null())
        .group_by(
            pl.when(pl.col("share") < 0.1)
            .then(pl.lit("under 10%"))
            .when(pl.col("share") < 0.25)
            .then(pl.lit("10% to 25%"))
            .otherwise(pl.lit("25% or more"))
            .alias("size")
        )
        .len()
    )
    print(sizes.sort("size"))
    conserved = frame.filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility") == "conserved",
    )
    missing = conserved.filter(pl.col("max_side_missing_share") > 0.5).height
    print(f"conserved operations with a side over half missing: {missing} of {conserved.height}")
    alarms = frame.filter(pl.col("kind") == "alarm", pl.col("visibility").is_in(list(VIS)))
    print("visible alarms: median MW", alarms["moved_mw_p50"].median(), "n", alarms.height)
    after = collections.Counter(
        frame.filter(pl.col("adjudicated"), ~pl.col("is_sentinel"))["visibility"].to_list()
    )
    print("after adjudication:", dict(after))


if __name__ == "__main__":
    main()
