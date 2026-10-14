"""Post hoc: the 14-day correction applied to the best recency arm and the best window arm.

Reads the scored rows `e00017_report.py` wrote. These contrasts were not pre-registered, so the
intervals are plain 95% ones.

    uv run python ml-experiments/00017-*/e00017_posthoc.py
"""

import e00017_fit as fit
import e00017_report as report
import polars as pl
from studies.bootstrap import bootstrap_absolute, bootstrap_difference

PAIRS: tuple[tuple[str, str], ...] = (("K_H0.5", "H0.5"), ("K_H0.5", "K"), ("K_W12", "W12"))


def main() -> None:
    """Print the absolute errors and the post hoc contrasts."""
    rows = pl.read_parquet(report.RESULTS_DIR / "scored_rows.parquet")
    rows = report.add_correction(rows=rows, arm="H0.5", name="K_H0.5")
    rows = report.add_correction(rows=rows, arm="W12", name="K_W12")
    arms = sorted({a for pair in PAIRS for a in pair})
    frame = report.loss_frame(rows=rows, arms=arms)
    out = []
    for arm in arms:
        r = bootstrap_absolute(losses=frame, arm=arm, metric="abs_error")
        out.append({"arm": arm, "mae": r["value"], "lower": r["lower_95"], "upper": r["upper_95"]})
    print(pl.DataFrame(out))
    for treatment, reference in PAIRS:
        r = bootstrap_difference(
            losses=frame, treatment=treatment, reference=reference, metric="abs_error"
        )
        print(treatment, "minus", reference, r["difference"], r["lower_95"], r["upper_95"])
    by_year = (
        rows.group_by(pl.col("valid_time").dt.year().alias("year"))
        .agg(
            [(pl.col(a) - pl.col("target")).abs().mean().mul(100).alias(f"mae_{a}") for a in arms]
            + [(pl.col("target") - pl.col(a)).mean().mul(100).alias(f"bias_{a}") for a in arms]
        )
        .sort("year")
    )
    print(by_year)
    print(fit.EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
