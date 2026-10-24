"""Compare the level-based first design with the final design on the Primaries both ran."""

from __future__ import annotations

from pathlib import Path

import e00295_common as c
import polars as pl

OLD = c.EXPERIMENT_DIR / "segmentation_level_only"
NEW = c.EXPERIMENT_DIR / "segmentation"
YEARS = 6.79


def summary(*, folder: Path, sites: list[str]) -> pl.DataFrame:
    """Return, per configuration, regime changes per Primary-year and recall by size and kind."""
    frame = pl.concat(
        [pl.read_parquet(folder / f"{s}.parquet") for s in sites], how="diagonal_relaxed"
    )
    clean = frame.filter((pl.col("record") == "clean_break") & (pl.col("label") != "temporary"))
    rate = clean.group_by("beta", "min_length").agg(rate=pl.len() / (len(sites) * YEARS))
    inj = frame.filter((pl.col("record") == "injection") & (pl.col("absorption") == "fit"))
    persistent = (
        inj.filter(pl.col("family") == "persistent")
        .group_by("beta", "min_length", "size")
        .agg(recall=(pl.col("label") == "persistent").mean())
        .pivot(on="size", index=["beta", "min_length"], values="recall")
        .rename({"0.1": "recall_10pct", "0.25": "recall_25pct"})
    )
    temporary = (
        inj.filter(pl.col("family") == "temporary")
        .group_by("beta", "min_length")
        .agg(recall_temporary=(pl.col("label") == "temporary").mean())
    )
    return (
        rate.join(persistent, on=["beta", "min_length"])
        .join(temporary, on=["beta", "min_length"])
        .sort("beta", "min_length")
    )


def main() -> None:
    """Print both designs on D10, D11, D12, D14, and D16, all onsets."""
    sites = ["D10", "D11", "D12", "D14", "D16"]
    for name, folder in (
        ("level only (first design)", OLD),
        ("distribution features (final)", NEW),
    ):
        print(f"\n== {name}, {sites} ==")
        with pl.Config(tbl_rows=30):
            print(summary(folder=folder, sites=sites).filter(pl.col("min_length") == 14))


if __name__ == "__main__":
    main()
