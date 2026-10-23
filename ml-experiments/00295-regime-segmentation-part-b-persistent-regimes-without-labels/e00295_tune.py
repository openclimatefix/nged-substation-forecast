# ruff: noqa: E501
"""Choose the penalty and minimum segment length on odd months and tuning Primaries.

Writes `chosen_config.json` and `usable_sites.json` in the experiment's data folder.
"""

from __future__ import annotations

import json
import pickle

import e00295_common as c
import numpy as np
import polars as pl


def site_years() -> dict[str, float]:
    """Return the span of each Primary's daily series in years."""
    states = pickle.loads((c.EXPERIMENT_DIR / "clean_states.pkl").read_bytes())
    return {
        name: float((s["series"].days[-1] - s["series"].days[0]) / np.timedelta64(1, "D") / 365.25)
        for name, s in states.items()
    }


def primaries() -> tuple[list[str], list[str]]:
    """Return the tuning and held-out Primaries: the usable ones other than D8, by sorted label."""
    states = pickle.loads((c.EXPERIMENT_DIR / "clean_states.pkl").read_bytes())
    usable = sorted(s for s in states if s != "D8")
    return usable[0::2], usable[1::2]


def load(kind: str) -> pl.DataFrame:
    """Read every Primary's rows of one record kind."""
    files = sorted((c.EXPERIMENT_DIR / "segmentation").glob("*.parquet"))
    frame = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return frame.filter(pl.col("record") == kind)


def false_breaks(*, clean: pl.DataFrame, sites: list[str], odd_months_only: bool) -> pl.DataFrame:
    """Return regime changes (boundaries not labelled temporary) per Primary-year per configuration."""
    rows = clean.filter(pl.col("site").is_in(sites) & (pl.col("label") != "temporary"))
    scale = 1.0
    if odd_months_only:
        rows = rows.filter(pl.col("date").str.slice(5, 2).cast(pl.Int32) % 2 == 1)
        scale = 2.0
    return (
        rows.group_by("beta", "min_length")
        .agg(n=pl.len())
        .with_columns(per_primary_year=pl.col("n") * scale / sum(site_years()[x] for x in sites))
    )


def recalls(*, injections: pl.DataFrame, sites: list[str], parity: str) -> pl.DataFrame:
    """Return persistent and temporary recall per configuration, for the fitted-absorption arm."""
    rows = injections.filter(
        pl.col("site").is_in(sites)
        & (pl.col("parity") == parity)
        & (pl.col("absorption") == "fit")
        & pl.col("family").is_in(["persistent", "temporary"])
    )
    return (
        rows.group_by("beta", "min_length", "family")
        .agg(recall=(pl.col("label") == pl.col("expected")).mean())
        .pivot(on="family", index=["beta", "min_length"], values="recall")
        .rename({"persistent": "persistent_recall", "temporary": "temporary_recall"})
    )


def main() -> None:
    """Pick the configuration and write it."""
    tune, hold = primaries()
    clean, injections = load("clean_break"), load("injection")
    table = (
        recalls(injections=injections, sites=tune, parity="odd")
        .join(
            false_breaks(clean=clean, sites=tune, odd_months_only=True),
            on=["beta", "min_length"],
            how="left",
        )
        .with_columns(
            pl.col("per_primary_year").fill_null(0.0),
            objective=(pl.col("persistent_recall") + pl.col("temporary_recall")) / 2,
        )
    )
    print(table.sort("beta", "min_length"))
    ok = table.filter(pl.col("per_primary_year") < 1.0)
    pick = (
        (ok if ok.height else table.sort("per_primary_year").head(3))
        .sort("objective", descending=True)
        .row(0, named=True)
    )
    print("chosen:", pick, "constraint met:", bool(ok.height))
    (c.EXPERIMENT_DIR / "chosen_config.json").write_text(
        json.dumps(
            {
                "beta": pick["beta"],
                "min_length": pick["min_length"],
                "constraint_met": bool(ok.height),
                "tune_sites": tune,
                "hold_sites": hold,
            }
        )
    )
    (c.EXPERIMENT_DIR / "usable_sites.json").write_text(json.dumps(["D8", *tune, *hold]))


if __name__ == "__main__":
    main()
