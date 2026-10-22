# ruff: noqa: E501
"""Part (a): per source and threshold, alarms per matched load shift, matched count, AFTER share, harm.

Reads `scores_`, `meta_`, `after_`, and `spans_` files of `e00227_run.py` and `found35.parquet` of
`e00227_found.py`. Writes `part_a.parquet` and prints every number the page quotes.

    OMP_NUM_THREADS=1 uv run --no-sync python e00227_report.py
"""

from __future__ import annotations

import json
from typing import Final

import e00227_alarms as al
import numpy as np
import polars as pl
from e00227_common import WORK_DIR

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
N_DRAWS: Final[int] = 20_000
LIMIT: Final[float] = 0.05
FOLDLESS: Final[frozenset[str]] = frozenset({"K0", "ORACLE", "PAIR_NEW", "PAIR_GATED"})


def scalar(value: object) -> float:
    """A polars scalar as a float."""
    return float(value)  # ty: ignore[invalid-argument-type]


def foldless(arm: str) -> bool:
    """Whether an arm has one score for every fold."""
    return arm in FOLDLESS or al.is_foldless(arm)


def boot_ratio(
    *, num: np.ndarray, den: np.ndarray, groups: np.ndarray, seed: int = 227
) -> tuple[float, float, float]:
    """Ratio of sums with a 95% interval from resampling whole groups."""
    labels, inverse = np.unique(groups, return_inverse=True)
    n = np.bincount(inverse, weights=num, minlength=len(labels))
    d = np.bincount(inverse, weights=den, minlength=len(labels))
    picks = np.random.default_rng(seed).integers(0, len(labels), size=(N_DRAWS, len(labels)))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = n[picks].sum(axis=1) / d[picks].sum(axis=1)
    return (
        float(n.sum() / d.sum()),
        float(np.nanpercentile(draws, 2.5)),
        float(np.nanpercentile(draws, 97.5)),
    )


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The meta, after-row, and score tables of every site."""

    def cat(name: str) -> pl.DataFrame:
        return pl.concat(
            [
                pl.read_parquet(WORK_DIR / f"{name}_{s}.parquet").with_columns(
                    **({} if name == "meta" else {"site": pl.lit(s)})
                )
                for s in SITES
            ],
            how="diagonal_relaxed",
        )

    return cat("meta"), cat("after"), cat("scores")


def main() -> None:
    """Print the part (a) table."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    meta, after, scores = load()
    meta = meta.with_columns(
        mo=pl.col("valid_time").dt.month(), month=pl.col("valid_time").dt.strftime("%Y-%m")
    )
    found = pl.read_parquet(WORK_DIR / "found35.parquet")
    summary = (
        found.group_by("source", maintain_order=True)
        .agg(matched=pl.col("matched").sum(), alarms=pl.col("alarms").first())
        .with_columns(alarms_per_match=pl.col("alarms") / pl.col("matched"))
    )

    # ---- ordinary-day harm: arm minus K0, learned arms averaged over the three folds ----
    ordinary = meta.filter(pl.col("ordinary")).select("site", "row", "month", "mo")
    sc = scores.join(ordinary, on=["site", "row"], how="inner")
    wide = (
        sc.group_by("site", "row", "arm")
        .agg(qs=pl.col("qs").mean())
        .pivot(on="arm", index=["site", "row"], values="qs")
        .join(ordinary, on=["site", "row"])
    )
    print("ordinary rows:", wide.height, " K0 absolute:", round(scalar(wide["K0"].mean()), 3))
    odd = pl.col("mo") % 2 == 1

    def harm(frame: pl.DataFrame, arm: str) -> tuple[float, float, float]:
        diff = (frame[arm] - frame["K0"]).to_numpy().astype(float)
        return boot_ratio(num=diff, den=np.ones(len(diff)), groups=frame["month"].to_numpy())

    rows = []
    # ---- AFTER share recovered (logged-oracle gain as the denominator) ----
    ev = after.join(meta.select("site", "row", "run_period"), on=["site", "row"], how="inner")
    ev = ev.filter(pl.col("run_period").is_in(["scored", "replicate"]))
    long = []
    for arm in scores["arm"].unique().to_list():
        s = scores.filter(pl.col("arm") == arm)
        if foldless(arm):
            j = ev.join(s.drop("fold"), on=["site", "row"], how="left")
        else:
            j = ev.join(s, on=["site", "row", "fold"], how="left")
        long.append(j.select("site", "card", "row", pl.lit(arm).alias("arm"), "qs"))
    longf = pl.concat(long).pivot(on="arm", index=["site", "card", "row"], values="qs")
    print(
        "AFTER rows:",
        longf.height,
        "cards:",
        longf["card"].n_unique(),
        " K0",
        round(scalar(longf["K0"].mean()), 3),
        " ORACLE",
        round(scalar(longf["ORACLE"].mean()), 3),
        " PAIR_NEW",
        round(scalar(longf["PAIR_NEW"].mean()), 3),
    )

    for source in [*al.source_ids(), "PAIR_NEW", "PAIR_GATED"]:
        h_all = harm(wide, source)
        h_even = harm(wide.filter(~odd), source)
        h_odd = harm(wide.filter(odd), source)
        share = boot_ratio(
            num=(longf["K0"] - longf[source]).to_numpy().astype(float),
            den=(longf["K0"] - longf["ORACLE"]).to_numpy().astype(float),
            groups=longf["card"].to_numpy(),
        )
        rows.append(
            {
                "source": source,
                "harm": h_all[0],
                "harm_lo": h_all[1],
                "harm_hi": h_all[2],
                "harm_even": h_even[0],
                "harm_even_lo": h_even[1],
                "harm_even_hi": h_even[2],
                "harm_odd": h_odd[0],
                "share": share[0],
                "share_lo": share[1],
                "share_hi": share[2],
                "after_score": scalar(longf[source].mean()),
            }
        )
    table = (
        pl.DataFrame(rows)
        .join(summary, on="source", how="left")
        .with_columns(
            inside=(pl.col("harm_lo") >= -LIMIT) & (pl.col("harm_hi") <= LIMIT),
            inside_even=(pl.col("harm_even_lo") >= -LIMIT) & (pl.col("harm_even_hi") <= LIMIT),
        )
    )
    table.write_parquet(WORK_DIR / "part_a.parquet")
    print(
        table.select(
            "source",
            "matched",
            "alarms",
            "alarms_per_match",
            "harm",
            "harm_lo",
            "harm_hi",
            "harm_even",
            "harm_odd",
            "share",
            "share_lo",
            "share_hi",
            "inside",
            "inside_even",
        ).with_columns(pl.col(pl.Float64).round(3))
    )
    print(
        "K0 absolute on AFTER rows, ORACLE:",
        scalar(longf["K0"].mean()),
        scalar(longf["ORACLE"].mean()),
    )
    (WORK_DIR / "part_a.json").write_text(json.dumps(table.to_dicts(), default=str, indent=1))


if __name__ == "__main__":
    main()
