"""Part (b) on the real record: share of the AFTER gain recovered and ordinary-day harm per use.

Reads `scores_`, `scores_b_`, `scores_b_extra_`, `meta_`, `after_`, and `spans_` files. Writes
`part_b.parquet` and prints every number the page quotes.

    OMP_NUM_THREADS=1 uv run --no-sync python e00227_report_b.py
"""

from __future__ import annotations

import json
from typing import Final

import e00227_alarms as al
import e00227_report as ra
import numpy as np
import polars as pl
from e00227_common import WORK_DIR

USES: Final[tuple[str, ...]] = (
    "NONE",
    "MASK",
    "SOFT_2",
    "SOFT_4",
    "SOFT_16",
    "HALF",
    "SUB",
    "PAIRSUM",
)


def scores_of_site(site: str) -> pl.DataFrame:
    """The part (a) and part (b) scores of one site, with arm names `<source>|<use>`."""
    a = pl.read_parquet(WORK_DIR / f"scores_{site}.parquet")
    parts = [pl.read_parquet(p) for p in sorted(WORK_DIR.glob(f"scores_b*_{site}.parquet"))]
    b = pl.concat(parts)
    sources = b["arm"].str.split("|").list.get(0).unique().to_list()
    mask = a.filter(pl.col("arm").is_in(sources)).with_columns(arm=pl.col("arm") + pl.lit("|MASK"))
    k0 = a.filter(pl.col("arm").is_in(["K0", "ORACLE"]))
    return pl.concat([k0, mask, b], how="diagonal_relaxed").with_columns(
        site=pl.lit(site), qs=pl.col("qs").cast(pl.Float64)
    )


def main() -> None:
    """Print the part (b) table."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    meta = pl.concat(
        [pl.read_parquet(WORK_DIR / f"meta_{s}.parquet") for s in ra.SITES], how="diagonal_relaxed"
    ).with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
    after = pl.concat(
        [
            pl.read_parquet(WORK_DIR / f"after_{s}.parquet").with_columns(site=pl.lit(s))
            for s in ra.SITES
        ]
    )
    scores = pl.concat([scores_of_site(s) for s in ra.SITES], how="diagonal_relaxed")
    ordinary = meta.filter(pl.col("ordinary")).select("site", "row", "month")
    sc = scores.join(ordinary, on=["site", "row"], how="inner")
    wide = (
        sc.group_by("site", "row", "arm")
        .agg(qs=pl.col("qs").mean())
        .pivot(on="arm", index=["site", "row"], values="qs")
        .join(ordinary, on=["site", "row"])
    )
    n_ordinary = wide.height
    ev = after.join(
        meta.select("site", "row", "run_period"), on=["site", "row"], how="inner"
    ).filter(pl.col("run_period").is_in(["scored", "replicate"]))
    long = []
    for arm in scores["arm"].unique().to_list():
        s = scores.filter(pl.col("arm") == arm)
        foldless = ra.foldless(arm.split("|")[0])
        j = (
            ev.join(s.drop("fold"), on=["site", "row"], how="left")
            if foldless
            else ev.join(s, on=["site", "row", "fold"], how="left")
        )
        long.append(j.select("site", "card", "row", pl.lit(arm).alias("arm"), "qs"))
    longf = pl.concat(long).pivot(on="arm", index=["site", "card", "row"], values="qs")
    spans = pl.concat([pl.read_parquet(WORK_DIR / f"spans_{s}.parquet") for s in ra.SITES])
    sources = sorted({a.split("|")[0] for a in scores["arm"].unique().to_list() if "|" in a})
    rows = []
    for source in sources:
        n_alarms = float(
            spans.filter(pl.col("arm") == source).group_by("fold").len()["len"].mean()  # ty: ignore[invalid-argument-type]
        )
        mask_harm = None
        for use in USES:
            arm = "K0" if use == "NONE" else f"{source}|{use}"
            diff = (wide[arm] - wide["K0"]).to_numpy().astype(float)
            h = ra.boot_ratio(num=diff, den=np.ones(len(diff)), groups=wide["month"].to_numpy())
            share = ra.boot_ratio(
                num=(longf["K0"] - longf[arm]).to_numpy().astype(float),
                den=(longf["K0"] - longf["ORACLE"]).to_numpy().astype(float),
                groups=longf["card"].to_numpy(),
            )
            if use == "MASK":
                mask_harm = h[0]
            rows.append(
                {
                    "source": source,
                    "use": use,
                    "alarms": n_alarms,
                    "harm": h[0],
                    "harm_lo": h[1],
                    "harm_hi": h[2],
                    "harm_per_alarm": h[0] * n_ordinary / n_alarms,
                    "harm_per_alarm_lo": h[1] * n_ordinary / n_alarms,
                    "harm_per_alarm_hi": h[2] * n_ordinary / n_alarms,
                    "share": share[0],
                    "share_lo": share[1],
                    "share_hi": share[2],
                    "after_score": float(longf[arm].mean()),  # ty: ignore[invalid-argument-type]
                    "harm_vs_mask": None if mask_harm in (None, 0) else h[0] / mask_harm,
                }
            )
    table = pl.DataFrame(rows)
    mask_share = table.filter(pl.col("use") == "MASK").select("source", mask_share=pl.col("share"))
    table = table.join(mask_share, on="source").with_columns(
        share_of_mask=pl.col("share") / pl.col("mask_share")
    )
    table.write_parquet(WORK_DIR / "part_b.parquet")
    print("ordinary rows", n_ordinary, " AFTER rows", longf.height)
    print(table.with_columns(pl.col(pl.Float64).round(3)).drop("mask_share"))
    (WORK_DIR / "part_b.json").write_text(json.dumps(table.to_dicts(), default=str, indent=1))
    _ = al


if __name__ == "__main__":
    main()
