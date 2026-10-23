# ruff: noqa: E501, RUF015
"""Experiment 304: scores the node-sum forms and prints every number the write-up quotes.

Reads the tables `e00304_run.py` wrote, writes `scores.csv`, `contrasts.csv`, `correlations.csv`,
`checks.csv`, `one_farm.csv` and `printed_report.md` under the private data folder, and prints the
report. Run with `OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run --no-sync python e00304_report.py`.
"""

import e00304_data as data
import numpy as np
import polars as pl

N_RESAMPLES = 2_000
INTERVAL = 0.9875
"""Bonferroni over the four planned contrasts."""
SEED = 0
FORMS = ("a_sum_of_quantiles", "b_independent", "c_member_wise", "c_shuffled", "d_copula")
TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)


def pinball(*, actual: pl.Expr, quantile: pl.Expr, level: float) -> pl.Expr:
    """Return the pinball loss of one quantile forecast."""
    return pl.max_horizontal(level * (actual - quantile), (level - 1.0) * (actual - quantile))


def per_row_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a coverage indicator and a pinball loss column per level, plus the median error."""
    columns = {}
    for level in data.OUTPUT_LEVELS:
        name = f"q{round(100 * level):02d}"
        columns[f"hit_{name}"] = (pl.col("actual") <= pl.col(name)).cast(pl.Float64)
        columns[f"pin_{name}"] = pinball(
            actual=pl.col("actual"), quantile=pl.col(name), level=level
        )
    columns["abs_err_median"] = (pl.col("actual") - pl.col("q50")).abs()
    return frame.with_columns(**columns)


def month_sums(frame: pl.DataFrame, *, keys: list[str]) -> pl.DataFrame:
    """Sum every metric column by month and keys, with the row count."""
    metric_columns = [c for c in frame.columns if c.startswith(("hit_", "pin_", "abs_"))]
    return frame.group_by([*keys, "month"]).agg(
        pl.len().alias("n"), *[pl.col(c).sum() for c in metric_columns]
    )


def bootstrap_means(
    sums: pl.DataFrame, *, metric: str, forms: list[str], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Resample months with replacement and return each form's metric mean per draw."""
    months = sorted(sums["month"].unique().to_list())
    draws = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    out = {}
    for form in forms:
        part = sums.filter(pl.col("form") == form).sort("month")
        total = part[metric].to_numpy()
        count = part["n"].to_numpy()
        out[form] = total[draws].sum(axis=1) / count[draws].sum(axis=1)
    return out


def interval(values: np.ndarray) -> tuple[float, float]:
    """Return the central `INTERVAL` interval of resampled values."""
    tail = (1.0 - INTERVAL) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def score_arm(frame: pl.DataFrame, *, arm: str, rng: np.random.Generator) -> tuple[list, list]:
    """Return (score rows, contrast rows) for one arm pooled over the two nodes."""
    rows = per_row_metrics(frame.filter(pl.col("arm") == arm))
    sums = month_sums(rows, keys=["form"])
    scores: list[dict] = []
    forms = list(FORMS)
    boot = {}
    for level in TAIL_LEVELS:
        name = f"q{round(100 * level):02d}"
        boot[f"hit_{name}"] = bootstrap_means(
            sums, metric=f"hit_{name}", forms=forms, rng=np.random.default_rng(SEED)
        )
        boot[f"pin_{name}"] = bootstrap_means(
            sums, metric=f"pin_{name}", forms=forms, rng=np.random.default_rng(SEED)
        )
    boot["abs_err_median"] = bootstrap_means(
        sums, metric="abs_err_median", forms=forms, rng=np.random.default_rng(SEED)
    )
    for form in forms:
        part = sums.filter(pl.col("form") == form)
        n = part["n"].sum()
        for metric in boot:
            value = float(part[metric].sum()) / float(n)
            lo, hi = interval(boot[metric][form])
            scores.append(
                {"arm": arm, "form": form, "metric": metric, "value": value, "lo": lo, "hi": hi}
            )
    contrasts = []
    pairs = [
        ("c_member_wise", "a_sum_of_quantiles"),
        ("c_member_wise", "b_independent"),
        ("d_copula", "c_member_wise"),
    ]
    for left, right in pairs:
        for metric in ("pin_q99", "pin_q95", "abs_err_median"):
            diff = boot[metric][left] - boot[metric][right]
            lo, hi = interval(diff)
            base = [s for s in scores if s["form"] == right and s["metric"] == metric][0]["value"]
            point = [s for s in scores if s["form"] == left and s["metric"] == metric][0][
                "value"
            ] - base
            contrasts.append(
                {
                    "arm": arm,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "diff": point,
                    "relative": point / base,
                    "lo": lo,
                    "hi": hi,
                }
            )
    del rng
    return scores, contrasts


def band_names() -> dict[int, str]:
    """Return the clear-sky band labels."""
    return {0: "overcast", 1: "broken", 2: "clear"}


def residual_correlations(residuals: pl.DataFrame) -> pl.DataFrame:
    """Return the Pearson correlation of each farm pair's residuals by clear-sky band and lead."""
    out = []
    frame = residuals.filter(pl.col("arm") == "blend").with_columns(
        lead_band=pl.when(pl.col("lead_hours") <= 24)
        .then(pl.lit("12-24 h"))
        .otherwise(pl.lit("24-36 h"))
    )
    for node, sites in data.NODES.items():
        wide = frame.filter(pl.col("node") == node).pivot(
            on="site", index=["time", "lead_hours", "band", "lead_band"], values="residual_fraction"
        )
        for i, a in enumerate(sites):
            for b in sites[i + 1 :]:
                for band in (None, 0, 1, 2):
                    for lead in (None, "12-24 h", "24-36 h"):
                        part = wide
                        if band is not None:
                            part = part.filter(pl.col("band") == band)
                        if lead is not None:
                            part = part.filter(pl.col("lead_band") == lead)
                        out.append(
                            {
                                "node": node,
                                "pair": f"{a}-{b}",
                                "band": "all" if band is None else band_names()[band],
                                "lead": "all" if lead is None else lead,
                                "n": part.height,
                                "correlation": float(
                                    np.corrcoef(part[a].to_numpy(), part[b].to_numpy())[0, 1]
                                ),
                            }
                        )
    return pl.DataFrame(out)


def one_farm_control() -> pl.DataFrame:
    """Return each farm's own coverage and pinball at the tail levels (a one-farm node)."""
    out = []
    for node in data.NODES:
        for arm in data.ARMS:
            for site, farm in zip(
                data.NODES[node], data.node_rows(node=node, arm=arm), strict=True
            ):
                for level in TAIL_LEVELS:
                    q = farm.quantiles[:, data.LEVELS.index(level)]
                    err = farm.actual - q
                    out.append(
                        {
                            "arm": arm,
                            "site": site,
                            "level": level,
                            "coverage": float(np.mean(err <= 0)),
                            "pinball": float(np.mean(np.maximum(level * err, (level - 1.0) * err))),
                            "n": len(q),
                        }
                    )
    return pl.DataFrame(out)


def leave_one_node_out(frame: pl.DataFrame) -> pl.DataFrame:
    """Return coverage at 0.95 and 0.99 of each form for each node alone (audit trigger 4)."""
    rows = per_row_metrics(frame)
    return (
        rows.group_by("arm", "node", "form")
        .agg(
            pl.len().alias("n"),
            pl.col("hit_q95").mean(),
            pl.col("hit_q99").mean(),
            pl.col("pin_q99").mean(),
        )
        .sort("arm", "node", "form")
    )


def main() -> None:
    """Score every arm and print the report."""
    quantiles = pl.read_parquet(data.DATA_DIR / "node_quantiles.parquet")
    residuals = pl.read_parquet(data.DATA_DIR / "farm_residuals.parquet")
    rng = np.random.default_rng(SEED)
    scores, contrasts = [], []
    for arm in data.ARMS:
        s, c = score_arm(quantiles, arm=arm, rng=rng)
        scores += s
        contrasts += c
    score_frame = pl.DataFrame(scores)
    contrast_frame = pl.DataFrame(contrasts)
    corr = residual_correlations(residuals)
    one_farm = one_farm_control()
    per_node = leave_one_node_out(quantiles)
    counts = (
        quantiles.filter((pl.col("arm") == "p2") & (pl.col("form") == "a_sum_of_quantiles"))
        .group_by("node")
        .agg(pl.len().alias("rows"))
    )
    score_frame.write_csv(data.DATA_DIR / "scores.csv")
    contrast_frame.write_csv(data.DATA_DIR / "contrasts.csv")
    corr.write_csv(data.DATA_DIR / "correlations.csv")
    one_farm.write_csv(data.DATA_DIR / "one_farm.csv")
    per_node.write_csv(data.DATA_DIR / "per_node.csv")
    with pl.Config(tbl_rows=500, tbl_cols=20, fmt_str_lengths=40, tbl_width_chars=200):
        print("Rows per node (daylight half-hours with all three farms present)")
        print(counts)
        for arm in data.ARMS:
            print(
                f"\n## Coverage and pinball, arm {arm} (pooled over both nodes; 98.75% month-block interval)"
            )
            wide = score_frame.filter(
                (pl.col("arm") == arm)
                & pl.col("metric").is_in(
                    [
                        "hit_q01",
                        "hit_q05",
                        "hit_q95",
                        "hit_q99",
                        "pin_q95",
                        "pin_q99",
                        "abs_err_median",
                    ]
                )
            ).with_columns(
                cell=pl.format(
                    "{} [{}, {}]",
                    pl.col("value").round(4),
                    pl.col("lo").round(4),
                    pl.col("hi").round(4),
                )
            )
            print(wide.pivot(on="metric", index="form", values="cell"))
        print("\n## Planned and extra contrasts")
        print(contrast_frame.with_columns(pl.col("diff", "relative", "lo", "hi").round(4)))
        print("\n## Residual correlations (blend arm), pairs pooled over lead, by band")
        print(
            corr.filter(pl.col("lead") == "all")
            .pivot(on="band", index=["node", "pair"], values="correlation")
            .with_columns(pl.exclude("node", "pair").round(3))
        )
        print("\n## Residual correlations by lead (all bands)")
        print(
            corr.filter(pl.col("band") == "all")
            .pivot(on="lead", index=["node", "pair"], values="correlation")
            .with_columns(pl.exclude("node", "pair").round(3))
        )
        print("\n## One-farm control (a farm's own marginal)")
        print(
            one_farm.filter(pl.col("arm") == "blend")
            .pivot(on="level", index="site", values="coverage")
            .with_columns(pl.exclude("site").round(4))
        )
        print("\n## Per node, arm and form (leave-one-node-out)")
        print(per_node.with_columns(pl.exclude("arm", "node", "form", "n").round(4)))


if __name__ == "__main__":
    main()
