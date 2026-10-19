"""Audit checks of experiment 00102: per-site contrasts, leave-one-out, and bias.

Reads the arms, scores the planned contrasts site by site, leaves each site out of the verdict
contrasts, and reports each arm's mean error of the median (actual minus median, in points of
capacity), so that the write-up's "Checks" bullets rest on printed numbers. Writes
`work/results/checks.txt`.

    uv run python ml-experiments/00102-*/e00102_checks.py
"""

import e00102_assemble as assemble
import e00102_common as common
import e00102_score as score
import numpy as np
import polars as pl

PRIMARY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("C1 weekly no-lag minus BG2", "p_nl_weekly", "p_s6_bg2"),
    ("weekly lags: no-lag minus with lags", "p_nl_weekly", "p_s4_weekly"),
    ("monthly lags: no-lag minus with lags", "p_nl_monthly", "p_s3_monthly"),
)
OTHER_PAIRS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "BSP/GSP": (("C2 weekly no-lag minus BG2", "b_nl_weekly", "b_weekly"),),
    "solar": (("C3 no-lag minus BG2", "g_nl_monthly", "g_ens_weekly"),),
    "wind": (("C4 no-lag minus BG2", "g_nl_monthly", "g_ens_weekly"),),
}
BIAS_ARMS: dict[str, tuple[str, ...]] = {
    "Primary": ("p_s6_bg2", "p_s4_weekly", "p_nl_weekly", "p_nl_monthly", "ref_manual_heuristic"),
    "BSP/GSP": ("b_weekly", "b_nl_weekly", "ref_manual_heuristic"),
    "solar": ("g_ens_weekly", "g_nl_monthly", "ref_manual_heuristic"),
    "wind": ("g_ens_weekly", "g_nl_monthly", "ref_manual_heuristic"),
}


def site_means(*, cases: score.ClassCases, values: np.ndarray) -> pl.DataFrame:
    """Return the mean of a per-case value in each site over the gap-run cases."""
    use = cases.gap & np.isfinite(values)
    return (
        pl.DataFrame({"site": cases.keys["site"].filter(pl.Series(use)), "value": values[use]})
        .group_by("site")
        .agg(pl.col("value").mean())
        .sort("site")
    )


def loo_interval(*, cases: score.ClassCases, values: np.ndarray, site: str) -> tuple[float, float]:
    """Return the 99% interval of a contrast with one site left out of the gap-run cases."""
    sites = cases.keys["site"].to_numpy()
    keep = cases.gap & np.isfinite(values) & (sites != site)
    n_months = len(cases.months)
    sums = np.bincount(cases.month[keep], weights=values[keep], minlength=n_months)[None, :]
    counts = np.bincount(cases.month[keep], minlength=n_months).astype(float)[None, :]
    lower, upper = score.month_intervals(sums=sums, counts=counts, level=score.VERDICT_LEVEL)
    return float(lower[0]), float(upper[0])


def mean_forecast(*, arm: str, cases: score.ClassCases) -> np.ndarray:
    """Return an arm's mean forecast per case in points of capacity (mass-weighted quantiles)."""
    frame = cases.keys.select(assemble.KEY_COLUMNS).join(
        pl.read_parquet(
            score.arm_path(name=arm), columns=[*assemble.KEY_COLUMNS, *common.LEVEL_COLUMNS]
        ),
        on=list(assemble.KEY_COLUMNS),
        how="left",
        maintain_order="left",
    )
    return frame.select(common.LEVEL_COLUMNS).to_numpy().astype(float) @ common.ATOM_MASSES * 100.0


def pooled_interval(
    *, cases: score.ClassCases, values: np.ndarray, keep: np.ndarray, level: float
) -> tuple[float, float, float]:
    """Return the mean and the month-resampled interval of a contrast over the kept cases."""
    use = cases.gap & np.isfinite(values) & keep
    n_months = len(cases.months)
    sums = np.bincount(cases.month[use], weights=values[use], minlength=n_months)[None, :]
    counts = np.bincount(cases.month[use], minlength=n_months).astype(float)[None, :]
    lower, upper = score.month_intervals(sums=sums, counts=counts, level=level)
    return float(values[use].mean()), float(lower[0]), float(upper[0])


def main() -> None:
    """Print and write the checks."""
    lines: list[str] = []
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        for cls, pairs in {"Primary": PRIMARY_PAIRS, **OTHER_PAIRS}.items():
            cases = score.ClassCases(cls=cls)
            names = {a for _, t, r in pairs for a in (t, r)} | set(BIAS_ARMS[cls])
            names |= {"p_s5_blend_no_offset", "p_s6_bg2"} if cls == "Primary" else set()
            for name in sorted(names):
                cases.add_arm(name=name)
            lines.append(f"=== {cls}: per-site contrasts (qs, points of capacity) ===")
            for label, treatment, reference in pairs:
                values = cases.qs[treatment] - cases.qs[reference]
                per_site = site_means(cases=cases, values=values)
                lines.append(f"{label}\n{per_site}")
                if cls == "Primary":
                    lines.extend(
                        _primary_extras(
                            cases=cases,
                            treatment=treatment,
                            reference=reference,
                            label=label,
                            per_site=per_site,
                        )
                    )
                lines.extend(
                    _mean_error_lines(
                        cases=cases, treatment=treatment, reference=reference, label=label
                    )
                )
                loo = [
                    (site, *loo_interval(cases=cases, values=values, site=site))
                    for site in per_site["site"].to_list()
                ]
                lines.append("leave one site out, 99% interval of the pooled contrast:")
                lines.extend(f"  without {s}: {lo:.3f} to {hi:.3f}" for s, lo, hi in loo)
            if cls == "Primary":
                values = cases.qs["p_s6_bg2"] - cases.qs["p_s5_blend_no_offset"]
                offset_rows = cases.masks["midday_summer|offset5"] & np.isfinite(values)
                per_site = (
                    pl.DataFrame(
                        {
                            "site": cases.keys["site"].filter(pl.Series(offset_rows)),
                            "value": values[offset_rows],
                        }
                    )
                    .group_by("site")
                    .agg(pl.col("value").mean())
                    .sort("site")
                )
                lines.append(
                    f"C5 offset minus no offset, midday summer, per offset site\n{per_site}"
                )
            lines.append(f"=== {cls}: mean error of the median, actual minus median (points) ===")
            bias = {
                arm: float(
                    np.nanmean(
                        np.where(
                            cases.gap,
                            (cases.keys["actual"].to_numpy() * 100.0) - _median(arm, cases),
                            np.nan,
                        )
                    )
                )
                for arm in BIAS_ARMS[cls]
                if arm in cases.qs
            }
            lines.extend(f"  {arm}: {value:.3f}" for arm, value in bias.items())
    text = "\n".join(lines)
    (common.WORK_DIR / "results" / "checks.txt").write_text(text)
    print(text)


def _median(arm: str, cases: score.ClassCases) -> np.ndarray:
    """Return an arm's median forecast per case in points of capacity, NaN where missing."""
    frame = cases.keys.select(assemble.KEY_COLUMNS).join(
        pl.read_parquet(score.arm_path(name=arm), columns=[*assemble.KEY_COLUMNS, "q50"]),
        on=list(assemble.KEY_COLUMNS),
        how="left",
        maintain_order="left",
    )
    return frame["q50"].to_numpy().astype(float) * 100.0


def _mean_error_lines(
    *, cases: score.ClassCases, treatment: str, reference: str, label: str
) -> list[str]:
    """Return the contrast on the absolute error of the mean forecast, beside the score contrast."""
    actual = cases.keys["actual"].to_numpy().astype(float) * 100.0
    error = np.abs(actual - mean_forecast(arm=treatment, cases=cases)) - np.abs(
        actual - mean_forecast(arm=reference, cases=cases)
    )
    mean, lower, upper = pooled_interval(
        cases=cases, values=error, keep=np.ones(len(error), dtype=bool), level=score.LEVEL_95
    )
    return [
        (
            f"{label}: error of the MEAN forecast, treatment minus reference {mean:.3f} "
            f"({lower:.3f} to {upper:.3f}, 95%)"
        )
    ]


def _primary_extras(
    *, cases: score.ClassCases, treatment: str, reference: str, label: str, per_site: pl.DataFrame
) -> list[str]:
    """Return the leave-two-out intervals and write the per-site lag costs with anonymous labels."""
    values = cases.qs[treatment] - cases.qs[reference]
    sites = cases.keys["site"].to_numpy()
    ranked = per_site.sort("value", descending=True)["site"].to_list()
    out = [f"{label}: leave the two largest series ({ranked[0]}, {ranked[1]}) out"]
    mean, lower, upper = pooled_interval(
        cases=cases,
        values=values,
        keep=~np.isin(sites, ranked[:2]),
        level=score.VERDICT_LEVEL,
    )
    out.append(f"  pooled contrast {mean:.3f} (99%: {lower:.3f} to {upper:.3f})")
    primaries = (
        common.series_table().filter(pl.col("substation_type") == "Primary")["site"].to_list()
    )
    labelled = per_site.with_columns(
        label=pl.col("site").map_elements(
            lambda site: f"Primary {chr(ord('A') + primaries.index(site))}", return_dtype=pl.String
        ),
        contrast=pl.lit(label),
    )
    labelled.write_parquet(
        common.WORK_DIR / "results" / f"checks_by_site_{treatment}_{reference}.parquet"
    )
    return out


if __name__ == "__main__":
    main()
