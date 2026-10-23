# ruff: noqa: E501
"""Score every arm of experiment 264 and print each number the write-up quotes.

Quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. Intervals resample the 12 calendar months of the scored year (20,000 draws).

    uv run --no-sync python ml-experiments/00264-*/e00264_score.py
"""

import e00264_asof as asof
import e00264_common as common
import e00264_mixture as mixture
import numpy as np
import polars as pl

N_RESAMPLES = 20_000
SITE_ARMS: dict[str, tuple[str, ...]] = {
    "D8": ("arm1", "arm2", "arm2b", "arm3", "arm4", "arm3_forced", "arm_emp"),
    "D10": ("arm1", "arm2", "arm2b", "arm3", "arm3_forced", "arm_emp"),
    "D16": ("arm1", "arm2", "arm2b", "arm3", "arm3_forced", "arm_emp"),
}
QUARTERS: dict[str, tuple[int, int]] = {
    "2025Q3": (7, 9),
    "2025Q4": (10, 12),
    "2026Q1": (1, 3),
    "2026Q2": (4, 6),
}
LEAD_BANDS: tuple[tuple[float, float], ...] = ((0.0, 24.0), (24.0, 72.0), (72.0, 168.0))
TAILS: tuple[tuple[str, float], ...] = (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))


def load_arm(*, arm: str, site: str) -> pl.DataFrame:
    """Return one arm's stored forecast frame."""
    return pl.read_parquet(common.PREDS_DIR / arm / f"{site}.parquet")


def row_losses(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row arrays: quantile score, mean absolute error, coverage, and tail losses."""
    q = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - q
    pinball = np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)
    out = {
        "qs": 2.0 * pinball.mean(axis=1),
        "ae": np.abs(actual - frame["mean"].to_numpy().astype(np.float64)),
        "in90": ((actual >= q[:, 1]) & (actual <= q[:, 9])).astype(np.float64),
        "median_minus_mean": q[:, 5] - frame["mean"].to_numpy().astype(np.float64),
    }
    for name, level in TAILS:
        j = common.QUANTILES.index(name)
        e = actual - q[:, j]
        out[f"pin_{name}"] = np.maximum(level * e, (level - 1.0) * e)
        out[f"below_{name}"] = (actual < q[:, j]).astype(np.float64)
    return out


def month_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return year*100 + month of each row's valid time."""
    return (frame["valid_time"].dt.year() * 100 + frame["valid_time"].dt.month()).to_numpy()


def month_block_interval(
    *, difference: np.ndarray, months: np.ndarray, level: float, leave_out: bool = False
) -> tuple[float, float, float]:
    """Return the mean of `difference` and its month-resampled interval at `level` (two-sided)."""
    keys = np.unique(months)
    sums = np.array([difference[months == m].sum() for m in keys])
    counts = np.array([(months == m).sum() for m in keys], dtype=np.float64)
    rng = np.random.default_rng(264)
    draws = rng.integers(0, len(keys), size=(N_RESAMPLES, len(keys)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def leave_one_month_out(*, difference: np.ndarray, months: np.ndarray) -> list[float]:
    """Return the mean difference with each calendar month dropped in turn."""
    return [float(difference[months != m].mean()) for m in np.unique(months)]


def brier_low_flow(*, frame: pl.DataFrame, threshold: float) -> float:
    """Return the Brier score of the event "reading below `threshold`" from the 11 quantiles."""
    q = mixture.make_monotone(
        quantiles=frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    )
    y = np.full((len(q), 1), threshold)
    probability = np.concatenate(
        [
            mixture.cdf_and_density(y=y[i : i + 20_000], quantiles=q[i : i + 20_000])[0]
            for i in range(0, len(q), 20_000)
        ]
    )[:, 0]
    event = (frame["actual"].to_numpy().astype(np.float64) < threshold).astype(np.float64)
    return float(((probability - event) ** 2).mean())


def switch_history(*, site: str, on: float, off: float) -> list[str]:
    """Return the dates on which the visibility switch changes state at the given thresholds."""
    reference = common.load_reference(site=site)
    resolved = asof.resolved_from_reference(reference=reference)
    origins = np.unique(reference["power_fcst_init_time"].dt.epoch("ns").to_numpy())
    shares = resolved.drop_share(at_ns=origins, window_days=common.SWITCH_WINDOW_DAYS)
    state, flips = False, []
    for origin, share in zip(origins.tolist(), shares, strict=True):
        if np.isnan(share):
            continue
        new = True if share >= on else (False if share < off else state)
        if new != state:
            flips.append(
                f"{np.datetime64(origin, 'ns').astype('datetime64[D]')} -> {'on' if new else 'off'}"
            )
        state = new
    return flips


def summarise_arm(
    *,
    frame: pl.DataFrame,
    loss: dict[str, np.ndarray],
    months: np.ndarray,
    post_burn: np.ndarray,
    lead: np.ndarray,
) -> dict:
    """Return one arm's absolute scores, coverage by quarter and lead band, and tail shares."""
    row = {
        "qs": float(loss["qs"].mean()),
        "ae": float(loss["ae"].mean()),
        "cov90_all": float(loss["in90"].mean()),
        "cov90_after_burn_in": float(loss["in90"][post_burn].mean()),
        "median_minus_mean": float(loss["median_minus_mean"].mean()),
    }
    for name, (m0, m1) in QUARTERS.items():
        in_q = np.array([(m0 <= m % 100 <= m1) for m in months])
        row[f"cov90_{name}_full"] = float(loss["in90"][in_q].mean())
        row[f"cov90_{name}_burnin_rule"] = float(loss["in90"][in_q & post_burn].mean())
        row[f"qs_{name}"] = float(loss["qs"][in_q].mean())
    for lo, hi in LEAD_BANDS:
        in_band = (lead > lo) & (lead <= hi)
        row[f"cov90_lead_{int(lo)}_{int(hi)}"] = float(loss["in90"][in_band].mean())
        row[f"qs_lead_{int(lo)}_{int(hi)}"] = float(loss["qs"][in_band].mean())
    for name, _ in TAILS:
        row[f"below_{name}"] = float(loss[f"below_{name}"].mean())
        row[f"pin_{name}"] = float(loss[f"pin_{name}"].mean())
    if "switch_on" in frame.columns:
        on = frame["switch_on"].to_numpy()
        row["switch_on_share"] = float(on.mean())
        row["p_any_mean_when_on"] = (
            float(frame["p_any"].to_numpy()[on].mean()) if on.any() else None
        )
    return row


def main() -> None:
    """Print the absolute scores, contrasts, and checks."""
    report: dict = {}
    for site, arms in SITE_ARMS.items():
        frames = {arm: load_arm(arm=arm, site=site) for arm in arms}
        keys = frames["arm1"].select(common.KEYS)
        for arm, frame in frames.items():
            assert frame.select(common.KEYS).equals(keys), f"{site} {arm} row keys differ from arm1"
        base = frames["arm1"]
        months = month_of(frame=base)
        valid = base["valid_time"]
        post_burn = (valid >= common.BURN_IN_END).to_numpy()
        lead = base["lead_hours"].to_numpy()
        losses = {arm: row_losses(frame=frame) for arm, frame in frames.items()}
        report[site] = {}
        print(f"\n=== {site}: {base.height:,} rows")
        for arm in arms:
            row = summarise_arm(
                frame=frames[arm], loss=losses[arm], months=months, post_burn=post_burn, lead=lead
            )
            report[site][arm] = row
            print(
                f"{arm:11s} qs={row['qs']:.4f} mae={row['ae']:.3f} cov90={row['cov90_all']:.3f} "
                f"(after burn-in {row['cov90_after_burn_in']:.3f}) quarters "
                + " ".join(f"{q}={row[f'cov90_{q}_burnin_rule']:.3f}" for q in QUARTERS)
                + f" med-mean={row['median_minus_mean']:+.3f}"
            )
            print(
                "            tails "
                + " ".join(
                    f"{n}: below={row[f'below_{n}']:.3f} pin={row[f'pin_{n}']:.3f}"
                    for n, _ in TAILS
                )
            )
            print(
                "            leads "
                + " ".join(
                    f"{lo}-{hi}h cov={row[f'cov90_lead_{int(lo)}_{int(hi)}']:.3f}"
                    for lo, hi in LEAD_BANDS
                )
            )
        contrasts = [(arm, "arm1") for arm in arms if arm != "arm1"]
        if site == "D8":
            contrasts += [
                ("arm3", "arm2b"),
                ("arm4", "arm3"),
                ("arm3", "arm_emp"),
                ("arm_emp", "arm2b"),
            ]
        for arm, ref in contrasts:
            difference = losses[arm]["qs"] - losses[ref]["qs"]
            mean, low, high = month_block_interval(difference=difference, months=months, level=0.95)
            _, bl, bh = month_block_interval(difference=difference, months=months, level=0.992)
            ref_mean = float(losses[ref]["qs"].mean())
            loo = leave_one_month_out(difference=difference, months=months)
            ae_diff = float((losses[arm]["ae"] - losses[ref]["ae"]).mean())
            report[site][f"{arm}_minus_{ref}"] = {
                "qs": mean, "low95": low, "high95": high, "low99_2": bl, "high99_2": bh,
                "pct": mean / ref_mean * 100.0, "pct_high95": high / ref_mean * 100.0,
                "loo_signs": [int(np.sign(v)) for v in loo], "ae_diff": ae_diff,
            }  # fmt: skip
            print(
                f"  {arm}-{ref} qs {mean:+.4f} [{low:+.4f}, {high:+.4f}] 99.2%[{bl:+.4f}, {bh:+.4f}] "
                f"= {mean / ref_mean * 100:+.2f}% (upper {high / ref_mean * 100:+.2f}%); "
                f"leave-one-month-out signs {[int(np.sign(v)) for v in loo]}; mean-forecast MAE diff {ae_diff:+.4f}"
            )
        if site == "D8":
            actual = base["actual"].to_numpy().astype(np.float64)
            threshold = float(np.quantile(actual, 0.02))
            report[site]["brier_threshold_points"] = threshold
            report[site]["brier"] = {
                arm: brier_low_flow(frame=frames[arm], threshold=threshold) for arm in arms
            }
            print(
                f"  low-flow Brier (event: reading < {threshold:.1f} points, the 2nd percentile of the scored year): "
                + " ".join(f"{a}={v:.5f}" for a, v in report[site]["brier"].items())
            )
            by_month = {}
            for arm in ("arm1", "arm2b", "arm3", "arm_emp"):
                by_month[arm] = {
                    int(m): float(losses[arm]["in90"][months == m].mean())
                    for m in np.unique(months)
                }
            report[site]["cov90_by_month"] = by_month
            print(
                "  coverage by month (arm1/arm2b/arm3): "
                + "; ".join(
                    f"{m}: {by_month['arm1'][m]:.2f}/{by_month['arm2b'][m]:.2f}/{by_month['arm3'][m]:.2f}"
                    for m in sorted(by_month["arm1"])
                )
            )
    print("\n=== visibility switch changes")
    for site in common.SITE_IDS:
        for on, off in ((0.05, 0.03), (0.07, 0.04), (0.03, 0.02)):
            flips = switch_history(site=site, on=on, off=off)
            report.setdefault("switch", {})[f"{site}_{on}_{off}"] = flips
            print(f"{site} on {on:.2f} off {off:.2f}: {len(flips)} flips {flips}")
    common.write_json_atomic(path=common.WORK_DIR / "score.json", content=report)


if __name__ == "__main__":
    main()
