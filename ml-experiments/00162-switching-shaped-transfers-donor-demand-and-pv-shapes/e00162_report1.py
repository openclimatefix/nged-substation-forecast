# ruff: noqa: E501
"""Part 1 report: shape fits on real transfers, detected intervals, controls, and placebos.

```bash
uv run --no-sync python e00162_report1.py | tee <private folder>/printed_part1.txt
```
"""

from __future__ import annotations

from typing import Final

import e00162_part1 as p1
import e00162_shape as shape
import numpy as np
import polars as pl
from e00162_common import BLOCK, WORK_DIR
from scipy.stats import beta as beta_dist

OUT = WORK_DIR / "part1"
DRAWS: Final[int] = 20_000
LEVEL: Final[float] = 0.05 / 9
SEED: Final[int] = 1620
DURATION_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 1 d", 0.0, 1.0),
    ("1 to 3 d", 1.0, 3.0),
    ("3 to 7 d", 3.0, 7.0),
    ("over 7 d", 7.0, 1e9),
)


def with_reductions(frame: pl.DataFrame) -> pl.DataFrame:
    """Held-out reductions in residual sum of squares against the constant, mean delta, duration."""
    n = pl.col("n_inside")
    return frame.with_columns(
        red_shape=1 - pl.col("held_M1") / pl.col("held_M0"),
        red_both=1 - pl.col("held_M2") / pl.col("held_M0"),
        days=(pl.col("end") - pl.col("onset")) / BLOCK,
        mean_delta=pl.col("constant")
        + pl.col("alpha2") * pl.col("demand_energy") / n
        + pl.col("beta2") * pl.col("solar_energy") / n,
        f_hat=-pl.col("alpha"),
        g_hat=pl.col("beta"),
    )


def boot_median(
    *, values: np.ndarray, seed: int = SEED
) -> tuple[float, float, float, float, float]:
    """Median and 95% and Bonferroni-level intervals, resampling transfers."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(DRAWS, len(values)))
    draws = np.median(values[picks], axis=1)
    lo95, hi95 = np.percentile(draws, [2.5, 97.5])
    lo, hi = np.percentile(draws, [100 * LEVEL / 2, 100 * (1 - LEVEL / 2)])
    return float(np.median(values)), float(lo95), float(hi95), float(lo), float(hi)


def show(label: str, r: tuple[float, ...]) -> None:
    """Print a point with intervals."""
    print(
        f"  {label}: {r[0]:+.3f} (95% {r[1]:+.3f} to {r[2]:+.3f}; {100 * (1 - LEVEL):.1f}% {r[3]:+.3f} to {r[4]:+.3f})"
    )


def clopper(*, k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval of a share."""
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def band_of(days: float) -> int:
    """Index of the duration band of a length in days."""
    return next(i for i, (_, lo, hi) in enumerate(DURATION_BANDS) if lo <= days < hi)


def calibrated(
    frame: pl.DataFrame, *, column: str, positive: bool, null: pl.DataFrame
) -> pl.Series:
    """True where the coefficient lies beyond the 2.5th or 97.5th percentile of the placebos' in its band."""
    cuts = {}
    for i, (_, lo, hi) in enumerate(DURATION_BANDS):
        v = null.filter((pl.col("days") >= lo) & (pl.col("days") < hi))[column]
        cuts[i] = float(v.quantile(0.975 if positive else 0.025)) if v.len() >= 10 else float("nan")  # ty: ignore[invalid-argument-type]
    values = frame[column].to_numpy()
    out = np.array(
        [
            (v > cuts[band_of(d)]) if positive else (v < cuts[band_of(d)])
            for v, d in zip(values, frame["days"].to_numpy(), strict=True)
        ]
    )
    return pl.Series(out)


def main_series(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per transfer: the series with the largest absolute mean effect."""
    return (
        frame.with_columns(abs_delta=pl.col("mean_delta").abs())
        .sort("abs_delta", descending=True)
        .group_by("key", maintain_order=True)
        .first()
    )


def real_part(frame: pl.DataFrame, *, label: str, null: pl.DataFrame) -> None:
    """Held-out reductions, signs, and coefficient intervals for one set of intervals."""
    main = main_series(frame)
    print(
        f"\n== {label}: {frame.height} series fits over {frame['key'].n_unique()} intervals ({main.height} main series) =="
    )
    for column, name in (
        ("red_both", "P1: constant plus shape (M2, primary)"),
        ("red_shape", "shape alone (M1)"),
    ):
        v = main[column].to_numpy()
        show(f"median held-out reduction in squared error, {name}", boot_median(values=v))
        print(
            f"    share of intervals with a positive reduction: {float((v > 0).mean()):.3f}; quartiles {np.percentile(v, [25, 75]).round(3)}"
        )
    if "donor" in frame.columns and frame["donor"].null_count() < frame.height:
        donors = frame.filter(pl.col("series") == pl.col("donor"))
        print(
            f"  P2: logged donors with a fit: {donors.height} (donor named in the log, not chosen by the fit)"
        )
    else:
        donors = main.filter(pl.col("mean_delta") < 0)
        print(
            f"  P2 (no logged donor; main series with a negative fitted effect, so chosen by the fit): {donors.height}"
        )
    n = donors.height
    cal_f = calibrated(donors, column="alpha", positive=False, null=null)
    cal_g = calibrated(donors, column="beta", positive=True, null=null)
    k = int(cal_f.sum())
    lo, hi = clopper(k=k, n=n)
    print(
        f"      demand share f = -alpha beyond the placebo 2.5th percentile in {k} of {n} ({k / max(n, 1):.2f}; exact 95% {lo:.2f} to {hi:.2f}); for reference, the 95% bootstrap interval excludes zero in {donors.filter(pl.col('alpha_hi') < 0).height}"
    )
    print(
        f"      solar share g = beta beyond the placebo 97.5th percentile in {int(cal_g.sum())} of {n}; bootstrap interval above zero in {donors.filter(pl.col('beta_lo') > 0).height}"
    )
    print(f"      median f {donors['f_hat'].median():.3f}, median g {donors['g_hat'].median():.3f}")
    print(
        f"  placebo false-positive rate of the rule by construction: 0.025 per tail; main series with a positive mean effect (receivers): {main.filter(pl.col('mean_delta') > 0).height}"
    )
    long = frame.filter(pl.col("days") >= 3)
    print(
        f"  fits of intervals of 3 days or more: {long.height}; f beyond the placebo cut {int(calibrated(long, column='alpha', positive=False, null=null).sum())}; g beyond {int(calibrated(long, column='beta', positive=True, null=null).sum())}"
    )


def controls_part() -> None:
    """Recovery of known shaped injections and the null distribution of the placebos."""
    c = with_reductions(pl.read_parquet(OUT / "controls.parquet"))
    print(
        f"\n== controls: {c.height} fits ({c.group_by('scenario').len().sort('scenario').rows()}) =="
    )
    for scenario in ("placebo", "demand_only", "solar_only", "both"):
        sub = c.filter(pl.col("scenario") == scenario)
        print(
            f"  {scenario}: median held-out reduction shape {sub['red_shape'].median():+.3f}, both {sub['red_both'].median():+.3f}; 95th percentile of shape reduction {sub['red_shape'].quantile(0.95):+.3f}"
        )
    plac = c.filter(pl.col("scenario") == "placebo")
    threshold = float(plac["red_shape"].quantile(0.95))  # ty: ignore[invalid-argument-type]
    print(
        f"  placebo false positives (95% interval excludes zero): demand {float((plac['alpha_hi'] < 0).mean() + (plac['alpha_lo'] > 0).mean()):.3f}, solar {float((plac['beta_lo'] > 0).mean() + (plac['beta_hi'] < 0).mean()):.3f}"  # ty: ignore[unsupported-operator]
    )
    cv_cut = c["solar_day_cv"].drop_nulls().quantile(0.5)
    print(
        f"  median solar day-to-day variation (CV of daily solar energy) over controls {cv_cut:.3f}"
    )
    print(
        "\n  identification by duration and by day-to-day solar variation (share of fits whose 95% interval excludes zero in the right direction; M1):"
    )
    for scenario, column, positive in (
        ("demand_only", "alpha", False),
        ("solar_only", "beta", True),
        ("both", "alpha", False),
        ("both", "beta", True),
    ):
        for name, lo_, hi_ in DURATION_BANDS:
            for cv_name, cv_mask in (
                ("low CV", pl.col("solar_day_cv") < cv_cut),
                ("high CV", pl.col("solar_day_cv") >= cv_cut),
            ):
                sub = c.filter(
                    (pl.col("scenario") == scenario)
                    & (pl.col("days") >= lo_)
                    & (pl.col("days") < hi_)
                    & cv_mask
                )
                if not sub.height:
                    continue
                ok = (sub[f"{column}_lo"] > 0) if positive else (sub[f"{column}_hi"] < 0)
                print(
                    f"    {scenario:11s} {column:5s} {name:9s} {cv_name:7s}: {float(ok.mean()):.2f} of {sub.height}"  # ty: ignore[invalid-argument-type]
                )
    print(
        "\n  identification with a placebo-calibrated rule (coefficient beyond the 2.5th or 97.5th percentile of the placebos' coefficient in the same duration band):"
    )
    for scenario, column, positive in (
        ("demand_only", "alpha", False),
        ("solar_only", "beta", True),
        ("both", "alpha", False),
        ("both", "beta", True),
    ):
        out = []
        for name, lo_, hi_ in DURATION_BANDS:
            band = (pl.col("days") >= lo_) & (pl.col("days") < hi_)
            null = plac.filter(band)[column]
            sub = c.filter((pl.col("scenario") == scenario) & band)
            if null.len() < 10 or not sub.height:
                continue
            cut = float(null.quantile(0.975)) if positive else float(null.quantile(0.025))  # ty: ignore[invalid-argument-type]
            ok = (sub[column] > cut) if positive else (sub[column] < cut)
            out.append(f"{name} {float(ok.mean()):.2f} of {sub.height} (null {null.len()})")  # ty: ignore[invalid-argument-type]
        print(f"    {scenario:11s} {column:5s}: " + "; ".join(out))
    c = c.with_columns(
        e_d=(pl.col("alpha").abs() * pl.col("demand_energy")),
        e_s=(pl.col("beta").abs() * pl.col("solar_energy")),
    ).with_columns(solar_share=pl.col("e_s") / (pl.col("e_d") + pl.col("e_s")))
    both = c.filter(pl.col("scenario") == "both")
    print(
        f"  both: median recovered f {both['f_hat'].median():.3f} (true {p1.FRACTION_DEMAND}) and g {both['g_hat'].median():.3f} (true {p1.FRACTION_SOLAR}); true solar share of the moved energy is {float((p1.FRACTION_SOLAR * both['solar_energy'] / (p1.FRACTION_SOLAR * both['solar_energy'] + p1.FRACTION_DEMAND * both['demand_energy'])).median()):.2f}, recovered {float(both['solar_share'].median()):.2f}"  # ty: ignore[invalid-argument-type]
    )
    recovered = c.filter(pl.col("scenario") == "demand_only")
    print(
        f"  recovered demand share f (true {p1.FRACTION_DEMAND}) in demand-only controls: median {recovered['f_hat'].median():.3f}; solar g (true {p1.FRACTION_SOLAR}) in solar-only controls: median {c.filter(pl.col('scenario') == 'solar_only')['g_hat'].median():.3f}"
    )
    _ = (threshold, shape)


def receiver_check(real: pl.DataFrame) -> None:
    """Fit each receiver's step with its own terms and with the donor's terms."""
    main = with_reductions(real)
    rng = np.random.default_rng(SEED)
    rows = []
    for key, group in main.group_by("key"):
        if group.height < 2:
            continue
        donor = group.sort("mean_delta").row(0, named=True)
        for recv in group.filter(pl.col("mean_delta") > 0).iter_rows(named=True):
            own = p1.make_window(series=recv["series"], onset=recv["onset"], end=recv["end"])
            don = p1.make_window(series=donor["series"], onset=recv["onset"], end=recv["end"])
            swapped = shape.Window(
                residual=own.residual,
                demand=don.demand,
                solar_output=don.solar_output,
                inside=own.inside,
                day=own.day,
            )
            fit_own = shape.fit_window(window=own, rng=rng, n_boot=50)
            fit_don = shape.fit_window(window=swapped, rng=rng, n_boot=50)
            if fit_own is None or fit_don is None:
                continue
            rows.append(
                {
                    "key": key[0],
                    "receiver": recv["series"],
                    "donor": donor["series"],
                    "own": shape.held_out_reduction(fit=fit_own, model="M1"),
                    "donor_terms": shape.held_out_reduction(fit=fit_don, model="M1"),
                    "own_c": shape.held_out_reduction(fit=fit_own, model="M2"),
                    "donor_c": shape.held_out_reduction(fit=fit_don, model="M2"),
                }
            )
    t = pl.DataFrame(rows)
    print(f"\n== receivers' opposite step: {t.height} receiver fits ==")
    if t.height:
        print(
            f"  median held-out reduction with the receiver's own terms (shape alone) {t['own'].median():+.3f}, with the donor's terms {t['donor_terms'].median():+.3f}; constant plus shape: own {t['own_c'].median():+.3f}, donor {t['donor_c'].median():+.3f}"
        )
        print(
            f"  receivers where the donor's terms fit better: {int((t['donor_terms'] > t['own']).sum())} of {t.height}"
        )
        t.write_parquet(OUT / "receiver_check.parquet")


def terms_check() -> None:
    """The solar and demand terms must sum to experiment 130's stored forecast."""
    gam = WORK_DIR.parent / "00130-ro-nc-detector-on-the-gam-k-residual" / "gam"
    worst = 0.0
    for series in (3, 8, 10, 16, 17, 18):
        t = np.load(WORK_DIR / "terms" / f"terms_{series}.npz")
        f = np.load(gam / f"forecast_{series}_1.npy")
        both = np.isfinite(f) & np.isfinite(t["total"])
        worst = max(
            worst,
            float(np.abs(f[both] - t["total"][both]).max()),
            float(np.abs(t["solar"][both] + t["demand"][both] - t["total"][both]).max()),
        )
    print(
        f"known-answer check: largest gap between solar + demand terms and experiment 130's stored forecast, six series: {worst:.2e} MW"
    )


def main() -> None:
    """Print part 1."""
    terms_check()
    real = with_reductions(pl.read_parquet(OUT / "real.parquet"))
    detected = with_reductions(pl.read_parquet(OUT / "detected.parquet"))
    spec = p1.real_specs()
    print(
        f"logged transfers over 1 day {len(spec)}; with at least one series fitted {real['key'].n_unique()}; series without convex-model terms are excluded (12, 13)"
    )
    null = with_reductions(pl.read_parquet(OUT / "controls.parquet")).filter(
        pl.col("scenario") == "placebo"
    )
    real_part(real, label="P1/P2: logged real transfers over 1 day", null=null)
    real_part(
        detected,
        label="RO-NC detected intervals over 1 day (experiment 147, setting R0)",
        null=null,
    )
    controls_part()
    receiver_check(pl.read_parquet(OUT / "real.parquet"))
    plac = with_reductions(pl.read_parquet(OUT / "controls.parquet")).filter(
        pl.col("scenario") == "placebo"
    )
    print(
        f"\nplacebo median held-out reduction (shape alone) {plac['red_shape'].median():+.3f}; (constant plus shape) {plac['red_both'].median():+.3f}"
    )


if __name__ == "__main__":
    main()
