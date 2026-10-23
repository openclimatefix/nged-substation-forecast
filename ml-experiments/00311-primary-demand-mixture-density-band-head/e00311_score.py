# ruff: noqa: E501, PLR0915
"""Build the arms of experiment 311 from the stored fits and score them.

Arm M keeps the default's median and replaces every other quantile by the head's offsets about its
own median (averaged over 3 seeds). Arm MW applies the default's cold-day widening to arm M's upper
offsets. The scored year compares with Primary demand research default 1 (arm 1 of experiments 220
and 251). The replicate year (the even months of 2024-25) compares with experiment 169's arm A
bands around arm A's own median, because the default's neural median head and solar summary exist for
the scored year only. Intervals resample the months of the forecast run's origin, 20,000 draws.

    PYTHONPATH=ml-experiments/00278-stack-retraining-cadence \
        uv run --no-sync python ml-experiments/00311-*/e00311_score.py
"""

from typing import Final

import numpy as np
import polars as pl

import e00311_common as common  # isort: skip

N_RESAMPLES: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = ("default", "M", "MW")
CONTRAST_ARMS: Final[tuple[str, ...]] = ("M", "MW")
BENCHMARKS: Final[tuple[str, ...]] = ("bg2", "manual_heuristic", "climatology")
TAILS: Final[tuple[tuple[str, int], ...]] = (("q01", 0), ("q05", 1), ("q95", 9), ("q99", 10))
UPPER_INDEX: Final[dict[str, int]] = {"q75": 7, "q90": 8, "q95": 9, "q99": 10}
GATE_SIX: Final[float] = 2.579
GATE_FIVE: Final[float] = 2.617
GATE_TOLERANCE: Final[float] = 0.005
GAIN_LEVEL: Final[float] = 1.0 - 0.01 / 2.0
SCORED_START: Final = common.E251_SCORED


def read_fits(*, cuts: tuple[str, ...]) -> pl.DataFrame:
    """Return the seed-averaged head quantiles per (site, run, valid_time), asserting 3 seeds each."""
    pieces = []
    for label in cuts:
        pieces.extend(
            pl.read_parquet(
                common.FIT_DIR / f"cut{label}_seed{seed}.parquet",
                columns=["site", "run", "valid_time", *common.QUANTILES],
            )
            for seed in common.SEEDS
        )
    stacked = pl.concat(pieces)
    out = stacked.group_by(["site", "run", "valid_time"]).agg(
        [pl.col(q).mean() for q in common.QUANTILES] + [pl.len().alias("n_seeds")]
    )
    assert (out["n_seeds"] == len(common.SEEDS)).all(), "a row is missing a seed"
    return out.drop("n_seeds")


def centred_arms(
    *,
    default_q: np.ndarray,
    head_q: np.ndarray,
    ens_mean_c: np.ndarray,
    widened_default: bool = False,
) -> dict[str, np.ndarray]:
    """Return arms M and MW: the head's offsets about the default's median, then widening."""
    head = np.sort(head_q.astype(np.float64), axis=1)
    offsets = head - head[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1]
    median = default_q[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1].astype(np.float64)
    arm_m = median + offsets
    wide = offsets.copy()
    cold = ens_mean_c < common.WIDENING_COLD_C
    for name, factor in common.WIDENING_FACTORS.items():
        idx = UPPER_INDEX[name]
        wide[cold, idx] = offsets[cold, idx] * factor
    arm_mw = median + wide
    assert np.array_equal(
        arm_m[:, common.MEDIAN_INDEX], default_q[:, common.MEDIAN_INDEX].astype(np.float64)
    ), "median moved"
    return {
        "M": np.sort(arm_m, axis=1).astype(np.float32),
        "MW": np.sort(arm_mw, axis=1).astype(np.float32),
    }


def thresholds() -> dict[str, float]:
    """Return each site's historical 95th percentile of power before the scored year, in points."""
    pieces = [
        pl.read_parquet(
            common.E169_PREDS / f"ref_{period}_{site}.parquet", columns=["actual", "valid_time"]
        ).with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
        for period in ("tuning", "replicate")
    ]
    frame = pl.concat(pieces).filter(pl.col("valid_time") < pl.datetime(2025, 7, 2, time_zone="UTC")).unique(subset=["site", "valid_time"])  # fmt: skip
    return {
        str(site): float(np.quantile(group["actual"].to_numpy().astype(np.float64), 0.95))
        for (site,), group in frame.group_by("site")
    }


def per_row(*, actual: np.ndarray, q: np.ndarray, edge: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row score columns: quantile score, absolute error, tw95, tail pinball and coverage."""
    q = q.astype(np.float64)
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)
    out = {"qs": 2.0 * pin.mean(axis=1), "mae": np.abs(error[:, common.MEDIAN_INDEX])}
    e_tw = np.maximum(actual, edge)[:, None] - np.maximum(q, edge[:, None])
    out["tw95"] = 2.0 * np.maximum(common.LEVELS[None, :] * e_tw, (common.LEVELS[None, :] - 1.0) * e_tw).mean(axis=1)  # fmt: skip
    for name, pos in TAILS:
        out[f"pin_{name}"] = pin[:, pos]
        out[f"cov_{name}"] = (error[:, pos] <= 0).astype(np.float64)
    for k, name in enumerate(common.QUANTILES):
        out[f"lvl_{name}"] = 2.0 * pin[:, k] / len(common.LEVELS)
    return out


class Blocks:
    """Month-block bootstrap of a mean, or of a relative difference of two sums."""

    def __init__(self, *, blocks: np.ndarray, seed: int) -> None:
        """Hold the block labels and one set of resampling indices."""
        self.labels, self.codes = np.unique(blocks, return_inverse=True)
        self.draws = np.random.default_rng(seed).integers(0, len(self.labels), size=(N_RESAMPLES, len(self.labels)))  # fmt: skip

    def sums(self, *, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the per-block sums and counts over `mask`."""
        n = len(self.labels)
        return (
            np.bincount(self.codes[mask], weights=values[mask], minlength=n),
            np.bincount(self.codes[mask], minlength=n).astype(np.float64),
        )

    def mean(self, *, values: np.ndarray, mask: np.ndarray, level: float = 0.95) -> dict:
        """Mean over `mask` with a two-sided interval at `level` and a one-sided 95% upper bound."""
        sums, counts = self.sums(values=values, mask=mask)
        valid = np.flatnonzero(counts > 0)
        draws = (
            valid[self.draws[:, : len(valid)] % len(valid)]
            if len(valid) < len(counts)
            else self.draws
        )
        res = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        tail = 100.0 * (1.0 - level) / 2.0
        lo, hi = np.percentile(res, [tail, 100.0 - tail])
        return {"est": float(sums.sum() / counts.sum()), "lo": float(lo), "hi": float(hi), "upper95_one_sided": float(np.percentile(res, 95.0)), "n": int(mask.sum())}  # fmt: skip

    def relative(self, *, treat: np.ndarray, base: np.ndarray, mask: np.ndarray) -> dict:
        """Relative difference (treat - base) / base of sums over `mask`, with a one-sided bound."""
        t_sum, _ = self.sums(values=treat, mask=mask)
        b_sum, _ = self.sums(values=base, mask=mask)
        res = (t_sum[self.draws].sum(axis=1) - b_sum[self.draws].sum(axis=1)) / b_sum[
            self.draws
        ].sum(axis=1)
        return {"est": float((t_sum.sum() - b_sum.sum()) / b_sum.sum()), "upper95_one_sided": float(np.percentile(res, 95.0))}  # fmt: skip


def main() -> None:
    """Score the scored year and the replicate year and write the report."""
    rows = pl.read_parquet(common.E251_SCORED / "rows.parquet")
    keys = ["site", "run", "valid_time"]
    fits = read_fits(cuts=common.SCORED_CUTS)
    joined = rows.select("row", *keys).join(fits, on=keys, how="left", validate="1:1")
    assert joined["q50"].null_count() == 0, "a scored row has no head forecast"
    head_q = joined.select(list(common.QUANTILES)).to_numpy()
    default_q = np.load(common.E251_SCORED / "arm1.npy")
    ens = rows["ens_mean_c"].to_numpy()
    arms = {
        "default": default_q.astype(np.float32),
        **centred_arms(default_q=default_q, head_q=head_q, ens_mean_c=ens),
    }
    for name in BENCHMARKS:
        arms[name] = np.load(common.E251_SCORED / f"{name}.npy")
    out_dir = common.WORK_DIR / "arms" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(out_dir / "rows.parquet")
    for name, q in arms.items():
        np.save(out_dir / f"{name}.npy", q)

    site = rows["site"].to_numpy()
    th = thresholds()
    edge = np.array([th[s] for s in site], dtype=np.float64)
    actual = rows["actual"].to_numpy().astype(np.float64)
    m = {a: per_row(actual=actual, q=arms[a], edge=edge) for a in arms}
    boot = Blocks(
        blocks=rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
        seed=common.EXPERIMENT_ID,
    )
    daylight = rows["daylight"].to_numpy()
    cold_night = rows["cold2"].to_numpy() & ~daylight
    sunny = rows["sunny"].to_numpy() & daylight
    local_hh = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy() * 2 + rows["valid_time"].dt.convert_time_zone("Europe/London").dt.minute().to_numpy() // 30  # fmt: skip
    lead = rows["lead_hours"].to_numpy()
    every = np.ones(rows.height, dtype=bool)
    ordinary = ~cold_night & ~sunny
    masks: dict[str, np.ndarray] = {
        "all rows": every, "cold winter nights": cold_night, "sunny daylight": sunny,
        "ordinary rows": ordinary, "Christmas": rows["christmas"].to_numpy(),
        "bank holidays": rows["bank_holiday"].to_numpy(), "Economy 7 night (00:30-07:30)": (local_hh >= 1) & (local_hh <= 14),
        "leads 12-36 h": (lead >= 12) & (lead < 36),
        **{f"site {s}": site == s for s in common.SCORED_SITES},
    }  # fmt: skip
    rep: dict = {"n_rows": rows.height, "thresholds_points": th}
    rep["absolute"] = {a: {**{k: float(m[a][k].mean()) for k in ("qs", "mae", "tw95")}, **{f"{k}_cov": float(m[a][f"cov_{k}"].mean()) for k, _ in TAILS}, **{f"{k}_pin": float(m[a][f"pin_{k}"].mean()) for k, _ in TAILS}} for a in arms}  # fmt: skip
    five = np.isin(site, ["D5", "D10", "D11", "D14", "D16"])
    rep["gate"] = {
        "default_six": float(m["default"]["qs"].mean()), "default_five": float(m["default"]["qs"][five].mean()),
    }  # fmt: skip
    rep["gate"]["pass"] = bool(abs(rep["gate"]["default_six"] - GATE_SIX) <= GATE_TOLERANCE and abs(rep["gate"]["default_five"] - GATE_FIVE) <= GATE_TOLERANCE)  # fmt: skip
    # absolute scores by mask, all arms
    rep["by_mask"] = {a: {cut: {k: float(m[a][k][mask].mean()) for k in ("qs", "mae", "tw95", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q01", "pin_q99")} for cut, mask in masks.items() if mask.any()} for a in arms}  # fmt: skip
    contrasts: dict = {}
    for treat in CONTRAST_ARMS:
        c: dict = {"gain_99_5": {}, "groups": {}, "noninferiority": {}}
        diff = m[treat]["qs"] - m["default"]["qs"]
        c["gain_99_5"] = boot.mean(values=diff, mask=every, level=GAIN_LEVEL)
        c["gain_95"] = boot.mean(values=diff, mask=every, level=0.95)
        month_means = []
        for label in boot.labels:
            sel = every & (boot.labels[boot.codes] == label)
            month_means.append(float(diff[sel].mean()))
        month_means = np.array(month_means)
        c["monthly_differences"] = dict(
            zip(boot.labels.tolist(), month_means.tolist(), strict=True)
        )
        c["months_won"] = int((month_means < 0).sum())
        for cut, mask in masks.items():
            if mask.any():
                c["groups"][cut] = boot.mean(values=diff, mask=mask)
        for metric, cut in (("qs", "all rows"), ("pin_q01", "all rows"), ("pin_q99", "all rows"), ("qs", "cold winter nights"), ("qs", "sunny daylight")):  # fmt: skip
            d = m[treat][metric] - m["default"][metric]
            c["noninferiority"][f"{metric} on {cut}"] = boot.mean(values=d, mask=masks[cut])
        c["noninferiority"]["tw95 relative"] = boot.relative(
            treat=m[treat]["tw95"], base=m["default"]["tw95"], mask=every
        )
        # share of the gain: levels q01 and q99, and cold winter night rows
        total = float(diff.mean())
        tails_part = float(
            (
                (m[treat]["lvl_q01"] - m["default"]["lvl_q01"])
                + (m[treat]["lvl_q99"] - m["default"]["lvl_q99"])
            ).mean()
        )
        c["share_of_gain"] = {"total": total, "levels_q01_q99": tails_part, "levels_q01_q99_share": tails_part / total if total else None, "cold_night_rows": float(diff[cold_night].sum() / rows.height), "cold_night_rows_share": float(diff[cold_night].sum() / diff.sum()) if diff.sum() else None}  # fmt: skip
        c["by_level"] = {name: float((m[treat][f"lvl_{name}"] - m["default"][f"lvl_{name}"]).mean()) for name in common.QUANTILES}  # fmt: skip
        first6 = np.isin(
            boot.labels[boot.codes],
            ["2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"],
        )
        c["leave_one_site_out"] = {
            s: boot.mean(values=diff, mask=site != s) for s in common.SCORED_SITES
        }
        c["first_six_months"] = boot.mean(values=diff, mask=first6)
        for ref in BENCHMARKS:
            d = m[treat]["qs"] - m[ref]["qs"]
            c[f"minus_{ref}"] = boot.mean(values=d, mask=every)
        contrasts[treat] = c
    contrasts["M_minus_MW"] = boot.mean(values=m["M"]["qs"] - m["MW"]["qs"], mask=every)
    contrasts["default_minus_benchmarks"] = {ref: boot.mean(values=m["default"]["qs"] - m[ref]["qs"], mask=every) for ref in BENCHMARKS}  # fmt: skip
    rep["contrasts"] = contrasts
    # controls
    cold_gate = ens < common.WIDENING_COLD_C
    rep["controls"] = {
        "MW_equals_M_off_cold": bool(np.array_equal(arms["M"][~cold_gate], arms["MW"][~cold_gate])),
        "median_equals_default": bool(np.array_equal(arms["M"][:, 5], arms["default"][:, 5])),
        "cold_forecast_rows": int(cold_gate.sum()),
    }
    # P3: coverage of q95 on ordinary rows
    rep["p3_ordinary_cov_q95"] = {a: float(m[a]["cov_q95"][ordinary].mean()) for a in arms}
    # replicate year
    rep["replicate"] = replicate(th=th)
    common.write_json_atomic(path=common.REPORT_DIR / "report_scored.json", content=rep)
    print("gate", rep["gate"])
    print({a: round(v["qs"], 4) for a, v in rep["absolute"].items()})
    for treat in CONTRAST_ARMS:
        print(
            treat,
            "gain99.5",
            contrasts[treat]["gain_99_5"],
            "first6",
            contrasts[treat]["first_six_months"],
        )
    print("controls", rep["controls"], "p3", rep["p3_ordinary_cov_q95"])
    print("replicate", rep["replicate"])


def replicate(*, th: dict[str, float]) -> dict:
    """Score the even months of 2024-25: arm A bands against arm A's median plus the head's offsets."""
    pieces = [
        pl.read_parquet(common.E169_PREDS / f"a_replicate_{site}.parquet").with_columns(
            site=pl.lit(site)
        )
        for site in common.SCORED_SITES
    ]
    frame = pl.concat(pieces).filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
    frame = frame.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")).filter(pl.col("month").is_in(list(common.REPLICATE_CUTS)))  # fmt: skip
    keys = ["site", "run", "valid_time"]
    fits = read_fits(cuts=common.REPLICATE_CUTS)
    joined = frame.join(fits, on=keys, how="left", validate="1:1", suffix="_h")
    assert joined["q50_h"].null_count() == 0, "a replicate row has no head forecast"
    default_q = joined.select(list(common.QUANTILES)).to_numpy()
    head_q = joined.select([f"{q}_h" for q in common.QUANTILES]).to_numpy()
    ens = joined["ens_mean_c"].to_numpy()
    arms = {
        "default": default_q.astype(np.float32),
        **centred_arms(default_q=default_q, head_q=head_q, ens_mean_c=ens),
    }
    actual = joined["actual"].to_numpy().astype(np.float64)
    edge = np.array([th[s] for s in joined["site"].to_numpy()], dtype=np.float64)
    m = {a: per_row(actual=actual, q=arms[a], edge=edge) for a in arms}
    boot = Blocks(
        blocks=joined["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy(),
        seed=common.EXPERIMENT_ID + 1,
    )
    every = np.ones(joined.height, dtype=bool)
    out: dict = {"n_rows": joined.height, "absolute": {a: float(m[a]["qs"].mean()) for a in arms}}
    for treat in CONTRAST_ARMS:
        out[treat] = boot.mean(values=m[treat]["qs"] - m["default"]["qs"], mask=every)
    return out


if __name__ == "__main__":
    main()
