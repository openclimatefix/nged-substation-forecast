# ruff: noqa: E501
"""Score each stage of the GAM stack and each BG2 member at D8 (and the stack's stages at D2 and D4).

Stages, each scored with the own-residual band of experiment 354 (arm 3) built from that stage's own
median (D8), or with the stored arm-D spread offsets (placebos D2, D4):

| Stage | Median |
|---|---|
| `S0` | the plain GAM: stored `mean - level - anomaly` |
| `S1` | `S0` plus the Kalman level: `mean - anomaly` |
| `S2` | `S1` plus the Kalman transient: stored `mean` (the stage-2 arm A) |
| `S3` | `S2` plus the winter night offset (stage-2 arm B) |
| `S4` | `S3` plus recipe R (stage-2 arm D): the stack's median |

`S4_stored` is `S4` with the stored (donated) spread, before the own-residual band. Gate: `S4` must
score 8.338 (weekly) or 8.382 (monthly) on the 96,986 rows, and BG2 7.593.

    PYTHONPATH=$E252:$E264:$E354:$E361:$E87 OMP_NUM_THREADS=2 \
        uv run --no-sync python -W ignore ml-experiments/00364-*/e00364_score.py > printed_report.txt
"""

import dataclasses
import json
import os
from pathlib import Path
from typing import Final

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import e00087_assemble as asm  # ty: ignore[unresolved-import]
import e00252_common as common252  # ty: ignore[unresolved-import]
import e00252s2_common as common2  # ty: ignore[unresolved-import]
import e00252s2_corrections as corr  # ty: ignore[unresolved-import]
import e00252s2_run as stage2  # ty: ignore[unresolved-import]
import e00354_bands as bands  # ty: ignore[unresolved-import]
import e00354_lib as lib  # ty: ignore[unresolved-import]
import e00354_step as step  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

DATA: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data/ml-experiments")
WORK: Final[Path] = DATA / "00364-d8-stage-blame-and-analogue-recent-power" / "work"
FOLDERS: Final[dict[tuple[str, str], Path]] = {
    ("D8", "monthly"): DATA / "00354-d8-own-residual-bands-and-longer-fit-history/work/fits_b",
    ("D8", "weekly"): DATA / "00355-d8-weekly-refit-and-median-band-swap/work/D8_weekly",
    ("D2", "monthly"): DATA / "00355-d8-weekly-refit-and-median-band-swap/work/D2_monthly",
    ("D4", "monthly"): DATA / "00355-d8-weekly-refit-and-median-band-swap/work/D4_monthly",
}
MEMBER_FRAME: Final[str] = "00360-bg2-member-drop-one-at-d8/work/frame_{site}.parquet"
MEMBER_NAMES: Final[dict[str, str]] = {
    "only_W": "W", "only_M": "M", "only_AN": "AN", "only_PT": "PT", "only_ST": "ST", "full": "BG2_reblend",
}  # fmt: skip
STAGES: Final[tuple[str, ...]] = ("S0", "S1", "S2", "S3", "S4")
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-24h": (0.0, 24.0), "24-72h": (24.0, 72.0), "72-168h": (72.0, 168.0),
}  # fmt: skip
N_RESAMPLES: Final[int] = 20_000
GATES: Final[dict[str, float]] = {"weekly": 8.338, "monthly": 8.382}
ANALOGUE_ARMS: Final[tuple[str, ...]] = ("control", "nolag", "norecent")


def stage_medians(*, folder: Path, site: str) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the cleaned arm-D frame and each stage's median on its rows (points of capacity)."""
    raw_all = pl.read_parquet(folder / f"base_raw_scored_{site}.parquet")
    if site == "D8":
        arms = stage2.build_arms(site=site, frame=corr.add_calendar(frame=raw_all))
    else:
        arms = {
            n: pl.read_parquet(common2.out_path(arm=n, period="scored", site=site)) for n in "abd"
        }
        stored_a = arms["a"].join(
            raw_all.select(*lib.KEY, raw_mean="mean"), on=lib.KEY, how="inner"
        )
        gap = float((stored_a["mean"] - stored_a["raw_mean"]).abs().max())  # ty: ignore[invalid-argument-type]
        if stored_a.height < 90_000 or gap > 1e-3:
            msg = f"{site}: stored stage-2 arm A differs from the base frame (gap {gap}, rows {stored_a.height})"
            raise ValueError(msg)
    final = lib.clean(frame=arms["d"])
    keys = final.select(*lib.KEY)
    aligned = {
        "raw": raw_all.join(keys, on=lib.KEY, how="inner", validate="1:1"),
        "b": arms["b"].join(keys, on=lib.KEY, how="inner", validate="1:1"),
    }
    aligned = {
        name: keys.join(f, on=lib.KEY, how="left", validate="1:1") for name, f in aligned.items()
    }
    raw, b = aligned["raw"], aligned["b"]
    mean = raw["mean"].to_numpy().astype(np.float64)
    level = raw["level"].to_numpy().astype(np.float64)
    anomaly = raw["anomaly"].to_numpy().astype(np.float64)
    medians = {
        "S0": mean - level - anomaly,
        "S1": mean - anomaly,
        "S2": mean,
        "S3": b["q50"].to_numpy().astype(np.float64),
        "S4": final["q50"].to_numpy().astype(np.float64),
    }
    return final, medians


def with_median(*, template: pl.DataFrame, median: np.ndarray) -> pl.DataFrame:
    """Return the template with every quantile moved to `median` plus the template's offsets."""
    offsets = template.select(list(lib.QUANTILES)).to_numpy().astype(np.float64)
    offsets = offsets - template["q50"].to_numpy().astype(np.float64)[:, None]
    values = median[:, None] + offsets
    return template.with_columns(
        **{q: pl.Series(values[:, i].astype(np.float32)) for i, q in enumerate(lib.QUANTILES)},
        mean=pl.Series(median.astype(np.float32)),
    )


def member_frames(*, site: str) -> dict[str, pl.DataFrame]:
    """Return BG2's members from experiment 360's stored frame, in points of capacity."""
    stored = pl.read_parquet(DATA / MEMBER_FRAME.format(site=site))
    out = {}
    for column, name in MEMBER_NAMES.items():
        frame = stored.select(
            *lib.KEY,
            *[(pl.col(f"{column}__{q}") * 100.0).alias(q) for q in lib.QUANTILES],
        ).with_columns(mean=pl.col("q50"))
        out[name] = frame
    return out


def analogue_frames() -> dict[str, pl.DataFrame]:
    """Return the refitted analogue arms at D8, finished as experiment 87 finishes a member."""
    ctx = asm.make_context(smoke=False, store=False)
    out = {}
    for arm in ANALOGUE_ARMS:
        local = dataclasses.replace(ctx, demand_root=WORK / "analogue_preds" / arm)
        raw = asm.raw_member(ctx=local, member="AN", offset="vintage", sites=["D8"])
        table = asm.finish_member(ctx=local, raw=raw, correct=False, spread=True)
        out[f"AN_{arm}"] = table.select(
            *lib.KEY, *[(pl.col(q) * 100.0).alias(q) for q in lib.QUANTILES]
        ).with_columns(mean=pl.col("q50"))
    return out


def month_stats(
    *, x: np.ndarray, y: np.ndarray, months: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Return per-month (n, sx, sy, sxx, syy, sxy) for the correlation bootstrap."""
    out = np.zeros((labels.size, 6))
    for i, label in enumerate(labels):
        m = months == label
        out[i] = (
            m.sum(),
            x[m].sum(),
            y[m].sum(),
            (x[m] ** 2).sum(),
            (y[m] ** 2).sum(),
            (x[m] * y[m]).sum(),
        )
    return out


def corr_from(*, stats: np.ndarray) -> np.ndarray:
    """Return the Pearson correlation from summed (n, sx, sy, sxx, syy, sxy), rows or one vector."""
    n, sx, sy, sxx, syy, sxy = (stats[..., k] for k in range(6))
    cov = sxy / n - (sx / n) * (sy / n)
    return cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))


class Bootstrap:
    """Month-resampled intervals for one row subset (shared draws, so contrasts are paired)."""

    def __init__(self, *, months: np.ndarray, seed: int = 0) -> None:  # noqa: D107
        self.labels = np.unique(months)
        self.months = months
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.labels.size, size=(N_RESAMPLES, self.labels.size))
        self.counts = (draws[:, :, None] == np.arange(self.labels.size)).sum(axis=1).astype(float)

    def mean(self, *, values: np.ndarray) -> list[float]:
        """Return the mean and its 95% interval."""
        sums = np.array([values[self.months == m].sum() for m in self.labels])
        n = np.array([(self.months == m).sum() for m in self.labels], dtype=float)
        draws = (self.counts @ sums) / (self.counts @ n)
        return [float(sums.sum() / n.sum()), *np.quantile(draws, [0.025, 0.975]).tolist()]

    def correlation(self, *, x: np.ndarray, y: np.ndarray) -> list[float]:
        """Return the Pearson correlation and its 95% interval."""
        stats = month_stats(x=x, y=y, months=self.months, labels=self.labels)
        draws = corr_from(stats=self.counts @ stats)
        return [
            float(corr_from(stats=stats.sum(axis=0))),
            *np.quantile(draws, [0.025, 0.975]).tolist(),
        ]

    def correlation_difference(
        self, *, x: np.ndarray, y: np.ndarray, base: np.ndarray
    ) -> list[float]:
        """Return corr(x, y) minus corr(base, y) and its 95% interval, on shared draws."""
        a = month_stats(x=x, y=y, months=self.months, labels=self.labels)
        b = month_stats(x=base, y=y, months=self.months, labels=self.labels)
        draws = corr_from(stats=self.counts @ a) - corr_from(stats=self.counts @ b)
        point = corr_from(stats=a.sum(axis=0)) - corr_from(stats=b.sum(axis=0))
        return [float(point), *np.quantile(draws, [0.025, 0.975]).tolist()]


def arm_metrics(*, wide: pl.DataFrame, arm: str, capacity_mw: float, boot: Bootstrap) -> dict:
    """Return score, MAE, bias, and correlation of one arm on the rows of `wide`."""
    actual = wide["actual"].to_numpy().astype(np.float64)
    q50 = wide[f"{arm}__q50"].to_numpy().astype(np.float64)
    scale = capacity_mw / 100.0
    return {
        "score": boot.mean(values=wide[f"s_{arm}"].to_numpy()),
        "mae_mw": boot.mean(values=np.abs(q50 - actual) * scale),
        "bias_mw": boot.mean(values=(q50 - actual) * scale),
        "corr": boot.correlation(x=q50, y=actual),
    }


def contrast(
    *, wide: pl.DataFrame, arm: str, base: str, capacity_mw: float, boot: Bootstrap
) -> dict:
    """Return arm minus base: score, MAE, bias, and correlation, each with a 95% interval."""
    actual = wide["actual"].to_numpy().astype(np.float64)
    qa, qb = (
        wide[f"{arm}__q50"].to_numpy().astype(np.float64),
        wide[f"{base}__q50"].to_numpy().astype(np.float64),
    )
    scale = capacity_mw / 100.0
    return {
        "score": boot.mean(values=wide[f"s_{arm}"].to_numpy() - wide[f"s_{base}"].to_numpy()),
        "mae_mw": boot.mean(values=(np.abs(qa - actual) - np.abs(qb - actual)) * scale),
        "bias_mw": boot.mean(values=(qa - qb) * scale),
        "corr": boot.correlation_difference(x=qa, y=actual, base=qb),
    }


def step_days(*, valid: pl.Series) -> np.ndarray:
    """Return a flag for valid times on a UTC day touched by any of experiment 361's detected segments."""
    folder = DATA / "00361-d8-step-catalogue-detector-injection-and-gam-stack"
    segments = pl.concat(
        [
            pl.read_parquet(folder / f"{n}.parquet")
            for n in ("d8_negative_segments", "d8_positive_segments", "d8_drops")
        ]
    )
    days: set = set()
    for start, end in segments.select("start", "end").iter_rows():
        days.update(np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1).tolist())
    mine = valid.dt.replace_time_zone(None).dt.date().to_list()
    return np.array([d in days for d in mine])


def slices_of(*, wide: pl.DataFrame, site: str) -> dict[str, np.ndarray]:
    """Return the boolean row masks to score: all, lead bands, step days, calendar flags."""
    lead = wide["lead_hours"].to_numpy()
    out = {"all": np.ones(wide.height, bool)}
    out |= {name: (lead > lo) & (lead <= hi) for name, (lo, hi) in LEAD_BANDS.items()}
    if site == "D8":
        flag = step_days(valid=wide["valid_time"])
        out |= {"step_days": flag, "no_step_days": ~flag}
    out |= {k: wide[k].to_numpy() for k in ("sunny", "cold_night", "winter_night")}
    return out


def score_wide(
    *,
    wide: pl.DataFrame,
    site: str,
    arms: list[str],
    capacity_mw: float,
    pairs: list[tuple[str, str]],
) -> dict:
    """Score every arm and contrast in every slice."""
    out: dict = {"rows": wide.height, "arms": {}, "contrasts": {}, "monthly": {}}
    months_all = wide["month"].to_numpy()
    for name, mask in slices_of(wide=wide, site=site).items():
        sub = wide.filter(pl.Series(mask))
        boot = Bootstrap(months=months_all[mask])
        out["arms"][name] = {
            a: arm_metrics(wide=sub, arm=a, capacity_mw=capacity_mw, boot=boot) for a in arms
        }
        out["contrasts"][name] = {
            f"{a}-{b}": contrast(wide=sub, arm=a, base=b, capacity_mw=capacity_mw, boot=boot)
            for a, b in pairs
        }
    actual = wide["actual"].to_numpy().astype(np.float64)
    for month in np.unique(months_all):
        sel = months_all == month
        out["monthly"][str(month)] = {
            a: {
                "score": float(wide[f"s_{a}"].to_numpy()[sel].mean()),
                "corr": float(np.corrcoef(wide[f"{a}__q50"].to_numpy()[sel], actual[sel])[0, 1]),
            }
            for a in arms
        }
    return out


def build_wide(
    *, site: str, cadence: str, cache: object, with_analogue: bool
) -> tuple[pl.DataFrame, list[str]]:
    """Return the shared-row wide frame of the stages and the members, and the arm names."""
    final, medians = stage_medians(folder=FOLDERS[(site, cadence)], site=site)
    frames: dict[str, pl.DataFrame] = {}
    for name in STAGES:
        staged = with_median(template=final, median=medians[name])
        frames[name] = lib.cell_residual_band(frame=staged) if site == "D8" else staged
    if site == "D8":
        frames["S4_stored"] = final
    frames |= member_frames(site=site)
    if with_analogue:
        frames |= analogue_frames()
    return bands.wide_frame(site=site, cache=cache, frames=frames), list(frames)


def main() -> None:
    """Score everything and write `work/scores.json`."""
    step.install()
    cache = common252.load_cache()
    results: dict = {}
    for site, cadence in FOLDERS:
        wide, arms = build_wide(
            site=site,
            cadence=cadence,
            cache=cache,
            with_analogue=site == "D8"
            and cadence == "weekly"
            and not os.environ.get("E364_SKIP_ANALOGUE"),
        )
        capacity_mw = float(cache.sites[site].table.effective_capacity_mw)
        pairs = [(STAGES[i + 1], STAGES[i]) for i in range(len(STAGES) - 1)] + [
            ("S4", "S0"),
            ("S2", "S0"),
        ]
        if site == "D8":
            pairs += [("S4", "S4_stored"), ("BG2_reblend", "S4"), ("AN", "S4"), ("W", "S4")]
            if cadence == "weekly" and not os.environ.get("E364_SKIP_ANALOGUE"):
                pairs += [
                    ("AN_nolag", "AN_control"),
                    ("AN_norecent", "AN_control"),
                    ("AN_control", "AN"),
                    ("AN_norecent", "AN_nolag"),
                    ("M", "AN"),
                ]
        key = f"{site}_{cadence}"
        results[key] = score_wide(
            wide=wide, site=site, arms=arms, capacity_mw=capacity_mw, pairs=pairs
        )
        results[key]["capacity_mw"] = capacity_mw
        results[key]["gate"] = {
            "S4_all": results[key]["arms"]["all"]["S4"]["score"][0] if site == "D8" else None,
            "expected": GATES[cadence],
            "bg2_stored": float(wide["s_bg2"].mean()) if "s_bg2" in wide.columns else None,  # ty: ignore[invalid-argument-type]
        }
        wide.write_parquet(WORK / f"wide_{key}.parquet")
        if site == "D8":
            (WORK / "step_day_flags.json").write_text(
                json.dumps(step_days(valid=wide["valid_time"]).tolist())
            )
        print(
            key,
            wide.height,
            {a: round(v["score"][0], 3) for a, v in results[key]["arms"]["all"].items()},
            flush=True,
        )
    (WORK / "scores.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
