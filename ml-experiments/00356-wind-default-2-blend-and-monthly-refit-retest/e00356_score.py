# ruff: noqa: E501, PLR0912
"""Step 6 of experiment 00356: score the arms and read the planned contrasts.

Reads `arms/*.npy` (5 seeds, scored rows, 11) and `arms/rows.parquet`. Every score is in points of
capacity (experiment 33's quantile score over the nine inner levels, and single-level pinball
losses), on the kept rows, and the seed mean of each row's score is used. The scored year is the 53
Tuesday runs of 2025-07 to 2026-06 (blocks: calendar months of the run); the replicate is the
Tuesday runs of 2026-07 to 2026-10 (blocks: single runs). Contrasts are `arm - comparator`, so a
negative number means the arm is better. Writes `score.json` and `report.txt` to the data folder.

    PYTHONPATH=<see e00356_common> uv run --no-sync python e00356_score.py
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_arms as arms  # noqa: E402
import e00356_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ARM_NAMES: Final[tuple[str, ...]] = ("d1", "d2", "u1", "u2", "u2p", "u3", "u4")
CONTRASTS: Final[tuple[str, ...]] = ("u2", "u1", "u4", "u3")
"""The adoption order of the plan."""
MIN_GAIN: Final[dict[str, float]] = {"u1": 0.02, "u2": 0.02, "u4": 0.0, "u3": 0.0}
LEVEL: Final[float] = 1 - 0.01 / 4
RESAMPLES: Final[int] = 200_000
MEAN_COLUMNS: Final[tuple[str, ...]] = ("qs", "p01", "p05", "p95", "p99", "mae_median", "mae_mean")
INDICATORS: Final[tuple[str, ...]] = ("below_q01", "below_q05", "above_q95", "above_q99", "covered")


def row_scores(*, q_all: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return the seed mean of each per-row metric (rows,), in points of capacity or shares."""
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in (*MEAN_COLUMNS, *INDICATORS, "med_minus_mean")
    }
    scale = c303.POINTS_PER_FRACTION
    for q in q_all.astype(np.float64):
        mean_forecast = q[:, c303.NINE_INDEX] @ c303.NINE_MASSES
        out["qs"].append(c303.quantile_score(quantiles=q, actual=actual) * scale)
        for col, name in (("q01", "p01"), ("q05", "p05"), ("q95", "p95"), ("q99", "p99")):
            out[name].append(c303.pinball_at(quantiles=q, actual=actual, column=col) * scale)
        out["mae_median"].append(np.abs(actual - q[:, c303.MEDIAN_INDEX]) * scale)
        out["mae_mean"].append(np.abs(actual - mean_forecast) * scale)
        out["med_minus_mean"].append((q[:, c303.MEDIAN_INDEX] - mean_forecast) * scale)
        out["below_q01"].append((actual < q[:, 0]).astype(np.float64))
        out["below_q05"].append((actual < q[:, 1]).astype(np.float64))
        out["above_q95"].append((actual > q[:, 9]).astype(np.float64))
        out["above_q99"].append((actual > q[:, 10]).astype(np.float64))
        out["covered"].append(c303.covered(quantiles=q, actual=actual).astype(np.float64))
    return {k: np.mean(v, axis=0) for k, v in out.items()}


class Blocks:
    """Block bootstrap of a mean over kept rows, with whole blocks resampled."""

    def __init__(self, *, labels: np.ndarray, seed: int = 0) -> None:
        """Draw the resamples once so that every contrast uses the same draws."""
        self.unique, self.inverse = np.unique(labels, return_inverse=True)
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, self.unique.size, size=(RESAMPLES, self.unique.size))

    def means(self, *, values: np.ndarray, keep: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the point estimate and the resampled means of `values` over `keep`."""
        sums = np.bincount(self.inverse[keep], weights=values[keep], minlength=self.unique.size)
        counts = np.bincount(self.inverse[keep], minlength=self.unique.size).astype(np.float64)
        resampled = sums[self.draws].sum(axis=1) / np.maximum(counts[self.draws].sum(axis=1), 1)
        return float(sums.sum() / max(counts.sum(), 1)), resampled

    def interval(
        self, *, values: np.ndarray, keep: np.ndarray, level: float
    ) -> tuple[float, float, float]:
        """Return the mean and the two-sided interval at `level`."""
        point, resampled = self.means(values=values, keep=keep)
        low, high = np.quantile(resampled, [(1 - level) / 2, 1 - (1 - level) / 2])
        return point, float(low), float(high)

    def upper95(self, *, values: np.ndarray, keep: np.ndarray) -> float:
        """Return the one-sided 95% upper bound of the mean."""
        return float(np.quantile(self.means(values=values, keep=keep)[1], 0.95))


def main() -> None:  # noqa: PLR0915
    """Score every arm, read the contrasts, and write the report."""
    rows = pl.read_parquet(arms.ARMS_DIR / "rows.parquet")
    actual = rows["actual_norm"].to_numpy().astype(np.float64)
    kept = rows["kept"].to_numpy()
    hours = rows["lead_hours"].to_numpy().astype(np.float64)
    scored = rows["scored"].to_numpy()
    oot = ~scored
    scopes = {
        "headline": (hours >= 24) & (hours <= 240),
        "days_1_2": (hours >= 24) & (hours < 72),
        "days_3_10": (hours >= 72) & (hours <= 240),
        "all_leads": np.ones_like(kept),
    }
    metrics = {
        name: row_scores(q_all=np.load(arms.ARMS_DIR / f"{name}.npy"), actual=actual)
        for name in ARM_NAMES
    }
    seed_qs = {
        name: score303.row_metrics(quantiles=np.load(arms.ARMS_DIR / f"{name}.npy"), actual=actual)[
            "qs"
        ]
        for name in ("d2", "u3", "u4", "u2", "u1")
    }
    months = rows["month"].to_numpy()
    runs = rows["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy()
    blocks_scored = Blocks(labels=months)
    blocks_oot = Blocks(labels=runs)
    lines: list[str] = []
    result: dict[str, object] = {}

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    def keep_for(*, scope: str, year: bool) -> np.ndarray:
        return kept & scopes[scope] & (scored if year else oot)

    say("== absolute scores (points of capacity; kept rows)")
    say(
        "arm | headline S | days1-2 S | days3-10 S | all leads S | headline R | seed sd (headline S)"
    )
    for name in ARM_NAMES:
        parts = [
            f"{metrics[name]['qs'][keep_for(scope=s, year=True)].mean():.3f}"
            for s in ("headline", "days_1_2", "days_3_10", "all_leads")
        ]
        rep = metrics[name]["qs"][keep_for(scope="headline", year=False)].mean()
        sd = ""
        if name in seed_qs:
            sd = f"{seed_qs[name][:, keep_for(scope='headline', year=True)].mean(axis=1).std():.3f}"
        say(f"{name} | " + " | ".join(parts) + f" | {rep:.3f} | {sd}")
        result[f"abs_{name}"] = {
            s: float(metrics[name]["qs"][keep_for(scope=s, year=True)].mean()) for s in scopes
        } | {"replicate_headline": float(rep)}

    say()
    say("== gate: reproduction (headline and all leads, scored year, kept)")
    say(
        f"d2 {metrics['d2']['qs'][keep_for(scope='headline', year=True)].mean():.3f} (347: 10.518), "
        f"all leads {metrics['d2']['qs'][keep_for(scope='all_leads', year=True)].mean():.3f} (347: 11.081); "
        f"d1 {metrics['d1']['qs'][keep_for(scope='headline', year=True)].mean():.3f} (10.752); "
        f"u1 {metrics['u1']['qs'][keep_for(scope='headline', year=True)].mean():.3f} (348 z5: 10.605)"
    )

    say()
    say("== contrasts vs default 2 (scored year, headline; 99.75% month-block interval)")
    say(
        "arm | days | gain point | low | high | vs default 1 point | low | high | replicate point | 95% low | 95% high"
    )
    adoption: dict[str, dict[str, object]] = {}
    for name in CONTRASTS:
        row = [name]
        entry: dict[str, object] = {}
        for scope in ("headline", "days_1_2", "days_3_10", "all_leads"):
            keep = keep_for(scope=scope, year=True)
            diff = metrics[name]["qs"] - metrics["d2"]["qs"]
            point, low, high = blocks_scored.interval(values=diff, keep=keep, level=LEVEL)
            entry[f"vs_d2_{scope}"] = [point, low, high]
            if scope == "headline":
                row += [scope, f"{point:+.3f}", f"{low:+.3f}", f"{high:+.3f}"]
                d1diff = metrics[name]["qs"] - metrics["d1"]["qs"]
                p1, l1, h1 = blocks_scored.interval(values=d1diff, keep=keep, level=LEVEL)
                entry["vs_d1_headline"] = [p1, l1, h1]
                row += [f"{p1:+.3f}", f"{l1:+.3f}", f"{h1:+.3f}"]
                keep_r = keep_for(scope="headline", year=False)
                pr, lr, hr = blocks_oot.interval(values=diff, keep=keep_r, level=0.95)
                entry["replicate"] = [pr, lr, hr]
                row += [f"{pr:+.3f}", f"{lr:+.3f}", f"{hr:+.3f}"]
        say(" | ".join(row))
        for scope in ("days_1_2", "days_3_10", "all_leads"):
            say(f"   {scope}: {entry['vs_d2_' + scope]}")
        adoption[name] = entry
    result["contrasts"] = adoption

    say()
    say("== tails (scored year, headline scope)")
    say(
        "arm | below q01 % | below q05 % | above q95 % | above q99 % | p01 | p05 | p95 | p99 | 5-95 coverage %"
    )
    keep = keep_for(scope="headline", year=True)
    tails = {}
    for name in ARM_NAMES:
        m = metrics[name]
        vals = [
            100 * m["below_q01"][keep].mean(),
            100 * m["below_q05"][keep].mean(),
            100 * m["above_q95"][keep].mean(),
            100 * m["above_q99"][keep].mean(),
            m["p01"][keep].mean(),
            m["p05"][keep].mean(),
            m["p95"][keep].mean(),
            m["p99"][keep].mean(),
            100 * m["covered"][keep].mean(),
        ]
        tails[name] = vals
        say(f"{name} | " + " | ".join(f"{v:.3f}" for v in vals))
    result["tails"] = tails
    keep_r = keep_for(scope="headline", year=False)
    say("tails, replicate rows (headline scope):")
    for name in ARM_NAMES:
        m = metrics[name]
        say(
            f"{name} | above q99 {100 * m['above_q99'][keep_r].mean():.2f} % | p99 {m['p99'][keep_r].mean():.3f}"
            f" | p01 {m['p01'][keep_r].mean():.3f}"
        )

    say()
    say("== harm (one-sided 95% upper bound of arm minus comparator; headline, p01, p99)")
    harm: dict[str, dict[str, float]] = {}
    for name in CONTRASTS:
        for comparator in ("d2", "d1"):
            out = {}
            for metric in ("qs", "p01", "p99"):
                diff = metrics[name][metric] - metrics[comparator][metric]
                out[metric] = blocks_scored.upper95(values=diff, keep=keep)
            harm[f"{name}_vs_{comparator}"] = out
            say(
                f"{name} vs {comparator}: headline {out['qs']:+.3f}, p01 {out['p01']:+.3f}, "
                f"p99 {out['p99']:+.3f}"
            )
    result["harm"] = harm

    say()
    say("== placebo and added value")
    for name, comp in (("u2p", "d2"), ("u2", "d2"), ("u2", "u2p")):
        diff = metrics[name]["qs"] - metrics[comp]["qs"]
        point, low, high = blocks_scored.interval(values=diff, keep=keep, level=LEVEL)
        say(f"{name} minus {comp}: {point:+.3f} ({low:+.3f}, {high:+.3f})")
    for name in ("d2", "d1", "u1", "u2", "u3", "u4"):
        wide = []
        for comp in ("d2",):
            diff = metrics[name]["qs"] - metrics[comp]["qs"]
            _, low, high = blocks_scored.interval(values=diff, keep=keep, level=LEVEL)
            wide.append(high - low)
        result[f"width_{name}"] = wide[0]
    say(f"interval widths vs d2: u3 {result['width_u3']:.3f}, u4 {result['width_u4']:.3f}")

    say()
    say("== by farm (scored year, headline): qs / above q99 %")
    for name in ARM_NAMES:
        parts = []
        for farm in common.FARMS:
            k = keep & (rows["site"].to_numpy() == farm)
            parts.append(
                f"{farm} {metrics[name]['qs'][k].mean():.3f} / {100 * metrics[name]['above_q99'][k].mean():.2f}"
            )
        say(f"{name} | " + " | ".join(parts))

    say()
    say("== checks: median minus mean forecast per farm (points; scored year, headline)")
    for name in ("d1", "d2", "u1", "u2", "u3", "u4"):
        parts = [
            f"{farm} {metrics[name]['med_minus_mean'][keep & (rows['site'].to_numpy() == farm)].mean():+.2f}"
            for farm in common.FARMS
        ]
        say(f"{name} | " + " | ".join(parts))
    say("mean-forecast absolute error / median absolute error (scored year, headline):")
    for name in ("d1", "d2", "u1", "u2", "u3", "u4"):
        say(
            f"{name} | mean {metrics[name]['mae_mean'][keep].mean():.3f}"
            f" | median {metrics[name]['mae_median'][keep].mean():.3f}"
        )
    say("leave-one-farm-out sign check of the headline gain vs d2:")
    for name in CONTRASTS:
        diff = metrics[name]["qs"] - metrics["d2"]["qs"]
        parts = [
            f"without {farm}: {diff[keep & (rows['site'].to_numpy() != farm)].mean():+.3f}"
            for farm in common.FARMS
        ]
        say(f"{name} | " + " | ".join(parts))

    say()
    say("== blend weights (mean over months; plant or default 2 side) by quantile level")
    for s in (0,):
        w = np.load(arms.ARMS_DIR / f"u2_weights_s{s}.npy")
        say("u2, weight on default 2, mean over months, by lead band (rows) and level (cols):")
        say(str(np.round(w.mean(axis=0), 2).tolist()))
        w1 = np.load(arms.ARMS_DIR / f"u1_weights_s{s}.npy")
        say("u1, weight on the plant band, mean over months:")
        say(str(np.round(w1.mean(axis=0), 2).tolist()))
    (common.DATA_DIR / "score.json").write_text(json.dumps(result, indent=1))
    (common.DATA_DIR / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
