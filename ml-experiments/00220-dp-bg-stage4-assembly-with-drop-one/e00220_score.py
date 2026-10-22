# ruff: noqa: E501
"""Score every arm and the planned contrasts; write `work/report/report.json` and print it.

The quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. Contrasts are paired differences with month-resampled intervals (months are the months of
the forecast run's initialisation time), 20,000 resamples, drawn once and shared by every contrast
so that the stacking ratio is resampled consistently.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00220-*/e00220_score.py
"""

from typing import Final

import numpy as np
import polars as pl

import e00220_common as common  # isort: skip

N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 220
TAILS: Final[tuple[tuple[str, float, int], ...]] = (
    ("q01", 0.01, 0),
    ("q05", 0.05, 1),
    ("q95", 0.95, 9),
    ("q99", 0.99, 10),
)
NIGHT_BANDS: Final[tuple[int, ...]] = (0, 3)
"""Hour bands night and evening, in `studies.dp_calendar`'s order (night, morning, midday, evening)."""


def load() -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the common rows and every arm's quantiles (float64)."""
    rows = pl.read_parquet(common.ARM_DIR / "rows.parquet")
    arms = {
        name: np.load(common.ARM_DIR / f"{name}.npy").astype(np.float64) for name in common.ARMS
    }
    return rows, arms


def row_metrics(*, rows: pl.DataFrame, quantiles: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row score columns: the quantile score, point error, and the four tail columns."""
    actual = rows["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    pinball = np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)
    out = {
        "qs": 2.0 * pinball.mean(axis=1),
        "abs_median": np.abs(error[:, 5]),
    }
    for name, _level, position in TAILS:
        out[f"pin_{name}"] = pinball[:, position]
        out[f"cov_{name}"] = (actual <= quantiles[:, position]).astype(np.float64)
    return out


def subsets(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return boolean row masks: all, the decision substations, and the breakouts."""
    lead = rows["lead_hours"].to_numpy()
    decision = rows["site"].is_in(list(common.DECISION_SITES)).to_numpy()
    night = rows["hour_band"].is_in(list(NIGHT_BANDS)).to_numpy()
    winter = rows["valid_month"].is_in(list(common.WINTER_MONTHS)).to_numpy()
    cold = winter & night & rows["cold2"].to_numpy()
    low, high = common.DAY_AHEAD_LEADS
    day_ahead = (lead >= low) & (lead <= high)
    daylight = rows["daylight"].to_numpy()
    return {
        "five sites, all rows": decision,
        "five sites, leads 15.5-39 h": decision & day_ahead,
        "five sites, cold nights": decision & cold,
        "five sites, daylight": decision & daylight,
        "six sites, all rows": np.ones(rows.height, dtype=bool),
        "six sites, leads 15.5-39 h": day_ahead,
    }


class Bootstrap:
    """Paired month-block bootstrap with draws shared across every contrast."""

    def __init__(self, *, months: np.ndarray) -> None:
        """Fix the month codes and draw the resamples once."""
        labels, self.codes = np.unique(months, return_inverse=True)
        self.n_months = len(labels)
        rng = np.random.default_rng(SEED)
        self.draws = rng.integers(0, self.n_months, size=(N_RESAMPLES, self.n_months))

    def month_sums(self, *, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the per-month sum of `values` and row count over the masked rows."""
        sums = np.bincount(self.codes[mask], weights=values[mask], minlength=self.n_months)
        counts = np.bincount(self.codes[mask], minlength=self.n_months).astype(np.float64)
        return sums, counts

    def difference(
        self, *, treatment: np.ndarray, reference: np.ndarray, mask: np.ndarray, level: float
    ) -> dict:
        """Return the paired mean difference with its interval at `level` percent."""
        sums, counts = self.month_sums(values=treatment - reference, mask=mask)
        estimate = float(sums.sum() / counts.sum())
        resampled = sums[self.draws].sum(axis=1) / counts[self.draws].sum(axis=1)
        tail = (100.0 - level) / 2.0
        low, high = np.nanpercentile(resampled, [tail, 100.0 - tail])
        return {"estimate": estimate, "lower": float(low), "upper": float(high)}

    def ratio(
        self,
        *,
        numerator: np.ndarray,
        parts: list[np.ndarray],
        mask: np.ndarray,
        level: float,
    ) -> dict:
        """Return numerator / sum(parts) and the interaction, each with its interval."""
        n_sums, counts = self.month_sums(values=numerator, mask=mask)
        p_sums = np.sum([self.month_sums(values=p, mask=mask)[0] for p in parts], axis=0)
        n_draw = n_sums[self.draws].sum(axis=1)
        p_draw = p_sums[self.draws].sum(axis=1)
        total = counts[self.draws].sum(axis=1)
        tail = (100.0 - level) / 2.0
        ratio = n_draw / p_draw
        interaction = (n_draw - p_draw) / total
        return {
            "ratio": float(n_sums.sum() / p_sums.sum()),
            "ratio_lower": float(np.nanpercentile(ratio, tail)),
            "ratio_upper": float(np.nanpercentile(ratio, 100.0 - tail)),
            "interaction": float((n_sums.sum() - p_sums.sum()) / counts.sum()),
            "interaction_lower": float(np.nanpercentile(interaction, tail)),
            "interaction_upper": float(np.nanpercentile(interaction, 100.0 - tail)),
        }


def read_arm_d(*, rows: pl.DataFrame) -> np.ndarray:
    """Return experiment 155's arm D quantiles on the common rows (for gate 5)."""
    frames = [
        pl.read_parquet(common.E155_DIR / f"d_scored_{site}.parquet").with_columns(
            site=pl.lit(site)
        )
        for site in common.SCORED_SITES
    ]
    joined = rows.select(common.KEYS).join(
        pl.concat(frames).select([*common.KEYS, *common.QUANTILES]),
        on=common.KEYS,
        how="left",
        validate="1:1",
    )
    return joined.select(list(common.QUANTILES)).to_numpy().astype(np.float64)


def main() -> None:
    """Score the arms, the contrasts, the tails, and the gates."""
    rows, arms = load()
    masks = subsets(rows=rows)
    boot = Bootstrap(months=rows["month"].to_numpy())
    metrics = {name: row_metrics(rows=rows, quantiles=q) for name, q in arms.items()}
    qs = {name: m["qs"] for name, m in metrics.items()}
    report: dict = {"n_rows": {k: int(v.sum()) for k, v in masks.items()}}
    months = rows["month"].to_numpy()
    report["n_months"] = {k: len(np.unique(months[v])) for k, v in masks.items()}

    # Absolute scores and tail tables.
    report["absolute"] = {
        scope: {name: float(qs[name][mask].mean()) for name in arms}
        for scope, mask in masks.items()
    }
    tail_scopes = {
        "all rows": masks["five sites, all rows"],
        "cold nights": masks["five sites, cold nights"],
        "daylight": masks["five sites, daylight"],
        "leads 15.5-39 h": masks["five sites, leads 15.5-39 h"],
    }
    report["tails"] = {
        scope: {
            name: {
                f"{col}_{tail}": float(metrics[name][f"{col}_{tail}"][mask].mean())
                for tail, _, _ in TAILS
                for col in ("cov", "pin")
            }
            for name in arms
        }
        for scope, mask in tail_scopes.items()
    }

    # Contrasts.
    pairs = {
        "C1 A+H+S - A": ("a_hs", "a"),
        "C4a A+H - A": ("a_h", "a"),
        "C4b A+S - A": ("a_s", "a"),
        "C3 H-own - A+H": ("h_own", "a_h"),
        "H-own - A": ("h_own", "a"),
        "cost of removing H (A+S - A+H+S)": ("a_s", "a_hs"),
        "cost of removing S (A+H - A+H+S)": ("a_h", "a_hs"),
        "cost of removing widening (drop_widen - A+H+S)": ("drop_widen", "a_hs"),
        "cost of removing R, frozen network (drop_r - A+H+S)": ("drop_r", "a_hs"),
        "cost of removing offset, frozen network (drop_offset - A+H+S)": ("drop_offset", "a_hs"),
        "cost of removing R, no network (A+S drop R - A+S)": ("a_s_drop_r", "a_s"),
        "cost of removing offset, no network (A+S drop offset - A+S)": ("a_s_drop_offset", "a_s"),
    }
    report["contrasts"] = {}
    for scope, mask in masks.items():
        report["contrasts"][scope] = {
            label: {
                "level95": boot.difference(treatment=qs[t], reference=qs[r], mask=mask, level=95.0),
                "level99": boot.difference(treatment=qs[t], reference=qs[r], mask=mask, level=99.0),
                "abs_median": boot.difference(
                    treatment=metrics[t]["abs_median"],
                    reference=metrics[r]["abs_median"],
                    mask=mask,
                    level=95.0,
                ),
            }
            for label, (t, r) in pairs.items()
        }
    for scope in ("five sites, all rows", "five sites, leads 15.5-39 h", "six sites, all rows"):
        mask = masks[scope]
        report["contrasts"][scope]["C2 stacking"] = boot.ratio(
            numerator=qs["a_hs"] - qs["a"],
            parts=[qs["a_h"] - qs["a"], qs["a_s"] - qs["a"]],
            mask=mask,
            level=95.0,
        )
    # Tail contrasts on the cold nights and all rows for the main pairs.
    report["tail_contrasts"] = {}
    for scope in ("five sites, all rows", "five sites, cold nights", "five sites, daylight"):
        mask = masks[scope]
        report["tail_contrasts"][scope] = {
            f"{label} {col}_{tail}": boot.difference(
                treatment=metrics[t][f"{col}_{tail}"],
                reference=metrics[r][f"{col}_{tail}"],
                mask=mask,
                level=95.0,
            )
            for label, (t, r) in {
                "A+H+S - A": ("a_hs", "a"),
                "H-own - A+H": ("h_own", "a_h"),
            }.items()
            for tail, _, _ in TAILS
            for col in ("pin", "cov")
        }

    # Per-seed H-own (scores of the single-seed networks are not stored here; the seed mean is used).
    # Per-site and leave-one-out contrasts for the headline.
    report["by_site"] = {}
    for site in common.SCORED_SITES:
        mask = rows["site"].to_numpy() == site
        report["by_site"][site] = {
            "A": float(qs["a"][mask].mean()),
            "A+H+S - A": float((qs["a_hs"] - qs["a"])[mask].mean()),
            "A+H - A": float((qs["a_h"] - qs["a"])[mask].mean()),
            "A+S - A": float((qs["a_s"] - qs["a"])[mask].mean()),
            "H-own - A+H": float((qs["h_own"] - qs["a_h"])[mask].mean()),
        }
    report["leave_one_out"] = {}
    decision = masks["five sites, all rows"]
    for site in common.DECISION_SITES:
        mask = decision & (rows["site"].to_numpy() != site)
        report["leave_one_out"][site] = {
            "C1": boot.difference(treatment=qs["a_hs"], reference=qs["a"], mask=mask, level=95.0),
            "H-own - A+H": boot.difference(
                treatment=qs["h_own"], reference=qs["a_h"], mask=mask, level=95.0
            ),
            "cost of removing H": boot.difference(
                treatment=qs["a_s"], reference=qs["a_hs"], mask=mask, level=95.0
            ),
        }

    # Checks: median minus mean per site at 12 to 36 hours, where the mean is known.
    lead = rows["lead_hours"].to_numpy()
    mid = (lead >= 12.0) & (lead <= 36.0)
    mean_a = rows["mean"].to_numpy().astype(np.float64)
    delta = (arms["a_h"] - arms["a"])[:, 0]
    report["median_minus_mean"] = {
        site: {
            "A": float((arms["a"][:, 5] - mean_a)[(rows["site"].to_numpy() == site) & mid].mean()),
            "A+H": float(
                (arms["a_h"][:, 5] - mean_a - 0.0)[(rows["site"].to_numpy() == site) & mid].mean()
            ),
            "A+S": float(
                (arms["a_s"][:, 5] - mean_a)[(rows["site"].to_numpy() == site) & mid].mean()
            ),
            "mean_delta": float(delta[(rows["site"].to_numpy() == site) & mid].mean()),
        }
        for site in common.SCORED_SITES
    }

    # Gates.
    d_quantiles = read_arm_d(rows=rows)
    d_metrics = row_metrics(rows=rows, quantiles=d_quantiles)
    six = masks["six sites, all rows"]
    five = masks["five sites, all rows"]
    s_dl = masks["five sites, daylight"]
    report["gates"] = {
        "gate1_A_score": float(qs["a"][five].mean()),
        "gate1_target": 2.7519,
        "gate4_R2_daylight_change_percent": float(
            100.0 * (qs["a_s"][s_dl].sum() - qs["a"][s_dl].sum()) / qs["a"][s_dl].sum()
        ),
        "gate4_target": -2.30,
        "gate5_D_six_sites": float(d_metrics["qs"][six].mean()),
        "gate5_H_own_six_sites": float(qs["h_own"][six].mean()),
        "gate5_H_own_minus_D_six_sites": float((qs["h_own"] - d_metrics["qs"])[six].mean()),
        "gate5_target": -0.104,
        "gate5_D_five_sites": float(d_metrics["qs"][five].mean()),
        "gate5_D_five_target": 2.756,
        "gate6_direct_vs_additive_max": float(
            np.abs(
                arms["a_hs"]
                - np.sort(arms["a"] + (arms["a_s"] - arms["a"]) + delta[:, None], axis=1)
            ).max()
        ),
    }
    # Library cross-check of C1's interval (same resampling scheme, different draws).
    report["rows_by_arm_sorted_ok"] = {
        name: bool((np.diff(q, axis=1) >= 0).all()) for name, q in arms.items()
    }
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=report)

    def show(label: str, value: dict) -> None:
        print(f"{label}: {value['estimate']:+.4f} ({value['lower']:+.4f} to {value['upper']:+.4f})")

    scope = "five sites, all rows"
    print("rows", report["n_rows"])
    print("absolute", {k: round(v, 4) for k, v in report["absolute"][scope].items()})
    print("gates", report["gates"])
    for label in pairs:
        show(label + " [95%]", report["contrasts"][scope][label]["level95"])
    show("C1 [99%]", report["contrasts"][scope]["C1 A+H+S - A"]["level99"])
    print("C2", report["contrasts"][scope]["C2 stacking"])
    print(
        "cold-night q95 coverage",
        {k: round(v["cov_q95"], 3) for k, v in report["tails"]["cold nights"].items()},
    )


if __name__ == "__main__":
    main()
