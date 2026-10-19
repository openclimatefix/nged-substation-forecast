"""Score every arm of experiment 00102 and its contrasts with BG2, by scope.

Reads experiment 00102's arms (`work/arms/<name>.parquet`) and experiment 00087's arms, scores each
on the cases of the 292 runs of `fold_a_daily_gap_v1`, and writes tidy tables under `work/results/`
and `work/results/report.txt`, which holds every number the page quotes. All scores are in points of
capacity (fraction of capacity times 100). The month-resampled intervals and the per-case metric
are copied from experiment 00087's `e00087_score.py`.

**Cases.** A case is a series, a run, and a valid time before 2026-07-01 with a finite observation.
A contrast uses the cases that both arms hold. The 73 runs of `fold_a_every5d_v1` choose the best
pair and triple of blend members (the pair or triple with the lowest mean quantile score there) and
are never scored for a result.

**Scopes.** `all` is every case. `midday_summer` is valid times from 10:00 to 14:00 UK time in
April to September. `night` is valid times from 23:00 to 03:59 UK time all year. `other` is every
case in neither. Each scope is also taken for the 5 Primary series that have a solar offset
(`|offset5`). The lead bands of `e00102_common.LEAD_BANDS` are scopes too.

**Intervals.** Every interval resamples whole origin months (20,000 resamples). The five verdict
contrasts are read at 99% (5% divided by five, Bonferroni); every other contrast at 95%.

    uv run python ml-experiments/00102-*/e00102_score.py
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00102_assemble as assemble
import e00102_common as common
import numpy as np
import polars as pl

from studies import bootstrap

N_RESAMPLES: Final[int] = 20_000
POINTS: Final[float] = 100.0
LEVEL_95: Final[float] = 95.0
VERDICT_COUNT: Final[int] = 5
VERDICT_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / VERDICT_COUNT)
"""99 percent: 5 percent divided by the five verdict contrasts (Bonferroni)."""
THRESHOLDS: Final[dict[str, float]] = {
    "Primary": 0.15,
    "BSP/GSP": 0.30,
    "solar": 0.10,
    "wind": 0.30,
}
"""A simplification is safe when the upper end of its interval is below its class's threshold."""
TAIL_LEVELS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
CLASS_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Primary": ("Primary",),
    "BSP/GSP": ("BSP", "GSP"),
    "solar": ("solar",),
    "wind": ("wind",),
}
BG2_OF_CLASS: Final[dict[str, str]] = {
    "Primary": "p_s6_bg2",
    "BSP/GSP": "b_weekly",
    "solar": "g_ens_weekly",
    "wind": "g_ens_weekly",
}
MIDDAY_HOURS: Final[tuple[int, ...]] = (10, 11, 12, 13)
NIGHT_HOURS: Final[tuple[int, ...]] = (23, 0, 1, 2, 3)
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
BLEND_MEMBERS: Final[tuple[str, ...]] = assemble.MEMBER_NAMES
REFERENCES: Final[tuple[str, ...]] = (
    "ref_manual_heuristic",
    "ref_climatology",
    "ref_xgboost_baseline",
)
ARM_COLUMNS_READ: Final[tuple[str, ...]] = (*assemble.KEY_COLUMNS, *common.LEVEL_COLUMNS)


@dataclass(frozen=True)
class Contrast:
    """One planned comparison: treatment minus reference, in one class.

    Attributes:
        name: A short label for the tables.
        cls: The class (`Primary`, `BSP/GSP`, `solar`, or `wind`).
        treatment: The arm whose score is compared.
        reference: The arm it is compared against.
        verdict: Whether the contrast decides the verdict (read at 99% in its verdict scope).
        verdict_scope: The scope the verdict reads (only for a verdict contrast).
    """

    name: str
    cls: str
    treatment: str
    reference: str
    verdict: bool = False
    verdict_scope: str = "all"


def arm_path(*, name: str) -> Path:
    """Return the file of an arm: this experiment's if it has one, else experiment 00087's."""
    own = assemble.common.WORK_DIR / "arms" / f"{name}.parquet"
    return own if own.exists() else assemble.BG2_ARMS_DIR / f"{name}.parquet"


@functools.cache
def draw_counts(*, n_months: int, n_resamples: int) -> np.ndarray:
    """Return, per resample, how many times each of `n_months` months is drawn.

    The stream is that of `studies.bootstrap`: for each resample one draw of the seed (there is one
    seed here), then one draw of the months with replacement.
    """
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    counts = np.empty((n_resamples, n_months))
    for resample in range(n_resamples):
        generator.integers(0, 1)
        drawn = generator.integers(0, n_months, size=n_months)
        counts[resample] = np.bincount(drawn, minlength=n_months)
    return counts


def month_intervals(
    *, sums: np.ndarray, counts: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the interval of each scope's mean, resampling whole months.

    Args:
        sums: Shape (scopes, months): each scope's sum of the metric in each month.
        counts: Shape (scopes, months): each scope's row count in each month.
        level: The interval's coverage in percent.

    Returns:
        The lower and upper bounds, each of shape (scopes,), NaN for a scope with no rows.

    Raises:
        ValueError: If `level` is not above 1 and below 100.
    """
    if not 1.0 < level < 100.0:
        msg = f"level is a coverage in percent, above 1 and below 100: {level}"
        raise ValueError(msg)
    lower = np.full(sums.shape[0], np.nan)
    upper = np.full(sums.shape[0], np.nan)
    tail = (100.0 - level) / 2.0
    patterns: dict[tuple[bool, ...], list[int]] = {}
    for row in range(sums.shape[0]):
        if (counts[row] > 0).any():
            patterns.setdefault(tuple((counts[row] > 0).tolist()), []).append(row)
    for pattern, rows in patterns.items():
        columns = np.flatnonzero(np.array(pattern))
        draws = draw_counts(n_months=len(columns), n_resamples=N_RESAMPLES)
        numerator = sums[np.ix_(rows, columns)] @ draws.T
        denominator = counts[np.ix_(rows, columns)] @ draws.T
        bounds = np.percentile(numerator / denominator, [tail, 100.0 - tail], axis=1)
        lower[rows], upper[rows] = bounds[0], bounds[1]
    return lower, upper


class ClassCases:
    """The cases of one class, their scopes, and every arm's per-case scores.

    Attributes:
        cls: The class name.
        keys: The cases' keys, `site`, `lead_hours`, and `actual`, sorted.
        gap: Whether each case belongs to a run of `fold_a_daily_gap_v1`.
        every5d: Whether each case belongs to a run of `fold_a_every5d_v1`.
        month: Each case's origin-month index.
        months: The origin-month labels in index order.
        masks: Each scope's boolean mask over the cases (the gap runs only).
        qs: Each arm's per-case quantile score in points of capacity, NaN where it has no case.
        mae: Each arm's per-case absolute error of the median in points of capacity.
        tails: Each arm's tail table (coverage with its interval, and pinball loss).
    """

    def __init__(self, *, cls: str) -> None:
        """Load the class's cases from the BG2 arm, which holds every series and run."""
        self.cls = cls
        reference = (
            pl.read_parquet(
                assemble.BG2_ARMS_DIR / f"{self._reference_arm(cls=cls)}.parquet",
                columns=[*assemble.KEY_COLUMNS, "actual", "site", "group", "lead_hours"],
            )
            .filter(
                pl.col("group").is_in(CLASS_GROUPS[cls])
                & pl.col("actual").is_finite()
                & (pl.col("valid_time") < common.HOLDOUT)
            )
            .sort(*assemble.KEY_COLUMNS)
        )
        self.keys = reference
        local = reference["valid_time"].dt.convert_time_zone("Europe/London")
        runs_gap = set(common.origins_of_subset(subset="gap"))
        runs_5d = set(common.origins_of_subset(subset="every5d"))
        origin = reference["power_fcst_init_time"]
        self.gap = origin.is_in(runs_gap).to_numpy()
        self.every5d = origin.is_in(runs_5d).to_numpy()
        month_labels = origin.dt.strftime("%Y-%m").to_numpy()
        self.months, self.month = np.unique(month_labels, return_inverse=True)
        hour, calendar_month = local.dt.hour().to_numpy(), local.dt.month().to_numpy()
        midday = np.isin(hour, MIDDAY_HOURS) & np.isin(calendar_month, SUMMER_MONTHS)
        night = np.isin(hour, NIGHT_HOURS)
        base = {
            "all": np.ones(len(reference), dtype=bool),
            "midday_summer": midday,
            "night": night,
            "other": ~midday & ~night,
        }
        lead = reference["lead_hours"].to_numpy()
        for name, low, high in common.LEAD_BANDS:
            base[f"lead {name}"] = (lead >= low) & (lead < high)
        site5 = reference["site"].is_in(assemble.OFFSET_SITES).to_numpy()
        self.masks = {name: mask & self.gap for name, mask in base.items()}
        if cls == "Primary":
            self.masks.update(
                {f"{name}|offset5": mask & self.gap & site5 for name, mask in base.items()}
            )
        self.qs: dict[str, np.ndarray] = {}
        self.mae: dict[str, np.ndarray] = {}
        self.tails: dict[str, pl.DataFrame] = {}

    @staticmethod
    def _reference_arm(*, cls: str) -> str:
        return (
            "p_s6_bg2" if cls == "Primary" else "b_weekly" if cls == "BSP/GSP" else "g_ens_weekly"
        )

    def add_arm(self, *, name: str) -> bool:
        """Score an arm on the class's cases; return False if the arm's file is missing."""
        path = arm_path(name=name)
        if not path.exists():
            return False
        frame = (
            self.keys.select(assemble.KEY_COLUMNS)
            .join(
                pl.read_parquet(path, columns=list(ARM_COLUMNS_READ)),
                on=list(assemble.KEY_COLUMNS),
                how="left",
                validate="1:1",
                maintain_order="left",
            )
            .select(common.LEVEL_COLUMNS)
        )
        quantiles = frame.to_numpy().astype(np.float64)
        actual = self.keys["actual"].to_numpy().astype(np.float64)
        present = np.isfinite(quantiles).all(axis=1)
        qs = np.full(len(actual), np.nan)
        qs[present] = (
            common.pinball_quantile_score(quantiles=quantiles[present], actual=actual[present])
            * POINTS
        )
        mae = np.full(len(actual), np.nan)
        mae[present] = np.abs(quantiles[present, common.MEDIAN] - actual[present]) * POINTS
        self.qs[name], self.mae[name] = qs, mae
        self.tails[name] = self._tails(quantiles=quantiles, actual=actual, present=present)
        return True

    def _tails(
        self, *, quantiles: np.ndarray, actual: np.ndarray, present: np.ndarray
    ) -> pl.DataFrame:
        """Return the coverage (with a 95% interval) and pinball loss at the four tail levels."""
        use = present & self.gap
        rows = []
        for column, level in TAIL_LEVELS:
            q = quantiles[use, common.LEVEL_COLUMNS.index(column)]
            a = actual[use]
            covered = (a <= q).astype(np.float64)
            diff = a - q
            pinball = np.where(diff >= 0, level * diff, (level - 1) * diff) * POINTS
            sums, counts = self._month_sums(values=covered, use=use)
            lower, upper = month_intervals(sums=sums, counts=counts, level=LEVEL_95)
            rows.append(
                {
                    "level": level,
                    "coverage": float(covered.mean()),
                    "coverage_lower": float(lower[0]),
                    "coverage_upper": float(upper[0]),
                    "pinball": float(pinball.mean()),
                }
            )
        return pl.DataFrame(rows)

    def _month_sums(self, *, values: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.months)
        months = self.month[use]
        sums = np.bincount(months, weights=values, minlength=n)[None, :]
        counts = np.bincount(months, minlength=n).astype(np.float64)[None, :]
        return sums, counts

    def scope_stats(self, *, values: np.ndarray, level: float) -> pl.DataFrame:
        """Return the mean of a per-case value in every scope, with a month-resampled interval.

        Args:
            values: One value per case, NaN where a case is missing.
            level: The interval's coverage in percent.

        Returns:
            One row per scope: `scope`, `n`, `mean`, `lower`, `upper`.
        """
        n_months = len(self.months)
        finite = np.isfinite(values)
        names, sums, counts = [], [], []
        for name, mask in self.masks.items():
            use = mask & finite
            names.append(name)
            sums.append(np.bincount(self.month[use], weights=values[use], minlength=n_months))
            counts.append(np.bincount(self.month[use], minlength=n_months).astype(np.float64))
        sums_a, counts_a = np.array(sums), np.array(counts)
        lower, upper = month_intervals(sums=sums_a, counts=counts_a, level=level)
        total = counts_a.sum(axis=1)
        mean = np.divide(
            sums_a.sum(axis=1), total, out=np.full(len(total), np.nan), where=total > 0
        )
        return pl.DataFrame(
            {
                "scope": names,
                "n": total.astype(np.int64),
                "mean": mean,
                "lower": lower,
                "upper": upper,
            }
        )


def contrast_values(
    *, cases: ClassCases, treatment: str, reference: str, metric: str
) -> np.ndarray:
    """Return treatment minus reference per case, NaN where either arm lacks the case."""
    table = cases.qs if metric == "qs" else cases.mae
    return table[treatment] - table[reference]


def choose_best(*, cases: ClassCases, size: int) -> str:
    """Return the blend of `size` members with the lowest mean score on the 73 every5d runs."""
    best, best_score = "", np.inf
    for subset in assemble.itertools.combinations(BLEND_MEMBERS, size):
        name = f"p_b_{assemble.subset_label(members=subset).lower()}"
        if name not in cases.qs:
            continue
        use = cases.every5d & np.isfinite(cases.qs[name])
        score = float(cases.qs[name][use].mean())
        if score < best_score:
            best, best_score = name, score
    return best


def primary_contrasts(*, best_pair: str, best_triple: str) -> list[Contrast]:
    """Return the contrasts of the Primary class, the first being the verdict contrast C1."""
    bg2 = "p_s6_bg2"
    out = [
        Contrast("C1 simplest candidate - BG2", "Primary", "p_nl_weekly", bg2, True),
        Contrast(
            "C5 offset: BG2 - no offset",
            "Primary",
            bg2,
            "p_s5_blend_no_offset",
            True,
            "midday_summer|offset5",
        ),
        Contrast(
            "lags, weekly: no-lag - weekly with lags", "Primary", "p_nl_weekly", "p_s4_weekly"
        ),
        Contrast(
            "lags, monthly: no-lag - monthly with lags", "Primary", "p_nl_monthly", "p_s3_monthly"
        ),
        Contrast(
            "spread off, simplest: no spread - spread",
            "Primary",
            "p_nl_weekly_nospread",
            "p_nl_weekly",
        ),
        Contrast("weekly with lags - BG2", "Primary", "p_s4_weekly", bg2),
        Contrast("weekly with lags and offset - BG2", "Primary", "p_a2_no_blend", bg2),
        Contrast("BG2 without offset - BG2", "Primary", "p_s5_blend_no_offset", bg2),
        Contrast("BG2 without spread factor - BG2", "Primary", "p_a5_spread_off", bg2),
        Contrast("BG2 with ST uncorrected - BG2", "Primary", "p_b_bg2_st_uncorrected", bg2),
        Contrast("BG2 rebuilt - BG2 (reproduction)", "Primary", "p_b_bg2_rebuilt", bg2),
        Contrast("best pair - BG2", "Primary", best_pair, bg2),
        Contrast("best triple - BG2", "Primary", best_triple, bg2),
        Contrast("stack - BG2", "Primary", "p_s2_stack", bg2),
        Contrast("manual heuristic - BG2", "Primary", "ref_manual_heuristic", bg2),
        Contrast("simplest candidate - stack", "Primary", "p_nl_weekly", "p_s2_stack"),
        Contrast(
            "simplest candidate - manual heuristic",
            "Primary",
            "p_nl_weekly",
            "ref_manual_heuristic",
        ),
    ]
    out += [
        Contrast(f"member {m} alone - BG2", "Primary", f"p_m_{m.lower()}", bg2)
        for m in BLEND_MEMBERS
    ]
    for size in (2, 3):
        for subset in assemble.itertools.combinations(BLEND_MEMBERS, size):
            label = assemble.subset_label(members=subset).lower()
            out.append(Contrast(f"blend {label} - BG2", "Primary", f"p_b_{label}", bg2))
    return out


def other_contrasts() -> list[Contrast]:
    """Return the contrasts of the supply point, solar, and wind classes."""
    return [
        Contrast("C2 simplest candidate - BG2", "BSP/GSP", "b_nl_weekly", "b_weekly", True),
        Contrast(
            "lags, monthly: no-lag - monthly with lags", "BSP/GSP", "b_nl_monthly", "b_monthly"
        ),
        Contrast("with 14-day correction - BG2", "BSP/GSP", "b_a_correction", "b_weekly"),
        Contrast(
            "simplest candidate - manual heuristic",
            "BSP/GSP",
            "b_nl_weekly",
            "ref_manual_heuristic",
        ),
        Contrast("simplest candidate - stack", "BSP/GSP", "b_nl_weekly", "b_stack"),
        Contrast("C3 simplest candidate - BG2", "solar", "g_nl_monthly", "g_ens_weekly", True),
        Contrast("with fitted spread factor - BG2", "solar", "g_a_spread_on", "g_ens_weekly"),
        Contrast(
            "simplest candidate - manual heuristic", "solar", "g_nl_monthly", "ref_manual_heuristic"
        ),
        Contrast("C4 simplest candidate - BG2", "wind", "g_nl_monthly", "g_ens_weekly", True),
        Contrast("with fitted spread factor - BG2", "wind", "g_a_spread_on", "g_ens_weekly"),
        Contrast("with power-curve margin - BG2", "wind", "g_a_wind_margin", "g_ens_weekly"),
        Contrast(
            "simplest candidate - manual heuristic", "wind", "g_nl_monthly", "ref_manual_heuristic"
        ),
    ]


def arms_of_class(*, cls: str, contrasts: Sequence[Contrast]) -> list[str]:
    """Return every arm the class's contrasts and absolute table need."""
    names = {BG2_OF_CLASS[cls], *REFERENCES}
    for contrast in contrasts:
        names |= {contrast.treatment, contrast.reference}
    if cls == "Primary":
        names |= {
            "p_s1_frozen",
            "p_s2_stack",
            "p_s3_monthly",
            "p_s4_weekly",
            "p_s5_blend_no_offset",
        }
        for size in (2, 3):
            for subset in assemble.itertools.combinations(BLEND_MEMBERS, size):
                names.add(f"p_b_{assemble.subset_label(members=subset).lower()}")
    return sorted(names)


def score_class(
    *, cls: str
) -> tuple[list[pl.DataFrame], list[pl.DataFrame], list[pl.DataFrame], list[str]]:
    """Score every arm and contrast of one class.

    Args:
        cls: The class name.

    Returns:
        The absolute-score tables, the contrast tables, the tail tables, and report notes.
    """
    notes: list[str] = []
    cases = ClassCases(cls=cls)
    if cls == "Primary":
        contrasts = primary_contrasts(best_pair="p_b_w_m", best_triple="p_b_w_m_an")
    else:
        contrasts = [c for c in other_contrasts() if c.cls == cls]
    notes.extend(
        f"{cls}: arm {name} has no file, skipped"
        for name in arms_of_class(cls=cls, contrasts=contrasts)
        if not cases.add_arm(name=name)
    )
    if cls == "Primary":
        best_pair, best_triple = choose_best(cases=cases, size=2), choose_best(cases=cases, size=3)
        notes.append(f"Best pair on every5d runs: {best_pair}; best triple: {best_triple}")
        contrasts = primary_contrasts(best_pair=best_pair, best_triple=best_triple)
    absolute, tails = [], []
    for name in cases.qs:
        for metric, values in (("qs", cases.qs[name]), ("mae", cases.mae[name])):
            absolute.append(
                cases.scope_stats(values=values, level=LEVEL_95).with_columns(
                    cls=pl.lit(cls), arm=pl.lit(name), metric=pl.lit(metric)
                )
            )
        tails.append(cases.tails[name].with_columns(cls=pl.lit(cls), arm=pl.lit(name)))
    rows = []
    for contrast in contrasts:
        if contrast.treatment not in cases.qs or contrast.reference not in cases.qs:
            notes.append(f"{cls}: contrast {contrast.name} skipped, an arm is missing")
            continue
        for metric in ("qs", "mae"):
            verdict = contrast.verdict and metric == "qs"
            level = VERDICT_LEVEL if verdict else LEVEL_95
            values = contrast_values(
                cases=cases,
                treatment=contrast.treatment,
                reference=contrast.reference,
                metric=metric,
            )
            rows.append(
                cases.scope_stats(values=values, level=level).with_columns(
                    cls=pl.lit(cls),
                    contrast=pl.lit(contrast.name),
                    treatment=pl.lit(contrast.treatment),
                    reference=pl.lit(contrast.reference),
                    metric=pl.lit(metric),
                    verdict=pl.lit(verdict),
                    verdict_scope=pl.lit(contrast.verdict_scope),
                    level=pl.lit(level),
                )
            )
    return absolute, rows, tails, notes


SCOPES_OF_INTEREST: Final[tuple[str, ...]] = (
    "all",
    "midday_summer",
    "night",
    "other",
    "all|offset5",
    "midday_summer|offset5",
    "night|offset5",
    "other|offset5",
)
TAIL_ARMS: Final[tuple[str, ...]] = (
    "p_s6_bg2",
    "p_nl_weekly",
    "b_weekly",
    "b_nl_weekly",
    "g_ens_weekly",
    "g_nl_monthly",
    "ref_manual_heuristic",
)


def report_lines(
    *, absolute: pl.DataFrame, contrasts: pl.DataFrame, tails: pl.DataFrame
) -> list[str]:
    """Return the report text: verdicts, absolute scores, contrasts, scopes, and tails."""
    lines = ["== Verdict contrasts (qs, 99%, in the verdict scope) =="]
    verdicts = contrasts.filter(pl.col("verdict") & (pl.col("scope") == pl.col("verdict_scope")))
    lines.append(str(verdicts.select("cls", "contrast", "scope", "n", "mean", "lower", "upper")))
    for row in verdicts.iter_rows(named=True):
        if row["contrast"].startswith("C5"):
            lines.append(f"C5: the offset helps if the upper end is below 0: {row['upper']:.3f}")
            continue
        threshold = THRESHOLDS[row["cls"]]
        verdict = "safe" if row["upper"] < threshold else "not shown safe"
        lines.append(
            f"{row['contrast']} ({row['cls']}): upper {row['upper']:.3f} against threshold "
            f"{threshold:.2f}: {verdict}"
        )
    lines.append("== Absolute qs, all rows ==")
    lines.append(
        str(
            absolute.filter((pl.col("metric") == "qs") & (pl.col("scope") == "all"))
            .select("cls", "arm", "n", "mean", "lower", "upper")
            .sort("cls", "mean")
        )
    )
    lines.append("== Contrasts, qs, all rows (95% unless a verdict contrast) ==")
    lines.append(
        str(
            contrasts.filter((pl.col("metric") == "qs") & (pl.col("scope") == "all"))
            .select("cls", "contrast", "n", "mean", "lower", "upper", "level")
            .sort("cls")
        )
    )
    lines.append("== Scopes: the offset, no-offset, simplest candidate, and heuristic arms ==")
    lines.append(
        str(
            absolute.filter(
                (pl.col("metric") == "qs")
                & pl.col("arm").is_in(
                    ["p_s6_bg2", "p_s5_blend_no_offset", "p_nl_weekly", "ref_manual_heuristic"]
                )
                & pl.col("scope").is_in(list(SCOPES_OF_INTEREST))
            )
            .select("arm", "scope", "n", "mean", "lower", "upper")
            .sort("arm", "scope")
        )
    )
    lines.append("== Tails of selected arms ==")
    lines.append(
        str(tails.filter(pl.col("arm").is_in(list(TAIL_ARMS))).sort("cls", "arm", "level"))
    )
    return lines


def analyse(*, results_dir: Path) -> None:
    """Score every arm of every class and write the tables and the report."""
    results_dir.mkdir(parents=True, exist_ok=True)
    absolute_rows, contrast_rows, tail_rows, notes = [], [], [], []
    for cls in CLASS_GROUPS:
        absolute, rows, tails, class_notes = score_class(cls=cls)
        absolute_rows += absolute
        contrast_rows += rows
        tail_rows += tails
        notes += class_notes
    absolute_table = pl.concat(absolute_rows)
    contrast_table = pl.concat(contrast_rows)
    tail_table = pl.concat(tail_rows)
    absolute_table.write_parquet(results_dir / "absolute.parquet")
    contrast_table.write_parquet(results_dir / "contrasts.parquet")
    tail_table.write_parquet(results_dir / "tails.parquet")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80, tbl_width_chars=250):
        lines = [
            *notes,
            *report_lines(absolute=absolute_table, contrasts=contrast_table, tails=tail_table),
        ]
    (results_dir / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    """Score everything."""
    analyse(results_dir=common.WORK_DIR / "results")


if __name__ == "__main__":
    main()
