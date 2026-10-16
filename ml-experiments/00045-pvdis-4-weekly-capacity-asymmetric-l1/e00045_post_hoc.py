"""Score the post hoc pair `alt_post_hoc` (a = 2, r = 10) on the scoring units (experiment 00045).

The pair was chosen after the frozen pair failed the injection test, so every number here is post
hoc: the frozen configuration, the tuning code, the limits, and the verdict logic are untouched,
and this script calls no verdict. It runs the frozen scoring's own code on the same scoring units
(Primaries, farms D to F, path seeds 1000 to 1003) and the same coverage units (one path seed,
100 bootstrap draws per unit), with one fit (the post hoc asymmetric pair) in place of the five.

The fits and ratio tables go in `scores/post_hoc/`. The top-level outputs end in `_post_hoc`:
`unit_scores_post_hoc.parquet`, `coverage_summary_post_hoc.parquet`,
`coverage_report_post_hoc.txt`, and `score_report_post_hoc.txt`, which `e00045_score.py` appends to
`score_report.txt` as a separate block. Condition (ii) compares the post hoc fit with the
season-year constant fit that `e00045_score.py` already scored, because that fit does not depend on
the penalty pair.

Stages: `jobs` (unit and placebo fits), `ratios`, `score`, `coverage`, `report`.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_post_hoc.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Final

import e00045_bootstrap as bootstrap
import e00045_common as common
import e00045_inject as inject
import e00045_score as score
import polars as pl

STAGES: Final[tuple[str, ...]] = ("jobs", "ratios", "score", "coverage", "report")
COVERAGE_DRAWS: Final[int] = 100
"""Bootstrap draws per unit, as in the frozen coverage run."""
POST_HOC_DIR: Final = score.SCORES_DIR / score.POST_HOC_DIR_NAME
HEADING: Final[str] = (
    "== POST HOC: alt_post_hoc (a=2, r=10), scored after the frozen pair failed; no verdict =="
)


@dataclass(frozen=True)
class PostHocPair(score.Frozen):
    """The post hoc penalty pair, whose only fit is the asymmetric one."""

    def fits(self) -> tuple[inject.FitSpec, ...]:
        """Return the single post hoc asymmetric fit."""
        return (("asymmetric", self.a, self.r),)


def _coverage_job(args: tuple[str, str, PostHocPair]) -> str:
    primary, farm, pair = args
    bootstrap.bootstrap_job(
        primary=primary,
        farm=farm,
        split="scoring",
        out_dir=POST_HOC_DIR,
        n_draws=COVERAGE_DRAWS,
        pair=pair,
    )
    return f"coverage {primary} {farm}"


def run_coverage(*, run: score.Run, pair: PostHocPair) -> pl.DataFrame:
    """Bootstrap the coverage units and write the top-level coverage files.

    Args:
        run: What is scored.
        pair: The post hoc pair.

    Returns:
        The coverage summary.
    """
    jobs = [(p, f, pair) for f in run.farms for p in run.primaries]
    with ProcessPoolExecutor(max_workers=run.workers) as pool:
        for line in pool.map(_coverage_job, jobs):
            print(line, flush=True)
    tables = pl.concat(
        [pl.read_parquet(f) for f in sorted(POST_HOC_DIR.glob("coverage_scoring_*.parquet"))]
    )
    summary = bootstrap.coverage_summary(tables=tables)
    score.atomic_write(frame=summary, path=score.SCORES_DIR / "coverage_summary_post_hoc.parquet")
    (score.SCORES_DIR / "coverage_report_post_hoc.txt").write_text(
        f"Coverage of the 90% residual-block-bootstrap interval of the post hoc pair "
        f"({COVERAGE_DRAWS} draws per unit; "
        f"{tables.select('primary', 'farm', 'case').unique().height} units).\n"
        + score.table_text(summary)
        + "\n"
    )
    return summary


def build_block(*, run: score.Run, pair: PostHocPair, coverage: pl.DataFrame) -> str:
    """Compute decision conditions (i) to (vi) for the post hoc pair, without a verdict.

    The conditions use the same code, limits, and statistics as `score.build_report`.

    Args:
        run: What is scored.
        pair: The post hoc pair.
        coverage: The post hoc coverage summary.

    Returns:
        The text of the post hoc block.
    """
    asym = "asymmetric"
    scores = pl.read_parquet(run.out_dir / "unit_scores.parquet")
    frozen_scores = pl.read_parquet(score.SCORES_DIR / "unit_scores.parquet")
    placebo = score.read_placebo(run=run)
    draws = score.Draws(n_primaries=scores["primary"].n_unique())
    lines = [
        HEADING,
        (
            f"Pair a={pair.a:g}, r={pair.r:g}; the same {run.farms} farms, path seeds "
            f"{inject.SCORING_SEEDS}, {len(placebo['perm_seed'].unique())} placebo permutations "
            f"per Primary, and coverage units ({COVERAGE_DRAWS} draws) as the frozen scoring. Same "
            "statistics and limits as the conditions above; the limits and the frozen "
            "configuration are unchanged and no verdict is called."
        ),
    ]

    errors = {
        v: score.median_of_medians(
            score.tensor_of(
                frame=scores,
                fit=asym,
                cases=score.CONDITION_I_CASES,
                column="path_error",
                variant=v,
            )
        )
        for v in score.VARIANTS
    }
    flat = {
        v: score.median_of_medians(
            score.tensor_of(
                frame=scores,
                fit=asym,
                cases=score.CONDITION_I_CASES,
                column="path_error_flat",
                variant=v,
            )
        )
        for v in score.VARIANTS
    }
    worst_i = max(errors.values())
    lines.append(
        f"(i) median path error on ramps and steps <= {score.LIMIT_I}: semi-synthetic "
        f"{errors['semi_synthetic']:.3f}, real plus injection {errors['real_plus_injection']:.3f}, "
        f"worse {worst_i:.3f} -> {'PASS' if worst_i <= score.LIMIT_I else 'FAIL'}"
        f" (an estimator that detects nothing: semi-synthetic {flat['semi_synthetic']:.3f}, "
        f"real plus injection {flat['real_plus_injection']:.3f})"
    )

    left = score.tensor_of(frame=scores, fit=asym, cases=score.C2_CASES, column="path_error")
    right = score.tensor_of(
        frame=frozen_scores, fit="season_year", cases=score.C2_CASES, column="path_error"
    )
    c2 = score.contrast(
        left=left, right=right, statistic=score.median_of_medians, draws=draws, relative=True
    )
    c2_primaries = score.contrast(
        left=left,
        right=right,
        statistic=score.median_of_medians,
        draws=draws,
        relative=True,
        resample_columns=False,
    )
    lines.append(
        f"(ii) upper limit of C2 < {score.LIMIT_II}: C2 {c2}; Primaries-only interval "
        f"[{c2_primaries.low:+.3f}, {c2_primaries.high:+.3f}] -> "
        f"{'PASS' if c2.high < score.LIMIT_II else 'FAIL'}"
    )

    constant_rate, sky_rate = score.false_step_tensors(scores=scores, placebo=placebo, fit=asym)
    stats = {
        "constant path": (
            score.median_of_means(constant_rate),
            score.worst_of_means(constant_rate),
        ),
        "placebo sky": (score.median_of_means(sky_rate), score.worst_of_means(sky_rate)),
    }
    passed_iii = all(
        m <= score.LIMIT_FALSE_MEDIAN and w <= score.LIMIT_FALSE_WORST for m, w in stats.values()
    )
    lines.append(
        f"(iii) false steps of 10% per Primary-year, median <= {score.LIMIT_FALSE_MEDIAN} and "
        f"worst <= {score.LIMIT_FALSE_WORST}, on both skies: "
        + "; ".join(f"{sky} median {m:.2f}, worst {w:.2f}" for sky, (m, w) in stats.items())
        + f" -> {'PASS' if passed_iii else 'FAIL'}"
    )
    inj_rate = score.tensor_of(
        frame=scores, fit=asym, cases=("constant_path",), column="steps_10_inj"
    )
    lines.append(
        "(iii) with the constant-path threshold at 10% of the injected constant capacity: "
        f"median {score.median_of_means(inj_rate):.2f}, worst {score.worst_of_means(inj_rate):.2f}"
    )

    constant_drift, sky_drift = score._drift_tensors(scores=scores, placebo=placebo, fit=asym)
    drift = {
        "constant path": score.median_of_medians(constant_drift),
        "placebo sky": score.median_of_medians(sky_drift),
    }
    lines.append(
        f"(iv) ratchet drift <= {score.LIMIT_DRIFT} of capacity per 5 years, on both skies: "
        + "; ".join(f"{sky} {value:+.3f}" for sky, value in drift.items())
        + f" -> {'PASS' if all(v <= score.LIMIT_DRIFT for v in drift.values()) else 'FAIL'}"
    )

    fall_26 = min(
        score.median_of_medians(
            score.tensor_of(
                frame=scores, fit=asym, cases=("fall_30_26",), column="fall_recovered", variant=v
            )
        )
        for v in score.VARIANTS
    )
    fall_8 = max(
        score.median_of_medians(
            score.tensor_of(
                frame=scores, fit=asym, cases=("fall_30_8",), column="fall_recovered", variant=v
            )
        )
        for v in score.VARIANTS
    )
    passed_v = fall_26 >= score.LIMIT_FALL_26 and fall_8 < score.LIMIT_FALL_8
    lines.append(
        f"(v) 30% fall of 26 weeks recovered >= {score.LIMIT_FALL_26} and of 8 weeks < "
        f"{score.LIMIT_FALL_8}: 26 weeks {fall_26:.3f} (worse baseline), 8 weeks {fall_8:.3f} "
        f"(worse baseline) -> {'PASS' if passed_v else 'FAIL'}"
    )

    overall = coverage.filter(pl.col("group") == "all")
    by_kind = {r["weeks"]: r for r in overall.iter_rows(named=True)}
    non_step, step = by_kind["non-step"], by_kind["step"]
    lo, hi = score.LIMIT_COVERAGE_NON_STEP
    passed_vi = (
        lo <= non_step["coverage_90"] <= hi and step["coverage_90"] >= score.LIMIT_COVERAGE_STEP
    )
    lines.append(
        f"(vi) 90% interval covers non-step weeks in {score.LIMIT_COVERAGE_NON_STEP} and step "
        f"weeks >= {score.LIMIT_COVERAGE_STEP}: non-step weeks {non_step['coverage_90']:.3f}, step "
        f"weeks {step['coverage_90']:.3f} -> {'PASS' if passed_vi else 'FAIL'}"
        f" (80% interval: {non_step['coverage_80']:.3f} and {step['coverage_80']:.3f}; 50% "
        f"interval: {non_step['coverage_50']:.3f} and {step['coverage_50']:.3f}; "
        f"{non_step['n_weeks']} non-step and {step['n_weeks']} step weeks in "
        f"{non_step['n_units']} units)"
    )
    lines.append("")
    lines.append(score.table_text(coverage))
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pair = PostHocPair(status="post_hoc", a=score.POST_HOC_PAIR[0], r=score.POST_HOC_PAIR[1])
    run = score.Run(
        split="scoring",
        primaries=common.PRIMARIES,
        farms=inject.SCORING_FARMS,
        out_dir=POST_HOC_DIR,
        origins=(),
        workers=args.workers,
    )
    run.out_dir.mkdir(parents=True, exist_ok=True)
    if "jobs" in args.stages:
        score.run_fit_jobs(run=run, frozen=pair)
    if "ratios" in args.stages:
        score.run_ratio_jobs(run=run, frozen=pair)
    if "score" in args.stages:
        scores = score.run_scoring(run=run)
        score.atomic_write(frame=scores, path=score.SCORES_DIR / "unit_scores_post_hoc.parquet")
    coverage = None
    if "coverage" in args.stages:
        coverage = run_coverage(run=run, pair=pair)
    if "report" in args.stages:
        if coverage is None:
            coverage = pl.read_parquet(score.SCORES_DIR / "coverage_summary_post_hoc.parquet")
        text = build_block(run=run, pair=pair, coverage=coverage)
        print(text)
        (score.SCORES_DIR / "score_report_post_hoc.txt").write_text(text)


if __name__ == "__main__":
    main()
