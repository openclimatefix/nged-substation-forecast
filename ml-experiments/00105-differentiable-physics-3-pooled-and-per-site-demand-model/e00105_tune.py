"""Tune the pooled and per-site DP on the tuning year, by quantile score.

Everything is fitted on data before 2024-07-01 and scored from 2024-07-01 to 2025-06-30, so the
scored year (from 2025-07-01) is never used. The score is the mean quantile score of the forecast,
the fitted mean plus the centred quantiles of a replica fit's residuals, over the eligible MW
substations D5, D10, D11, D14, and D16 (D9 and D18 are left out because they hold too little
history to tune on).

1. The number of shared daily shapes K in {2, 4, 6}, for the variational pooled model.
2. The population scale of the per-site deviation in {0.1, 0.3, 1.0}: for the MAP pooled model
   (at the chosen K), and for the per-site model.

The best of each is written to `work/tuned.json`.

    uv run --no-sync python ml-experiments/00105-*/e00105_tune.py
"""

import json
import time
from typing import Final

import e00105_arms as arms
import e00105_common as common
import e00105_onboard as onboard
import numpy as np
import torch
from studies.dp_onboarding import after_mask, predict_pooled, select_rows

TUNE_SITES: Final[tuple[str, ...]] = common.ELIGIBLE_SITES
K_GRID: Final[tuple[int, ...]] = (2, 4, 6)
DEVIATION_GRID: Final[tuple[float, ...]] = (0.1, 0.3, 1.0)


def score_pooled(*, cache, config) -> float:  # noqa: ANN001
    """Return the mean headline quantile score of a pooled configuration on the tuning year."""
    forecasts, _ = arms.arm_forecast(
        cache=cache,
        config=config,
        fit_sites=common.POOL,
        score_sites=TUNE_SITES,
        months=common.WINDOW_MONTHS,
        cut=common.TUNING_CUT,
        through=common.TUNING_END,
        all_present_rows=False,
    )
    scores = [
        arms.quantile_score_of(
            frame=arms.frames_from(cache=cache, forecasts={site: forecast}), headline=True
        )
        for site, forecast in forecasts.items()
    ]
    return float(sum(scores) / len(scores))


def score_per_site(*, cache, config) -> float:  # noqa: ANN001
    """Return the mean headline quantile score of the per-site model on the tuning year."""
    scores = []
    for site in TUNE_SITES:
        forecasts, _ = arms.arm_forecast(
            cache=cache,
            config=config,
            fit_sites=(site,),
            score_sites=(site,),
            months=common.WINDOW_MONTHS,
            cut=common.TUNING_CUT,
            through=common.TUNING_END,
            all_present_rows=False,
        )
        scores.append(
            arms.quantile_score_of(frame=arms.frames_from(cache=cache, forecasts=forecasts))
        )
    return float(sum(scores) / len(scores))


def short_history_error(*, cache, config) -> float:  # noqa: ANN001
    """Return the mean absolute error of the per-site model on 4 weeks and on 3 months of history.

    The truncation is applied at the tuning cut, and the error is that of the mean forecast over
    the tuning year. The spread is the same for every candidate penalty, so the mean forecast
    carries the comparison.
    """
    errors = []
    for site in TUNE_SITES:
        full = common.pooled_batch_for(cache=cache, sites=(site,), cut=common.TUNING_CUT)
        score = common.scoring_batch_for(
            cache=cache, sites=(site,), after=common.TUNING_CUT, through=common.TUNING_END
        )
        for weeks in (4, 13):
            rows = onboard.weeks_rows(
                batch=full, cache=cache, site=site, weeks=weeks, cut=common.TUNING_CUT
            )
            model, _, _ = arms.fit(
                batch=select_rows(
                    batch=rows,
                    keep=after_mask(
                        batch=rows, cache=cache, after=np.datetime64("1970-01-01", "us")
                    ),
                ),
                config=config,
            )
            predicted = predict_pooled(model=model, batch=score)
            errors.append(float(np.abs(score.y.cpu().numpy() - predicted).mean()))
    return float(np.mean(errors))


def main() -> None:
    """Run the three grids and write `work/tuned.json`."""
    torch.set_num_threads(4)
    cache = common.load_cache()
    settings = dict(arms.DEFAULT_TUNED)
    report: dict[str, dict[str, float]] = {
        "k": {},
        "map_deviation": {},
        "site_deviation": {},
        "short_deviation": {},
    }
    for k in K_GRID:
        started = time.perf_counter()
        report["k"][str(k)] = score_pooled(
            cache=cache, config=arms.config_for(kind="p", settings={**settings, "k": k})
        )
        print(
            f"K={k}: {report['k'][str(k)]:.4f} ({time.perf_counter() - started:.0f} s)", flush=True
        )
    settings["k"] = int(min(report["k"], key=lambda name: report["k"][name]))
    for scale in DEVIATION_GRID:
        trial = {**settings, "map_deviation": scale, "site_deviation": scale}
        report["map_deviation"][str(scale)] = score_pooled(
            cache=cache, config=arms.config_for(kind="p_map", settings=trial)
        )
        print(f"P-map deviation {scale}: {report['map_deviation'][str(scale)]:.4f}", flush=True)
        report["site_deviation"][str(scale)] = score_per_site(
            cache=cache, config=arms.config_for(kind="s", settings=trial)
        )
        print(f"S deviation {scale}: {report['site_deviation'][str(scale)]:.4f}", flush=True)
        report["short_deviation"][str(scale)] = short_history_error(
            cache=cache, config=arms.config_for(kind="s", settings=trial)
        )
        print(f"S short-history deviation {scale}: {report['short_deviation'][str(scale)]:.4f}")
    settings["map_deviation"] = float(
        min(report["map_deviation"], key=lambda name: report["map_deviation"][name])
    )
    settings["site_deviation"] = float(
        min(report["site_deviation"], key=lambda name: report["site_deviation"][name])
    )
    settings["short_deviation"] = float(
        min(report["short_deviation"], key=lambda name: report["short_deviation"][name])
    )
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    arms.TUNED_PATH.write_text(json.dumps({**settings, "scores": report}, indent=2) + "\n")
    print(json.dumps(settings))


if __name__ == "__main__":
    main()
