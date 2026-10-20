"""Refit experiment 105's gradient arms P-map and S at the replicate cut (2024-07-01).

Experiment 105 stored the scored year only, so the replicate year's P-map and S are fitted here by
the same code (K shapes and the MAP deviation scale from its `tuned.json`, 1,200 Adam steps, a
window of 2 years, the spread from the replica fitted 12 months earlier). The sets are `p_map_rep`
and `s_rep`. The scored year's P, P-map, and S are experiment 105's stored sets.

    uv run --no-sync python ml-experiments/00112-*/e00112_refs.py
"""

import e00112_common as common
import e00112_gradient as gradient
import torch


def main() -> None:
    """Fit and store the replicate year's P-map and S."""
    torch.set_num_threads(4)
    cache = common.load_cache()
    settings = common.tuned()
    forecasts, record = gradient.arm_forecast(
        cache=cache,
        config=gradient.config_for(kind="p_map", settings=settings),
        fit_sites=common.POOL,
        score_sites=common.REPORT_SITES,
        months=common.WINDOW_MONTHS,
        cut=common.REPLICATE_CUT,
        through=common.FINAL_CUT,
    )
    common.store_arm(name="p_map_rep", frame=gradient.frames_from(cache=cache, forecasts=forecasts))
    print("p_map_rep", record, flush=True)
    forecasts, record = gradient.per_site_forecasts(
        cache=cache,
        config=gradient.config_for(kind="s", settings=settings),
        sites=common.REPORT_SITES,
        months=common.WINDOW_MONTHS,
        cut=common.REPLICATE_CUT,
        through=common.FINAL_CUT,
    )
    common.store_arm(name="s_rep", frame=gradient.frames_from(cache=cache, forecasts=forecasts))
    print("s_rep", record, flush=True)


if __name__ == "__main__":
    main()
