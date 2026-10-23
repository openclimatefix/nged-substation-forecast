"""Audit the step start: refit A2's model with the step window starting at each candidate half-hour.

For each decision substation the script fits the `e2` design (T-all, 20 night levels, and the
monotone temperature step) at the cut 2025-07-01 00:00 UTC, so the 24-month training window holds
no row of the scored year, once for each UTC half-hour from 00:00 to 03:00 as the start of the step
window. The fitted start is the one with the lowest unpenalised training loss. Experiment 123
placed the step at the half-hour starting 00:30 UTC at all 14 series; the verdict is capped at
`inconclusive` if the fitted start differs by more than 30 minutes at 2 or more of the 5
substations.

    uv run --no-sync python ml-experiments/00281-*/e00281_start_audit.py
"""

from datetime import UTC, datetime
from typing import Final

import e00281_common as common
import e00281_tall as tall
from studies.dp_pinned_capacity import read_vintages

CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
CANDIDATE_HALF_HOURS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 6)
"""The step window starts at 00:00, 00:30, ... 03:00 UTC."""
CARD_START_HALF_HOUR: Final[int] = 1
TOLERANCE_HALF_HOURS: Final[int] = 1


def main() -> None:
    """Fit every candidate start at every decision substation and write the table."""
    cache = common.load_cache()
    vintages = read_vintages()
    table: dict[str, dict] = {}
    agree = 0
    for site in common.DECISION_SITES:
        losses = {}
        for half_hour in CANDIDATE_HALF_HOURS:
            fit = tall.fit_tall(
                cache=cache,
                vintages=vintages,
                site=site,
                cut=CUT,
                penalty_weight=common.T_ALL_LAMBDA,
                variant="e2",
                step_start_half_hour=half_hour,
            )
            losses[half_hour] = fit.fit.unpenalised_loss
        best = min(losses, key=losses.__getitem__)
        within = abs(best - CARD_START_HALF_HOUR) <= TOLERANCE_HALF_HOURS
        agree += int(within)
        table[site] = {
            "fitted_start_half_hour": best,
            "fitted_start_utc": f"{best // 2:02d}:{30 * (best % 2):02d}",
            "within_30_minutes_of_card": within,
            "loss_relative_to_best": {
                f"{h // 2:02d}:{30 * (h % 2):02d}": losses[h] / losses[best] - 1.0 for h in losses
            },
        }
        print(site, table[site]["fitted_start_utc"], table[site]["within_30_minutes_of_card"])
    table["summary"] = {"agree_at": agree, "n_sites": len(common.DECISION_SITES)}
    print(f"within 30 minutes of the card's 00:30 start at {agree} of 5 substations")
    common.write_json_atomic(path=common.WORK_DIR / "start_audit.json", content=table)


if __name__ == "__main__":
    main()
