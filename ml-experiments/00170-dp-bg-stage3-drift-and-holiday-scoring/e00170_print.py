"""Print every number the write-up of experiment 170 quotes, from the three report files.

uv run --no-sync python ml-experiments/00170-*/e00170_print.py \
        > ml-experiments/00170-*/printed_report.txt
"""

import json

import e00170_stage2_common as common
import polars as pl

ARMS: tuple[str, ...] = (
    "a155", "c155", "d155", "s24_a", "s24_b", "s24_c", "s24_d", "sall_a", "sall_b", "sall_c",
    "sall_d", "r_a", "r_d", "k111", "bg2", "manual_heuristic", "climatology",
)  # fmt: skip


def contrast_line(*, label: str, entry: dict) -> str:
    """Return one contrast as a printed line."""
    return (
        f"{label}: {entry['estimate']:+.3f} ({entry['lower']:+.3f} to {entry['upper']:+.3f}); "
        f"sites better {entry['sites_better']}/{entry['n_sites']}, "
        f"months negative {entry['months_negative']}/{entry['n_months']}"
    )


def main() -> None:  # noqa: PLR0912
    """Print the report."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    folder = common.WORK_DIR / "report"
    report = json.loads((folder / "report.json").read_text())
    nofilter = json.loads((folder / "nofilter.json").read_text())
    events = json.loads((folder / "events.json").read_text())
    for period in ("scored", "replicate"):
        data = report[period]
        print(f"== {period}: {data['rows']:,} common rows (all arms), decision substations")
        for band, items in data["tables"]["by_lead"].items():
            print(f"-- mean quantile score, leads {band}")
            print({i["arm"]: round(i["qs"], 3) for i in items if i["arm"] in ARMS})
        for label, entry in data["tables"]["breakouts"].items():
            print(f"-- breakout {label} ({entry['rows']:,} rows)")
            print({i["arm"]: round(i["qs"], 3) for i in entry["scores"] if i["arm"] in ARMS})
        print("-- contrasts, all leads (99% interval from resampling calendar months)")
        for label, entry in data["contrasts"].items():
            print(contrast_line(label=label, entry=entry["all"]["qs"]))
            print(f"   mean-forecast absolute error: {entry['all']['abs_mean']['estimate']:+.3f}")
        print("-- contrasts at day D+1")
        for label, entry in data["contrasts"].items():
            print(f"{label}: {entry['D+1']['qs']['estimate']:+.3f}")
        print("-- tails (coverage q01 q05 q95 q99; pinball q01 q05 q95 q99)")
        for scope, items in data["tail"].items():
            if scope == "by hour band":
                continue
            for i in items:
                if i["arm"] in ARMS:
                    cover = [round(i[f"cov_{q}"], 3) for q in ("q01", "q05", "q95", "q99")]
                    pinball = [round(i[f"pin_{q}"], 3) for q in ("q01", "q05", "q95", "q99")]
                    print(f"{scope} | {i['arm']}: {cover} {pinball}")
    print("== leave one site out, scored year (all-lead contrast)")
    for label in ("C1", "C2", "C4", "C5"):
        print(label, {k: round(v, 3) for k, v in report["scored"]["leave_one_out"][label].items()})
    print("== site gains, scored year")
    for label in ("C1", "C2", "C4", "C5"):
        print(label, {k: round(v, 3) for k, v in report["scored"]["site_gains"][label].items()})
    print("== median minus mean forecast, 12 to 36 h (per arm and site)")
    print(
        pl.DataFrame(report["checks"]["median_minus_mean"]).filter(
            pl.col("arm").is_in(["r_a", "s24_d", "sall_d"])
        )
    )
    print("== lead check (growth from 0.5-3.5 h to 3.5-9.5 h minus arm A's)")
    for arm, entry in report["scored"]["lead_check"].items():
        print(arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items()})
    print("== stage 1 with and without the filter, scored year, leads 0.5-168 h")
    for row in nofilter["table"]:
        print(row["arm"], round(row["qs"], 3), round(row["abs_mean"], 3), row["rows"])
    for label, entry in nofilter["contrasts"].items():
        print(contrast_line(label=label, entry=entry["qs"]))
    print("== events: rows found")
    print(events["events_found"])
    ratios = pl.DataFrame(events["ratios"]).filter(
        pl.col("arm").is_in(["a155", "c155", "d155", "s24_a", "s24_c", "s24_d", "sall_d"])
    )
    print("== event-day ratios (event mean over ordinary mean at the same half-hours)")
    print(
        ratios.select(
            "arm", "scope", "period", "event", "reference", "n_event_rows", "event_mean", "ratio"
        ).sort("scope", "event", "reference", "period", "arm")
    )
    print("== with-minus-without differences on event days")
    print(
        pl.DataFrame(events["differences"])
        .select("difference", "scope", "event", "period", "rows", "dates", "mean_qs",
                "mean_abs_error_of_mean", "dates_worse", "sites_worse", "n_sites")
        .sort("scope", "event", "difference", "period")
    )  # fmt: skip


if __name__ == "__main__":
    main()
