# ruff: noqa: E501
"""Experiment 194: PS5 census, and PS1's register distances from G2.

Prints counts and megawatts by D-label only, never a customer name, a series identifier or
coordinates beside a series. Reads NGED's series table, the August 2026 Embedded Capacity Register
(both Part 1 sheets and the demand-side response sheet), the Capacity Market components file, and
the weekly left-tail shares written by `e00194_changepoint.py`.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python e00194_census.py
"""

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00194_changepoint as cp
import e00194_common as common
import numpy as np
import polars as pl
from pyproj import Transformer
from studies.ecr_reader import ECR_PATH, REGISTER_SHEETS, read_sheet
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

METADATA: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
CM_COMPONENTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00186-switching-d8-gas-peaker-spark-spread-and-triads"
    / "public"
    / "registers"
    / "component.csv"
)
EXCEL_EPOCH: Final[date] = date(1899, 12, 30)
EAST: Final[str] = "Location (X-coordinate):Eastings (where data is held)"
NORTH: Final[str] = "Location (y-coordinate):Northings (where data is held)"
PRIMARY_NAMES: Final[dict[str, str]] = {
    "D2": "Warth Lane Skegness", "D3": "Sleaford Road Boston", "D4": "Ingoldmells", "D5": "Kirton",
    "D8": "Marsh Lane", "D9": "Trusthorpe", "D10": "Spilsby", "D11": "Horncastle",
    "D12": "Stickney", "D14": "Mount Bridge Boston", "D15": "Wrangle T1", "D16": "Donington",
    "D18": "Wrangle T2", "D19": "Chapel St Leonards", "D21": "Alford", "D22": "Langrick",
}  # fmt: skip
REPORT: list[str] = []
CM_GRID_SQUARES: Final[dict[str, tuple[int, int]]] = {
    "TF": (500_000, 300_000), "TA": (500_000, 400_000), "SK": (400_000, 300_000),
    "TG": (600_000, 300_000), "TL": (500_000, 200_000), "TQ": (500_000, 100_000),
}  # fmt: skip


def say(text: str = "") -> None:
    """Print a line and keep it for the printed report."""
    print(text, flush=True)
    REPORT.append(text)


def asset_class(*, source: str, technology: str, chp: str) -> str | None:
    """Return the price-following class of one technology slot, or None for other plant."""
    text = f"{source} {technology}".lower()
    if "storage" in text and "electrochemical" in text:
        return "battery"
    if "gas" in text and "biofuel" in text:
        return None
    if ("gas" in text or "oil" in text or "fossil" in text) and "engine" in text:
        return "gas or oil engine, CHP" if chp == "Yes" else "gas or oil engine, not CHP"
    if "fossil" in text and ("turbine" in text or "gas" in text or "oil" in text):
        return "other gas or oil turbine or plant"
    return None


def ecr_slots() -> pl.DataFrame:
    """Return one row per technology slot of every ECR register row (no names)."""
    pieces = []
    for sheet in REGISTER_SHEETS:
        frame = read_sheet(path=ECR_PATH, sheet_name=sheet).with_columns(sheet=pl.lit(sheet[-8:]))
        for slot in (1, 2, 3):
            suffix = "" if slot == 1 else f" {slot}"
            pieces.append(
                frame.select(
                    primary=pl.col("Primary"),
                    source=pl.col(f"Energy Source {slot}"),
                    technology=pl.col(f"Energy Conversion Technology {slot}"),
                    chp=pl.col(f"CHP Cogeneration{suffix} (Yes/No)"),
                    mw=pl.col(
                        f"Energy Source & Energy Conversion Technology {slot} - Registered Capacity (MW)"
                    ),
                    storage_mwh=pl.col(f"Storage Capacity {slot} (MWh)"),
                    status=pl.col("Connection Status"),
                    connected=pl.col("Date Connected"),
                    east=pl.col(EAST),
                    north=pl.col(NORTH),
                )
            )
    slots = pl.concat(pieces).filter(pl.col("source").is_not_null())
    return slots.with_columns(
        mw=pl.col("mw").cast(pl.Float64, strict=False),
        storage_mwh=pl.col("storage_mwh").cast(pl.Float64, strict=False),
        east=pl.col("east").cast(pl.Float64, strict=False),
        north=pl.col("north").cast(pl.Float64, strict=False),
        connected_date=pl.col("connected")
        .cast(pl.Float64, strict=False)
        .map_elements(
            lambda v: EXCEL_EPOCH + timedelta(days=int(v)), return_dtype=pl.Date, skip_nulls=True
        ),
    ).with_columns(
        asset_class=pl.struct("source", "technology", "chp").map_elements(
            lambda r: asset_class(source=r["source"], technology=r["technology"], chp=r["chp"]),
            return_dtype=pl.String,
        )
    )


def cm_metres(*, ref: str) -> tuple[float, float] | None:
    """Return easting and northing of a six-figure Ordnance Survey grid reference."""
    match = re.match(r"^\s*([A-Z]{2})\s*(\d{3})\s*(\d{3})\s*$", ref.upper())
    if match is None or match.group(1) not in CM_GRID_SQUARES:
        return None
    east0, north0 = CM_GRID_SQUARES[match.group(1)]
    return east0 + int(match.group(2)) * 100.0, north0 + int(match.group(3)) * 100.0


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the PS5 census and the register distances."""
    meta = pl.read_parquet(METADATA)
    to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

    say(
        "PS5 (a): series whose `information` field names an added-back generator or a reading that is not working"
    )
    say(
        f"  series in the table: {meta.height}; with an information note: {meta.filter(pl.col('information').is_not_null() & (pl.col('information') != '')).height}"
    )
    info = meta.filter(pl.col("information").is_not_null() & (pl.col("information") != ""))
    for r in info.iter_rows(named=True):
        text = r["information"].lower()
        kind = (
            "added-back generators and analogues not working"
            if "disaggregated" in text and "not working" in text
            else "analogue not working"
            if "not working" in text
            else "analogue suspect"
            if "suspect" in text
            else "customer not exporting"
            if "not exported" in text
            else "other"
        )
        label = (
            next(
                (k for k, v in PRIMARY_NAMES.items() if v.lower() in r["time_series_name"].lower()),
                None,
            )
            if r["time_series_type"] == "Disaggregated Demand"
            else None
        )
        say(f"  {label or 'generator'} ({r['time_series_type']}, {r['substation_type']}): {kind}")
    prim = meta.filter(pl.col("time_series_type") == "Disaggregated Demand")
    say(
        f"  Primaries typed Disaggregated Demand: {prim.height}; with an added-back note: {prim.filter(pl.col('information').str.contains('Disaggregated')).height}"
    )
    generators = meta.filter(pl.col("time_series_type") != "Disaggregated Demand").filter(
        pl.col("substation_type").is_in(["EHV Customer", "HV Customer"])
    )
    say(
        f"  metered generators and batteries: {generators.height}; with a note: {generators.filter(pl.col('information').is_not_null() & (pl.col('information') != '')).height}"
    )

    say(
        "\nPS5 (b): reading availability of the metered generators and batteries since 2024-01-01 (anonymous labels)"
    )
    ids = generators["time_series_id"].to_list()
    raw = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(ids), pl.col("time") >= datetime(2024, 1, 1, tzinfo=UTC)
        )
        .group_by("time_series_id")
        .agg(n=pl.col("time").n_unique(), last=pl.col("time").max(), first=pl.col("time").min())
        .collect()
    )
    expected_n = int(
        (datetime(2026, 7, 1, tzinfo=UTC) - datetime(2024, 1, 1, tzinfo=UTC))
        / timedelta(minutes=30)
    )
    joined = generators.join(raw, on="time_series_id", how="left").sort(
        "time_series_type", "time_series_id"
    )
    counter: dict[str, int] = {}
    for r in joined.iter_rows(named=True):
        counter[r["time_series_type"]] = counter.get(r["time_series_type"], 0) + 1
        n = r["n"] or 0
        say(
            f"  {r['time_series_type']} {counter[r['time_series_type']]}: {n / expected_n:.2f} of half-hours since 2024-01-01 have a reading; last reading {r['last']:%Y-%m-%d}"
            if n
            else f"  {r['time_series_type']} {counter[r['time_series_type']]}: no readings since 2024-01-01"
        )

    say(
        "\nPS5 (c): left-tail share (residual below -2.5 MW) step of every Primary in the stored panel"
    )
    panel = common.load_panel()
    for _key, v in sorted(panel.items(), key=lambda kv: kv[1]["site"]):
        tail = (v["resid"] < common.LEFT_TAIL_MW)[np.isfinite(v["resid"])]
        present = np.isfinite(v["resid"]).mean()
        res = np.where(
            np.isfinite(v["resid"]), (v["resid"] < common.LEFT_TAIL_MW).astype(float), np.nan
        )
        # Largest absolute difference in mean share between consecutive 13-week blocks (a crude step screen).
        block = 13 * 336
        means = [
            np.nanmean(res[i : i + block])
            for i in range(0, res.size - block, block)
            if np.isfinite(res[i : i + block]).sum() > block // 2
        ]
        step = max(abs(np.diff(means))) if len(means) > 1 else float("nan")
        top = max(means) if means else float("nan")
        say(
            f"  {v['site']}: overall share {tail.mean() if tail.size else float('nan'):.3f}, half-hours present {present:.2f}, 13-week blocks with data {len(means)}, largest block-to-block change {step:.3f}, max block share {top:.3f}"
        )

    say(
        "\nPS5 (c2): weekly one-step fit (as for D8) on every Primary with data, weeks before 2026-01-01"
    )
    rng = np.random.default_rng(0)
    for label in sorted(PRIMARY_NAMES, key=lambda x: int(x[1:])):
        frame = common.residual_frame(panel=panel, label=label)
        wk = cp.share_by(
            frame=cp.tail_frame(frame=frame, threshold_mw=common.LEFT_TAIL_MW), every="1w"
        )
        usable = wk.filter(pl.col("n") >= cp.MIN_WEEK_HALF_HOURS, pl.col("time") < cp.PLATEAU_END)
        if usable.height < 30:
            say(f"  {label}: only {usable.height} usable weeks; not fitted")
            continue
        r = cp.fit_one(label=label, weekly=wk, rng=rng)
        peak = (
            wk.filter(pl.col("n") >= cp.MIN_WEEK_HALF_HOURS)
            .sort("share", descending=True)
            .row(0, named=True)
        )
        say(
            f"    highest single week {peak['time']:%Y-%m-%d}: {peak['share']:.3f}; step at least {cp.STEP_THRESHOLD}: {r['step'] >= cp.STEP_THRESHOLD}"
        )

    say("\nPS5 (d): price-following assets in the August 2026 ECR, by Primary D-label")
    slots = ecr_slots()
    say(f"  register slots with a technology: {slots.height}")
    dsr = read_sheet(path=ECR_PATH, sheet_name="Register Part 2 (DSR)")
    say(f"  demand-side response register rows: {dsr.height}; columns {len(dsr.columns)}")
    labelled = []
    for label, name in PRIMARY_NAMES.items():
        labelled.append(
            slots.filter(
                pl.col("primary").str.to_lowercase().str.contains(name.lower())
            ).with_columns(site=pl.lit(label))
        )
    table = pl.concat(labelled)
    say(f"  slots under the 16 Primaries: {table.height}")
    flagged = table.filter(pl.col("asset_class").is_not_null())
    summary = (
        flagged.group_by("site", "asset_class", "status")
        .agg(
            n=pl.len(),
            mw=pl.col("mw").sum(),
            mwh=pl.col("storage_mwh").sum(),
            first=pl.col("connected_date").min(),
            last=pl.col("connected_date").max(),
        )
        .sort("site", "asset_class", "status")
    )
    for r in summary.iter_rows(named=True):
        say(
            f"  {r['site']}: {r['asset_class']}, {r['status']}: {r['n']} slot(s), {r['mw']:.2f} MW, {r['mwh'] or 0:.2f} MWh, connected {r['first']} to {r['last']}"
        )
    say(
        f"  Primaries with no price-following asset: {sorted(set(PRIMARY_NAMES) - set(flagged['site'].to_list()))}"
    )
    conn = flagged.filter(pl.col("status") == "Connected")
    by_class = (
        conn.group_by("asset_class")
        .agg(n=pl.len(), mw=pl.col("mw").sum(), sites=pl.col("site").n_unique())
        .sort("asset_class")
    )
    for r in by_class.iter_rows(named=True):
        say(
            f"  connected, all 16 Primaries: {r['asset_class']}: {r['n']} slot(s), {r['mw']:.2f} MW, {r['sites']} Primaries"
        )
    big = conn.filter(pl.col("mw") >= 1.0)
    say(
        f"  connected price-following slots of at least 1 MW: {big.height}; by site: {big.group_by('site').agg(mw=pl.col('mw').sum()).sort('site').rows()}"
    )
    say(
        f"  those with a connection date after 2023-06-01 (the window that matters for D8's history): {big.filter(pl.col('connected_date') >= date(2023, 6, 1)).select('site', 'mw', 'connected_date').rows()}"
    )
    chp = table.filter((pl.col("chp") == "Yes") & (pl.col("status") == "Connected"))
    say(
        f"  connected CHP slots under the 16 Primaries: {chp.height}, {chp['mw'].sum():.2f} MW; by site {chp.group_by('site').agg(mw=pl.col('mw').sum()).sort('site').rows()}"
    )
    gas_plants_all = slots.filter(
        pl.col("asset_class").is_in(["gas or oil engine, CHP", "gas or oil engine, not CHP"]),
        pl.col("status") == "Connected",
        pl.col("mw") >= 1.0,
    )
    say(
        f"  whole register, connected gas or oil engines of at least 1 MW: {gas_plants_all.height}, {gas_plants_all['mw'].sum():.1f} MW (all Primaries, trial or not)"
    )

    say("\nPS1: register entries near G2 (MW and distance only)")
    g2 = meta.filter(pl.col("time_series_id") == common.G2_SERIES).row(0, named=True)
    e, n = to_bng.transform(g2["longitude"], g2["latitude"])
    ecr_near = (
        slots.filter(pl.col("east").is_not_null())
        .with_columns(km=(((pl.col("east") - e) ** 2 + (pl.col("north") - n) ** 2).sqrt() / 1000.0))
        .filter(pl.col("km") <= 1.0, pl.col("asset_class").is_not_null())
        .sort("km")
    )
    for r in ecr_near.iter_rows(named=True):
        say(
            f"  ECR: {r['asset_class']}, {r['mw']:.2f} MW, {r['status']}, connected {r['connected_date']}, {r['km']:.2f} km from G2's coordinates"
        )
    if ecr_near.height == 0:
        say("  ECR: no price-following slot within 1 km of G2's coordinates")
    comp = pl.read_csv(CM_COMPONENTS, encoding="utf8-lossy", infer_schema_length=0)
    seen = set()
    for r in comp.iter_rows(named=True):
        xy = cm_metres(ref=r["OS Grid Reference"] or "")
        text = f"{r['Primary Fuel of Component']} {r['Generating Technology Class']} {r['Description of CMU Components']}".lower()
        if xy is None or "gas" not in text or (r["CMU ID"], r["OS Grid Reference"]) in seen:
            continue
        km = float(np.hypot(xy[0] - e, xy[1] - n) / 1000.0)
        if km <= 1.0:
            seen.add((r["CMU ID"], r["OS Grid Reference"]))
            say(
                f"  Capacity Market: gas-fuelled component, connection capacity {r['Connection or DSR Capacity']} MW, {km:.2f} km from G2's coordinates (grid reference is six-figure, so about 0.07 km precision)"
            )
    if not seen:
        say("  Capacity Market: no gas-fuelled component within 1 km of G2's coordinates")
    # The D8 comparison: the ECR's own Primary for the engine near G2.
    if ecr_near.height:
        top = ecr_near.row(0, named=True)
        owner = next(
            (k for k, v in PRIMARY_NAMES.items() if v.lower() in (top["primary"] or "").lower()),
            "another Primary",
        )
        say(f"  the nearest ECR price-following slot is registered under Primary {owner}")
    Path(__file__).with_name("printed_census.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
