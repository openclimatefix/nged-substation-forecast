"""Fit the exploratory extra-lead arms on the GPU, and write their losses and report once.

One-off throwaway script for the extra lead days of
<https://github.com/openclimatefix/nged-substation-forecast/issues/912>. It reads the published
shared rows and folds (`nwp_forecast_comparison.rows` on the published inputs), joins the columns
`build_forecast_inputs.py --extra-leads` wrote onto them, fits each arm at the primary setting only
(every arm here is exploratory), and writes `<domain>_losses.parquet`,
`<domain>_predictions.parquet`, and `report.md` to a new `--output-dir`. It never writes to the
published folder. It reuses a domain's saved losses rather
than refitting, and refuses to overwrite `report.md`.

The arms:

- **New arms:** ENS mean at days 5, 10 and 14; GEFS mean at days 0, 5, 10 and 14; IFS 0.25° and GFS
  at days 0, 5 and 7; ICON global at days 0 and 5; every other Previous Runs product at day 0
  (Open-Meteo's freshest run covering each hour); and the ENS control member at day 0, whose
  columns the published inputs already hold. ARPEGE Europe and AROME France are solar only.
- **References:** every published arm the new arms are compared with, refitted here on the same
  device, because a GPU fit is not bit-identical to a CPU fit and a contrast must not mix them.
  Each reference's difference from its published CPU fit is the device noise floor.

`--check` fits one arm at one site twice, on the GPU, and stops unless the two runs produce the
same fingerprint. Run it before the full fit.

Every output carries only the anonymised `site` label.

Run it with `uv run python studies/nwp_forecast_comparison/fit_extra_leads.py --published-dir
PUBLISHED --output-dir DIR`, after `build_forecast_inputs.py --extra-leads --output-dir DIR` has
written the extra inputs there.
"""

import argparse
import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from build_forecast_inputs import PRODUCT_SLUGS, SOLAR_ONLY_PRODUCTS
from nwp_forecast_comparison import (
    METRIC,
    PERCENTAGE_POINTS,
    SETTINGS,
    TARGET,
    DomainType,
    arm_columns,
    assert_equal_rows,
    difference,
    fingerprint,
    leaderboard,
    losses_path,
    predictions_from_losses,
    predictions_path,
    rows,
)
from studies.bootstrap import bootstrap_absolute
from studies.cross_validation import DeviceType, out_of_fold_losses

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

DEVICE: Final[DeviceType] = "cuda"
"""The XGBoost device every fit here uses."""

SETTING: Final[str] = "primary"
"""The one hyperparameter setting fitted: every arm here is exploratory."""

NEW_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day5",
    "ens_mean_day10",
    "ens_mean_day14",
    "ens_control_day0",
    "gefs_mean_day0",
    "gefs_mean_day5",
    "gefs_mean_day10",
    "gefs_mean_day14",
    "ukv_day0",
    "ifs025_day0",
    "ifs025_day5",
    "ifs025_day7",
    "gfs_day0",
    "gfs_day5",
    "gfs_day7",
    "icon_global_day0",
    "icon_global_day5",
    "icon_d2_day0",
    "icon_eu_day0",
    "arpege_day0",
    "arome_day0",
    "knmi_harmonie_day0",
    "dmi_harmonie_day0",
)
"""The arms not fitted in the published run: `build_forecast_inputs.py --extra-leads` builds their
columns, except `ens_control_day0`, whose columns the published inputs already hold."""

REFERENCE_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day0",
    "ens_mean_day1",
    "ens_mean_day3",
    "ens_control_day1",
    "gefs_mean_day1",
    "gefs_mean_day3",
    "ifs025_day1",
    "ifs025_day3",
    "gfs_day1",
    "gfs_day3",
    "ukv_day1",
    "arpege_day1",
    "arome_day1",
    "knmi_harmonie_day1",
    "dmi_harmonie_day1",
    "icon_global_day1",
    "icon_global_day3",
    "icon_eu_day1",
    "icon_d2_day1",
)
"""The published arms refitted on the same device, as the new arms' references."""

SAME_PRODUCT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day5", "ens_mean_day3"),
    ("ens_mean_day10", "ens_mean_day3"),
    ("ens_mean_day14", "ens_mean_day3"),
    ("gefs_mean_day5", "gefs_mean_day3"),
    ("gefs_mean_day10", "gefs_mean_day3"),
    ("gefs_mean_day14", "gefs_mean_day3"),
    ("ifs025_day5", "ifs025_day3"),
    ("ifs025_day7", "ifs025_day3"),
    ("gfs_day5", "gfs_day3"),
    ("gfs_day7", "gfs_day3"),
    ("icon_global_day5", "icon_global_day3"),
    ("icon_d2_day0", "icon_d2_day1"),
    ("icon_eu_day0", "icon_eu_day1"),
    ("ens_control_day0", "ens_control_day1"),
    ("gefs_mean_day0", "gefs_mean_day1"),
    ("ukv_day0", "ukv_day1"),
    ("ifs025_day0", "ifs025_day1"),
    ("gfs_day0", "gfs_day1"),
    ("icon_global_day0", "icon_global_day1"),
    ("arpege_day0", "arpege_day1"),
    ("arome_day0", "arome_day1"),
    ("knmi_harmonie_day0", "knmi_harmonie_day1"),
    ("dmi_harmonie_day0", "dmi_harmonie_day1"),
)
"""Each new arm against the same product at the longest lead already fitted, as (treatment,
reference): the error's rise with lead."""

ENSEMBLE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("gefs_mean_day5", "ens_mean_day5"),
    ("gefs_mean_day10", "ens_mean_day10"),
    ("gefs_mean_day14", "ens_mean_day14"),
    ("ifs025_day5", "ens_mean_day5"),
    ("gefs_mean_day0", "ens_mean_day0"),
    ("ens_control_day0", "ens_mean_day0"),
    ("ukv_day0", "ens_mean_day0"),
    ("ifs025_day0", "ens_mean_day0"),
    ("gfs_day0", "ens_mean_day0"),
    ("icon_global_day0", "ens_mean_day0"),
    ("arpege_day0", "ens_mean_day0"),
    ("arome_day0", "ens_mean_day0"),
    ("knmi_harmonie_day0", "ens_mean_day0"),
    ("dmi_harmonie_day0", "ens_mean_day0"),
)
"""The other products against ENS at the same day. GEFS and the ENS control member have ENS's exact
lead. A Previous Runs product's day-0 lead follows its own run cycle, so its contrast with ENS mean
at day 0 mixes weather models and leads. IFS 0.25° day 5 does not have ENS's lead either."""

NEAR_ANALYSIS_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_d2_day0", "icon_eu_day0"),
    ("icon_d2_day1", "icon_eu_day1"),
)
"""ICON-D2 against ICON-EU at day 0 and day 1, also split by the hour of day modulo 3."""

ELSEWHERE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day10", "ens_mean_day5"),
    ("ens_mean_day14", "ens_mean_day5"),
    ("ens_mean_day14", "ens_mean_day10"),
    ("gefs_mean_day14", "gefs_mean_day10"),
    ("icon_d2_day0", "ens_mean_day0"),
    ("ens_mean_day0", "icon_eu_day0"),
)
"""Contrasts between two arms fitted here, as (treatment, reference): ENS's error rise from day 5 to
day 10 and day 14 and from day 10 to day 14, GEFS's fall from day 10 to day 14, ICON-D2 at day 0
against ENS at day 0, and ENS at day 0 against ICON-EU at day 0."""

CLIMATOLOGY_CONTRASTS: Final[tuple[str, ...]] = (
    "ens_mean_day10",
    "ens_mean_day14",
    "gefs_mean_day10",
    "gefs_mean_day14",
)
"""The arms compared with the no-weather climatology baseline, which is the published fit because
climatology involves no XGBoost model, so no device."""

HOUR_MODULO: Final[int] = 3
"""ICON-D2's and ICON-EU's runs start every 3 hours, so the freshest run's lead depends on the hour
of day modulo 3."""

MAX_MISSING_SHARE: Final[float] = 0.015
"""The largest share of rows with a missing weather value the report accepts for a new arm; the
published exploratory arms reach at most 1.48%."""


def joined_rows(*, published_dir: Path, output_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Return the published shared rows with the extra-lead columns joined on.

    Args:
        published_dir: The folder holding the published inputs.
        output_dir: The folder holding `<domain>_extra_lead_inputs.parquet`.
        domain: `solar` or `wind`.

    Returns:
        `nwp_forecast_comparison.rows`'s result, with the extra columns added.

    Raises:
        ValueError: If a shared row has no row in the extra inputs, or the join changes the row
            count.
    """
    frame = rows(input_dir=published_dir, domain=domain)
    extra = pl.read_parquet(output_dir / f"{domain}_extra_lead_inputs.parquet")
    keys = ["site", "time"]
    unmatched = frame.select(keys).join(extra.select(keys), on=keys, how="anti").height
    if unmatched:
        msg = f"{domain}: {unmatched} shared rows are missing from the extra-lead inputs"
        raise ValueError(msg)
    joined = frame.join(extra, on=keys, how="left")
    if joined.height != frame.height:
        msg = f"{domain}: joining the extra columns changed {frame.height} rows to {joined.height}"
        raise ValueError(msg)
    return joined


def domain_prefixes(*, domain: DomainType, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the arms of solar-only products from a wind run.

    Args:
        domain: `solar` or `wind`.
        prefixes: Arm prefixes such as `arpege_day0`.

    Returns:
        `prefixes` unchanged for solar; for wind, without the arms of `SOLAR_ONLY_PRODUCTS`.
    """
    if domain == "solar":
        return prefixes
    solar_only = tuple(f"{PRODUCT_SLUGS[product]}_day" for product in SOLAR_ONLY_PRODUCTS)
    return tuple(prefix for prefix in prefixes if not prefix.startswith(solar_only))


def missing_shares(*, frame: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Return each new arm's share of rows with any missing weather value.

    Args:
        frame: `joined_rows`'s result.
        domain: `solar` or `wind`.

    Returns:
        One row per new arm present in `frame`, with `arm` and `share`.

    Raises:
        ValueError: If a new arm's share exceeds `MAX_MISSING_SHARE`, which would confound the
            lead with coverage.
    """
    records = []
    for prefix in domain_prefixes(domain=domain, prefixes=NEW_PREFIXES):
        columns = arm_columns(domain=domain, prefixes=(prefix,))
        weather = [name for name in columns if name.startswith(f"{prefix}_")]
        if not weather or not all(name in frame.columns for name in weather):
            continue
        share = float(
            frame.select(
                pl.any_horizontal(pl.col(name).is_null() for name in weather).mean()
            ).item()
        )
        records.append({"arm": prefix, "share": share})
    result = pl.DataFrame(records, schema={"arm": pl.String, "share": pl.Float64})
    too_many = result.filter(pl.col("share") > MAX_MISSING_SHARE)
    if not too_many.is_empty():
        msg = f"{domain}: new arms with more than {MAX_MISSING_SHARE:.1%} missing: {too_many}"
        raise ValueError(msg)
    return result


def fit_arms(
    *, frame: pl.DataFrame, domain: DomainType, prefixes: tuple[str, ...], workers: int
) -> pl.DataFrame:
    """Fit every arm at every site, out of fold, on the GPU, and stack the losses.

    Args:
        frame: `joined_rows`'s result.
        domain: `solar` or `wind`.
        prefixes: The arms to fit.
        workers: How many (arm, site) fits run at once.

    Returns:
        Every fit's per-row losses, labelled with `arm` and `setting`.

    Raises:
        ValueError: If an arm's columns are absent from `frame`, which would otherwise drop the
            arm from the run silently.
    """
    sites = sorted(frame["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for prefix in prefixes:
            columns = arm_columns(domain=domain, prefixes=(prefix,))
            absent = [name for name in columns if name not in frame.columns]
            if absent:
                msg = f"{domain}: {prefix} has no columns {absent} in the joined rows"
                raise ValueError(msg)
            for site in sites:
                future = pool.submit(
                    out_of_fold_losses,
                    site_rows=frame.filter(pl.col("site") == site),
                    features=list(columns),
                    target=TARGET,
                    hyper_parameters=SETTINGS[SETTING],
                    with_quantiles=False,
                    device=DEVICE,
                )
                futures[future] = (prefix, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            prefix, site = futures[future]
            outputs.append(
                future.result().with_columns(arm=pl.lit(prefix), setting=pl.lit(SETTING))
            )
            _LOG.info("%s: %d/%d done: %s / site %s", domain, done, len(futures), prefix, site)
    return pl.concat(outputs)


def check_determinism(*, published_dir: Path, output_dir: Path) -> bool:
    """Fit one arm at one site twice on the GPU and compare the two runs' fingerprints.

    Args:
        published_dir: The folder holding the published inputs.
        output_dir: The folder holding the extra-lead inputs.

    Returns:
        Whether the two fingerprints agree.
    """
    frame = joined_rows(published_dir=published_dir, output_dir=output_dir, domain="wind")
    site = min(frame["site"].unique().to_list())
    fingerprints = []
    for _ in range(2):
        losses = fit_arms(
            frame=frame.filter(pl.col("site") == site),
            domain="wind",
            prefixes=("ens_mean_day1",),
            workers=1,
        )
        fingerprints.append(fingerprint(frame=losses))
    _LOG.info("two GPU runs of one arm at one site: %s", fingerprints)
    return fingerprints[0] == fingerprints[1]


def interval_text(*, point: float, lower: float, upper: float) -> str:
    """Format a difference or an error as `point [lower, upper]`, in percentage points."""
    scaled = [value * PERCENTAGE_POINTS for value in (point, lower, upper)]
    return f"{scaled[0]:+.3f} [{scaled[1]:+.3f}, {scaled[2]:+.3f}]"


def error_text(*, value: float, lower: float, upper: float) -> str:
    """Format an absolute error as `value [lower, upper]`, in percent of capacity."""
    scaled = [x * PERCENTAGE_POINTS for x in (value, lower, upper)]
    return f"{scaled[0]:.3f} [{scaled[1]:.3f}, {scaled[2]:.3f}]"


def contrast_line(*, losses: pl.DataFrame, treatment: str, reference: str) -> str | None:
    """Format one paired contrast as a Markdown table row, or None if an arm is absent.

    Args:
        losses: Per-row losses at one setting.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        `| treatment − reference | difference [interval] | rows | months |`.
    """
    present = set(losses["arm"].unique().to_list())
    if treatment not in present or reference not in present:
        return None
    result = difference(losses=losses, treatment=treatment, reference=reference)
    text = interval_text(
        point=result["difference"], lower=result["lower_95"], upper=result["upper_95"]
    )
    return f"| {treatment} − {reference} | {text} | {result['n_rows']} | {result['n_months']} |"


def served_lead(*, domain: DomainType, remainder: int) -> int:
    """Return a 3-hourly model's served lead in hours where `hour % 3 == remainder`.

    Radiation is a mean over the hour before its label, so its served lead is `((h - 1) % 3) + 1`;
    wind is an instantaneous value at its label, so its served lead is `h % 3`.

    Args:
        domain: `solar` or `wind`.
        remainder: The hour of day modulo `HOUR_MODULO`.

    Returns:
        The served lead in hours.
    """
    return ((remainder - 1) % HOUR_MODULO) + 1 if domain == "solar" else remainder


def by_hour_modulo(
    *, domain: DomainType, losses: pl.DataFrame, treatment: str, reference: str
) -> list[str]:
    """Format one contrast on each class of the hour of day modulo `HOUR_MODULO`.

    Args:
        domain: `solar` or `wind`, which decides each class's served lead.
        losses: Per-row losses at one setting.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        One table row per class that holds both arms, labelled with the class and its served lead.
    """
    lines = []
    for remainder in range(HOUR_MODULO):
        subset = losses.filter(pl.col("time").dt.hour() % HOUR_MODULO == remainder)
        line = contrast_line(losses=subset, treatment=treatment, reference=reference)
        if line is not None:
            lead = served_lead(domain=domain, remainder=remainder)
            label = f"(hour mod {HOUR_MODULO} = {remainder}, lead {lead} h)"
            lines.append(line.replace("| ", f"| {label} ", 1))
    return lines


def report_domain(
    *, domain: DomainType, losses: pl.DataFrame, published: pl.DataFrame, shares: pl.DataFrame
) -> list[str]:
    """Write one technology's report section.

    Args:
        domain: `solar` or `wind`.
        losses: The new fits' per-row losses at the primary setting.
        published: The published CPU fits' per-row losses at the primary setting.
        shares: `missing_shares`'s result.

    Returns:
        The section's Markdown lines.
    """
    arms = list(domain_prefixes(domain=domain, prefixes=(*NEW_PREFIXES, *REFERENCE_PREFIXES)))
    board = leaderboard(losses=losses, arms=arms)
    lines = [
        f"## {domain.capitalize()}",
        "",
        "### Absolute error of every arm fitted here (GPU, primary setting)",
        "",
        "| Arm | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|",
    ]
    for row in board.iter_rows(named=True):
        text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
        lines.append(
            f"| {row['arm']} | {text.split(' [')[0]} | [{text.split(' [')[1]} "
            f"| {row['n_rows']} | {row['n_months']} |"
        )
    lines += [
        "",
        "### Share of rows with a missing weather value, new arms",
        "",
        "| Arm | Share |",
        "|---|---|",
        *(f"| {row['arm']} | {row['share']:.4%} |" for row in shares.iter_rows(named=True)),
    ]
    header = [
        "",
        "| Contrast (points) | Difference [95% interval] | Rows | Months |",
        "|---|---|---|---|",
    ]
    for title, pairs in (
        (
            "Error's rise with lead: each new arm minus the same product at day 3 (or day 1)",
            SAME_PRODUCT_CONTRASTS,
        ),
        ("Other products against ENS at the same day", ENSEMBLE_CONTRASTS),
    ):
        table = [contrast_line(losses=losses, treatment=t, reference=r) for t, r in pairs]
        lines += ["", f"### {title}", *header, *(line for line in table if line)]
    near = [*header]
    for treatment, reference in NEAR_ANALYSIS_CONTRASTS:
        line = contrast_line(losses=losses, treatment=treatment, reference=reference)
        if line:
            near.append(line)
            near.extend(
                by_hour_modulo(
                    domain=domain, losses=losses, treatment=treatment, reference=reference
                )
            )
    lines += ["", "### ICON-D2 against ICON-EU, whole and by hour of day modulo 3", *near]
    climatology = published.filter(pl.col("arm") == "climatology")
    with_climatology = pl.concat([losses, climatology], how="vertical_relaxed")
    elsewhere = [
        contrast_line(losses=losses, treatment=t, reference=r) for t, r in ELSEWHERE_CONTRASTS
    ]
    elsewhere += [
        contrast_line(losses=with_climatology, treatment=arm, reference="climatology")
        for arm in CLIMATOLOGY_CONTRASTS
    ]
    lines += [
        "",
        "### Long leads against climatology, and day 0 against ENS at day 0",
        *header,
        *(line for line in elsewhere if line),
    ]
    lines += [
        "",
        "### Absolute error by hour of day modulo 3, ICON-D2 and ICON-EU at day 0",
        "",
        "| Arm | Hour of day mod 3 (served lead) | Error (% of capacity) | 95% interval | Rows |",
        "|---|---|---|---|---|",
    ]
    for arm in ("icon_d2_day0", "icon_eu_day0"):
        for remainder in range(HOUR_MODULO):
            subset = losses.filter(pl.col("time").dt.hour() % HOUR_MODULO == remainder)
            if arm not in set(subset["arm"].unique().to_list()):
                continue
            lead = served_lead(domain=domain, remainder=remainder)
            result = bootstrap_absolute(losses=subset, arm=arm, metric=METRIC)
            text = error_text(
                value=result["value"], lower=result["lower_95"], upper=result["upper_95"]
            )
            lines.append(
                f"| {arm} | {remainder} ({lead} h) | {text.split(' [')[0]} "
                f"| [{text.split(' [')[1]} | {result['n_rows']} |"
            )
    noise = ["", "### Device noise floor: GPU fit minus published CPU fit, same arm", *header]
    for prefix in domain_prefixes(domain=domain, prefixes=REFERENCE_PREFIXES):
        both = pl.concat(
            [
                losses.filter(pl.col("arm") == prefix).with_columns(arm=pl.lit("gpu")),
                published.filter(pl.col("arm") == prefix).with_columns(arm=pl.lit("cpu")),
            ],
            how="diagonal",
        )
        if (
            both.filter(pl.col("arm") == "cpu").is_empty()
            or both.filter(pl.col("arm") == "gpu").is_empty()
        ):
            continue
        assert_equal_rows(losses=both, treatment="gpu", reference="cpu")
        line = contrast_line(losses=both, treatment="gpu", reference="cpu")
        if line:
            noise.append(line.replace("| gpu − cpu", f"| {prefix} (GPU − CPU)", 1))
    return [*lines, *noise, ""]


def require_arms(*, frame: pl.DataFrame, domain: DomainType) -> None:
    """Raise unless every arm to fit has all its columns in `frame`.

    Args:
        frame: `joined_rows`'s result.
        domain: `solar` or `wind`.

    Raises:
        ValueError: If any arm is missing columns, which a GEFS gate that returned the keys
            unchanged would cause silently.
    """
    absent = [
        prefix
        for prefix in domain_prefixes(domain=domain, prefixes=(*NEW_PREFIXES, *REFERENCE_PREFIXES))
        if not all(name in frame.columns for name in arm_columns(domain=domain, prefixes=(prefix,)))
    ]
    if absent:
        msg = f"{domain}: arms with no columns in the joined rows: {absent}"
        raise ValueError(msg)


def main() -> int:
    """Fit the arms for both technologies and write the losses and report once."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, help="(arm, site) fits run at once.")
    parser.add_argument("--check", action="store_true", help="Compare two GPU runs of one arm.")
    parser.add_argument(
        "--report-only", action="store_true", help="Write the report from the saved losses."
    )
    args = parser.parse_args()
    if args.output_dir.resolve() == args.published_dir.resolve():
        msg = "the output folder must not be the published folder"
        raise ValueError(msg)
    if args.check:
        agree = check_determinism(published_dir=args.published_dir, output_dir=args.output_dir)
        sys.stdout.write(f"two GPU runs agree: {agree}\n")
        return 0 if agree else 1
    report_path = args.output_dir / "report.md"
    if report_path.exists():
        msg = f"{report_path} exists; the extra-lead report is write-once, move it first"
        raise FileExistsError(msg)
    frames = {
        domain: joined_rows(
            published_dir=args.published_dir, output_dir=args.output_dir, domain=domain
        )
        for domain in DOMAINS
    }
    shares = {domain: missing_shares(frame=frames[domain], domain=domain) for domain in DOMAINS}
    for domain in DOMAINS:
        require_arms(frame=frames[domain], domain=domain)
    report = [
        "# Extra lead days, GPU fits: report",
        "",
        (
            "Every arm is exploratory and fitted at the primary setting only, on the published "
            "shared rows and folds. Differences are first arm minus second, in percentage points "
            "of capacity; about 1 in 20 exploratory intervals reaches significance at the 5% level "
            "by chance."
        ),
        "",
    ]
    for domain in DOMAINS:
        path = losses_path(output_dir=args.output_dir, domain=domain)
        if path.exists():
            _LOG.info("%s exists; reporting from the saved losses", path)
            losses = pl.read_parquet(path)
        elif args.report_only:
            msg = f"{path} does not exist; --report-only needs both domains' losses"
            raise FileNotFoundError(msg)
        else:
            losses = fit_arms(
                frame=frames[domain],
                domain=domain,
                prefixes=(*NEW_PREFIXES, *REFERENCE_PREFIXES),
                workers=args.workers,
            )
            losses.write_parquet(path)
            predictions_from_losses(losses=losses, frame=frames[domain]).write_parquet(
                predictions_path(output_dir=args.output_dir, domain=domain)
            )
        published = pl.read_parquet(
            losses_path(output_dir=args.published_dir, domain=domain)
        ).filter(pl.col("setting") == SETTING)
        report += report_domain(
            domain=domain, losses=losses, published=published, shares=shares[domain]
        )
    report_path.write_text("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
