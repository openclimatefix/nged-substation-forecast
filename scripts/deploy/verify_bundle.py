"""Check the step-5 bundle inside the production image against its own manifest, then load it.

`build_and_verify_image.sh` pipes this file to `python -I -` inside the built image, with no
network, so the check reads the bundle the image will serve rather than the copy on the host. The
image leaves out the bundle's `fixtures/` folder and its `.pt` weights, which only research's
`reference_check.py` reads, so those entries of the manifest are expected to be absent.

Exits 1 with one line per fault if a file's SHA-256 differs from `manifest.json`, a file the
image needs is missing, the image holds a file the manifest does not list, or `load_bundle` or
`require_servable_bundle` rejects the bundle.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Final

from substation_demand_forecaster.bundle import load_bundle, require_servable_bundle

NOT_SHIPPED_PREFIX: Final[str] = "fixtures/"
NOT_SHIPPED_SUFFIX: Final[str] = ".pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_faults(bundle_dir: Path) -> list[str]:
    """Return one line per file of `bundle_dir` that disagrees with its `manifest.json`."""
    expected: dict[str, str] = json.loads((bundle_dir / "manifest.json").read_text())["sha256"]
    shipped = {
        name: digest
        for name, digest in expected.items()
        if not name.startswith(NOT_SHIPPED_PREFIX) and not name.endswith(NOT_SHIPPED_SUFFIX)
    }
    present = {
        str(path.relative_to(bundle_dir))
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    faults = [f"missing: {name}" for name in sorted(shipped.keys() - present)]
    faults += [f"not in the manifest: {name}" for name in sorted(present - expected.keys())]
    faults += [
        f"SHA-256 differs: {name}"
        for name in sorted(shipped.keys() & present)
        if _sha256(bundle_dir / name) != shipped[name]
    ]
    return faults


def main(bundle_dir: Path) -> int:
    """Print the bundle's faults and return 1 if it has any, otherwise load it and return 0."""
    faults = bundle_faults(bundle_dir)
    for fault in faults:
        print(f"  [FAIL] {fault}", file=sys.stderr)
    if faults:
        return 1
    bundle = load_bundle(bundle_dir)
    require_servable_bundle(bundle)
    print(f"  {len(bundle.primaries)} Primaries, {len(bundle.seeds)} seeds, every SHA-256 matches.")
    print(f"  model_release {bundle.model_release}, refit month {bundle.refit_month}")
    print(f"  research commit {bundle.research_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
