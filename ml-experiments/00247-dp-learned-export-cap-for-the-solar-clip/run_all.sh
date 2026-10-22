#!/usr/bin/env bash
# Usage: run_all.sh : run every learned-cap arm chain at the four sites, four jobs at a time.
set -euo pipefail
D=ml-experiments/00247-dp-learned-export-cap-for-the-solar-clip
L=/home/jack/dev/nged-substation-forecast/data/ml-experiments/00247-dp-learned-export-cap-for-the-solar-clip/logs
mkdir -p "$L"
one() {
  local arm=${1%%:*} site=${1##*:}
  systemd-run --user --scope -q -p MemoryMax=4G -p OOMPolicy=continue \
    bash "$D/run_arm.sh" "$arm" "$site" > "$L/${arm}_${site}.log" 2>&1
}
export -f one
export D L
printf '%s\n' p:D5 p:D9 e:D5 e:D9 sat:D5 sat:D9 p:D10 p:D16 e:D10 e:D16 sat:D10 sat:D16 |
  xargs -P4 -I{} bash -c 'one {}'
