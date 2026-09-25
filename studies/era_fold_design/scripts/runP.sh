#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out; S=../scratch/era-fold/scripts; L=$O/runP.log
while pgrep -f "partC_fit.py wind IIr 1 --sens" >/dev/null; do sleep 15; done
fit() { echo "=== [${DEVICE:-cpu}] $* $(uptime | sed 's/.*load/load/')" >> $L
  uv run python $S/$1 "${@:2}" 2>&1 | grep --line-buffered "^done\|rror\|Trace\|^{\|^Coverage" | cut -c1-400 >> $L; }
export DEVICE=cuda
fit partP.py solar_all D0
fit partP.py solar_all D1
uv run python $S/partP_analyse.py solar_all cuda pooled >> $L 2>&1
echo SOLARALLPRIMARYDONE >> $L
fit partP.py solar_all D0 --sens
fit partP.py solar_all D1 --sens
uv run python $S/partP_analyse.py solar_all cuda sensitivity >> $L 2>&1
echo SOLARALLSENSDONE >> $L
fit partP.py wind_dream D0
fit partP.py wind_dream D1
uv run python $S/partP_analyse.py wind_dream cuda pooled >> $L 2>&1
echo WINDDREAMDONE >> $L
