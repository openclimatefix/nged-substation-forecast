#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out; S=../scratch/era-fold/scripts; L=$O/runQ2.log
until grep -q I2DONE $O/runQ.log 2>/dev/null; do sleep 30; done
fit() { echo "=== $* $(uptime | sed 's/.*load/load/')" >> $L
  uv run python $S/$1 "${@:2}" 2>&1 | grep "^done\|rror\|Trace\|rows\|^{" | cut -c1-400 >> $L; }
fit partB_fit.py wind D1r3
fit partB_fit.py solar D1r3
echo D1R3DONE >> $L
for a in "solar_all D0" "solar_all D1" "wind_dream D0" "wind_dream D1"; do fit partP.py $a; done
echo ALLDONE >> $L
