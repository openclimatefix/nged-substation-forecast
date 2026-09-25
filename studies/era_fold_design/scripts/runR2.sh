#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out; S=../scratch/era-fold/scripts; L=$O/runR2.log
until grep -q WINDDREAMDONE $O/runP.log 2>/dev/null; do sleep 30; done
fit() { echo "=== [${DEVICE:-cpu}] $* $(uptime | sed 's/.*load/load/')" >> $L
  uv run python $S/$1 "${@:2}" 2>&1 | grep --line-buffered "^done\|rror\|Trace\|^rows\|^offsets\|^device\|^arms\|^{\|^Coverage" | cut -c1-500 >> $L; }
export DEVICE=cuda
fit partC_fit.py solar IIr 1 --sens
echo IIRDONE >> $L
fit partC_fit.py wind I2 1
fit partC_fit.py solar I2 1
echo I2DONE >> $L
export DEVICE=cpu
fit partB_fit.py wind D1r3
fit partB_fit.py solar D1r3
echo D1R3DONE >> $L
fit partP.py solar_all D0
fit partP.py solar_all D1
fit partP.py wind_dream D0
fit partP.py wind_dream D1
uv run python $S/partP_analyse.py solar_all cpu pooled >> $L 2>&1
uv run python $S/partP_analyse.py wind_dream cpu pooled >> $L 2>&1
echo ALLDONE >> $L
