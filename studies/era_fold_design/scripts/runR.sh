#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out; S=../scratch/era-fold/scripts; L=$O/runR.log
fit() { echo "=== [${DEVICE:-cpu}] $* $(uptime | sed 's/.*load/load/')" >> $L
  uv run python $S/$1 "${@:2}" 2>&1 | grep --line-buffered "^done\|rror\|Trace\|^rows\|^offsets\|^device\|^arms\|^{\|Uncovered\|cells" | cut -c1-500 >> $L; }
export DEVICE=cuda
fit partC_fit.py solar D0trim 1 --sens
fit partC_fit.py solar I 1 --sens
echo SOLARSENSDONE >> $L
DEVICE=cpu fit partC_fit.py wind D0trim 1 --extra-only
fit partC_fit.py wind IIr 1
fit partC_fit.py solar IIr 1
fit partC_fit.py wind IIr 1 --sens
fit partC_fit.py solar IIr 1 --sens
echo IIRDONE >> $L
fit partC_fit.py wind I2 1
fit partC_fit.py solar I2 1
echo I2DONE >> $L
export DEVICE=cpu
fit partB_fit.py wind D1r3
fit partB_fit.py solar D1r3
echo D1R3DONE >> $L
for a in "solar_all D0" "solar_all D1" "wind_dream D0" "wind_dream D1"; do fit partP.py $a; done
echo ALLDONE >> $L
