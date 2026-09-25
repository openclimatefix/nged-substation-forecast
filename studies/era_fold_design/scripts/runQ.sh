#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out; S=../scratch/era-fold/scripts; L=$O/runQ.log
while pgrep -f "partC_fit.py solar D0 1" >/dev/null; do sleep 20; done
fit() { echo "=== $* $(uptime | sed 's/.*load/load/')" >> $L
  uv run python $S/$1 "${@:2}" 2>&1 | grep -v "done:\|INFO\|UserWarning\|join_asof" >> $L; }
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
