#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out
S=../scratch/era-fold/scripts
L=$O/runS.log
until grep -q ALLDONE $O/runC.log; do sleep 20; done
for st in wind solar; do for d in D0 D1 D2; do
  echo "=== B $st $d" >> $L
  uv run python $S/partB_fit.py $st $d sensitivity 2>&1 | grep "^done\|rows\|Error\|Trace" >> $L
done; done
echo "=== B sens fits finished" >> $L
uv run python $S/partC_analyse.py pooled >> $L 2>&1
uv run python $S/nearline.py >> $L 2>&1
echo "=== primary C analysis + near-line list done" >> $L
for a in "wind D0" "wind D0trim" "wind I" "wind II" "solar D0" "solar D0trim" "solar I" "solar II"; do
  echo "=== C $a" >> $L
  uv run python $S/partC_fit.py $a 1 --sens 2>&1 | grep -v "done:\|INFO\|UserWarning\|join_asof" >> $L
done
echo ALLDONE >> $L
