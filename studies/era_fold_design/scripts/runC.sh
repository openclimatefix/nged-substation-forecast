#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
for a in "wind I" "wind II" "wind D0trim" "solar D0" "solar I" "solar II" "solar D0trim"; do
  echo "=== $a" >> ../scratch/era-fold/out/runC.log
  uv run python ../scratch/era-fold/scripts/partC_fit.py $a 1 2>&1 | grep -v "done:\|INFO\|UserWarning\|join_asof" >> ../scratch/era-fold/out/runC.log
done
echo ALLDONE >> ../scratch/era-fold/out/runC.log
