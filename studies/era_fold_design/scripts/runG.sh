#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out; S=../scratch/era-fold/scripts; L=$O/runG.log
while pgrep -f "fit_B_rot.py" >/dev/null; do sleep 30; done
for d in D0 D0trim I II; do
  echo "=== solar $d (cuda) $(uptime | sed 's/.*load/load/')" >> $L
  DEVICE=cuda uv run python $S/partC_fit.py solar $d 1 --sens 2>&1 | grep -v "done:\|INFO\|UserWarning\|join_asof" >> $L
done
echo ALLDONE >> $L
