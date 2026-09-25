#!/bin/bash
cd /home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design
O=../scratch/era-fold/out; S=../scratch/era-fold/scripts; L=$O/runW.log
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -ge 20 ]; do sleep 30; done; }
run() { wait_load; echo "=== $* $(uptime | sed 's/.*load/load/')" >> $L
  uv run python $S/partC_fit.py "$@" 2>&1 | grep -v "done:\|INFO\|UserWarning\|join_asof" >> $L; }
run wind I 1 --sens
run wind II 1 --sens
run wind D0 1 --extra-only
echo WINDDONE >> $L
run solar D0 1 --sens
run solar D0trim 1 --sens
run solar I 1 --sens
run solar II 1 --sens
echo ALLDONE >> $L
