#!/bin/bash
# usage: stats_stack.sh RUNDIR  -> paired stats for every stacking comparison, both tasks
R=$1; D=$(dirname "$0")
for t in toolalpaca science; do
  for p in "base tool" "base science" "base union" "base intersection" "tool union" "science union"; do
    set -- $p
    echo "== $t: $1 -> $2"
    "$D/../.venv/bin/python" "$D/stats.py" "$R/${t}__$1.json" "$R/${t}__$2.json"
  done
done
