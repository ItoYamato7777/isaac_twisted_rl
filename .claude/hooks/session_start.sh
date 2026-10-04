#!/usr/bin/env bash
# SessionStart: git の状態・docs/STATUS.md の要点・走っている GPU ジョブを文脈に出す。
# 表示だけで、何も止めない。どのコマンドが失敗しても exit 0。
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" 2>/dev/null || exit 0

echo "## git"
git status -sb 2>/dev/null | head -15
git log --oneline -1 2>/dev/null

echo
echo "## docs/STATUS.md の要点"
if [ -f docs/STATUS.md ]; then
  awk '/^## 要点/{p=1;next} /^## /{if(p)exit} p' docs/STATUS.md | sed '/^$/d' | head -8
else
  echo "(docs/STATUS.md が無い)"
fi

echo
echo "## GPU のジョブ (自分が起動していないものは触らない。止める必要があれば PID をユーザーに伝える)"
if command -v nvidia-smi >/dev/null 2>&1; then
  out=$(timeout -k 5 10 nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)
  [ -n "$out" ] && echo "$out" | head -10 || echo "(なし)"
fi
ps -eo pid,etime,args 2>/dev/null \
  | grep -E 'scripts/rsl_rl/(train|play|eval_[a-z]+)\.py|bin/train |overhand_script|divergence_bench|bench_physics' \
  | grep -v grep | cut -c1-160 | head -10
exit 0
