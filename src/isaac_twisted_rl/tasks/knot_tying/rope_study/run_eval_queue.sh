#!/bin/bash
# 紐モデルの比較の評価 (eval_robust.py) をジョブ一覧から 1 本ずつ回す。出力済みのジョブは飛ばす。
#
#   bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_eval_queue.sh outputs/rope_study/eval/jobs.txt cuda:1
#
# ジョブ一覧は make_eval_jobs.py が作る (1 行 = <tag> <Play タスク> <チェックポイント> <環境数> [追加の引数...])。
# 出力: outputs/rope_study/eval/<tag>.json と .log (tag は "<方策>/<評価>")。
# 複数の GPU で回すときは、同じ一覧を別の GPU で同時に回してよい (始める前に .log を作って取り合わない)。
set -u
JOBS=$1
DEV=${2:-cuda:0}
ROOT=/home/itoyama/work/isaac_twisted_rl
cd $ROOT
OUT=$ROOT/outputs/rope_study/eval
while read -r tag task ckpt nenv extra; do
  [ -z "$tag" ] && continue
  mkdir -p $OUT/$(dirname $tag)
  [ -f $OUT/$tag.json ] && continue
  # 別の GPU のキューが先に取ったジョブは飛ばす (noclobber で .log を排他的に作る)
  ( set -o noclobber; : > $OUT/$tag.log ) 2>/dev/null || continue
  echo "$(date +%T) start $tag on $DEV"
  timeout 5400 .venv/bin/python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/eval_robust.py --task $task \
    --checkpoint $ckpt --num_envs $nenv --seed 7 --headless --device $DEV --out $OUT/$tag.json \
    agent.device=$DEV $extra >> $OUT/$tag.log 2>&1
  if [ -f $OUT/$tag.json ]; then
    echo "$(date +%T) done  $tag: $(grep -E '"knot"|"diverged"' $OUT/$tag.json | tr -d ' \n')"
  else
    echo "$(date +%T) FAIL  $tag (see $OUT/$tag.log)"
  fi
done < $JOBS
echo "$(date +%T) queue done"
