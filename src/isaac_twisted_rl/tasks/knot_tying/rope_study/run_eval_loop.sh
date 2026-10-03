#!/bin/bash
# 学習中にチェックポイントが増えるたびに評価を回す (make_eval_jobs.py -> run_eval_queue.sh を繰り返す)。
#
#   bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_eval_loop.sh cuda:0 600 [make_eval_jobs.py への引数...]
#
# 2 つの GPU で同時に回してよい (run_eval_queue.sh が .log を排他的に作ってジョブを取り合わない)。
# 新しいジョブが無ければ <間隔> 秒待って一覧を作り直す。止めるときは outputs/rope_study/eval/STOP を作る
# (回っているジョブは最後まで走る)。
set -u
DEV=${1:-cuda:0}
WAIT=${2:-600}
shift 2
ROOT=/home/itoyama/work/isaac_twisted_rl
cd $ROOT
EV=$ROOT/outputs/rope_study/eval
HERE=src/isaac_twisted_rl/tasks/knot_tying/rope_study
mkdir -p $EV
JOBS=$EV/jobs_auto_${DEV/:/}.txt
while [ ! -f $EV/STOP ]; do
  # 読んでいる途中の一覧を書き換えないよう、別名で作ってから置き換える
  .venv/bin/python $HERE/make_eval_jobs.py --out $JOBS.tmp "$@" > /dev/null && mv $JOBS.tmp $JOBS
  before=$(find $EV -name "*.log" | wc -l)
  bash $HERE/run_eval_queue.sh $JOBS $DEV | grep --line-buffered -v "queue done"
  after=$(find $EV -name "*.log" | wc -l)
  [ "$after" -eq "$before" ] && sleep $WAIT
done
echo "$(date +%T) eval loop stopped ($EV/STOP)"
