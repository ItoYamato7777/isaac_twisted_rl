#!/bin/bash
# 紐モデルの学習を 1 本ずつ順に回す (4096 環境・seed 42・100 iteration、報告書 §6.3)。GPU の学習の枠 1 つぶん。
#
#   bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_train_chain.sh cuda:0 <待つ PID (0 なら待たない)> XpbdC28 XpbdMS28
#
# 先に走っている学習 (など) の PID が終わるのを待ってから始める。初期アーカイブ
# (outputs/archive_seed/rope_study/<モデル>.pt、run_physics_suite.sh のゲート 1 が作る) が無ければできるまで待つ。
# ログ: outputs/rope_study/logs/train_<モデル>.log、run: logs/rsl_rl/knot_tying_rs_<モデル小文字>/<日時>_rs_<モデル>_s42/
set -u
DEV=$1
WAIT_PID=$2
shift 2
ROOT=/home/itoyama/work/isaac_twisted_rl
cd $ROOT
LOG=$ROOT/outputs/rope_study/logs
SEED=$ROOT/outputs/archive_seed/rope_study
if [ "$WAIT_PID" != "0" ]; then
  echo "$(date +%T) waiting for PID $WAIT_PID"
  while ps -p $WAIT_PID > /dev/null; do sleep 60; done
fi
for M in "$@"; do
  until [ -f $SEED/$M.pt ]; do sleep 60; done
  echo "$(date +%T) [$M] train on $DEV"
  .venv/bin/train --task Isaac-Twisted-RL-RS-$M-v4 --headless --device $DEV --seed 42 --max_iterations 100 \
      --run_name rs_${M}_s42 agent.device=$DEV agent.save_interval=10 > $LOG/train_$M.log 2>&1
  echo "$(date +%T) [$M] exit $? ($(grep -c 'Learning iteration' $LOG/train_$M.log) iterations logged)"
done
echo "$(date +%T) chain done"
