#!/bin/bash
# 紐モデルの物理の比較 (台本のゲート 1 + 物理ベンチマーク) をモデルごとに順に回す。
#
#   bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_physics_suite.sh cuda:0 "Art28 XpbdC28 ..."
#
# モデルごとに:
#   1. ゲート 1 の台本 (遅い尺 7.8 s/手、128 環境、±25%): 固結びの成功率。段 0〜2 の形を初期アーカイブ
#      (outputs/archive_seed/rope_study/<モデル>.pt) に、最後の固結びを <モデル>_knots.pt に保存する
#   2. 物理ベンチマーク (bench_physics.py, 128 環境)
#   3. ゲート 1 の台本を学習と同じ尺 (3.16 s/手) で
# 出力: outputs/rope_study/{gate1,gate1_fast,physics}/<モデル>/
set -u
DEV=${1:-cuda:0}
MODELS=${2:-"Art28 Art28LowMu Art56 XpbdC28 XpbdC56 XpbdC28LowMu XpbdMS28"}
NENV=${3:-128}
ROOT=/home/itoyama/work/isaac_twisted_rl
cd $ROOT
OUT=$ROOT/outputs/rope_study
SEED=$ROOT/outputs/archive_seed/rope_study
mkdir -p $OUT/logs $SEED
PY=".venv/bin/python -u"
for M in $MODELS; do
  T=Isaac-Twisted-RL-RS-$M-v4
  echo "$(date +%T) [$M] gate1 (slow script) on $DEV"
  [ -f $OUT/gate1/$M/summary.log ] && grep -q "\[final\]" $OUT/gate1/$M/summary.log || \
  timeout 7200 $PY src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task $T --num_envs $NENV --jitter 0.25 \
      --headless --device $DEV --out $OUT/gate1/$M --dump_archive $SEED/$M.pt --dump_knots $SEED/${M}_knots.pt \
      > $OUT/logs/gate1_$M.log 2>&1
  grep -E "\[final\]|\[dump" $OUT/gate1/$M/summary.log
  echo "$(date +%T) [$M] physics bench"
  [ -f $OUT/physics/$M/shapes.pt ] || \
  timeout 7200 $PY src/isaac_twisted_rl/tasks/knot_tying/rope_study/bench_physics.py --model $M --num_envs $NENV \
      --headless --device $DEV --out $OUT/physics/$M > $OUT/logs/physics_$M.log 2>&1
  grep -E "\[bench\] (wrote|.* done)" $OUT/logs/physics_$M.log | cut -c1-200
  echo "$(date +%T) [$M] gate1 (fast = training durations)"
  [ -f $OUT/gate1_fast/$M/summary.log ] && grep -q "\[final\]" $OUT/gate1_fast/$M/summary.log || \
  timeout 7200 $PY src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task $T --num_envs $NENV --jitter 0.25 \
      --headless --device $DEV --out $OUT/gate1_fast/$M --fast > $OUT/logs/gate1_fast_$M.log 2>&1
  grep -E "\[final\]" $OUT/gate1_fast/$M/summary.log
done
echo "$(date +%T) suite done"
