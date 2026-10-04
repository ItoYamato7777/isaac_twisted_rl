#!/usr/bin/env bash
# run_job.sh (P4-1) の試験。GPU も Isaac Sim も使わない短い python3 で試す。記録は一時フォルダに書く。
R=/home/itoyama/work/isaac_twisted_rl/src/isaac_twisted_rl/scripts/run_job.sh
export RUN_JOB_ROOT="$(mktemp -d)"
unset PYTHONUNBUFFERED  # settings.json の env が入っていても、ラッパー自身が設定することを確かめる
ok=0; ng=0
check() { if eval "$2"; then echo "OK  $1"; ok=$((ok+1)); else echo "NG  $1"; ng=$((ng+1)); fi; }
last() { ls -d "$RUN_JOB_ROOT"/*_"$1" | tail -1; }

$R --name normal --timeout 20s -- python3 -c "import sys; print('hello'); print('err', file=sys.stderr)" 2>/dev/null; rc=$?
d=$(last normal)
check "正常終了の終了コード 0" '[ $rc = 0 ]'
check "stdout と stderr が job.log に入る" 'grep -q hello $d/job.log && grep -q err $d/job.log'
check "記録のファイルが揃う" '[ -s $d/cmd.txt ] && [ -s $d/git.txt ] && [ -s $d/pid ] && grep -q "exit: 0" $d/exit.txt'
check "cmd.txt にコマンドと cwd" 'grep -q "^cmd: python3 -c" $d/cmd.txt && grep -q "^cwd: " $d/cmd.txt'
check "git.txt の 1 行目が HEAD" '[ "$(head -1 $d/git.txt)" = "$(git -C /home/itoyama/work/isaac_twisted_rl rev-parse HEAD)" ]'
check "未追跡のファイルが git.diff に入る (run_job.sh 自身が未追跡のとき)" 'git -C /home/itoyama/work/isaac_twisted_rl ls-files --error-unmatch src/isaac_twisted_rl/scripts/run_job.sh >/dev/null 2>&1 || grep -q "b/src/isaac_twisted_rl/scripts/run_job.sh" $d/git.diff'

$R --name code5 --timeout 20s -- python3 -c "raise SystemExit(5)" 2>/dev/null; rc=$?
check "終了コードをそのまま返す (5)" '[ $rc = 5 ] && grep -q "exit: 5" $(last code5)/exit.txt'

$R --name tmo --timeout 1s -- python3 -c "import time; time.sleep(30)" 2>/dev/null; rc=$?
check "timeout で 124" '[ $rc = 124 ] && grep -q "timeout:" $(last tmo)/exit.txt'

t0=$(date +%s)
$R --name hang --timeout 1s --kill-after 1s -- python3 -c "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)" 2>/dev/null; rc=$?
dt=$(( $(date +%s) - t0 ))
check "SIGTERM を無視するものは kill-after で 137 (${dt} s)" '[ $rc = 137 ] && [ $dt -le 5 ]'

$R --name sigterm --timeout 20s -- python3 -c "import signal, sys, time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); time.sleep(30)" 2>/dev/null & rp=$!
sleep 1; kill -TERM $rp; wait $rp; rc=$?
check "外からの SIGTERM: 子が 0 で終わっても exit.txt に signal が残る" '[ $rc = 0 ] && grep -q "exit: 0" $(last sigterm)/exit.txt && grep -q "signal: run_job が SIGTERM" $(last sigterm)/exit.txt'
check "正常終了の exit.txt には signal の行が無い" '! grep -q signal: $(last normal)/exit.txt'

$R --name nogpu --timeout 5s --min-free-mib 99999999 -- true 2>/dev/null; rc=$?
check "GPU の空きが足りなければ 3 で起動しない" '[ $rc = 3 ] && ! ls -d "$RUN_JOB_ROOT"/*_nogpu >/dev/null 2>&1'
$R --name dev1 --timeout 5s --min-free-mib 1 -- python3 -c "import os; print('CVD', os.environ.get('CUDA_VISIBLE_DEVICES'))" --device cuda:1 2>/dev/null
check "--device cuda:1 から GPU 1 を確かめ、CUDA_VISIBLE_DEVICES は変えない" 'grep -q "checked GPU: 1" $(last dev1)/cmd.txt && grep -q "CVD None" $(last dev1)/job.log'
msg=$($R --name nodev --timeout 5s --check-gpu 7 -- true --device cuda:0 2>&1); rc=$?
check "--check-gpu が --device より優先 (無い GPU 7 は読めずに 3)" '[ $rc = 3 ] && echo "$msg" | grep -q "GPU 7 の空きメモリを読めなかった"'

out=$($R --name det --timeout 20s --detach -- python3 -c "print('early'); import time; time.sleep(3); print('late')")
d=$(last det)
sleep 1.5
check "--detach はすぐ戻り、job と pid を出す" 'echo "$out" | grep -q "^job: " && echo "$out" | grep -qE "^pid: [0-9]+"'
check "バッファされない (sleep の途中で early が見える)" 'grep -q early $d/job.log && ! grep -q late $d/job.log'
sleep 3
( python3 -c "print('early'); import time; time.sleep(3)" > "$RUN_JOB_ROOT/ctl.log" 2>&1 & ); sleep 1.5
check "対照: ラッパーなしではバッファされて early がまだ見えない" '! grep -q early "$RUN_JOB_ROOT/ctl.log"'
sleep 1.5
check "--detach でも終わったら exit.txt" 'grep -q "exit: 0" $d/exit.txt && grep -q late $d/job.log'

$R --name bad 2>/dev/null; rc=$?
check "引数が足りなければ 2" '[ $rc = 2 ]'
echo "$ok passed, $ng failed"
rm -rf "$RUN_JOB_ROOT"
[ $ng = 0 ]
