#!/usr/bin/env bash
# 長時間のジョブ (学習・評価・物理の計測) を起動する共通の入口。
#
#   src/isaac_twisted_rl/scripts/run_job.sh --name v6_s42 --timeout 10h [--min-free-mib 4096] [--detach] \
#       -- uv run train --task Isaac-Twisted-RL-Knot-v6 --headless --device cuda:0 --seed 42
#
# まとめて行うこと (どれも過去に踏んだ失敗への対策):
# - PYTHONUNBUFFERED=1 で起動する。Python のログがバッファに溜まり、途中経過が見えなくなるため
#   (console script の `train` には `python -u` を付けられない)。
# - `timeout -k` を付ける。Kit・FEM は固まると SIGTERM で止まらないことがあり、-k が無いと残り続ける。
# - 出力は job.log に書く。記録 (コマンド・cwd・commit・未追跡を含む差分・GPU の状態・PID・終了コード) を
#   outputs/jobs/<時刻>_<name>/ に残す。未追跡のファイルは run の git/*.diff に残らず、再現できなくなるため。
# - 起動の前に GPU の空きメモリを確かめる。GPU は他のジョブと共有している。確かめる GPU はコマンドの
#   `--device cuda:N` から読む (無ければ 0、`--check-gpu N` で上書き)。CUDA_VISIBLE_DEVICES は変えない
#   (変えると cuda:N の番号の対応が変わり、学習と評価で同じ --device を使う約束が崩れる)。
# - 止めるときは job の pid のファイルにある PID (timeout のプロセス) に SIGTERM を送る。timeout が子に伝え、
#   kill-after の後に SIGKILL する。自分が起動していないジョブの PID は、止めずにユーザーに伝える。
set -euo pipefail

usage() { sed -n '2,4p' "$0" >&2; exit 2; }

name="" duration="" kill_after="60s" gpu="" min_free=4096 detach=0 jobdir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --timeout) duration="$2"; shift 2 ;;
    --kill-after) kill_after="$2"; shift 2 ;;
    --check-gpu) gpu="$2"; shift 2 ;;
    --min-free-mib) min_free="$2"; shift 2 ;;
    --detach) detach=1; shift ;;
    --_jobdir) jobdir="$2"; shift 2 ;;  # --detach が自分を呼び直すときだけ使う
    --) shift; break ;;
    *) usage ;;
  esac
done
[ -n "$name" ] && [ -n "$duration" ] && [ $# -gt 0 ] || usage
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run_job: --name は英数字と . _ - だけ" >&2; exit 2; }
if [ -z "$gpu" ]; then
  gpu=0
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--device" ] && [[ "$a" =~ ^cuda:([0-9]+)$ ]]; then gpu="${BASH_REMATCH[1]}"; fi
    if [[ "$a" =~ ^--device=cuda:([0-9]+)$ ]]; then gpu="${BASH_REMATCH[1]}"; fi
    prev="$a"
  done
fi

repo="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"

if [ -z "$jobdir" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [ "$min_free" -gt 0 ]; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || true)
    if ! [[ "$free" =~ ^[0-9]+$ ]]; then
      echo "run_job: GPU $gpu の空きメモリを読めなかった" >&2; exit 3
    elif [ "$free" -lt "$min_free" ]; then
      echo "run_job: GPU $gpu の空きは ${free} MiB (< ${min_free} MiB)。他のジョブを確かめる (nvidia-smi)。" >&2; exit 3
    fi
  fi

  jobdir="${RUN_JOB_ROOT:-$repo/outputs/jobs}/$(date +%Y-%m-%d_%H-%M-%S)_$name"  # RUN_JOB_ROOT は試験用
  mkdir -p "$jobdir"
  {
    printf 'cwd: %s\n' "$PWD"
    printf 'cmd:'; printf ' %q' "$@"; printf '\n'
    printf 'timeout: %s (kill-after %s)\n' "$duration" "$kill_after"
    printf 'checked GPU: %s (min free %s MiB)\n' "$gpu" "$min_free"
    printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES-(未設定)}"
    printf 'start: %s\n' "$(date --iso-8601=seconds)"
    printf 'host: %s\n' "$(hostname)"
  } > "$jobdir/cmd.txt"
  {
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --porcelain --untracked-files=all
  } > "$jobdir/git.txt" 2>&1 || true
  {
    git -C "$repo" diff HEAD
    git -C "$repo" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
      if [ "$(stat -c %s "$repo/$f")" -le 1048576 ]; then
        (cd "$repo" && git diff --no-index -- /dev/null "$f") || true
      else
        printf '# 1 MiB を超える未追跡のファイルは省いた: %s\n' "$f"
      fi
    done
  } > "$jobdir/git.diff" 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv > "$jobdir/gpu_start.csv" 2>&1 || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv >> "$jobdir/gpu_start.csv" 2>&1 || true
  fi

  if [ "$detach" = 1 ]; then
    setsid nohup "$0" --name "$name" --timeout "$duration" --kill-after "$kill_after" \
      --_jobdir "$jobdir" -- "$@" > /dev/null 2>&1 < /dev/null &
    for _ in $(seq 50); do [ -s "$jobdir/pid" ] && break; sleep 0.1; done
    echo "job: $jobdir"
    echo "pid: $(cat "$jobdir/pid" 2>/dev/null || echo '(未確認)')  log: $jobdir/job.log"
    exit 0
  fi
fi

export PYTHONUNBUFFERED=1
echo "job: $jobdir" >&2
timeout -k "$kill_after" "$duration" "$@" > "$jobdir/job.log" 2>&1 &
child=$!
echo "$child" > "$jobdir/pid"
echo "pid: $child  log: $jobdir/job.log" >&2
trap 'kill -TERM "$child" 2>/dev/null || true' INT TERM
set +e
wait "$child"; rc=$?
# trap で wait が途中で返ったときは、子の終わりをもう一度待つ
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
set -e
printf 'exit: %s\nend: %s\n' "$rc" "$(date --iso-8601=seconds)" > "$jobdir/exit.txt"
[ "$rc" = 124 ] && echo "timeout: ${duration} で止めた" >> "$jobdir/exit.txt"
[ "$rc" = 137 ] && echo "SIGKILL で止まった (timeout の kill-after か、外からの kill -9)" >> "$jobdir/exit.txt"
exit "$rc"
