#!/usr/bin/env bash
# README の Sim2Real Transfer 用に、実機 (左) とシミュ (右) の横並び動画を作る。
#   使い方: video/make_sim2real.sh 01   (real/real_01.mp4 + real/sim_01.mp4 -> highlights/sim2real_01.{mp4,webp})
#
# 時間の合わせ方: 実機は約 21 s、シミュは約 4 s と長さが違う。どちらかを極端に変えるとどちらかが
# 不自然になるので、実機を REAL_SPEED 倍速、シミュを 1/SIM_SLOW 倍速にして約 8 s で揃える。
# 倍率は動画の左上・右上のラベルに出す (実機は早送りであると分かるように)。
# シミュは元動画の上端の黒帯 (動画名の表示) を切り落とし、5 env 全体を使う。
# 入力 (video/real/*.mp4, 計 80 MB) は .gitignore で追跡されない。出力は highlights/ に置く。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ID="${1:?usage: make_sim2real.sh <01|02>}"

REAL="$HERE/real/real_${ID}.mp4"
SIM="$HERE/real/sim_${ID}.mp4"
OUT="$HERE/highlights/sim2real_${ID}"

# クリップごとの設定 (REAL_SS: 実機の切り出し開始 [s]、REAL_SPEED: 実機の倍速)
case "$ID" in
  01) REAL_SS=0; REAL_SPEED=2.5 ;;
  02) REAL_SS=0; REAL_SPEED=2.5 ;;
  *) echo "unknown id: $ID" >&2; exit 1 ;;
esac
SIM_SLOW=2          # シミュを 1/2 倍速にする (0.5x)
FPS=15              # WebP を軽くするため。0.5x のシミュは元の 30 fps を 1 枚ずつ使うので 15 fps でちょうどよい
H=480               # 各動画の高さ (シミュの綱は小さいので、400 より大きくして見えやすくする)
WEBP_Q="${WEBP_Q:-75}"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

LABEL="fontfile=$FONT:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6:x=10:y=10"

# 長い方に合わせて短い方の最終フレームを延ばす (tpad)。stop=-1 は無限に延ばし、最後の -t で切る。
FILTER="
[0:v]trim=start=${REAL_SS},setpts=(PTS-STARTPTS)/${REAL_SPEED},fps=${FPS},scale=-2:${H},
     drawtext=text='Real (${REAL_SPEED}x)':${LABEL},tpad=stop_mode=clone:stop=-1[r];
[1:v]crop=1280:922:0:38,setpts=${SIM_SLOW}*PTS,fps=${FPS},scale=-2:${H},
     drawtext=text='Sim (0.5x)':${LABEL},tpad=stop_mode=clone:stop=-1[s];
[r][s]hstack=inputs=2:shortest=0[v]"

# 出力の長さは 2 本の長い方 (shortest=0 は tpad で無限に延ばしているので -t で止める)
DUR=$(python3 - <<EOF
import subprocess
def d(p):
    return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p]))
print(round(max((d("$REAL")-$REAL_SS)/$REAL_SPEED, d("$SIM")*$SIM_SLOW), 2))
EOF
)

mkdir -p "$HERE/highlights"
ffmpeg -v error -y -i "$REAL" -i "$SIM" -filter_complex "$FILTER" -map "[v]" -t "$DUR" -an \
  -c:v libx264 -crf 28 -pix_fmt yuv420p -movflags +faststart "$OUT.mp4"
ffmpeg -v error -y -i "$REAL" -i "$SIM" -filter_complex "$FILTER" -map "[v]" -t "$DUR" -an \
  -c:v libwebp_anim -loop 0 -q:v "$WEBP_Q" -compression_level 6 "$OUT.webp"
ls -l "$OUT.mp4" "$OUT.webp"
