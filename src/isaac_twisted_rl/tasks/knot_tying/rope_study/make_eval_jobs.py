"""紐モデルの比較の評価のジョブ一覧を作る (Isaac Sim 不要)。`run_eval_queue.sh` が 1 行ずつ回す。

    .venv/bin/python src/isaac_twisted_rl/tasks/knot_tying/rope_study/make_eval_jobs.py --iters 50,100 \
        --policies Art28,Art28LowMu,XpbdC28 --out outputs/rope_study/eval/jobs.txt

1 行 = `<tag> <Play タスク> <チェックポイント> <環境数> [eval_robust.py への追加の引数...]`。出力は
`outputs/rope_study/eval/<方策>/<tag>.json` (`run_eval_queue.sh` が決める)。

ジョブの種類 (報告書 §6.4):

* `self_it<N>`: 方策を学習したモデルで評価 (512 環境)
* `on_<モデル>_it<N>`: 別のモデルで評価 = モデル間の転移 (256 環境。iteration は `--transfer_iter`、既定は最後)
* `obs<mm>_it<N>` / `exec<mm>_it<N>`: 自分のモデルで、観測ノイズ・実行誤差 (`--noise_mm`、既定 10 mm。256 環境)。
  iteration は `--perturb_iters` (既定は --iters の最大)
* `on_<摂動先>_it<N>`: どの方策の学習モデルでもない物理の摂動 (`PERTURB_TARGETS`。XPBD の曲げ剛性 x0.5 / x2)。
  転移と同じく 256 環境・`--transfer_iter`。全ての方策で回す (XpbdC28 の方策にとっては自分のモデルの摂動)

方策の run は `logs/rsl_rl/knot_tying_rs_<モデル小文字>/` の最新の `*_rs_<モデル>_s42` を使う (`INVALID.txt` がある
run は飛ばす。XPBD を world 座標で解いていた版の学習。報告書 §7.1.6)。`--extra_policy name=path`
で別の方策 (例: v4 の `Knot` の run) を足せる。その方策は `--extra_home` のモデル (既定 Art28) を「自分のモデル」とみなす。
"""

from __future__ import annotations

import argparse
import glob
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))
MODELS = ("Art28", "Art28LowMu", "XpbdC28", "XpbdC28LowMu", "XpbdC56", "XpbdMS28")
"""転移の評価先。`Art56` (反復を上げても発散 33% 以上) と `Fem` (起動できない) は入れない。"""

_EI = 7.854e-4
PERTURB_TARGETS = {
    "XpbdC28EIx0.5": ("XpbdC28", f"--xpbd bend_stiffness={0.5 * _EI:.4e} --xpbd twist_stiffness={0.25 * _EI:.4e}"),
    "XpbdC28EIx2": ("XpbdC28", f"--xpbd bend_stiffness={2.0 * _EI:.4e} --xpbd twist_stiffness={1.0 * _EI:.4e}"),
}
"""物理の摂動先: 名前 -> (Play のモデル, eval_robust.py への追加の引数)。実機の紐の曲げ剛性は較正しないと 2 倍
程度は外れる (報告書 §7.1.1 の離散化の誤差だけで 0.4〜0.6 倍) ので、x0.5 / x2 を見る。GJ = EI / 2 は保つ。"""


def find_run(model: str, seed: int) -> str | None:
    runs = sorted(glob.glob(os.path.join(REPO, "logs", "rsl_rl", f"knot_tying_rs_{model.lower()}", f"*_rs_{model}_s{seed}")))
    runs = [r for r in runs if not os.path.exists(os.path.join(r, "INVALID.txt"))]
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default=",".join(MODELS))
    ap.add_argument("--iters", default="50,100", help="自分のモデルでの評価と摂動に使う iteration。")
    ap.add_argument("--transfer_iter", type=int, default=0, help="転移に使う iteration (0 なら --iters の最大)。")
    ap.add_argument("--targets", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra_policy", action="append", default=[], help="name=チェックポイントの run のフォルダ")
    ap.add_argument("--extra_home", default="Art28")
    ap.add_argument("--no_perturb", action="store_true")
    ap.add_argument("--perturb_iters", default="", help="観測ノイズ・実行誤差を回す iteration (空なら --iters の最大)。")
    ap.add_argument("--noise_mm", default="10", help="観測ノイズ・実行誤差の大きさ [mm] (カンマ区切り)。")
    ap.add_argument("--perturb_targets", default=",".join(PERTURB_TARGETS), help="物理の摂動先 (空なら回さない)。")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "rope_study", "eval", "jobs.txt"))
    a = ap.parse_args()
    iters = [int(v) for v in a.iters.split(",") if v]
    t_iter = a.transfer_iter or max(iters)
    p_iters = [int(v) for v in a.perturb_iters.split(",") if v] or [max(iters)]
    noise_mm = [int(v) for v in a.noise_mm.split(",") if v]
    runs = {}
    for name in filter(None, a.policies.split(",")):
        run = find_run(name, a.seed)
        if run:
            runs[name] = (run, name)
    for item in a.extra_policy:
        name, path = item.split("=", 1)
        runs[name] = (path, a.extra_home)
    lines = []
    for name, (run, home) in runs.items():
        for it in iters:
            ckpt = os.path.join(run, f"model_{it}.pt")
            if not os.path.isfile(ckpt):
                continue
            lines.append(f"{name}/self_it{it} Isaac-Twisted-RL-RS-{home}-Play-v4 {ckpt} 512")
            if a.no_perturb or it not in p_iters:
                continue
            for mm in noise_mm:
                lines.append(f"{name}/obs{mm}mm_it{it} Isaac-Twisted-RL-RS-{home}-Play-v4 {ckpt} 256 --obs_noise {mm / 1000}")
                lines.append(
                    f"{name}/exec{mm}mm_it{it} Isaac-Twisted-RL-RS-{home}-Play-v4 {ckpt} 256 "
                    f"env.actions.pick_place.exec_noise_xy={mm / 1000}"
                )
        ckpt = os.path.join(run, f"model_{t_iter}.pt")
        if os.path.isfile(ckpt):
            for tgt in filter(None, a.targets.split(",")):
                if tgt != home:
                    lines.append(f"{name}/on_{tgt}_it{t_iter} Isaac-Twisted-RL-RS-{tgt}-Play-v4 {ckpt} 256")
            for tgt in filter(None, a.perturb_targets.split(",")):
                base, extra = PERTURB_TARGETS[tgt]
                lines.append(f"{name}/on_{tgt}_it{t_iter} Isaac-Twisted-RL-RS-{base}-Play-v4 {ckpt} 256 {extra}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {a.out}")
    for line in lines:
        print("  " + line.split()[0])


if __name__ == "__main__":
    main()
