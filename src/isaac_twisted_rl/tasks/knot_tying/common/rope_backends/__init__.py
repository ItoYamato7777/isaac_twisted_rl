"""紐のシミュレーションモデル (バックエンド) を差し替えるための層。紐モデルの比較 (`rope_study/`) で追加。

v0〜v7 の紐は PhysX の Articulation (D6 関節でつないだカプセル剛体の鎖、`rope_model/`) だけだった。
ここには、それ以外の表現の紐を **同じ MDP 項 (`mdp/`) のまま** 使えるようにする部品を置く:

    centerline.py   中心線 (節点列) の弧長での再標本化・枠の構成・めり込みの計測 (純 torch)
    xpbd_rod.py     Warp で書いた XPBD の棒 (Cosserat 棒 / 質点-ばね) のソルバ (isaaclab を import しない)
    adapters.py     各バックエンドを「Articulation と同じ読み書き口」に見せるアダプタ (isaaclab 依存)
    env.py          アダプタをシーンに差し込み、外部ソルバを物理ステップごとに進める環境クラス

既存の項は `env.scene["rope"]` の `data.body_pos_w` などを読むだけなので、アダプタがその口を
「仮想リンク」(弧長で等分した点。既定 28 本で `Knot` の 28 リンクと同じ位置) で返せば、
観測・報酬・終了・アーカイブ・位相の計算はどのバックエンドでも同じものが使える。

注意: この __init__ は `tasks/__init__.py` の `import_packages` によって **Kit 起動前** に
実行されるので、docstring だけにしておく (`../__init__.py` と同じ理由)。
"""
