"""紐のバックエンドを差し替えられる環境クラス。紐モデルの比較 (`rope_study/`) で追加。

`ManagerBasedRLEnv` との違いは、アダプタ (`adapters.py`) の差し込みだけ:

1. **シーンを作った直後** (物理の開始の前) に `cfg.rope_backend` に従ってアダプタを作り、`scene["rope"]` で
   引けるようにする (`InteractiveScene.__getitem__` は `_extras` も探すので、そこへ入れる)。物理の開始時に
   初期化される項 (v4 のアーカイブ `ResetToStage`) が `scene["rope"].num_joints` を読むので、`load_managers` で
   入れるのでは間に合わない。そのため `ManagerBasedEnv.__init__` が使う `InteractiveScene` を `__init__` の
   間だけ包む。
2. `load_managers` の直前に `adapter.bind(env)` (グリッパーなど、物理が動いてから引けるものを掴む)。
3. `scene.update(dt)` の後にアダプタの `update(dt)` を呼ぶ。外部ソルバ (XPBD) はここで 1 物理ステップ
   進む。PhysX がグリッパーを動かした直後なので、ソルバはその手の姿勢と指の位置を読める。
4. `scene.reset(env_ids)` の後にアダプタの `reset(env_ids)` を呼ぶ (外部ソルバの紐を初期姿勢へ戻す)。
   `reset_scene_to_default` (`reset_all`) は PhysX の資産しか戻さないため。

`step()` は書き換えない。行動項 (`PickPlaceAction`) も台本 (`demos/overhand_script.py`) も、
グリッパーを書き込んで `sim.step()` / `scene.update()` を回すだけなので、そのまま動く。
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .adapters import FemRope, ResampledArticulationRope, XpbdRope


class RopeStudyEnv(ManagerBasedRLEnv):
    """`cfg.rope_backend` (`backend_cfg.RopeBackendCfg`) で紐の表現を選べる `ManagerBasedRLEnv`。"""

    rope_adapter = None

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # シーンを作った直後 (物理の開始の前) にアダプタを入れるため、ManagerBasedEnv が使う
        # InteractiveScene をこの __init__ の間だけ包む。物理の開始時に初期化される項
        # (v4 のアーカイブ `ResetToStage`) が `scene["rope"]` を引けるようにするため。
        from isaaclab.envs import manager_based_env as mbe

        original = mbe.InteractiveScene

        def make_scene(scene_cfg):
            scene = original(scene_cfg)
            self._install_rope_backend(scene, cfg)
            return scene

        mbe.InteractiveScene = make_scene
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            mbe.InteractiveScene = original

    def load_managers(self):
        if self.rope_adapter is not None:
            self.rope_adapter.bind(self)
        super().load_managers()

    def _install_rope_backend(self, scene, cfg):
        bcfg = getattr(cfg, "rope_backend", None)
        if bcfg is None or bcfg.kind == "articulation":
            return
        origins = scene.env_origins
        n = scene.num_envs
        if bcfg.kind == "articulation_resampled":
            adapter = ResampledArticulationRope(scene[bcfg.native_name], n, cfg.sim.device, bcfg.num_virtual_links, bcfg.radius, origins)
        elif bcfg.kind == "xpbd":
            from .xpbd_rod import XpbdRodParams

            params = XpbdRodParams(**dict(bcfg.xpbd_params))
            adapter = XpbdRope(n, cfg.sim.device, params, bcfg.num_virtual_links, bcfg.init_height, origins)
        elif bcfg.kind == "fem":
            adapter = FemRope(scene[bcfg.native_name], n, cfg.sim.device, bcfg.num_virtual_links, bcfg.radius, origins)
        else:
            raise ValueError(f"unknown rope backend kind {bcfg.kind!r}")
        if "rope" in scene.keys():
            raise RuntimeError("scene に 'rope' が既にある。アダプタを使う紐は native_name で別名にすること")
        scene._extras["rope"] = adapter
        self.rope_adapter = adapter

        scene_update = scene.update
        scene_reset = scene.reset

        def update(dt: float) -> None:
            scene_update(dt)
            adapter.update(dt)

        def reset(env_ids=None) -> None:
            scene_reset(env_ids)
            adapter.reset(env_ids)

        scene.update = update
        scene.reset = reset
        print(f"[rope_study] rope backend: {bcfg.kind} ({type(adapter).__name__})")
