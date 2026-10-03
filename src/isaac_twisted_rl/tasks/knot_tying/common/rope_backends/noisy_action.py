"""実行誤差つきの pick&place (紐モデルの比較の頑健性評価で追加)。

実機の手は、方策が狙った点を正確には掴めず、狙った点に正確には置けない (腕の位置決め誤差・カメラの
校正誤差・紐の位置の推定誤差)。`PickPlaceAction` が今の紐の形から決めた「掴む点」と「置く点」に、
手ごとに独立な等方ガウス誤差 (xy) を足す。既定 (`exec_noise_xy = 0`) では `PickPlaceAction` と同じ。
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from ..mdp.actions import PickPlaceAction, PickPlaceActionCfg


class NoisyPickPlaceAction(PickPlaceAction):
    """`PickPlaceAction` に、掴む点・置く点の xy の誤差と、手首の角度の誤差を足したもの。"""

    cfg: NoisyPickPlaceActionCfg

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        sxy, syaw = self.cfg.exec_noise_xy, self.cfg.exec_noise_yaw
        if sxy <= 0.0 and syaw <= 0.0:
            return
        n = self.num_envs
        valid = ~self._invalid
        if sxy > 0.0:
            dg = torch.randn(n, 2, device=self.device) * sxy
            dp = torch.randn(n, 2, device=self.device) * sxy
            dg = torch.where(valid.unsqueeze(-1), dg, torch.zeros_like(dg))
            dp = torch.where(valid.unsqueeze(-1), dp, torch.zeros_like(dp))
            self._grasp_xy = self._grasp_xy + dg
            self._place_pos[:, :2] += dp
            self._clear_pos[:, :2] += dp
        if syaw > 0.0:
            dy = torch.randn(n, device=self.device) * syaw
            self._grasp_yaw = self._grasp_yaw + torch.where(valid, dy, torch.zeros_like(dy))
        # approach の区間は super() の中で誤差の無い点で作られているので、作り直す
        self._begin_phase(torch.arange(n, device=self.device))


@configclass
class NoisyPickPlaceActionCfg(PickPlaceActionCfg):
    class_type: type = NoisyPickPlaceAction

    exec_noise_xy: float = 0.0
    """掴む点・置く点の xy の誤差の標準偏差 [m] (手ごとに独立)。"""

    exec_noise_yaw: float = 0.0
    """掴むときの手首の角度の誤差の標準偏差 [rad]。"""
