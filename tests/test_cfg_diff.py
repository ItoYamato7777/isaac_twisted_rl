"""`common/cfg_diff.py` (学習の env.yaml と評価の環境の設定の照合) の試験。Isaac Sim は要らない。"""

from cfg_diff import diff_flat, flatten, load_cfg_yaml, unexpected_diff

_TRAIN_YAML = """\
seed: 43
log_dir: /logs/run
viewer:
  eye: !!python/tuple
  - 1.2
  - 0.9
scene:
  num_envs: 4096
  rope:
    stiffness: 2.0
sim:
  device: cuda:1
  dt: 0.004
events:
  reset_to_stage:
    func: mod:reset
    params:
      straight_prob: 0.1
episode_length_s: 100.0
"""


def _cfgs():
    train = load_cfg_yaml(_TRAIN_YAML)
    train_default = load_cfg_yaml(_TRAIN_YAML.replace("seed: 43", "seed: 42"))
    play_default = load_cfg_yaml(
        _TRAIN_YAML.replace("straight_prob: 0.1", "straight_prob: 1.0").replace(
            "episode_length_s: 100.0", "episode_length_s: 57.0"
        )
    )
    return train, train_default, play_default


def test_load_cfg_yaml_reads_python_tuple_as_list():
    cfg = load_cfg_yaml(_TRAIN_YAML)
    assert cfg["viewer"]["eye"] == [1.2, 0.9]


def test_flatten_and_diff():
    a = {"x": {"y": 1, "z": [1, 2]}, "w": None}
    b = {"x": {"y": 2, "z": [1, 2], "n": 3}, "w": {"k": 0}}
    assert flatten(a) == {"x.y": 1, "x.z": [1, 2], "w": None}
    changed, only_a, only_b = diff_flat(flatten(a), flatten(b))
    assert changed == {"x.y": (1, 2)}
    assert only_a == ["w"]
    assert only_b == ["w.k", "x.n"]


def test_play_changes_and_ignored_keys_are_allowed():
    train, train_default, play_default = _cfgs()
    ev = load_cfg_yaml(
        _TRAIN_YAML.replace("straight_prob: 0.1", "straight_prob: 1.0")
        .replace("episode_length_s: 100.0", "episode_length_s: 57.0")
        .replace("num_envs: 4096", "num_envs: 512")
        .replace("seed: 43", "seed: 7")
        .replace("device: cuda:1", "device: cuda:0")
        .replace("/logs/run", "/logs/run2")
    )
    out = unexpected_diff(train, ev, train_default, play_default)
    assert out == {"changed": {}, "only_in_train": [], "only_in_eval": []}


def test_forgotten_override_is_reported():
    """学習で物理を上書きした (stiffness 3.0) のに、評価で付け忘れた。"""
    train, train_default, play_default = _cfgs()
    train["scene"]["rope"]["stiffness"] = 3.0
    ev = load_cfg_yaml(_TRAIN_YAML.replace("straight_prob: 0.1", "straight_prob: 1.0"))
    out = unexpected_diff(train, ev, train_default, play_default)
    assert out["changed"] == {"scene.rope.stiffness": (3.0, 2.0)}


def test_added_and_removed_keys_are_reported():
    train, train_default, play_default = _cfgs()
    ev = load_cfg_yaml(_TRAIN_YAML)
    ev["events"]["new_term"] = {"func": "mod:f"}
    del ev["sim"]["dt"]
    out = unexpected_diff(train, ev, train_default, play_default)
    assert out["only_in_eval"] == ["events.new_term.func"]
    assert out["only_in_train"] == ["sim.dt"]


def test_play_disabling_a_term_is_allowed():
    """Play が項を None にする (辞書 -> None) 変更も、Play の意図した変更として通す。"""
    train, train_default, play_default = _cfgs()
    play_default["events"]["reset_to_stage"] = None
    ev = load_cfg_yaml(_TRAIN_YAML)
    ev["events"]["reset_to_stage"] = None
    out = unexpected_diff(train, ev, train_default, play_default)
    assert out == {"changed": {}, "only_in_train": [], "only_in_eval": []}


def test_custom_ignore_prefixes():
    train, train_default, play_default = _cfgs()
    ev = load_cfg_yaml(_TRAIN_YAML.replace("dt: 0.004", "dt: 0.002"))
    assert unexpected_diff(train, ev, train_default, play_default)["changed"] == {"sim.dt": (0.004, 0.002)}
    out = unexpected_diff(train, ev, train_default, play_default, ignore=("sim",))
    assert out["changed"] == {}
