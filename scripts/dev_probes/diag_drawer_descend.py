"""drawer descend 卡死几何观测：打印 descend 前后方块/抽屉/TCP 的空间关系。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from darwin.benchmarks import get_benchmark
from darwin.skills.primitives import (DescendSkill, LiftSkill, PlaceSkill,
                                      MoveToSkill, MoveToXYTopSkill,
                                      MoveAboveSkill, PullDrawerSkill)

_orig_desc = DescendSkill.execute

def desc(self, env, *a, **kw):
    pt = np.asarray(kw.get("point"), float)
    body = kw.get("body")
    r = _orig_desc(self, env, *a, **kw)
    try:
        bp = np.asarray(env.get_body_pos(body), float)
    except Exception:
        bp = None
    if not r.get("success"):
        end = np.asarray(r.get("end") or env.get_site_pos("0_grip_site"), float)
        stop = float(kw.get("stop_above", 0.0))
        dz = end[2] - (bp[2] + stop) if bp is not None else float("nan")
        print(f"  [DESC-FAIL] point={np.round(pt, 3)} block_now={bp is not None and np.round(bp, 3)} "
              f"end={np.round(end, 3)} dz_to_goal={dz:+.4f} "
              f"xy_err={np.linalg.norm(end[:2] - pt[:2]):.4f} "
              f"clear={r.get('min_clearance')} pair={r.get('coll_pair')} steps={r.get('steps')}")
    else:
        print(f"  [desc-ok] steps={r.get('steps')}")
    return r

DescendSkill.execute = desc

# 其余技能：失败时打印完整 result + 当时 body 位置（定位 attempt 内失败环节）
def _patch_fail_print(cls):
    orig = cls.execute
    def wrapped(self, env, *a, **kw):
        r = orig(self, env, *a, **kw)
        if not r.get("success"):
            bp = None
            try:
                bp = np.round(np.asarray(env.get_body_pos(kw.get("body")), float), 3) if kw.get("body") else None
            except Exception:
                pass
            print(f"  [{cls.spec.name}-FAIL] params={ {k: (np.round(np.asarray(v, float), 3).tolist() if isinstance(v, (list, tuple)) and len(v) == 3 else v) for k, v in kw.items() if k in ('point', 'target', 'goal', 'body', 'height', 'joint_name', 'target_qpos')} } body_now={bp} result={r}")
        return r
    cls.execute = wrapped

for _cls in (LiftSkill, PlaceSkill, MoveToSkill, MoveToXYTopSkill,
             MoveAboveSkill, PullDrawerSkill):
    _patch_fail_print(_cls)

entry = dict(get_benchmark('drawer_place'))
from darwin.agents.runner_dynamic import DynamicEpisodeRunner
from darwin.agents.chain_registry import default_registry
from darwin.agents.experience_store import ExperienceStore
from darwin.memory import RAGMemory

exp = ExperienceStore(root=tempfile.mkdtemp())
rag = RAGMemory(root=tempfile.mkdtemp())
reg = default_registry()
runner = DynamicEpisodeRunner(entry, rag=rag, max_attempts=2, verbose=True,
                              log_dir=tempfile.mkdtemp(), sample_every=20,
                              registry=reg, experience=exp)
cfg = {'hover': 0.12, 'k_descend': 2.0, 'lift_height': 0.52, 'jit': 0.0}
res = runner.run(cfg)
