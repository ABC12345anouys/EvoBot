"""诊断：spatial:1 抓取候选生成决策链复现。

直接调 RuleGraspPoseLibero.propose，并逐项打印：
  bowl bounds / is_container / GraspNet 候选数 / 两方向 off 与 clr / 最终候选。

用法: MUJOCO_GL=egl python scripts/probe_spatial1_cand.py
"""
import os
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

sys.path.insert(0, ".")
sys.path.insert(0, "/home/lifd/Public/LIBERO")

import numpy as np

from darwin.ipc.sim_worker import parse_task, task_id_of, config_env_of
from darwin.agents.runner_dynamic import DynamicEpisodeRunner
from darwin.agents.chain_registry import default_registry
from darwin.agents.experience_store import ExperienceStore
from darwin.skills.skill_config import SkillConfigStore
from darwin.policies.rules import RuleGraspPoseLibero
from darwin.policies.context import StepContext

entry = parse_task("libero_spatial:1")
runner = DynamicEpisodeRunner(
    entry, rag=None, max_attempts=1, verbose=False,
    log_dir=tempfile.mkdtemp(prefix="probe_cand_"),
    registry=default_registry(), experience=ExperienceStore(),
    blacklist_path=os.path.join(tempfile.gettempdir(), "probe_bl.json"),
    blacklist_write=False)
store = SkillConfigStore.load("ik_servo", config_env_of(entry),
                              task=task_id_of(entry))
handle = runner.prepare_episode(record_dir=None,
                                initial_cfg=dict(store.params))
env = handle.env

body = entry["body"]
b = env.object_bounds(body)
print(f"[cand] bowl bounds: center={np.round(b['center'],4)} "
      f"half=({b['half_x']:.4f},{b['half_y']:.4f}) "
      f"z=[{b['z_bottom']:.4f},{b['z_top']:.4f}]")
print(f"[cand] is_container={env._is_container(body)}")

from darwin.skills.perception.grasp import grasp_candidates_from_env
res = grasp_candidates_from_env(env, body, top_k=8)
all_cands = res.get("candidates") or []
print(f"[cand] graspnet success={res.get('success')} "
      f"n_candidates={len(all_cands)} source={res.get('source')}")
if res.get("success") is False:
    print(f"[cand] graspnet error={res.get('error')}")
for c in all_cands[:8]:
    print(f"[cand]   score={c.get('score'):.3f} width={c.get('width'):.3f} "
          f"pos={np.round(c.get('position'),4)}")

cfg = dict(store.params)
ctx = StepContext(env=env, entry=entry, cfg=cfg, step=0)
try:
    out = RuleGraspPoseLibero().propose(ctx)
    print(f"[cand] rule 产出 {len(out.get('candidates') or [])} 候选:")
    for c in out.get("candidates") or []:
        print(f"[cand]   {c}")
except Exception as e:
    import traceback
    traceback.print_exc()
