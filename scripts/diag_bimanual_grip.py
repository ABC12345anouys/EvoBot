"""复现 runner 的 cart 抓取链（move_above/descend/close/lift），bisect 失败参数。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from darwin.benchmarks import get_benchmark
from darwin.agents.env_utils import get_env, apply_bounds
from darwin.skills.primitives import (MoveAboveSkill, DescendSkill,
                                      CloseGripperSkill, LiftSkill)

entry = dict(get_benchmark("pickplace"))
env = get_env(entry["env_id"], entry["robot"])
apply_bounds(env, entry.get("bounds"))
actor, site, body = entry["actor"], entry["grip_site"], entry["body"]
cmn = {"actor": actor, "grip_site": site}

def probe(label):
    bp = env.get_body_pos(body)
    tcp = env.get_site_pos(site)
    print(f"[{label}] tcp={np.round(tcp,3)} block_z={bp[2]:.4f} dz={tcp[2]-bp[2]:+.4f}")

for trial, stop_above in enumerate([0.0, -0.01, -0.02]):
    env.reset()
    apply_bounds(env, entry.get("bounds"))
    bp0 = np.array(env.get_body_pos(body), float)
    r1 = MoveAboveSkill().execute(env, point=bp0.tolist(), hover=0.12, **cmn)
    r2 = DescendSkill().execute(env, point=bp0.tolist(), body=body, k=2.0,
                                stop_above=stop_above, **cmn)
    r3 = CloseGripperSkill().execute(env, **cmn)
    r4 = LiftSkill().execute(env, height=0.52, body=body, **cmn)
    print(f"trial{trial} stop_above={stop_above}: "
          f"descend_end={r2.get('end')} lift={'OK' if r4.get('success') else r4.get('reason')}")
    probe("after")
