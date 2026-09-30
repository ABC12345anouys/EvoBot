"""一次性探针：spatial 场景物体枚举 + 动态性判定 + 全场景点云。"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/lifd/Public/darwin-bot")
sys.path.insert(0, "/home/lifd/Public/LIBERO")

import numpy as np
import mujoco

from darwin.envs.libero_adapter import LiberoEnvAdapter
from darwin.skills.perception.grasp import sample_scene_point_cloud


def dynamic_objects(adapter):
    """有自由关节且未被 weld 的物体 = 可抓动态物体。"""
    m = adapter.mj_model
    # weld equality 约束固定的 body
    welded = set()
    for i in range(m.neq):
        if int(m.eq_type[i]) == mujoco.mjtEq.mjEQ_WELD:
            welded.add(int(m.eq_obj1id[i]))
            welded.add(int(m.eq_obj2id[i]))
    out = []
    for name in adapter.object_names:
        root = adapter._resolve_body(name)
        rid = m.body_name2id(root)
        bid = rid
        has_free = False
        while bid != 0:
            jadr, jnum = int(m.body_jntadr[bid]), int(m.body_jntnum[bid])
            for j in range(jadr, jadr + jnum):
                if int(m.jnt_type[j]) == 0:  # free joint
                    has_free = True
            bid = int(m.body_parentid[bid])
        b = adapter.object_bounds(name)
        out.append((name, has_free and rid not in welded,
                    round(b["z_top"] - b["z_bottom"], 3),
                    round(2 * max(b["half_x"], b["half_y"]), 3)))
    return out


for idx in [0, 3, 6, 8]:
    env = LiberoEnvAdapter("libero_spatial", idx)
    env.reset()
    print(f"--- task {idx} objs ---")
    for nm, dyn, h, w in dynamic_objects(env):
        print(f"  {nm:30s} dynamic={dyn} h={h} w={w}")
    pc = sample_scene_point_cloud(env, 20000)
    print("  cloud:", pc.shape, np.round(pc.mean(0), 3))
    env.close()
