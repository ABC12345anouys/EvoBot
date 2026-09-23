"""探测各演示环境的运行时几何与坐标系，为纯控制器策略校准假设。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/probe_envs.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import numpy as np
import mujoco
import robopal
from robopal.envs.base import MujocoEnv

MujocoEnv.close = lambda self: self.renderer.close()

SITES = {
    "PickAndPlace-v1": ["goal_site"],
    "MultiCubeStack-v1": ["red_goal", "green_goal", "blue_goal"],
    "Drawer-v1": ["drawer", "drawer_goal"],
    "LockedCabinet-v1": ["left_handle", "beam_left", "cabinet_mid", "cabinet_left_opened"],
    "DrawerBox-v1": ["drawer", "drawer_goal", "cube_goal"],
    "BimanualReach-v0": ["goal_site0", "goal_site1"],
    "BimanualPickAndPlace-v0": ["goal_site"],
    "BimanualTransport-v0": [],
}
BODIES = {
    "PickAndPlace-v1": ["green_block"],
    "MultiCubeStack-v1": ["red_block", "green_block", "blue_block"],
    "Drawer-v1": [],
    "LockedCabinet-v1": [],
    "DrawerBox-v1": ["green_block"],
    "BimanualReach-v0": [],
    "BimanualPickAndPlace-v0": ["green_block"],
    "BimanualTransport-v0": ["hammer", "carton"],
}
ROBOTS = {
    "PickAndPlace-v1": "PandaPickAndPlace",
    "MultiCubeStack-v1": "DianaTripleStack",
    "Drawer-v1": "DianaDrawer",
    "LockedCabinet-v1": "DianaCabinet",
    "DrawerBox-v1": "DianaDrawerCube",
    "BimanualReach-v0": "DualDianaReach",
    "BimanualPickAndPlace-v0": "DualPandaPickAndPlace",
    "BimanualTransport-v0": "DualPandaTransport",
}


def probe(env_id):
    robot = ROBOTS[env_id]
    env = robopal.make(env_id, robot=robot, render_mode=None, control_freq=20)
    env.reset()
    print("\n==== %s (%s) ====" % (env_id, robot), flush=True)
    print("agents:", env.agents, flush=True)
    for ag in env.agents:
        base = env.robot.base_link_name[ag]
        bid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, base)
        bpos = env.mj_data.xpos[bid].copy()
        bmat = env.mj_data.xmat[bid].reshape(3, 3).copy()
        end_pos = env.controller.forward_kinematics(env.robot.get_arm_qpos(ag), agent=ag)[0]
        grip_site = "%s_grip_site" % ag[-1]
        try:
            gs = env.get_site_pos(grip_site)
        except Exception:
            gs = None
        print("  %s base=%s" % (ag, bpos.round(3)), flush=True)
        print("    base_mat=\n%s" % bmat.round(3), flush=True)
        print("    FK_end(base frame)=%s" % end_pos.round(3), flush=True)
        if gs is not None:
            print("    %s(world)=%s" % (grip_site, gs.round(3)), flush=True)
            # world = base_pos + base_mat @ end_base; end_base = base_mat.T @ (world - base_pos)
            eb = bmat.T @ (gs - bpos)
            print("    grip_site_in_base=%s" % eb.round(3), flush=True)
    print("  pos_bounds:", flush=True)
    pmx = getattr(env, "pos_max_bound", None)
    pmn = getattr(env, "pos_min_bound", None)
    if isinstance(pmx, dict):
        for ag in env.agents:
            print("    %s max=%s min=%s" % (ag, pmx[ag], pmn[ag]), flush=True)
    else:
        print("    max=%s min=%s" % (pmx, pmn), flush=True)
    for s in SITES.get(env_id, []):
        try:
            print("  site %s = %s" % (s, env.get_site_pos(s).round(3)), flush=True)
        except Exception as e:
            print("  site %s ERR %s" % (s, e), flush=True)
    for b in BODIES.get(env_id, []):
        try:
            print("  body %s = %s" % (b, env.get_body_pos(b).round(3)), flush=True)
        except Exception as e:
            print("  body %s ERR %s" % (b, e), flush=True)
    env.close()


def main():
    for eid in ROBOTS:
        try:
            probe(eid)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("[ERR] %s: %s" % (eid, e), flush=True)


if __name__ == "__main__":
    main()
