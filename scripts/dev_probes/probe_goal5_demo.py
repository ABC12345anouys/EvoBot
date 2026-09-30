"""goal:5 demo 抓取位姿提取：从 demo_0 状态序列找夹爪闭合时刻的 eef 位姿，
作为 plate 抓取的几何真值（对照现有候选为何夹空）。

跑: MUJOCO_GL=egl python scripts/probe_goal5_demo.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

sys.path.insert(0, ".")

import numpy as np

from darwin.envs.libero_adapter import _task_info

info = _task_info("libero_goal", 5)
print("[probe] task:", info["task"].name, flush=True)

import h5py
with h5py.File(info["demo"], "r") as f:
    demo = f["data"]["demo_0"]
    acts = np.array(demo["actions"])
    states = np.array(demo["states"])
    model_xml = demo.attrs["model_file"]
    print("[probe] actions:", acts.shape, "states:", states.shape, flush=True)

# robosuite libero action: [dx,dy,dz, droll,dpitch,dyaw, gripper] 或绝对版；
# 找 gripper 维度（动作最后一维）的符号翻转点
g = acts[:, -1]
close_idx = [i for i in range(1, len(g)) if g[i - 1] < 0.5 <= g[i]]
open_idx = [i for i in range(1, len(g)) if g[i - 1] >= 0.5 > g[i]]
print(f"[probe] gripper close at steps: {close_idx}", flush=True)
print(f"[probe] gripper open at steps: {open_idx}", flush=True)

# 状态里的 eef pose：robosuite 状态拼接 [robot0_joint_pos(7), robot0_gripper_qpos(2),
# eef_pos(3), eef_quat(4), ...]；先按常见布局猜，打印闭合前后窗口
for ci in close_idx[:2]:
    lo, hi = max(0, ci - 8), min(len(states), ci + 4)
    print(f"[probe] --- 闭合点 {ci} 前后状态窗口 ---", flush=True)
    for i in range(lo, hi):
        s = states[i]
        print(f"  t={i} act={np.round(acts[i], 3)}", flush=True)
        # 尝试常见偏移：7+2=9 后是 eef_pos
        for off, name in ((9, "off9"), (16, "off16"), (19, "off19")):
            if off + 3 <= len(s):
                print(f"      {name}: pos={np.round(s[off:off+3], 4)}", flush=True)
