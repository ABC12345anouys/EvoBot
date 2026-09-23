"""为每个注册环境录制一段 MP4（无头 EGL 离屏渲染）。

运行（darwin 环境）：
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/render_episodes.py
输出：data/videos/<env>.mp4
"""
import os
import sys

# 无头引导：必须先于 import robopal
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import numpy as np
import cv2
import mujoco
import robopal
from robopal.envs.base import MujocoEnv

# robopal 的 MujocoEnv.close() 会在收尾时 os._exit(0)（为交互式演示设计），
# 会导致多环境顺序录制时进程被硬杀。这里仅在本脚本内替换为正常清理。
MujocoEnv.close = lambda self: self.renderer.close()

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "videos")
W, H, FPS = 640, 480, 20

ENV_SPECS = [
    # (env_id, robot, steps, 双臂?)
    ("PickAndPlace-v1", "PandaPickAndPlace", 150, False),
    ("MultiCubeStack-v1", "DianaTripleStack", 100, False),
    ("Drawer-v1", "DianaDrawer", 100, False),
    ("LockedCabinet-v1", "DianaCabinet", 100, False),
    ("DrawerBox-v1", "DianaDrawerCube", 100, False),
    ("BimanualReach-v0", "DualDianaReach", 100, True),
    ("BimanualPickAndPlace-v0", "DualPandaPickAndPlace", 100, True),
    ("BimanualTransport-v0", "DualPandaTransport", 100, True),
]


def sinusoid(t, period=40):
    return np.array([
        0.6 * np.sin(2 * np.pi * t / period),
        0.6 * np.sin(2 * np.pi * t / period + np.pi / 2),
        0.3 * np.sin(2 * np.pi * t / (2 * period)),
        1.0 if (t // (period // 2)) % 2 == 0 else -1.0,
    ])


def pick_and_place_policy(env, t):
    """脚本化抓取：移到方块上方 -> 下降 -> 闭合 -> 抬起。"""
    end = env.get_site_pos("0_grip_site")
    block = env.get_body_pos("green_block")
    if t < 40:
        target = block + np.array([0.0, 0.0, 0.12])
    elif t < 70:
        target = block + np.array([0.0, 0.0, 0.03])
    elif t < 100:
        target = block + np.array([0.0, 0.0, 0.03])
    else:
        target = block + np.array([0.0, 0.0, 0.20])
    gripper = 1.0 if t < 70 else -1.0
    vel = np.clip(10.0 * (target - end), -1.0, 1.0)[:3]
    return np.append(vel, gripper)


def make_action(env, spec, t):
    _, _, _, bimanual = spec
    if env.name == "PickAndPlaceEnv":
        return pick_and_place_policy(env, t)
    a = sinusoid(t)
    if bimanual:
        return {ag: a.copy() for ag in env.agents}
    return a


def make_camera(bimanual):
    """自由机位：双臂场景视野更远，单臂 3/4 视角。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if bimanual:
        cam.lookat[:] = [0.5, 0.0, 0.45]
        cam.distance, cam.azimuth, cam.elevation = 2.6, 90.0, -25.0
    else:
        cam.lookat[:] = [0.45, 0.0, 0.40]
        cam.distance, cam.azimuth, cam.elevation = 2.0, 135.0, -22.0
    return cam


def record(spec):
    env_id, robot, steps, bimanual = spec
    env = robopal.make(env_id, robot=robot, render_mode=None, control_freq=FPS)
    env.reset()
    renderer = mujoco.Renderer(env.mj_model, height=H, width=W)
    camera = make_camera(bimanual)
    out_path = os.path.abspath(os.path.join(OUT_DIR, f"{env_id}.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened(), "VideoWriter 打开失败"

    try:
        for t in range(steps):
            env.step(make_action(env, spec, t))
            renderer.update_scene(env.mj_data, camera=camera)
            frame = renderer.render()  # RGB (H, W, 3)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()
        env.close()
    return out_path


def main():
    import faulthandler
    faulthandler.enable()
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for spec in ENV_SPECS:
        env_id = spec[0]
        print(f"[STAGE] {env_id}: 开始", flush=True)
        try:
            path = record(spec)
            size = os.path.getsize(path) / 1024
            print(f"[OK]   {env_id:26s} -> {path} ({size:.0f} KB)", flush=True)
            results.append((env_id, True))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {env_id}: {type(e).__name__}: {e}", flush=True)
            results.append((env_id, False))
    print("\n===== 汇总 =====", flush=True)
    for env_id, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {env_id}", flush=True)
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
