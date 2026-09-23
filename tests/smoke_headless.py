"""无头冒烟测试：验证 robopal 在当前服务器（EGL）上可加载模型、reset、步进、离屏渲染。

运行（darwin 环境）：
    MUJOCO_GL=egl python tests/smoke_headless.py
"""
import os
import sys
import numpy as np

# 服务器无显示时强制走 EGL 离屏渲染
os.environ.setdefault("MUJOCO_GL", "egl")
# 多卡机器上 EGL 可能默认选到显存占满的卡而卡死，默认固定到较空闲的 GPU0，
# 可用 DARWIN_GPU 环境变量覆盖。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
# 远程服务器转发的 X11 DISPLAY 可达但不稳定，会使 robopal 导入链中的 GL/X
# 相关初始化偶发无限阻塞；无头 EGL 渲染不需要 X，导入 robopal 前移除 DISPLAY 即可规避。
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import cv2
import robopal
import robopal.envs


def render_offscreen(env, camera="frontview"):
    """直接用 MuJoCo 离屏 Renderer 出 RGB 图（不弹 GLFW 窗口）。"""
    env.renderer.image_renderer.update_scene(env.mj_data, camera=camera)
    rgb = env.renderer.image_renderer.render()
    return rgb


def run_env(env_name, robot, steps=100, save_img=None):
    print(f"\n=== 构建环境 {env_name} (robot={robot}) ===", flush=True)
    env = robopal.make(
        env_name,
        robot=robot,
        render_mode=None,                 # 无头，不创建窗口
        control_freq=20,
        is_render_camera_offscreen=True,  # 仍会创建离屏 Renderer
        is_randomize_end=False,
        is_randomize_object=False,
    )
    print("环境构建成功，开始 reset ...", flush=True)
    obs, info = env.reset()
    print(f"reset 成功，obs shape = {np.asarray(obs).shape}，info keys = {list(info.keys())}", flush=True)

    action_dim = getattr(env, "action_dim", 4)
    amin = getattr(env, "min_action", -1.0)
    amax = getattr(env, "max_action", 1.0)

    rng = np.random.default_rng(0)
    ep_returns = 0.0
    for t in range(steps):
        action = rng.uniform(np.asarray(amin), np.asarray(amax), action_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_returns += float(np.asarray(reward).sum())
        if terminated or truncated:
            print(f"  step {t}: episode 结束 (success={info.get('is_success')})，自动 reset", flush=True)
            env.reset()

    print(f"步进 {steps} 步正常，累计 reward={ep_returns:.2f}", flush=True)

    rgb = render_offscreen(env)
    print(f"离屏渲染帧 shape = {rgb.shape}，像素范围=[{rgb.min()},{rgb.max()}]", flush=True)
    if save_img is not None:
        bgr = rgb[:, :, ::-1]
        cv2.imwrite(save_img, bgr)
        print(f"已保存渲染图: {save_img}", flush=True)

    env.close()
    print(f"=== {env_name} 通过 ===", flush=True)


if __name__ == "__main__":
    out_dir = "/tmp/robopal_smoke"
    os.makedirs(out_dir, exist_ok=True)

    print("已注册环境:", list(robopal.envs.REGISTERED_ENVS.keys()), flush=True)

    run_env(
        "PickAndPlace-v1",
        robot="PandaPickAndPlace",
        steps=100,
        save_img=os.path.join(out_dir, "pickplace.png"),
    )

    print("\n全部冒烟测试通过", flush=True)
    sys.exit(0)
