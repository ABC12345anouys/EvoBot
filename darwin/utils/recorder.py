"""EGL 离屏录像工具：包装 env.step，逐帧渲染为 MP4。"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

from pathlib import Path
from typing import Optional


def make_camera(bimanual: bool):
    """自由机位：双臂场景视野更远，单臂 3/4 视角。"""
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if bimanual:
        cam.lookat[:] = [0.5, 0.0, 0.45]
        cam.distance, cam.azimuth, cam.elevation = 2.6, 90.0, -25.0
    else:
        cam.lookat[:] = [0.45, 0.0, 0.40]
        cam.distance, cam.azimuth, cam.elevation = 2.0, 135.0, -22.0
    return cam


class StepRecorder:
    """录制 env.step 序列为 MP4（用完必须 close，恢复原 step）。

    用法：
        rec = StepRecorder(env, "out.mp4", bimanual=True)
        try:
            ...  # 正常 env.step(...)，自动逐帧落盘
        finally:
            rec.close()
    """

    def __init__(self, env, path: str, bimanual: bool, fps: int = 20,
                 width: int = 640, height: int = 480) -> None:
        import cv2
        import mujoco
        self._cv2, self._mujoco = cv2, mujoco
        self.env = env
        self._orig_step = env.step
        self._cam = make_camera(bimanual)
        self._renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        assert self._writer.isOpened(), "VideoWriter 打开失败"
        env.step = self._wrapped_step  # type: ignore

    def _wrapped_step(self, action):
        self._orig_step(action)
        self._renderer.update_scene(self.env.mj_data, camera=self._cam)
        frame = self._renderer.render()
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.env.step = self._orig_step  # type: ignore
        self._writer.release()
        self._renderer.close()


__all__ = ["StepRecorder", "make_camera"]
