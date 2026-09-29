"""RGB-D → world 点云融合（RoboDojo 3 相机：cam_head / cam_left_wrist / cam_right_wrist）。

USD 相机约定（2026-09-24 用 /tmp/darwin_obs_dump.pkl 实证，勿凭 pinhole 直觉改）：
- extrinsic_matrix 是 camera-to-world（4x4），但相机沿 **-Z 轴**观察、+Y 朝上。
- depth = distance_to_image_plane，沿光轴，z>0 表示在相机前方。
- 反投影公式：P_cam = [(u-cx)·z/fx, -(v-cy)·z/fy, -z]，P_world = R @ P_cam + t。
- world 系 ≡ env 相对系（单 env 实证；多 env 需加 env_origins 平移，暂不支持）。

wrist 相机自遮挡实证：depth 最小 0.058m 是自家手指——近距剔除 + ee 球域剔除双保险。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# 默认参数（可被 cfg 覆盖；PARAM_SPEC 见 darwin/skills/skill_config.py）
_DEFAULTS = dict(
    pcd_voxel=0.005,
    pcd_depth_min=0.10,
    pcd_depth_max_head=1.20,
    pcd_depth_max_wrist=0.80,
    ws_x_min=-0.45, ws_x_max=0.45,
    ws_y_min=-0.35, ws_y_max=0.25,
    ws_z_min=0.50, ws_z_max=1.10,
    self_occl_min=0.10,   # wrist 近距剔除（自家手指/壳体）
    ee_clear=0.09,        # 距 ee 球域剔除半径
)


def backproject(depth: np.ndarray, K: np.ndarray, T: np.ndarray,
                depth_min: float = 0.10, depth_max: float = 1.20,
                self_occl_ee: np.ndarray | None = None,
                self_occl_min: float = 0.10,
                ee_clear: float = 0.09) -> np.ndarray:
    """单相机 depth(H,W 米) + 内参 K(3,3) + 外参 T(4x4 cam→world) → world 点云 (N,3)。

    USD -Z 约定见模块 docstring。self_occl_ee 给该臂 ee 坐标 (3,) 时做双重
    自遮挡剔除（z < self_occl_min 或距 ee < ee_clear）。
    """
    depth = np.asarray(depth, np.float32)
    K = np.asarray(K, np.float64)
    T = np.asarray(T, np.float64)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    z = depth
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    if self_occl_ee is not None:
        valid &= z > self_occl_min
    if not valid.any():
        return np.zeros((0, 3), np.float32)

    z = z[valid]
    us = u[valid].astype(np.float64)
    vs = v[valid].astype(np.float64)
    # USD 相机：-Z forward、+Y up（v 翻转在 -(v-cy) 中体现）
    pc = np.stack([(us - K[0, 2]) * z / K[0, 0],
                   -(vs - K[1, 2]) * z / K[1, 1],
                   -z], axis=1)
    R, t = T[:3, :3], T[:3, 3]
    pw = pc @ R.T + t

    if self_occl_ee is not None:
        ee = np.asarray(self_occl_ee, np.float64).reshape(1, 3)
        keep = np.linalg.norm(pw - ee, axis=1) > ee_clear
        pw = pw[keep]
    return pw.astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel: float = 0.005) -> np.ndarray:
    """numpy 体素下采样（体素内取首个点，dict 哈希，不引 open3d）。"""
    if len(points) == 0:
        return points
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


class RgbdFusion:
    """obs['vision'] → world 融合点云。head 为主，wrist 补近景（带自遮挡剔除）。"""

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(_DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in _DEFAULTS})

    def obs_to_cloud(self, vision: Dict[str, dict],
                     arms_ee: Dict[str, np.ndarray] | None = None
                     ) -> Tuple[np.ndarray, dict]:
        """融合 3 相机 → (points (N,3) float32 world, meta{每相机点数})。

        arms_ee: {臂前缀('left'/'right'/''  ): ee 坐标 (3,)}，wrist 自遮挡剔除用。
        """
        arms_ee = arms_ee or {}
        c = self.cfg
        cams = {
            "cam_head": (c["pcd_depth_min"], c["pcd_depth_max_head"], None),
            "cam_left_wrist": (c["pcd_depth_min"], c["pcd_depth_max_wrist"],
                               arms_ee.get("left")),
            "cam_right_wrist": (c["pcd_depth_min"], c["pcd_depth_max_wrist"],
                                arms_ee.get("right")),
        }
        clouds, meta = [], {}
        for cam, (dmin, dmax, ee) in cams.items():
            cd = vision.get(cam)
            if not cd or "depth" not in cd or "intrinsic_matrix" not in cd \
                    or "extrinsic_matrix" not in cd:
                meta[cam] = 0
                continue
            pts = backproject(cd["depth"], cd["intrinsic_matrix"],
                              cd["extrinsic_matrix"], dmin, dmax,
                              self_occl_ee=ee,
                              self_occl_min=c["self_occl_min"],
                              ee_clear=c["ee_clear"])
            meta[cam] = int(len(pts))
            if len(pts):
                clouds.append(pts)
        if not clouds:
            return np.zeros((0, 3), np.float32), meta
        pts = np.vstack(clouds)
        # 工作空间盒裁剪（去背景/天花板/臂根）
        box = ((pts[:, 0] > c["ws_x_min"]) & (pts[:, 0] < c["ws_x_max"])
               & (pts[:, 1] > c["ws_y_min"]) & (pts[:, 1] < c["ws_y_max"])
               & (pts[:, 2] > c["ws_z_min"]) & (pts[:, 2] < c["ws_z_max"]))
        pts = pts[box]
        pts = voxel_downsample(pts, c["pcd_voxel"])
        return pts, meta
