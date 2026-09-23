"""GraspNet 抓取位姿估计技能。

输入：点云（或深度图+相机内参）+ 可选物体 mask
输出：最优抓取位姿 {position, rotation_matrix, width, score}

设计：
- 优先用 GraspNet 模型（checkpoint-rs.tar）预测
- 若权重不可用，回退到几何方法（从点云 PCA 估计抓取位姿）
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence
from ...utils.download import ensure_weight

GRASPNET_ROOT = Path(os.environ.get("DARWIN_GRASPNET_ROOT",
                                     "/home/lifd/Public/graspnet_repo/graspnet-baseline"))
DEFAULT_CHECKPOINT = Path(os.environ.get("DARWIN_GRASPNET_CKPT",
                                          "/home/lifd/Public/checkpoint-rs.tar"))


def _graspnet_available() -> bool:
    return (GRASPNET_ROOT / "models" / "graspnet.py").exists() and \
           DEFAULT_CHECKPOINT.exists()


@lru_cache(maxsize=1)
def _load_graspnet():
    """加载 GraspNet 模型（带权重）。失败返回 None。"""
    if not _graspnet_available():
        return None
    try:
        sys.path.insert(0, str(GRASPNET_ROOT / "models"))
        sys.path.insert(0, str(GRASPNET_ROOT / "utils"))
        from graspnet import GraspNet, pred_decode
        import torch

        net = GraspNet(
            input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        ckpt = torch.load(str(DEFAULT_CHECKPOINT), map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        return net, pred_decode, device
    except Exception as e:
        print(f"[grasp] GraspNet 加载失败，使用几何 fallback: {e}")
        return None


def _points_to_grasp_geometry(points: np.ndarray) -> Dict[str, Any]:
    """从点云用 PCA 几何方法估计抓取位姿（fallback）。

    对物体点云做 PCA，第二主方向为夹爪开合方向，第一主方向为接近方向。
    """
    if len(points) < 3:
        return {"success": False, "error": "点云点数不足"}
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    approach = eigvecs[:, 2]  # 最小方差方向（物体最短轴）
    if approach[2] > 0:
        approach = -approach  # 确保从上往下

    closing = eigvecs[:, 0]  # 最大方差方向（夹爪开合）
    binormal = np.cross(approach, closing)
    closing = np.cross(binormal, approach)
    closing = closing / (np.linalg.norm(closing) + 1e-8)

    rot = np.stack([closing, binormal, approach], axis=1)

    extents = np.sqrt(eigvals[order])
    width = float(np.clip(extents[0] * 1.2, 0.01, 0.12))

    return {
        "success": True,
        "position": centroid.tolist(),
        "rotation_matrix": rot.tolist(),
        "width": width,
        "score": 0.5,
        "source": "geometry_fallback",
    }


class GraspPoseSkill(Skill):
    """GraspNet 抓取位姿估计技能。

    输入：点云 (N,3) 或 深度图+内参，可选物体 mask
    输出：最优抓取位姿 {position, rotation_matrix, width, score}
    """
    spec = SkillSpec(
        name="grasp_pose",
        description="用 GraspNet 从点云预测抓取位姿（位置、旋转、夹爪宽度、置信度）。权重不可用时回退到几何 PCA 方法。",
        kind=SkillKind.PERCEPTION,
        confidence=Confidence.PROBABLE,
        applies_when="需要计算物体抓取位姿时；通常在 detect+segment 之后调用",
        parameters={
            "point_cloud": {"type": "array", "description": "场景点云 (N,3)"},
            "mask": {"type": "array", "description": "物体 mask（与点云等长的 bool），可选"},
            "depth": {"type": "array", "description": "深度图 (H,W)，与 intrinsic 二选一"},
            "intrinsic": {"type": "array", "description": "相机内参 3x3"},
        },
        required=["point_cloud"],
    )

    def execute(self, point_cloud=None, mask=None, depth=None,
                intrinsic=None, top_k=1) -> Dict[str, Any]:
        # 1. 准备点云
        if point_cloud is None and depth is not None and intrinsic is not None:
            point_cloud = self._depth_to_cloud(depth, intrinsic)
        if point_cloud is None:
            return {"success": False, "error": "需要 point_cloud 或 depth+intrinsic"}

        points = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
        if mask is not None:
            mask = np.asarray(mask).reshape(-1).astype(bool)
            points = points[mask]
        points = points[np.isfinite(points).all(axis=1)]

        if len(points) == 0:
            return {"success": False, "error": "过滤后点云为空"}

        # 2. 尝试 GraspNet 模型
        loaded = _load_graspnet()
        if loaded is not None:
            res = self._graspnet_inference(loaded, points, top_k=top_k)
            if res.get("success"):
                return res
            # GraspNet 对个别物体可能产出 0 候选：不崩，回退几何法
            print(f"[grasp] {res.get('error', 'GraspNet 无候选')}，回退几何法")

        # 3. fallback: 几何方法
        return _points_to_grasp_geometry(points)

    def _depth_to_cloud(self, depth, intrinsic):
        h, w = depth.shape
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def _graspnet_inference(self, loaded, points, top_k=1):
        import torch
        net, pred_decode, device = loaded
        num_point = 20000
        if len(points) >= num_point:
            idxs = np.random.choice(len(points), num_point, replace=False)
        else:
            idxs = np.random.choice(len(points), num_point, replace=True)
        sampled = points[idxs].astype(np.float32)

        end_points = {"point_clouds": torch.from_numpy(sampled[None]).to(device)}
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)

        gg = grasp_preds[0].detach().cpu().numpy()
        # gg 列: score, width, height, depth, rot(9), center(3), obj_id
        scores = gg[:, 0]
        # top-K 候选（按 score 降序）
        k = max(1, int(top_k))
        order = np.argsort(scores)[::-1][:k]
        candidates = []
        for idx in order:
            g = gg[idx]
            candidates.append({
                "position": g[13:16].tolist(),
                "rotation_matrix": g[5:14].reshape(3, 3).tolist(),
                "width": float(g[1]),
                "score": float(g[0]),
            })
        if not candidates:
            return {"success": False, "error": "GraspNet 产出 0 个抓取候选"}
        best = candidates[0]
        result = {
            "success": True,
            "position": best["position"],
            "rotation_matrix": best["rotation_matrix"],
            "width": best["width"],
            "score": best["score"],
            "source": "graspnet",
        }
        if k > 1:
            result["candidates"] = candidates
        return result


def grasp_from_depth_mask(depth, mask, intrinsic):
    """便捷函数：从深度图 + mask 计算抓取位姿。"""
    skill = GraspPoseSkill()
    return skill.execute(depth=depth, mask=mask.flatten(), intrinsic=intrinsic)


def sample_object_point_cloud(env, body_name: str, n_points: int = 2048) -> np.ndarray:
    """从仿真真值采样物体点云（替代深度相机）。

    遍历物体的所有 geom，按表面积比例采样表面点。
    返回 (n_points, 3) 世界坐标点云。

    跨 env 适配：
    - LIBERO（robosuite 装配体）：BDDL 名≠mujoco body 名，且碰撞体分布在
      root body 子树多个子 body 上（如 cream_cheese 装配体）。用 env._resolve_body
      解析根 body 名，按 body_rootid 遍历整个子树采 geom，避免采空。
    - robopal（单 body）：直接按 body_name 查 body_id，采其直属 geom。
    """
    m = env.mj_model
    d = env.mj_data

    geoms: List[int] = []
    if hasattr(env, "_resolve_body"):
        # LIBERO：BDDL 名 → root body，遍历子树全 geom（装配体兜底）
        root_name = env._resolve_body(body_name)
        rid = m.body_name2id(root_name)
        for bid in range(m.nbody):
            if int(m.body_rootid[bid]) != int(rid):
                continue
            gadr = int(m.body_geomadr[bid])
            gnum = int(m.body_geomnum[bid])
            for g in range(gadr, gadr + gnum):
                geoms.append(g)
    else:
        # robopal：单 body 直查
        body_id = None
        for i in range(m.nbody):
            if m.body(i).name == body_name:
                body_id = i
                break
        if body_id is None:
            raise ValueError(f"body '{body_name}' not found")
        gadr = int(m.body_geomadr[body_id])
        gnum = int(m.body_geomnum[body_id])
        for g in range(gadr, gadr + gnum):
            geoms.append(g)

    if not geoms:
        return np.zeros((0, 3), dtype=np.float32)

    points = []
    for g in geoms:
        geom_type = int(m.geom_type[g])
        size = m.geom_size[g]
        pos = d.geom_xpos[g]
        mat = d.geom_xmat[g].reshape(3, 3)
        n_g = max(8, n_points // max(len(geoms), 1))
        if geom_type == 6:  # box
            pts = _sample_box(size, n_g)
        elif geom_type == 2:  # sphere
            pts = _sample_sphere(size[0], n_g)
        elif geom_type == 5:  # cylinder
            pts = _sample_cylinder(size, n_g)
        else:
            pts = _sample_box(size, n_g)
        pts_world = (mat @ pts.T).T + pos
        points.append(pts_world)

    cloud = np.vstack(points).astype(np.float32)
    if len(cloud) > n_points:
        idx = np.random.choice(len(cloud), n_points, replace=False)
        cloud = cloud[idx]
    return cloud


def _sample_box(size, n):
    """采样立方体表面点。size=[hx,hy,hz] 半长。"""
    hx, hy, hz = size[:3]
    pts = []
    faces = [
        ([hx, 0, 0], [0, 1, 0], [0, 0, 1]),   # +x
        ([-hx, 0, 0], [0, 1, 0], [0, 0, 1]),  # -x
        ([0, hy, 0], [1, 0, 0], [0, 0, 1]),   # +y
        ([0, -hy, 0], [1, 0, 0], [0, 0, 1]),  # -y
        ([0, 0, hz], [1, 0, 0], [0, 1, 0]),   # +z
        ([0, 0, -hz], [1, 0, 0], [0, 1, 0]),  # -z
    ]
    per_face = max(1, n // 6)
    for center, d1, d2 in faces:
        u = np.random.uniform(-1, 1, per_face)
        v = np.random.uniform(-1, 1, per_face)
        p = np.array(center) + np.outer(u, d1) * np.array([hx, hy, hz]) + np.outer(v, d2) * np.array([hx, hy, hz])
        pts.append(p)
    return np.vstack(pts)


def _sample_sphere(radius, n):
    """采样球面点。"""
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(np.random.uniform(-1, 1, n))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _sample_cylinder(size, n):
    """采样圆柱表面点。size=[r, h/2]。"""
    r, h = size[0], size[1]
    theta = np.random.uniform(0, 2 * np.pi, n)
    z = np.random.uniform(-h, h, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def grasp_from_env(env, body_name: str) -> Dict[str, Any]:
    """便捷函数：从仿真环境直接获取物体抓取位姿。"""
    cloud = sample_object_point_cloud(env, body_name)
    skill = GraspPoseSkill()
    return skill.execute(point_cloud=cloud)


def grasp_candidates_from_env(env, body_name: str, top_k: int = 8) -> Dict[str, Any]:
    """便捷函数：从仿真环境获取 top-K 候选抓取位姿（供在线重试遍历）。"""
    cloud = sample_object_point_cloud(env, body_name)
    skill = GraspPoseSkill()
    return skill.execute(point_cloud=cloud, top_k=top_k)


# ============================================================
# 物体泛化特征与归一化（RAG 泛化记忆的基础）
# ============================================================

GEOM_TYPE_NAME = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def object_features(env, body_name: str) -> Dict[str, Any]:
    """提取物体泛化特征：形状类型 + 尺寸（用于归一化与条件匹配）。"""
    m = env.mj_model
    for i in range(m.nbody):
        if m.body(i).name == body_name:
            gadr = int(m.body_geomadr[i])
            gnum = int(m.body_geomnum[i])
            if gnum == 0:
                continue
            g = gadr
            gtype = int(m.geom_type[g])
            size = np.array(m.geom_size[g][:3], float)
            shape = GEOM_TYPE_NAME.get(gtype, f"type{gtype}")
            if shape == "mesh" or size.max() < 1e-6:
                # mesh 没有有意义 size，用点云包围盒近似
                cloud = sample_object_point_cloud(env, body_name, n_points=512)
                ext = cloud.max(axis=0) - cloud.min(axis=0)
                size = np.maximum(ext / 2, 1e-3)
                shape = "mesh"
            return {"shape": shape, "size": size.tolist()}
    return {"shape": "unknown", "size": [0.02, 0.02, 0.02]}


def rel_offset_of(grasp_pos, obj_center, size) -> list:
    """抓取点 → 物体坐标系归一化偏移（单位 = half-size）。泛化核心。"""
    off = np.asarray(grasp_pos, float) - np.asarray(obj_center, float)
    sz = np.maximum(np.asarray(size, float), 1e-3)
    return (off / sz).tolist()


def abs_pos_of(rel_offset, obj_center, size) -> np.ndarray:
    """归一化偏移 → 世界坐标（用旧经验在新物体/新位置上重建抓取点）。"""
    sz = np.maximum(np.asarray(size, float), 1e-3)
    return np.asarray(obj_center, float) + np.asarray(rel_offset, float) * sz


def score_band_of(score: float) -> str:
    if score < 1.0:
        return "low"
    if score < 1.3:
        return "mid"
    return "high"
