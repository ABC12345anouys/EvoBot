"""场景图构建：world 点云 → 桌面去除 → 实例聚类 → SAM 提纯 → 叠层拆分。

实例唯一来源是几何聚类；SAM 只提纯点云，永不创建/删除实例（同物 3 实例
是 SAM 过/欠分割高发场景，几何聚类才是权威）。感知失败抛 PerceptionFail，
调用方沿用上次 SceneGraph（stale 好过崩）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

_DEFAULTS = dict(
    table_margin=0.006,        # 桌面去除阈值（z > table_z + margin）
    cluster_radius=0.012,      # 欧式聚类体素
    cluster_min_pts=80,
    rim_band=0.012,            # 沿口环带（z > z_top - rim_band）
    cavity_depth_ratio=0.5,    # 内腔下凹 > 0.5*高度 → 容器
    sam_enable=0,
    sam_min_ratio=0.30,        # SAM mask 与聚类点交集 <30% → 弃用 SAM
    stack_split_ratio=1.5,     # cluster 高 > 1.5×单碗高 → 试拆层
    merge_radius=0.05,         # 过分割实例合并半径
)

# 实例数兜底聚类档位（radius, min_pts）逐级放松
_CLUSTER_LADDER = [(None, None), (0.008, 40), (0.005, 20)]


class PerceptionFail(RuntimeError):
    """场景感知失败（调用方应沿用上次 SceneGraph）。"""


@dataclass
class Instance:
    cloud: np.ndarray            # (N,3) float32 world
    centroid: np.ndarray         # (3,)
    z_top: float
    z_bottom: float
    aabb_half: np.ndarray        # (3,) 轴对齐半边长
    rim_ring: np.ndarray = field(default=None, repr=False)   # 沿口环点 (M,3)
    mask_px: Dict | None = None  # cam → (H,W) bool（SAM 可用时）
    is_container: bool = False
    source: str = "cluster"      # cluster / sam_refined / layer

    @property
    def center_xy(self) -> np.ndarray:
        return self.centroid[:2]


@dataclass
class SceneGraph:
    table_z: float
    instances: list[Instance] = field(default_factory=list)


def _pixel_of(points: np.ndarray, K: np.ndarray, T: np.ndarray):
    """world 点 → 像素 (u, v)、深度 z（USD -Z 约定，见 rgbd_fusion）。"""
    R, t = np.asarray(T, np.float64)[:3, :3], np.asarray(T, np.float64)[:3, 3]
    pc = (np.asarray(points, np.float64) - t) @ R
    z = -pc[:, 2]
    ok = z > 1e-6
    K = np.asarray(K, np.float64)
    u = K[0, 2] + K[0, 0] * pc[:, 0] / np.where(ok, z, 1)
    v = K[1, 2] - K[1, 1] * pc[:, 1] / np.where(ok, z, 1)
    return u, v, z, ok


def _make_instance(cloud: np.ndarray, rim_band: float, source="cluster") -> Instance:
    lo = cloud.min(axis=0)
    hi = cloud.max(axis=0)
    centroid = (lo + hi) / 2.0
    z_top = float(hi[2])
    ring = cloud[cloud[:, 2] > z_top - rim_band]
    return Instance(cloud=cloud, centroid=centroid, z_top=z_top,
                    z_bottom=float(lo[2]),
                    aabb_half=((hi - lo) / 2.0).astype(np.float32),
                    rim_ring=ring, source=source)


class SceneGraphBuilder:
    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(_DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in _DEFAULTS})
        self._sam = None

    # ---- 桌面 ----

    def _fit_table(self, cloud: np.ndarray) -> float:
        """z 直方图（1cm bin）取最密集平面。Isaac 桌面平整且点数占优。"""
        if len(cloud) < 200:
            raise PerceptionFail(f"点云过少 ({len(cloud)})，无法拟合桌面")
        zmin, zmax = float(cloud[:, 2].min()), float(cloud[:, 2].max())
        bins = np.arange(np.floor(zmin * 100), np.ceil(zmax * 100) + 1) / 100.0
        hist, edges = np.histogram(cloud[:, 2], bins=bins)
        i = int(np.argmax(hist))
        if hist[i] < 0.03 * len(cloud):
            raise PerceptionFail(f"无密集平面（最大 bin {hist[i]}/{len(cloud)}）")
        return float((edges[i] + edges[i + 1]) / 2.0)

    # ---- 聚类 ----

    def _cluster(self, cloud: np.ndarray, radius: float,
                 min_pts: int) -> list[np.ndarray]:
        """体素网格 26 邻接 BFS 欧式聚类。"""
        keys = np.floor(cloud / radius).astype(np.int64)
        vox: Dict[tuple, list[int]] = {}
        for i, k in enumerate(map(tuple, keys)):
            vox.setdefault(k, []).append(i)
        seen_vox = set()
        clusters = []
        for k0 in vox:
            if k0 in seen_vox:
                continue
            comp, stack = [], [k0]
            seen_vox.add(k0)
            while stack:
                k = stack.pop()
                comp.extend(vox[k])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            kn = (k[0] + dx, k[1] + dy, k[2] + dz)
                            if kn not in seen_vox and kn in vox:
                                seen_vox.add(kn)
                                stack.append(kn)
            if len(comp) >= min_pts:
                clusters.append(cloud[np.sort(comp)])
        clusters.sort(key=len, reverse=True)
        return clusters

    # ---- 容器判别 ----

    def _is_container(self, inst: Instance) -> bool:
        """沿口环径向剖面：内环最高点显著低于外环 → 空腔容器（碗）。"""
        ring = inst.rim_ring
        if ring is None or len(ring) < 10:
            return False
        axis = inst.centroid[:2]
        r = np.linalg.norm(ring[:, :2] - axis, axis=1)
        r_out = float(np.percentile(r, 95))
        if r_out < 1e-3:
            return False
        inner_all = inst.cloud[np.linalg.norm(inst.cloud[:, :2] - axis, axis=1)
                               < 0.4 * r_out]
        if len(inner_all) == 0:
            return False
        sink = inst.z_top - float(inner_all[:, 2].max())
        height = max(inst.z_top - inst.z_bottom, 1e-3)
        return bool(sink > self.cfg["cavity_depth_ratio"] * height)

    # ---- SAM 提纯 ----

    def _get_sam(self):
        if self._sam is None:
            from darwin.skills.perception.segment import SegmentObjectSkill
            self._sam = SegmentObjectSkill()
        return self._sam

    def _sam_refine(self, inst: Instance, head_rgb, K, T) -> Instance | None:
        """点提示 SAM 分割 → mask∩点云提纯。失败/交集过低返回 None（不建实例）。"""
        try:
            u, v, _z, ok = _pixel_of(inst.centroid.reshape(1, 3), K, T)
            if not ok[0]:
                return None
            res = self._get_sam().execute(image=head_rgb, point=[float(u[0]), float(v[0])])
            if not res.get("success"):
                return None
            mask = res["mask"]
            H, W = mask.shape
            if mask.shape != head_rgb.shape[:2]:
                return None
            pu, pv, pz, pok = _pixel_of(inst.cloud, K, T)
            iu = np.clip(np.round(pu).astype(int), 0, W - 1)
            iv = np.clip(np.round(pv).astype(int), 0, H - 1)
            infront = pok & (pz < 1.5)
            inmask = np.zeros(len(inst.cloud), bool)
            inmask[infront] = mask[iv[infront], iu[infront]]
            ratio = float(inmask.mean())
            if ratio < self.cfg["sam_min_ratio"]:
                return None
            refined = _make_instance(inst.cloud[inmask], self.cfg["rim_band"],
                                     source="sam_refined")
            refined.is_container = inst.is_container
            refined.mask_px = {"cam_head": mask}
            return refined
        except Exception:
            return None

    # ---- 叠层拆分 ----

    def split_stack_layers(self, inst: Instance,
                           bowl_height: float | None = None) -> list[Instance]:
        """高 cluster（已叠碗合并）按沿口环 z 层级拆成多层实例。"""
        ring = inst.rim_ring
        height = inst.z_top - inst.z_bottom
        ref_h = bowl_height or height
        if ring is None or len(ring) < 10 or height < self.cfg["stack_split_ratio"] * ref_h:
            return [inst]
        zs = np.sort(ring[:, 2])
        levels = [float(zs[0])]
        for z in zs[1:]:
            if z - levels[-1] > 0.02:
                levels.append(float(z))
        if len(levels) < 2:
            return [inst]
        cuts = [float(np.mean([levels[i], levels[i + 1]]))
                for i in range(len(levels) - 1)]
        bounds = [inst.z_bottom - 1e-3] + cuts + [inst.z_top + 1e-3]
        layers = []
        for i in range(len(levels)):
            seg = inst.cloud[(inst.cloud[:, 2] > bounds[i])
                             & (inst.cloud[:, 2] <= bounds[i + 1])]
            if len(seg) < 30:
                continue
            li = _make_instance(seg, self.cfg["rim_band"], source="layer")
            li.is_container = inst.is_container
            layers.append(li)
        return layers or [inst]

    # ---- 主入口 ----

    def build(self, cloud: np.ndarray,
              head_rgb=None, head_K=None, head_T=None,
              expected: int = 3,
              bowl_height: float | None = None,
              arms_ee: Dict | None = None) -> SceneGraph:
        cfg = self.cfg
        table_z = self._fit_table(cloud)
        above = cloud[cloud[:, 2] > table_z + cfg["table_margin"]]
        if len(above) < 50:
            raise PerceptionFail(f"桌面以上点过少 ({len(above)})")

        def _is_object(cl: np.ndarray) -> bool:
            """桌面操作先验：贴桌（z_bottom≈table_z）的簇才是物体。

            悬空大簇是臂/夹爪（dump 实证 z∈[0.89,0.99] 会挤掉真碗）——注意
            不能用「近 ee」判夹持物：夹爪本身就贴着 ee 会被误留。被夹持的碗
            不需要进实例表（model 用 held_off 预测其位姿）；已叠碗物理接触
            连成单簇（z_bottom 贴桌）→ split_stack_layers 负责拆层。
            """
            return bool(float(cl[:, 2].min()) < table_z + 0.03)

        clusters: list = []
        for radius, min_pts in _CLUSTER_LADDER:
            radius = radius or cfg["cluster_radius"]
            min_pts = min_pts or cfg["cluster_min_pts"]
            cs = [c for c in self._cluster(above, radius, min_pts)
                  if _is_object(c)]
            if len(cs) >= 2:
                clusters = cs
                break
            if len(cs) > len(clusters):
                clusters = cs      # 兜底记住最多簇的一轮（默认轮优先）
        if not clusters:
            raise PerceptionFail("桌面以上无可聚类实例")
        # 单簇合法：嵌套叠塔增高极小（5cm 碗叠后 ~5.5cm），不能按高度断言；
        # 调用方（model）用 stack_label 逻辑占位跟踪叠塔。

        # >expected：近邻合并，仍多则取点数 top-N
        if len(clusters) > expected:
            merged: list[np.ndarray] = []
            for cl in clusters:
                c = cl.mean(axis=0)
                hit = next((m for m in merged
                            if np.linalg.norm(m.mean(axis=0)[:2] - c[:2])
                            < cfg["merge_radius"]), None)
                if hit is not None:
                    merged[merged.index(hit)] = np.vstack([hit, cl])
                else:
                    merged.append(cl)
            if len(merged) > expected:
                print(f"[scene_graph] WARN 合并后仍 {len(merged)} 簇，取点数 top-{expected}")
                clusters = sorted(merged, key=len, reverse=True)[:expected]
            else:
                clusters = merged

        insts = [_make_instance(c, cfg["rim_band"]) for c in clusters]
        for it in insts:
            it.is_container = self._is_container(it)

        # 超高簇试拆层（bowl_height 已知时；嵌套叠塔增高极小不会误拆）
        if bowl_height and len(insts) > 1:
            tall = max(insts, key=lambda i: i.z_top - i.z_bottom)
            layers = self.split_stack_layers(tall, bowl_height)
            if len(layers) > 1:
                insts.remove(tall)
                insts.extend(layers)

        # SAM 提纯（可选；只提纯不增删实例）
        if cfg["sam_enable"] and head_rgb is not None and head_K is not None:
            refined = []
            for it in insts:
                r = self._sam_refine(it, head_rgb, head_K, head_T)
                refined.append(r if r is not None else it)
            insts = refined

        insts.sort(key=lambda i: (round(float(i.centroid[0]), 2),
                                  round(float(i.centroid[1]), 2)))
        return SceneGraph(table_z=table_z, instances=insts)
