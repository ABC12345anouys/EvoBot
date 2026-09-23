"""感知技能：YOLO 物体检测 + 真值深度。

- detect_objects：用 YOLO26n.pt 检测物体（输入 RGB 图像，输出 bbox + class + 置信度）
- estimate_depth：用 MuJoCo 真值深度（仿真）或深度相机（真实）
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence
from ...utils.download import ensure_weight

DETECT_WEIGHTS = ensure_weight("yolo26n.pt",
                                local_path=os.environ.get("DARWIN_YOLO_DET_WEIGHTS"))


@lru_cache(maxsize=1)
def _load_detector():
    from ultralytics import YOLO
    return YOLO(str(DETECT_WEIGHTS))


class DetectObjectsSkill(Skill):
    """用 YOLO 检测场景物体。

    输入：env（自动渲染）或 image
    输出：[{name, bbox, pos, size, conf, source}]
    - bbox: [x1,y1,x2,y2] 像素坐标
    - pos: 3D 位置（仿真下从 env 真值获取匹配物体，或从深度图反投影）
    """
    spec = SkillSpec(
        name="detect_objects",
        description="用 YOLO 检测场景物体，返回 bbox、类别、置信度和 3D 位置。",
        kind=SkillKind.PERCEPTION,
        confidence=Confidence.PROBABLE,
        applies_when="需要检测场景中物体时；先渲染 RGB 图像再 YOLO 推理",
        parameters={
            "image": {"type": "array", "description": "RGB 图像 (H,W,3)，不传则从 env 渲染"},
            "conf_threshold": {"type": "float", "description": "置信度阈值，默认 0.25"},
        },
        required=[],
    )

    def execute(self, env=None, image=None, conf_threshold=0.25) -> Dict[str, Any]:
        if image is None and env is not None:
            image = self._render(env)
        if image is None:
            return {"success": False, "error": "需要 env 或 image 参数"}

        model = _load_detector()
        results = model(image, verbose=False, conf=conf_threshold)[0]

        objects = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                name = results.names.get(cls_id, str(cls_id))
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                obj = {
                    "name": name,
                    "bbox": [float(x) for x in xyxy],
                    "center": [float(cx), float(cy)],
                    "conf": conf,
                    "source": "yolo26n",
                }
                if env is not None:
                    pos = self._match_gt_pos(env, name)
                    if pos is not None:
                        obj["pos"] = pos
                objects.append(obj)

        # YOLO 检测为空时（仿真物体非 COCO 类别），回退到仿真真值
        if not objects and env is not None:
            objects = self._gt_detect(env)

        return {"success": True, "objects": objects, "count": len(objects)}

    @staticmethod
    def _gt_detect(env) -> List[Dict]:
        """仿真真值检测（当 YOLO 检测不到时回退）。"""
        m = env.mj_model
        objects = []
        for i in range(m.nbody):
            name = m.body(i).name
            if name.startswith(("0_", "1_", "robot", "table", "floor", "world", "Base")):
                continue
            gadr = int(m.body_geomadr[i])
            gnum = int(m.body_geomnum[i])
            if gnum == 0:
                continue
            pos = env.mj_data.xpos[i]
            sizes = []
            for g in range(gadr, gadr + gnum):
                sz = m.geom_size[g]
                sizes.append(float(np.max(sz)))
            size = max(sizes) if sizes else 0.0
            objects.append({
                "name": name,
                "pos": [float(x) for x in pos],
                "size": size,
                "type": m.geom(gadr).type if gnum else -1,
                "conf": 1.0,
                "source": "sim_groundtruth",
            })
        return objects

    @staticmethod
    def _match_gt_pos(env, name: str) -> Optional[List[float]]:
        """在仿真真值中匹配同名物体的 3D 位置。"""
        m = env.mj_model
        for i in range(m.nbody):
            bname = m.body(i).name.lower()
            if name.lower() in bname or bname in name.lower():
                return [float(x) for x in env.mj_data.xpos[i]]
        return None

    @staticmethod
    def _render(env) -> np.ndarray:
        import mujoco
        renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.55, 0.0, 0.50]
        cam.distance, cam.azimuth, cam.elevation = 1.6, 135.0, -25.0
        renderer.update_scene(env.mj_data, camera=cam)
        img = renderer.render()
        renderer.close()
        return img


class EstimateDepthSkill(Skill):
    """深度图获取。

    仿真下用 MuJoCo 真值深度；真实环境下接深度相机。
    输出：depth_map (H,W) float32，单位米
    """
    spec = SkillSpec(
        name="estimate_depth",
        description="获取深度图。仿真下用 MuJoCo 真值深度，输出 (H,W) 深度图（米）。",
        kind=SkillKind.PERCEPTION,
        confidence=Confidence.VERIFIED,
        applies_when="需要精确深度信息时；仿真下直接用真值，真实环境接深度相机",
        parameters={
            "image": {"type": "array", "description": "RGB 图像（真实环境深度相机用）"},
        },
        required=[],
    )

    def execute(self, env=None, image=None) -> Dict[str, Any]:
        if env is not None:
            depth = self._mujoco_depth(env)
            return {"success": True, "depth_map": depth, "shape": list(depth.shape),
                    "min": float(depth.min()), "max": float(depth.max()),
                    "source": "mujoco_groundtruth"}
        return {"success": False, "error": "深度获取需要 env 参数（仿真）或深度相机接口"}

    @staticmethod
    def _mujoco_depth(env) -> np.ndarray:
        """从 MuJoCo 渲染器获取真值深度图（米）。"""
        import mujoco
        renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.55, 0.0, 0.50]
        cam.distance, cam.azimuth, cam.elevation = 1.6, 135.0, -25.0
        renderer.enable_depth_rendering()
        renderer.update_scene(env.mj_data, camera=cam)
        depth = renderer.render()
        renderer.close()

        # MuJoCo 深度渲染在有 GL context 时直接返回米为单位的深度
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)
        # 限制合理范围（过滤远背景）
        depth = np.clip(depth, 0.05, 50.0)
        return depth


def get_scene_graph(env=None, image=None) -> Dict[str, Any]:
    """文本化场景图（供 planner prompt 注入）。"""
    det = DetectObjectsSkill()
    result = det.execute(env=env, image=image)
    if not result.get("success"):
        return result
    lines = []
    for obj in result["objects"]:
        pos_str = ""
        if "pos" in obj:
            pos_str = f" @ pos=({obj['pos'][0]:.3f},{obj['pos'][1]:.3f},{obj['pos'][2]:.3f})"
        lines.append(f"- {obj['name']} (conf={obj.get('conf',0):.2f}){pos_str} [{obj['source']}]")
    text = "场景物体:\n" + "\n".join(lines) if lines else "场景中未检测到物体"
    return {"success": True, "scene_graph": text, "objects": result["objects"]}
