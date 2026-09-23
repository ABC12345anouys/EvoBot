"""MobileSAM 分割感知技能（用 ultralytics.SAM 加载 mobile_sam.pt）。

作为原子感知技能，与 YOLO 检测配合：YOLO 给 bbox → SAM 精确分割出可抓取区域与点云。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence
from ...utils.download import ensure_weight

DEFAULT_WEIGHTS = ensure_weight("mobile_sam.pt",
                                 local_path=os.environ.get("DARWIN_SAM_WEIGHTS"))


@lru_cache(maxsize=1)
def _load_sam():
    from ultralytics import SAM
    return SAM(str(DEFAULT_WEIGHTS))


class SegmentObjectSkill(Skill):
    """用 MobileSAM 分割指定物体。

    输入：image + bbox（来自 YOLO/检测）或 point 提示
    输出：mask (H,W bool) + 可抓取区域中心
    """
    spec = SkillSpec(
        name="segment_object",
        description="用 MobileSAM 精确分割物体，返回二值 mask 和可抓取中心坐标。",
        kind=SkillKind.PERCEPTION,
        confidence=Confidence.PROBABLE,
        applies_when="需要精确物体轮廓或可抓取区域时；通常先 detect_objects 拿 bbox 再调用",
        parameters={
            "image": {"type": "array", "description": "RGB 图像 (H,W,3)"},
            "bbox": {"type": "array", "description": "边界框 [x1,y1,x2,y2]"},
            "point": {"type": "array", "description": "点提示 [x,y]（二选一）"},
        },
        required=["image"],
    )

    def execute(self, image, bbox=None, point=None) -> Dict[str, Any]:
        if image is None:
            return {"success": False, "error": "需要 image 参数"}
        model = _load_sam()

        if bbox is not None:
            results = model(image, verbose=False, bboxes=[list(bbox)])
        elif point is not None:
            results = model(image, verbose=False, points=[list(point)])
        else:
            return {"success": False, "error": "需要 bbox 或 point 参数"}

        r = results[0]
        if r.masks is None or r.masks.data.numel() == 0:
            return {"success": False, "error": "分割结果为空"}
        mask = r.masks.data[0].cpu().numpy().astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return {"success": False, "error": "mask 为空", "mask": mask}
        center = [float(xs.mean()), float(ys.mean())]
        return {
            "success": True,
            "mask": mask,
            "center": center,
            "area": int(mask.sum()),
            "source": "mobile_sam",
        }


def segment_all_detected(image, detections: List[Dict]) -> Dict[str, Any]:
    """批量分割所有检测到的物体。"""
    seg = SegmentObjectSkill()
    results = []
    for det in detections:
        r = seg.execute(image=image, bbox=det.get("bbox"))
        if r.get("success"):
            r["class"] = det.get("class", det.get("name", "unknown"))
            results.append(r)
    return {"success": True, "segments": results, "count": len(results)}
