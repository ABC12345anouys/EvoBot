"""LIBERO BDDL 任务目标解析：从 (:goal ...) 提取"抓谁、放哪、怎么放"。

支持的谓词（LIBERO pick&place 主体）：
    (On A B)    抓取物体 A，放到 B（物体名 → <B>_default_site；或 region site）
    (In A R)    抓取物体 A，放进容器/抽屉 region R（region site）
    (Open R)    开合类（抽屉/柜门），IK 抓放链暂不支持 → 回落其他方法

多谓词任务（如 libero_10 同集两个 In）取第一个；解析失败返回 None。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional


def parse_bddl_goal(bddl_path: str) -> Optional[Dict[str, Any]]:
    """解析 BDDL goal 段为结构化目标。

    返回:
        {"kind": "place", "predicate": "On"/"In",
         "object": "<被抓物 BDDL 名>", "target": "<目标物体或 region 名>"}
        {"kind": "open", "region": "<region 名>"}
        None: 无法解析
    """
    text = open(bddl_path, encoding="utf-8").read()
    i = text.find("(:goal")
    if i < 0:
        return None
    # 只在 goal 段（BDDL 最后一个段）内匹配，避免命中 (:init ...) 中的同名词
    goal = text[i:]
    # BDDL 标识符只含字母/数字/下划线；按标识符边界匹配，不依赖括号严格闭合
    place = re.search(r"\((On|In)\s+(\w+)\s+(\w+)", goal)
    if place:
        return {"kind": "place", "predicate": place.group(1),
                "object": place.group(2), "target": place.group(3)}
    opened = re.search(r"\(Open\s+(\w+)", goal)
    if opened:
        return {"kind": "open", "region": opened.group(1)}
    return None


def resolve_site_name(model, name: str) -> Optional[str]:
    """BDDL 名 → mujoco site 名。

    解析顺序：region 原名（basket_1_contain_region / flat_stove_1_cook_region
    / wooden_cabinet_1_top_region 等）→ 桌面 region 加 main_table_ 前缀
    → 普通物体的 <name>_default_site。
    """
    candidates = [name, f"main_table_{name}", f"{name}_default_site"]
    for c in candidates:
        try:
            model.site_name2id(c)
            return c
        except Exception:
            continue
    return None
