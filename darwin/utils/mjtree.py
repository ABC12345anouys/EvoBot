"""MuJoCo 模型结构通用小工具（跨 env 适配层与感知层共享）。"""
from __future__ import annotations

from typing import List


def subtree_body_ids(model, root_id: int) -> List[int]:
    """root_id 子树的全部 body id（沿 body_parentid 链回溯判定）。

    mujoco 的 body_rootid 只把"自由关节装配根"标成子树根；焊在 world 下
    的装配子 body（LIBERO 柜架 wooden_cabinet_1_*、炉具 flat_stove_1_*）
    rootid==0，按 rootid 等价类收集 geom 会得到空集——柜架碰撞体对
    object_bounds / 点云采样整体隐身，抓取偏置失去柜侧净空约束、贴柜框
    棱楔停（spatial:6 实证：ik_unreachable，coll=gripper/cabinet_base）。
    parentid 链回溯是装配子树的通用判定，自由装配体与焊接子 body 等效。
    """
    n = int(model.nbody)
    par = [int(model.body_parentid[i]) for i in range(n)]
    out = []
    for bid in range(n):
        b = bid
        while b != 0 and b != root_id:
            b = par[b]
        if b == root_id:
            out.append(bid)
    return out
