"""rule 策略实现 = 现状逻辑的逐函数搬运（Phase A，行为不变）。

grasp_pose/rule：libero 候选生成（容器纯 Y straddle 偏置 + 净空翻向 /
实心物 GraspNet xy + z-override / 几何中心置顶 / 极薄物包边）。
搬运自 runner_dynamic._build_fresh_candidates 的 libero 分支——
runner 现在改调本模块，保证行为与搬运前完全一致（verify_30 回归）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from ..skills.perception.grasp import grasp_candidates_from_env
from .context import StepContext
from .registry import register_policy
from .spaces import ParamSpace


class RuleGraspPoseLibero:
    """libero 抓取候选生成（rule 实现，= 搬运前 runner_dynamic 行为）。"""

    name = "rule"

    def propose(self, ctx: StepContext,
                param_space: Optional[ParamSpace] = None) -> Dict[str, Any]:
        env, entry, cfg = ctx.env, ctx.entry, ctx.cfg
        body_name = entry["body"]
        feat = {"shape": "libero_object", "size": [0.02, 0.02, 0.02]}
        # 容器判定单一来源：libero_adapter._CONTAINER_HINTS（避免两份
        # 名单漂移）。注意不用几何启发式（扁平→容器）——会把盘类等
        # 扁平实心物误路由到偏心插指，当前通过任务有回归风险。
        is_container = env._is_container(body_name)
        res = grasp_candidates_from_env(env, body_name, top_k=8)
        all_cands = res.get("candidates") or []
        cands = []
        max_width = float(cfg.get("grasp_max_width", 0.075))
        b = env.object_bounds(body_name)
        cx, cy = float(b["center"][0]), float(b["center"][1])
        half_max = max(float(b["half_x"]), float(b["half_y"]))

        def _sanity_xy(pos) -> bool:
            # 候选 sanity：偏离几何中心超过物体本身尺寸的候选必假
            # （GraspNet 吃脏点云时的典型症状），丢弃。
            return np.linalg.norm(np.asarray(pos, float)[:2]
                                  - [cx, cy]) <= 1.2 * half_max + 1e-6

        if is_container and all_cands:
            # 容器 straddle 几何（akita_black_bowl 探针实证）：夹爪两指
            # 沿世界 Y 固定分离 80mm。偏置方向纯 Y 最优——内指越沿后径向
            # ≈40mm−off 落碗腔，外指远超碗沿半径，straddle 决定性；X 向
            # 双指都在沿半径上蹭沿磨停（spatial:2 回归根因）；45° 对角
            # 内指入腔过深。故废弃投票/45° snap，固定纯 Y 偏置，符号
            # 按两侧障碍净空翻向；两侧都不足才退回 12 方向搜索。
            half_y, z_top = float(b["half_y"]), float(b["z_top"])
            ratio = float(cfg.get("grasp_container_offset_ratio", 0.70))
            z_delta = float(cfg.get("grasp_container_z_delta", 0.030))
            hover = float(cfg.get("hover", 0.08))
            # straddle 偏置 = 双指下降走廊净空的 argmax（derives 单一来源）。
            # 固定 ratio（0.7·half_y）是"akita 碗腔径/沿厚恰好适配"的标定，
            # 腔径或沿厚一变内指即蹭沿磨停（spatial:0 实证：内指穿透沿
            # -16.5mm、阻塞在抓取点上方 25mm）。偏置是物体几何的函数，
            # 应由真值表面点云对双指 shaft 的净空决定；ratio 降级为点云
            # 不可得时的回落（标定成为下界而非规则）。
            from ..physics.derives import straddle_offset, straddle_offset_search
            from ..agents.runner_dynamic import DynamicEpisodeRunner as _DynR
            _cloud = np.zeros((0, 3), float)
            # 场景点云 = 目标自身 + 全障碍子树（柜架/邻物）。偏置搜索对
            # 双指 shaft 净空取 argmax——只喂目标云时柜架几何不可见
            # （object_bounds 对装配子 body 抛"子树无 geom"，AABB 通路
            # 整体跳过柜架），偏置落点可能贴柜框棱，descend shaft 蹭框
            # 楔停（spatial:4 实证：同 -y 向偏置差 6mm，一过一楔）。
            # 点云经 _resolve_body 取根子树全 geom，柜架表面进入 z 带
            # 参与净空计算，argmax 自然把柱挪到框口。采不到的障碍跳过
            # （行为同旧：该障碍不参与）。
            try:
                from ..skills.perception.grasp import sample_object_point_cloud
                _own_cloud = sample_object_point_cloud(env, body_name,
                                                       n_points=2048)
                _clouds = [_own_cloud]
                for _nm in (getattr(env, "obstacle_bodies", None) or ()):
                    if _nm == body_name:
                        continue
                    try:
                        _clouds.append(sample_object_point_cloud(
                            env, _nm, n_points=1024))
                    except Exception:
                        continue
                _cloud = np.vstack(_clouds)
                # 径向裁剪到目标邻域（3×half_max）：搜索网格半径由带内点
                # r_max 推导，远处场景点只会拉大网格步长、稀释平台中位
                # 稳定性；净空语义不变（远处点对 shaft 净空无竞争）。
                # 只裁剪混合云：own 云另行传入搜索（口沿覆盖判据用），
                # 不得被场景裁剪阉割。
                _near = _cloud[np.hypot(_cloud[:, 0] - cx,
                                        _cloud[:, 1] - cy)
                               < 3.0 * half_max]
                _cloud = _near if len(_near) else _cloud
            except Exception:
                _own_cloud = np.zeros((0, 3), float)
                pass
            _gp = str(entry.get("grip_site", "gripper0_grip_site")
                      ).split("_grip_site")[0]

            def _finger_half_spread(direction):
                # 两指 body 沿分离方向的世界距/2（现场状态观测，非标定）
                try:
                    m, d = env.mj_model, env.mj_data
                    p1 = d.body_xpos[m.body_name2id(f"{_gp}_leftfinger")]
                    p2 = d.body_xpos[m.body_name2id(f"{_gp}_rightfinger")]
                    return abs(float((np.asarray(p1) - np.asarray(p2)) @ np.asarray(
                        [direction[0], direction[1], 0.0]))) / 2.0
                except Exception:
                    return None

            def _separation_axis():
                # 夹爪两指分离轴（世界 xy 单位向量）。straddle 偏置沿此轴
                # 时搜索模型精确（双指 shaft 都在轴上）；垂直方向的 hs 投
                # 影≈0，搜索退化成"对中"语义、分数不可与 straddle 同榜比较
                # （spatial:4 实证：x 向 hs=0.0006，伪高分 0.0252 若混入
                # 排序会抢走 +y 的可用走廊）。轴向由 gripper 几何现场观测，
                # 换臂/换手爪自动跟随（derives.straddle_offset 的 P1-1
                # 注记同旨）。观测失败回落世界 ±Y。
                try:
                    m, d = env.mj_model, env.mj_data
                    p1 = d.body_xpos[m.body_name2id(f"{_gp}_leftfinger")]
                    p2 = d.body_xpos[m.body_name2id(f"{_gp}_rightfinger")]
                    s = np.asarray(p1, float)[:2] - np.asarray(p2, float)[:2]
                    nrm = float(np.linalg.norm(s))
                    if nrm > 1e-6:
                        return s / nrm
                except Exception:
                    pass
                return np.array([0.0, 1.0])

            def _off_for(direction):
                hs = _finger_half_spread(direction)
                if hs is not None and len(_cloud):
                    off = straddle_offset_search(
                        _cloud, [cx, cy], direction,
                        z_top - z_delta, z_top - z_delta + hover,
                        hs, _DynR.FINGER_R_M, own=_own_cloud)
                    if off is not None:
                        return off
                return straddle_offset(half_y, ratio)

            def _mk(direction, off):
                return [float(cx + off * direction[0]),
                        float(cy + off * direction[1]),
                        float(z_top - z_delta)]

            # 净空评估：只看 z 范围与插指走廊相交的障碍（z 过滤排除
            # 支撑面/桌面——它们在插指高度之下，不构成楔止）。防楔
            # 背景：偏置落点 lateral 贴障碍棱时插指卡死（spatial:4
            # 实证，descend 楔在 1.137 磨停 8/8）。
            # z 带 = 插指走廊本体 [抓取点 z, +0.30]：shaft 从 hover 下降
            # 到 p.z 为止，永不达 p.z 以下——支撑面顶低于 p.z 即几何上
            # 不可能阻挡（spatial:1 实证：z_delta 学到 0.045 后旧式
            # z_top−z_delta−0.02  slack 带漏进桌面 AABB，全方向 clr=0
            # 退化成 y_dirs[0] 默认朝障碍侧）。slack 与 z_delta 脱钩：
            # 过滤 operand 用走廊端点 p.z（物理量），不含任意余量。
            # 容纳谓词：障碍 AABB 在 xy 上包含目标中心、z 范围与目标
            # 相交 = 目标是坐/立在该家具（柜架/容器）之上或之内——其
            # 本体是支撑而非楔止障碍。AABB 把开放内部算成实体，参与
            # 净空只会把两侧都打成"嵌深 0"使方向选择退化成抽签
            # （spatial:4 实证：碗坐柜顶，柜框 footprint 含碗心，±y 柱
            # 全嵌，抽签朝柜深撞柜帮；绕入侧由云搜索的 shaft 走廊净空
            # 对真实柜面点云把关，带内柜面点密集）。
            def _clr(p):
                try:
                    obs_names = getattr(env, "obstacle_bodies", None)
                    if not (obs_names and hasattr(env, "object_bounds")):
                        return 1.0
                    z_low = float(p[2])
                    z_high = float(p[2] + 0.30)
                    tc = b["center"]
                    c = 1.0
                    for nm in obs_names:
                        if nm == body_name:
                            continue
                        try:
                            ob = env.object_bounds(nm)
                        except Exception:
                            continue
                        hx, hy = float(ob["half_x"]), float(ob["half_y"])
                        if hx <= 0.0 or hy <= 0.0:
                            continue
                        cc = ob["center"]
                        if (float(cc[0]) - hx <= float(tc[0])
                                <= float(cc[0]) + hx
                                and float(cc[1]) - hy <= float(tc[1])
                                <= float(cc[1]) + hy
                                and float(ob["z_bottom"]) <= float(b["z_top"])
                                and float(ob["z_top"]) >= float(b["z_bottom"])):
                            continue        # 容纳体（支撑），非楔止障碍
                        if float(ob["z_top"]) < z_low or \
                                float(ob["z_bottom"]) > z_high:
                            continue
                        dx = max(float(cc[0]) - hx - p[0], 0.0,
                                 p[0] - float(cc[0]) - hx)
                        dy = max(float(cc[1]) - hy - p[1], 0.0,
                                 p[1] - float(cc[1]) - hy)
                        d = float(np.hypot(dx, dy))
                        if d == 0.0:
                            # 抓取中心在障碍 AABB 内：带符号距离（出口距
                            # 离取负）。目标坐落于障碍顶面/开口内时，候选
                            # 柱都落在 footprint 内，旧式返回 0 使方向选择
                            # 退化成 y_dirs[0] 抽签（spatial:4 实证：碗坐
                            # 柜顶，-y 柱嵌深撞柜帮、+y 柱近柜口可下）；带
                            # 符号后"嵌得深"连续排在"靠近开口/边缘"之后，
                            # 绕入方向可辨，无需特判支撑面。
                            d = -min(
                                float(p[0]) - (float(cc[0]) - hx),
                                (float(cc[0]) + hx) - float(p[0]),
                                float(p[1]) - (float(cc[1]) - hy),
                                (float(cc[1]) + hy) - float(p[1]))
                        if d < c:
                            c = d
                    return c
                except Exception:
                    return 1.0

            def _clr_12dir_search(y_res):
                # 两侧净空都不足（柜塔夹缝里的碗）：12 方向按中心点
                # AABB 带符号净空搜最大（旧语义，距离带符号后嵌深
                # 连续可辨，见 _clr 注记）。
                best_dir, best_off, best_clr = y_res[0][0], y_res[0][1], \
                    y_res[0][2]
                for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    dd = np.array([np.cos(ang), np.sin(ang)])
                    off = _off_for(dd)
                    cl = _clr(_mk(dd, off))
                    if cl > best_clr + 1e-6:
                        best_dir, best_off, best_clr = dd, off, cl
                return best_dir, best_off

            u = _separation_axis()
            y_dirs = [u, -u]
            y_res = []
            for d in y_dirs:
                off = _off_for(d)
                y_res.append((d, off, _clr(_mk(d, off))))
            if max(t[2] for t in y_res) >= 0.02:
                direction, off = y_res[int(np.argmax([t[2] for t in y_res]))][:2]
            else:
                direction, off = _clr_12dir_search(y_res)
            pos = _mk(direction, off)
            cands.append({"position": pos, "score": 1.0,
                          "rel_offset": [0.0, 0.0, 0.0],
                          "score_band": "high", "source": "graspnet_dir"})
        else:
            # 实心物体：GraspNet position 的 xy 用，但 z 必须取物体顶面
            # 附近（GraspNet/几何法回质心=物体几何中心，薄物体中心 z 太
            # 低，手臂下不去 → contact_stop 让夹爪在物体上方闭合夹空）。
            thickness = float(b["z_top"] - b["z_bottom"])
            # z 推导单一来源（physics/derives）：实心 z-override + 极薄物
            # 包边（goal:5 实证：指尖降到盘顶下 2mm 闭合包沿，防压盘底
            # 以下 lift F=0）。
            from ..physics.derives import grasp_z_flat
            grasp_z = grasp_z_flat(thickness, float(b["z_top"]))
            # 高而规则的物体（橙汁盒/罐头，实测厚度>0.06）：几何中心候选
            # 置顶——夹爪对中夹住两侧平面最稳。GraspNet 候选 xy 常偏离
            # 中心 3-5cm，descend 后指尖在物体外侧闭合夹空气（F≈0），
            # lift 必判 lift_no_grip（libero_object:9：中心 F=53N 一次
            # 夹住，偏心 6/6 全夹空）。
            # 扁平物体（cream_cheese 厚 0.041）除外：对中直降两指尖会对称
            # 压在盒顶被架住，接触软停高位闭合夹空；GraspNet 的偏心点
            # （+jit 逐 attempt 换方向）让一指先越沿、指尖滑到盒侧中下
            # 部再闭合（object:1 实测偏心 19mm/end_z=0.014 成功）。
            _center_thr = float(os.environ.get("DARWIN_CENTER_THRESH", "0.06"))
            if thickness >= _center_thr:
                cands.append({
                    "position": [float(cx), float(cy), grasp_z],
                    "score": 1.0, "rel_offset": [0.0, 0.0, 0.0],
                    "score_band": "high", "source": "bounds_center"})
            for c in all_cands:
                w = float(c.get("width", 0.0))
                if w > max_width:
                    continue
                pos = c["position"]
                if not _sanity_xy(pos):
                    continue  # 离中心超过物体尺寸的候选必假
                score = float(c.get("score", 0.5))
                cands.append({
                    "position": [float(pos[0]), float(pos[1]), grasp_z],
                    "score": score,
                    "rel_offset": [0.0, 0.0, 0.0],
                    "score_band": "high" if score > 0.5 else "mid",
                })
        if not cands:
            pt = env.grasp_point(body_name)
            cands.append({"position": [float(x) for x in pt], "score": 1.0,
                          "rel_offset": [0.0, 0.0, 0.0], "score_band": "high"})
        return {"feat": feat, "candidates": cands}


register_policy("grasp_pose", "rule", RuleGraspPoseLibero(), default=True)
