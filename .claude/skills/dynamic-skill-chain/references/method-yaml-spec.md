# YAML 方法规格（method-yaml-spec）

本文是 SKILL.md 的按需参考：YAML 方法的完整 schema、$模板变量、可用条件类型、
可用 skill 名录与任务声明示例。所有代码位置以仓库根目录为基准。

## 1. YAML 方法 schema

```yaml
name: string                 # 必填，kebab/snake 均可，必须与文件名一致；
                             # 与内置方法同名 = 覆盖其执行函数
description: string          # 可选，chainctl show 展示
match:                       # 必填，声明本方法能达成哪类目标条件
  kind: string               #   joint_ge | body_lifted | body_near_site
  flavor: string             #   仅 body_near_site 需要：cart | pose
include_home: false          # 可选，链首是否自带 home；
                             # false 时引擎在"第一个 chunk"按 flavor 自动补
                             # home(pose flavor 补 pose_home)
steps:                       # 必填，非空有序列表
  - skill: string            #   skill 注册名（见第 4 节）
    params: { ... }          #   可选，键值支持 $模板（见第 2 节）
```

文件放 `darwin/agents/methods/<name>.yaml`，进程启动扫描；`examples/` 子目录与
`*.disabled` 不加载。安装/校验一律走 `scripts/chainctl.py`，不要手拷后忘记 validate。

## 2. $模板变量（白名单，无 eval）

params 的字符串值若以 `$` 开头，按整值替换（保留类型：float/list/str）；
list/dict 会递归渲染元素；普通数字/字符串原样。支持 `.x/.y/.z` 取三维向量分量。

| 写法 | 类型 | 含义 |
|------|------|------|
| `$safe_z` | float | 安全走廊高度 0.62（collision.SAFE_Z） |
| `$grasp_pt` `$grasp_pt.x/.y/.z` | list3/float | 当前抓取候选点（UCB/RAG 选出） |
| `$site:<name>` `$site:<name>.x` | list3/float | 规划当下 `env.get_site_pos(name)`，如 `$site:drawer` |
| `$goal_site` `.x/.y/.z` | list3/float | 条件自身 site 的现场坐标（BodyNearSite.site） |
| `$cond.<attr>` | any | 条件对象属性（见第 3 节属性表） |
| `$cfg.<key>` | any | EpisodeRunner.run(cfg) 的搜索空间字段（hover/k_descend/lift_height/stiffness/...） |
| `$entry.<key>` | any | benchmark entry 字段（task_name/body/actor/...） |

自动注入、**不要写**在 params 里的键：`actor`、`grip_site`（来自 entry）。
需要新变量：在 `darwin/agents/chain_registry.py` 的 `resolve_var()` 加分支，
不要在 YAML 里发明表达式。

典型渲染：

```yaml
target: ["$site:drawer.x", 0.0, "$safe_z"]   # -> [0.37, 0.0, 0.62]
point: "$grasp_pt"                            # -> [x, y, z] 整个 list
joint_name: "$cond.joint"                     # -> "drawer:joint"
stop_above: "$cond.stop_above"                # -> 0.025 (float)
```

## 3. 目标条件类型（GoalCond）

定义在 `darwin/agents/runner_dynamic.py`；判定方法都是 `check(env)`。

### joint_ge — JointAtLeast

滑动/转动关节达到阈值。属性：`joint: str`、`threshold: float`、
`target_qpos: float`（执行目标，可大于阈值）、`handle_site: str|None`。

声明：

```yaml
- kind: joint_ge
  joint: "drawer:joint"
  threshold: 0.08
  target_qpos: 0.12
  handle_site: "drawer"
```

### body_lifted — BodyLifted

物体被抬升到 z 高度以上（grasp-only 成功条件）。属性：`body`、`height`。

```yaml
- kind: body_lifted
  body: "green_block"
  height: 0.52
```

### body_near_site — BodyNearSite

物体进入目标 site 邻域。属性：`body`、`site`、`tol`、`flavor`（cart/pose）、
`stop_above`、`place_timeout`、`place_k`。check 用物体与 site 的三维距离 < tol。

```yaml
- kind: body_near_site
  body: "green_block"
  site: "cube_goal"
  tol: 0.05
  flavor: cart
  stop_above: 0.025
  place_timeout: 300
  place_k: 2.0
```

## 4. 可用 skill 名录（steps[].skill）

笛卡尔链（flavor=cart / 默认），darwin/skills/primitives/__init__.py：

| skill | 关键参数 | 说明 |
|-------|----------|------|
| `home` | timeout, k | 回安全 home（xy 判据） |
| `move_above` | point[3], hover, tol_xy, k | 到点上方悬停，走廊避障 |
| `descend` | point[3], body, stop_above, k | 竖直下降；有 body 时目标 z=body.z+stop_above |
| `close_gripper` | steps | 闭合夹爪 |
| `open_gripper` | steps | 张开夹爪 |
| `lift` | height, body, max_steps, k | 竖直抬起并验证确实抓住 |
| `move_to` | target[3], gripper, tol, k | 通用移动（保持夹爪状态），走廊避障 |
| `move_to_xy_top` | target[xy2], height, tol_xy | 搬到目标 xy 正上方 |
| `place` | goal[3], body, hold, tol, timeout, k | 下放到 goal 并判物体稳定 |
| `pull_drawer` | joint_name, target_qpos, handle_site, standoff_back, standoff_lift | 拉抽屉：先 standoff 走廊避让再驱动关节 |

位姿/力控链（flavor=pose），darwin/skills/primitives/control.py：
`pose_home` `pose_move_above` `pose_descend` `pose_close_gripper` `pose_lift`
`pose_move_to` `servo_align`(target={type:pose,name}) `guarded_move`
(direction, until.force_n, max_travel_m, speed_mps) `impedance_push`
(axis, until.force_n/depth_m, stiffness, damping) `spiral_search`
(radius, until=depth_reached) `pose_open_gripper`。

mplib backend 路线（需 profile 注入），motion.py：`path_plan`、`collision_check`。
无 backend 时返回 planner_unavailable，不要在 drawer_place 这类任务里依赖。

所有位移原语都返回 `min_clearance` / `coll_pair`；clearance < 4mm 中止并返回
reason=collision_risk。

## 5. 完整示例

darwin/agents/methods/examples/open_drawer.yaml（match joint_ge，链自带 home）：

```yaml
name: open_drawer
match: {kind: joint_ge}
include_home: true
steps:
  - skill: home
  - skill: move_to
    params:
      target: ["$site:drawer.x", 0.0, "$safe_z"]
      gripper: 0.0
  - skill: pull_drawer
    params:
      joint_name: "$cond.joint"
      target_qpos: "$cond.target_qpos"
      handle_site: "$cond.handle_site"
```

darwin/agents/methods/examples/transfer_cart.yaml（match body_near_site/cart，
move_above→descend→close→lift→xy_top→place→open，目标坐标全部现场渲染）。

## 6. entry 如何声明任务（新任务不改执行端代码）

benchmark dict（darwin/benchmarks/__init__.py）里写 objectives：

```python
"objectives": [
    {"kind": "joint_ge", "joint": "drawer:joint", "threshold": 0.08,
     "target_qpos": 0.12, "handle_site": "drawer"},
    {"kind": "body_near_site", "body": "green_block", "site": "cube_goal",
     "tol": 0.05, "flavor": "cart", "stop_above": 0.025,
     "place_timeout": 300, "place_k": 2.0},
],
"obstacles": ["cupboard", "drawer"],      # 可选：碰撞监测/走廊避障的障碍 body
"safe_home_qpos": [-0.614, -0.2586, ...], # 可选：reset 后覆盖构型并 hold 收敛
```

没有 objectives 字段时，按旧 mode（drawer/insert/full/grasp）自动适配等价条件。

## 7. 常见错误

| 现象（chainctl/probe） | 原因与处理 |
|------------------------|------------|
| validate: 缺少 match.kind / steps | YAML 顶层字段缺失或缩进错误 |
| TemplateError 未知模板变量 | 只允许第 2 节白名单；site 名拼错用 chainctl show 核 |
| TemplateError 条件没有属性 x | 该 cond.kind 不含此属性，换 $cfg/$entry 或给条件加字段 |
| probe: no method registered | match.kind/flavor 与条件不匹配；`chainctl list` 核对 |
| probe: skill unknown | steps 里 skill 名不在第 4 节名录（load_skills 加载失败时也会） |
| add 后行为没变 | 注册表启动时扫描；必须新起进程；用 list 确认 source 指向 yaml |
| collision_risk 中止 | 看 skill_end.coll_pair/min_clearance；优先调 $safe_z 走廊与 standoff |
