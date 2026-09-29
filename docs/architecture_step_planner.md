# StepPlanner 架构实施 TODO

> 本文档 = 执行清单 + 压缩知识库。完整设计推演与调研过程见
> `docs/archive/design_full_20260923.md`（历史档案，只读，以本文为准）。
> 每个 TODO 自包含：动作 / 涉及文件 / 验收 / 依据（附录）。
> 设计总纲一句话：**系统 = 一条认知循环 × 四个注册表**（附录 A），
> 机器人相对自动驾驶的结构性优势（低速可重试）必须被架构用足。

更新时间：2026-09-23 深夜 ｜ 当前基线：**24/30**（spatial+goal+object，
8 attempts 内通过；r10 批跑验证中，目标 30/30）

---

## 1. 当前状态

| 套件 | 通过 | 剩余失败 |
|---|---|---|
| libero_spatial | 9/10 | :4（descend 慢型 stall）、:9（place slipped） |
| libero_goal | 8/10 | :2（place slipped 酒瓶）、:3/:4（descend 余量边缘）、:5（plate 夹空） |
| libero_object | 10/10 | — |
| libero_10 | 未攻 | 全部 10 个（归因方法已验证，待跑一轮收集 fail_phase） |

r10 批跑（jit 归零 + F=0 判别子先行版 + 非累积重试序列）验证 7 任务中。
最大失败簇 = **place 族**（:9/:2/:8 滑脱/超时），刀已备好：滑移方向
判别子（附录 B 表 3）→ 摩擦锥割，不再盲试 place_vcap。

---

## 2. 实施 TODO

### P0 物理认知引擎（下一个主攻，顺序即依赖序）

- [x] **P0-1 skill evidence 字段** ✓ 2026-09-23
  - 动作：所有 skill 的返回 dict 增加 `evidence: {f_trace, z_trace,
    z_rate, xy_drift_trace, clearance_min}`；episode jsonl 的 skill_end
    事件落盘 evidence。成功与失败同样进学习信号，遥测零浪费。
  - 涉及：`darwin/skills/primitives/*.py`、`darwin/agents/runner_dynamic.py`
    日志管线
  - 验收：episode jsonl 每个 skill_end 含非空 evidence；现有 24 个通过
    任务 verify_30 回归不劣化
  - 依据：附录 A 循环的 observe 步、附录 B 表 3/4 的观测量清单

- [x] **P0-2 `darwin/physics/` 四合一知识引擎** ✓ 2026-09-23（六模块落地；reflection/methods/place 内置推导的收敛部分完成）
  - 动作：新建模块，内含四张表（约束判据+推导 / 割算子 / 判别子 /
    探针）+ 共享观测量基础设施（复用 `object_bounds`、
    `contact_force_on_body`）。把散落的同一物理推导收敛为单一来源：
    - `policies/rules.py`（straddle/z-override/包边——孤岛副本，改引用）
    - `agents/methods.py:324-333`（高位下降避让，残留场景坐标
      "微波炉 rim1.03"，附录 D 规则 1 违反者）
    - `skills/primitives/__init__.py:449-528`（place 物理默认値埋在
      skill 内部）
    - `agents/reflection.py:103-309`（十几条盲方向 if-elif，z_delta
      被推到 0.042 物理无意义值的元凶；降级为"ParamSpace 内扰动
      经验残差"，防漂移补丁随之删除）
  - 验收（四个现存病例同轮痊愈，跑 verify_30）：spatial:4（可达探针/
    预算推导）、spatial:9/goal:2（滑移判别子→摩擦锥割）、goal:3/4
    （接触探针实测 band，不靠 jit=0 手工清零）、z_delta 污染免疫
    （Θ 后验替代裸 YAML 值）
  - 依据：附录 B 全部四表、附录 D

- [ ] **P0-3 运行时约束监视器**
  - 动作：执行层每 N 步对 active constraints 求值（输入 P0-1 evidence
    的实时流），violation 即刻：① abort（不等 timeout）② 打包
    violation record ③ 走割算子更新 Θ 后验 ④ 触发换参重试。
  - 涉及：`darwin/physics/`（P0-2 同模块，监视器是表的第三列）、
    `runner_dynamic._execute_chunk`
  - 验收：人工注错场景（目标不可达 / μ 过低）在首个停滞窗口被截获
    （<60 步），而非跑满 120~400 步 timeout；学习信号从二元变连续
  - 依据：附录 B 表 3、§15.3

- [x] **P0-4 探针四件套** ✅ r11（2026-09-23）
  - 慢降接触探针 + 可达探针落地：`ik_servo.descend` 全量 evidence 采样
    （F/速率/xy/z trace），reach_limit 接受时短缩写进 Θ 后验
    （spatial:2/9 实测 reach_shortfall 0.15→0.0443→0.0367 真实收敛）；
    `physics/probes.py`（contact/reach/friction）就位。
  - 验收达成：`contact_stop_band` 已从全部 39 份 task YAML 的 params 区
    删除（r12，运行时实测 > 默认推导 两级来源接管）；
    agent_learner streak 回退的标定值例外同步删除。
  - 残留：摩擦探针的独立主动调度（当前为 f_hold 被动采样）。

- [x] **P0-5 重试采样器：假设栈 + 判别子驱动** ✅ r12（2026-09-23）
  - 已退役：`_DESCEND_SEQ/_LIFT_SEQ/_CARRY_SEQ/_PLACE_SEQ` 四张手工表
    与 `_apply`，机制驱动（retry.py `_mechanism_params`）为唯一物理
    方向来源；机制未知时回落"通用假设阶梯"（被挡 k/vcap 递减 × 慢
    timeout×2→6 交替——两假设竞争的最便宜判别序，非 per-action 表）。
  - 已退役：reflection.py 盲方向 if-elif（grip_failed 整块、timeout 的
    k_descend、collision、other/unknown"保守加深逼近"≈200 行）；
    attempt 级物理方向改由机制表 `_mechanism_adapt` 驱动（与步级同向），
    全链接线：runner attempt result → sim_worker → agent_learner →
    adapt_cfg(mechanism=…)。
  - 已退役：methods.py carry 高度 magic number（0.015/0.06 →
    derives.carry_lift_need 具名常数）；`_carry_timeout` 收敛到
    derives.carry_timeout 单一来源；PARAM_SPEC jit 默认 0.005→0.0
    （精密几何禁抖，探索须显式开）。
  - 保留（约束层非知识负债）：anchor 漂移裁剪、xy 滞环、stall 绕障
    余量、force_exceed 力控、遥测分流。
  - 待续：jit 判别子护航（仅"候选点偏差"假设 + 几何余量 2×jit 时
    允许）未做；实验选择"最便宜二分判别子"未做。

- [ ] **P0-6 YAML 从答案表到约束声明**
  - 动作：参数三要素声明化（推导式 derive / 输入特征 from / 物理夹紧
    域 clamp，§11.5 格式）；旧答案表包装成 `derive: const(x)` 过渡，
    随 champion-challenger 数据逐个替换真推导式；新 YAML 只写声明。
  - 验收：LLM/编码 agent 只读 YAML + 场景特征可现场反推全部参数
    （拿 spatial:4 做盲测：不给历史值，agent 推 timeout 与实际
    需求同阶）
  - 依据：附录 D 规则、§11.5

### P1 L2 显式化（跨臂，"机器人无关"的最硬实证）

- [ ] **P1-1 RobotAdapter 接口 + 双实现**
  - 动作：`gripper_geometry()`（指距/指宽/行程）、`ik_reachable(pose)`、
    `workspace_project(L1 可行集)`；libero_panda 与 robopal 各一实现。
  - 验收：接口单测；straddle 从"注释里的 80mm"变为
    `f(finger_span, rim_radius)` 且在两臂各自正确
- [ ] **P1-2 跨臂对照实验**
  - 动作：同一 L1 约束集 + 同一 Θ 后验跑两条臂，仅 L2 投影不同。
  - 验收：同一批抓放任务两臂成功率差 < 10%；L1/L0 代码零分叉

### P2 记忆与迁移

- [ ] **P2-1 EpisodeMemory**（`memory/episodic.py`，与 rag.py 平行）
  - 动作：键 = 物体中心特征（附录 D 规则 3，禁任务 ID）；存 Θ 后验 +
    签名→割的似然统计，**不存成功参数**；episode jsonl 为初始数据源。
  - 验收：同物体跨套件检索命中（akita_black_bowl 在 spatial/goal 的
    μ̂/到达极限后验互通）；LIBERO-10 陌生任务 day-0 有先验
- [ ] **P2-2 迁移三段式验收**
  - 动作：实例化（0 次）→ Θ 残差检索（0 次）→ 在线辨识（≤k 次）全流程
    在 LIBERO-10 跑通。
  - 验收：陌生任务个位数尝试内通过（预算论证见 §16.4：探针与首
    attempt 执行重叠，O(log H) 假设排除）；首个成功即冻结 Θ 写回
    记忆库（带置信度）

### P3 articulate 约束化 + runner 瘦身

- [ ] **P3-1 articulate 升级**
  - 动作：`articulation_info`（libero_adapter.py:274）已暴露关节轴/限位
    → articulate 从 demo 回放升级为"关节轴+限位+把手可达"约束声明
    （L1 表第 6 行），demo 退为 fallback。
  - 验收：换柜体布局（挪把手/换限位）不重录 demo 仍通过；原布局回归
    不劣化
- [ ] **P3-2 runner 瘦身**
  - 动作：`agents/runner_dynamic.py`（1078 行）瘦身为 StepExecutor 驱动
    壳，规划逻辑全部进 policies/physics。
  - 验收：行数减半；verify_30 不劣化

### P4 能力层（CAPABILITY_REGISTRY）

- [ ] **P4-1 注册表骨架 + fallback audit（只观测）**
  - 动作：`CAPABILITY_REGISTRY`（detect/segment/grasp_pose/depth/pose6d/
    open_vocab/vlm_scene），skill 声明能力由注册表统一供给，禁止私下
    torch.load；fallback 率按任务统计进回归。
  - 验收：audit 报告产出；静默回退清零（goal:5 教训：候选源降级必须
    计数告警）；`grasp_pose` 候选源 champion-challenger 进 POLICY_REGISTRY
- [ ] **P4-2 接 yolo26n-depth.pt**（已下载零引用，最划算）
- [ ] **P4-3 开词汇检测**（Grounding DINO / OWL-ViT；YOLO COCO 闭集一换
  物体就瞎，真机第一步）
- [ ] **P4-4 depth/pose6d 的 GT fallback 加 audit 告警**（上真机 readiness
  硬指标：fallback 率归零才允许切真机相机）
- [ ] **P4-5 RAG embedding 从 TF-IDF 切本地小模型**

### P5 编码 Agent 接入

- [ ] **P5-1 统一 headless 执行器** `dispatch(任务卡) → {结论, diff, 验收}`
  （包 codex/claude/kimi 非交互模式，可插拔，与 POLICY_REGISTRY 同型）
- [ ] **P5-2 任务卡模板库 + 失败升级归因**（复用 §3 归因分类：不可达/
  信息不足/验收不清；归因写回模板库）
- [ ] **P5-3 验收**：新 agent 空上下文只读 docs/ 完成一次调参任务；主循环
  token < 亲自做的 20%

### P6 语义层

- [ ] **P6-1 LLM 归因语言化**（辅助人工分析与 skill 库扩展决策）
- [ ] **P6-2 矛盾升级归因分析接口**（Θ 后验交集为空时，LLM 读任务卡
  做归因，§15.6）

---

## 3. 已完成日志（2026-09-23）

- [x] **P0 物理认知引擎主体落地**（深夜，r11 批跑验证中）：
  `darwin/physics/` 六模块（observables=证据采集 / discriminators=表3
  机制判别 / posterior=Θ 半空间后验 / cuts=表2 割算子 / derives=参数
  推导 / probes=表4 主动探针 / constraints=表1 判据库）。全链闭环：
  skill 产 evidence → sim 侧判别机制+算割 → result 回 agent 侧 →
  Θ 后验置信度加权求交落 YAML `theta:` 段 → 下次执行 band/预算从
  实测推导。skill_config 新增 theta 段（跨进程持久化）；retry 采样器
  机制驱动（budget_short→加时 / contact_blocked→更软更慢 /
  friction_slip→降速压稳 / ik_unreachable→增益上调），未知机制回落
  原序列；rules.py 的 straddle/包边推导收敛到 derives 单一来源。
  P0-3 运行时监视器（逐步求值 abort）与 P0-6（YAML 全声明化）未完工。

- [x] **P0-6 上半场：旧代码退役**（r12 批跑验证中）：retry 手工序列表
  （_DESCEND_SEQ 等 4 表 + _apply）删除，机制驱动为唯一物理方向来源，
  未知机制回落通用假设阶梯（被挡/慢两假设竞争序）；reflection.py 盲
  方向 if-elif 删除约 200 行（grip_failed 整块 / timeout 的 k_descend /
  collision / other/unknown），attempt 级改由机制表驱动且与步级同向；
  attempt result → sim_worker → agent_learner 全链接线 mechanism；
  39 份 task YAML 的 contact_stop_band 标定值删除；methods.py carry
  高度 magic number（0.015/0.06）收敛到 derives.carry_lift_need；
  jit 默认 0.0。P0-6 下半场（YAML 全声明化 derive/from/clamp）未开工。

- [x] **L1 判据库扩四条**（r13 并行，对准 r12 失败簇的主导机制）：
  `reach_feasible`（ik_unreachable 判据化，×Θ reach_shortfall——goal:3/4/5
  共 118 次的空转病因）、`grip_force_need_met`（close 后即判夹持力，
  lift_no_grip 提前可见——goal:5 ×24）、`insertion_clearance_ok`
  （peg-in-hole 间隙 < 感知容差 → 路由 transfer_pose 螺旋——goal:2 瓶架）、
  `approach_corridor_ok`（悬停→抓点走廊 clearance，descend xy_drift 判据化
  ——goal:3/4；采样须奇数否则漏中点，自测实证）。active_set 全部接线。
  待接线：判据 → 规划时否决/重路由（当前为库+步边界 force_closure 预警）。

- [x] **关节约束类**（`physics/articulation.py`，~230 行纯函数）：
  `JointDecl` 参数化声明（jtype/axis/point/q_range/handle_body）+
  `extract_joint_decl` 从 mj_model 白拿（jnt_type/axis/range/qposadr，
  零感知零学习）+ 三判据：`on_manifold_track`（拉歪=物体不在 h(q) 像上，
  MANIFOLD_TOL 15mm）、`within_limits`、`aligned_wrench`（力旋量轴对齐，
  参考点取作用点在轴上的投影——τ·axis 是不变量；径向对轴推 τ≡0 是
  零驱动不是对齐，退化分支自测实证）+ `mechanism_of_violation`（拉脱=
  friction_slip/卡死=contact_blocked/拉歪限位=geometry_squeeze，零新枚举）
  + articulate active_set。一种声明覆盖柜门/抽屉/翻盖/旋钮（参数实例≠
  类实例）。边界：已知轴+限位+单链刚体；待验收=LIBERO-10 关节任务。

- [x] **泛化方案：候选约束排序 + directive 执行器**（r14 验证中）：
  四场失败（goal:2/3/4/5）共同结构=死因在候选空间不在参数空间，而
  候选排序从不看物理约束（中心优先=几何偏好）。① `_constraint_rank`：
  候选生成（policies grasp_pose，libero+非 libero 两路径都接）后过
  L1 下降走廊检查（其他物体 AABB：z 重叠 + xy 距离 < 指半径+margin
  则降权，中心优先退居 tie-break）——goal:3/4 的 118 次 ik_unreachable
  是下降途中被楔偏，端点可达≠路径可达。② 黑名单=directive"换候选"的
  执行器：机制 ∈ {no_grip_air, geometry_squeeze, contact_blocked,
  ik_unreachable} 且失败在抓取相时，该候选 xy 进会话黑名单（IPC：agent
  写 sock 目录 JSON、sim 只读每 attempt 重读；进程内：内存集）。门控
  防误杀：place 相失败（goal:2/8 滑脱）不拉黑抓点。两个自测实证坑：
  libero rel_offset 恒 [0,0,0]（候选身份必须用 position xy）；盒内
  clearance 符号（边界距离=较近两维穿透深度，max 非 min）。

- [x] **18/30 → 24/30**：z_top 角点变换（mesh 顶点/旋转盒 8 角点）、
  straddle 纯 Y 偏置 + 净空翻向、容器 band 收紧 0.015、反思死亡螺旋
  硬守卫（k_descend≤6、hover≥0.06、z_delta≥0.04）、cfg 重置法
- [x] **Phase A 骨架**（行为不变）：`darwin/policies/`
  context/spaces/registry/rules；grasp_pose 规则从 runner 搬进注册表，
  runner 改调 `propose("grasp_pose", ctx)`
- [x] **Phase B 步级快照重试**：`_execute_chunk` 每步 save_physics，
  失败 restore + 换参重试 ≤5；白名单含 ik_servo；resume 机制保留
- [x] **非累积重试序列**（retry.py）：timeout ×1.5/2/3/4.5 绝对倍数，
  替代幂等叠加
- [x] **F=0 到达极限判别子先行版**（ik_servo.py descend）：双窗口停滞
  + 接触力≈0 + z 在 0.035 内 + xy 收敛 → 判成功（reach_limit）；
  F>0 仍走原接触软停/失败路径——§16.2 第一行的手工实现
- [x] **jit 归零**（goal:3/4/5、spatial_4）：精密几何容差 < 2×jit 时
  抖动物理不可行——§16.2 教训条目的手工执行
- [x] **架构文档**：§14 分层约束（L0/L1/L2）→ §15 对偶割学习 →
  §16 认知循环（本文附录 A/B 的完整版在 archive）

---

## 附录 A 认知循环与设计原则（压缩版）

```
hypothesize → experiment → observe → conclude → execute
  假设栈      实验选择     判别子求值   割/测量/成功  可行集重采样
```

- 系统 = 一条循环 × 四个注册表：约束（表1）/ 割（表2）/ 判别子（表3）/
  探针（表4）。表行是知识（可沉淀迁移），循环代码是水管（一次写好不动），
  新失败模式 = 新一行，不写新 if-else。
- 小模型插口：感知→判别子求值器；物理/几何→约束推导行；策略→假设排序；
  LLM→矛盾归因。**每格有非学习兜底**（解析推导或一次探针）——这就是
  "泛化边界拓展到小模型边界"。
- 知识单位 = 约束实例 (判据, 未知标量 Θ, 裕度)。只有 Θ 里的标量需要学
  （μ、质量、质心偏差、感知偏置，通常 k≤3~5），其余解析推导。
- 学习速度账：现状反思 O(∏log range)（盲二分，死亡螺旋根源）→ 对偶割
  O(k) 次信息量充分的失败 → +探针直测 +判别子切假设 = O(log H) 次实验。
  陌生任务预算 ≈ 实例化(0) + 检索(0) + ≤3 探针(多与首 attempt 重叠)
  + ≤2 假设排除 + 1 验证。
- 四条设计原则：①泛化 = 小模型边界的并集；②脚链 = 约束层不是策略层
  （物理/安全/预算约束对一切策略生效）；③已验证的东西全部保留
  （新架构包裹旧系统，rule = fallback）；④重试是一等公民（机器人撞了
  = 教学样本，自动驾驶撞了 = 事故）。

## 附录 B 四张知识表

### 表 1 约束全集（L1，机器人无关；✔=2026-09-23 已实证用过）

| 动作 | 约束 | 失效边界（诚实标注） |
|---|---|---|
| 抓 ✔ | 力封闭/形封闭：grasp wrench space 含重力螺旋；摩擦锥内可施夹持力 | 软体/可变形；μ 未知只能给裕度版 |
| 放/堆 ✔ | 质心投影 ∈ 支撑多边形；放置面共面 | 质心不可纯几何推出（需在线辨识/记忆库） |
| 推 | 稳定推动条件（Lynch & Mason：推点/摩擦角/前向锥） | 多接触推挤（深堆 clutter）退化需搜索 |
| 插 | 间隙比 + 倒角捕获：轴对正 ∧ 间隙>姿态误差 → 力控收敛 | 深孔长径比过大 → 多段对准（同型叠加） |
| 笼取 | 全部逃逸方向被几何封闭（caging） | — |
| 开关节体 ✔ | 关节轴方向 + 限位 + 把手可达点 + 拉力沿轴 | 多关节链逐关节串接同一约束 |

覆盖边界 = 刚体（LIBERO 30 + LIBERO-10 抓放推插开柜全落表内）。
布料/倒水等可变形/流体不在内（另一类模型，不进第一阶段）。
约束保证**可行/安全**不保证**成功**：{约束满足} ⊂ {任务成功}。

### 表 2 违背 → 对偶割算子（reflection 的替代物）

| 遥测签名 | 违背的约束 | 对偶割（写进 Θ 后验的不等式） |
|---|---|---|
| lift 时物体下滑 | 摩擦锥 | μ < m·g/F_grip（spatial:9/goal:2 的直接读法） |
| descend 零接触匀速停 | 时间预算 | timeout ≥ 距离/实测速率×1.3（spatial:4） |
| descend 接触后停 + xy 漂 | 几何干涉 | 抓取点移出障碍 AABB⊕margin |
| lift_no_grip F=0 | 力封闭 | 抓取构型 ∉ 封闭集 → 换候选点（几何推导，无 Θ） |
| place 后倾倒 | 支撑多边形 | 放置点修正；m̂/质心偏差进 Θ |
| place_timeout TCP 到不了 | 可达性 | 目标 ∉ 工作空间 → 换路径点（几何，无 Θ） |

每条割带置信度（签名越唯一越高），低置信软更新；签名二义给 competing
cuts（如 stall：慢 vs 挡，用 z 轨迹匀速性裁决）。

### 表 3 判别子（比约束稀缺的知识；每行有 2026-09-23 实证病例）

| 判别子 | 二分 | 病例 |
|---|---|---|
| contact_force(body) ≈0 vs >阈值 | 自由空间停滞（到达极限，可接受）vs 几何阻挡（真问题） | spatial:4 F=0 停 22mm → 接受；goal:3/4 沿口 F>0 → 修几何 |
| z_trace 匀速>0 vs 恒为 0 | 预算不足（加时有效）vs 真卡死（加时无效） | spatial:4（0.3mm/步）vs spatial:2 X 向蹭沿——同为 stall 对策相反，r9 烧错 5 次重试的教训 |
| lift 时 body_z 跟随 TCP_z? | 夹持成功 vs 夹空 | 全部 lift_no_grip 第一刀 |
| 滑移方向 vs 重力方向夹角 | 摩擦锥违背（加力/降速）vs 几何挤压（修放置点） | spatial:9/goal:2 place slipped——还没切这刀，place_vcap 才盲试 8/8 无效 |
| clearance 最小对的 pair_name | 撞的是哪个障碍 | goal:3/4 above 卡柜顶 vs 卡桌沿，绕障侧不同 |
| xy 漂移方向的时间一致性 | OSC 构型漂移（可补偿）vs 目标点不可达（换点） | goal:3/4 xy_drift |
| 精密几何容差 vs 2×jit | 允许探索抖动 vs 物理不可行必须 jit=0 | goal:3/4/5、spatial:4 容器 straddle |

### 表 4 探针（主动实验，把标定彩票变运行时测量）

| 探针 | 一次实验测什么 | 杀死的彩票 |
|---|---|---|
| 慢降接触（vcap↓ + F 阈值停） | 接触 z 真值 → band=实测短缩+裕度 | contact_stop_band 逐任务标定（0.066/0.108/0.12…） |
| 可达（指令深 z 读最大到达） | L2 到达极限（构型的函数，非常数） | spatial:4 的 22mm 靠人工诊断 |
| 摩擦（定力夹持 + 切向加载） | μ 一次直接测得 | place slipped 反复试错（各烧 8 次） |
| 几何核对（指距 vs 沿口半径） | straddle 可行性 + 最优偏置 | 容器方向投票抽签（09-22 bug 源） |

约束：全部经 Phase B 快照保护（可恢复）、低速小步进；多数探针 = 执行
本身，额外成本≈0。"机器人可以问物理引擎问题"——VLA 做不到（开环答案机）。

## 附录 C 六任务病例档案（hard-code 边界的 6 个标本）

| 任务 | 实证根因 | 已上的手工修复 | 架构正解（TODO 落点） |
|---|---|---|---|
| spatial:4 | descend 慢型 stall（0.3mm/步 × 预算 90 步，需 ~280 步）；F=0 无接触 | timeout 重试 ×3/×4.5；jit=0；F=0 接受 | P0-3 预算推导 + P0-4 可达探针 |
| spatial:9 | grasp→carry 全通，place slipped（摩擦边际） | place_vcap=0.05（未解决） | 表 3 滑移方向判别 → 表 2 摩擦锥割（P0-2/5） |
| goal:2 | 同上（酒瓶，8/8） | 同上 | 同上 |
| goal:3/4 | descend 余量边缘运行：band 0.015 vs OSC 短缩 8~11mm，jit 吃掉最后 3~4mm，stalled/xy_drift 交替 | jit=0 | P0-4 接触探针实测 band（运行时替代手工清零） |
| goal:5 | GraspNet 对 plate 产 0 候选 → 静默回退中心点 → 双指压盘顶夹空 F=0 8/8 | jit=0；包边规则（thickness<0.015） | P4-1 fallback audit + 薄扁物沿口推导进表 1 |
| goal:8 | place_timeout（r9b 回归，resume 机制本身正常） | — | place 族总攻（同 spatial:9 刀） |

提炼的 4 条架构修订：①一切静默 fallback 都是 bug 种子（必须计数+告警）；
②预算参数也是参数（timeout/kd/vcap 进约束声明，不许裸值）；③步级快照
重试是 place 族公共解（Phase B 已与 A 并列最高优先级）；④接近段是独立
决策点（approach 与 descend/carry 同级，各自的泛化边界互不干扰）。

## 附录 D 三层模型与工程规则

```
L0 物理定律（无条件通用）：刚体、摩擦锥 ‖f_t‖≤μ·f_n、法向单向、重力
L1 任务约束（机器人无关 ← 泛化主体）：以"物体×环境"为中心的关系式，
   不含机械臂变量、不含世界坐标
L2 机械臂适配层（唯一与臂有关，必须很薄）：只做投影/过滤——IK 可达、
   关节限位、力矩上限、夹爪几何
```

关键性质：策略可以蠢可以错，但物理上不可能的执行永不发生；L2 换臂即换，
L0/L1 原封不动。工程规则：①约束用物体中心坐标系写禁世界坐标（"沿口
上方 22mm"✔ "z=1.115"✘）；②技能用约束参数化不用轨迹参数化（"下降到
接触"✔ "下降 90 步"✘，轨迹是彩票约束是方程）；③记忆库键 = 物体中心
特征禁任务 ID（"高摩擦圆柱 0.5kg 级"✔ "libero_goal_2 成功参数"✘）。

## 附录 E 能力盘点与模型缺口（2026-09 审计）

真实加载仅 4 模型，多数有真值备胎：YOLO26n（COCO 闭集，回退真值）、
MobileSAM、GraspNet baseline（缺权重回退 PCA）、LLM API（无 key 回退
OfflinePlanner）；深度/6D 位姿全真值（**仿真专用，上真机即失明**）；
yolo26n-depth.pt / sam_b.pt / FastSAM-s.pt 已下载零代码引用。
分层缺口：开词汇检测（高，真机第一步）、深度（高，先接已下载权重）、
VLM 场景（中）、6D 位姿（中）。正确形态 = 每层恰好一个 champion +
注册表可换；模型不是越多越好，每个常驻模型都是显存/延迟/维护成本。
缺模型四步路径：几何兜底 → 换等价基础模型 → LLM 语义近似 → 标记缺口
攒数据事后补。**缺模型不是异常，是 learning loop 的输入。**

## 附录 F 风险登记册

| 风险 | 防线 |
|---|---|
| 小模型 garbage-in-out | 特征用真值几何；rule fallback 兜底；champion-challenger 数据驱动开放插拔 |
| 误分类割/判别子误判级联 | 置信度加权软更新；探针结果可交叉校验；签名唯一性随同型失败提升 |
| 割的矛盾（后验交集空） | 触发归因升级（P6-2 任务卡接口） |
| 快照恢复掩盖世界随机性 | memory 记"重试成功所需次数"作稳健度指标；一次成功 > 五次侥幸 |
| 配置矩阵爆炸 | 默认全 rule，数据证明更优才开放插拔 |
| 探针物理副作用 | 快照保护 + 低速小步进（探针设计约束） |
| sim2real | Θ 在线辨识即接口：仿真后验到真机继续收缩，非从零开始 |
| 约束≠成功 | {约束满足}⊂{任务成功}；成功靠重试+记忆在可行集内逼近 |
| 表膨胀 | 行格式五列标准化（签名/判据/对策/失效边界/实证来源），失效行退役不删 |

## 附录 G 与自动驾驶对照（用户原始问题的答案）

自动驾驶用了本方案前一半（MPC 车载标配 = 约束层；端到端+规则壳 =
小模型分解+约束层）；用不了后一半，四个结构性原因：①重试不可逆
（真世界无快照回滚）②时间尺度（10-100ms 控制周期无 per-step 推理
预算）③长尾位置（车的长尾在他车意图/交规博弈，物理约束压缩不了；
机器人的长尾恰在物理参数 μ/几何/质量里，约束声明杠杆大得多）
④认证责任（ISO 26262 禁在线学习热替换）。一句话：**约束层两车通用，
重试层只有机器人玩得起**——这是物理条件差异，不是想不到。

## 附录 H 文献（2026-09 调研）

- 物理约束 + LLM 动作推理：arxiv.org/html/2602.21161v1
- 闭环状态反馈分层 LLM 规划：advanced.onlinelibrary.wiley.com/doi/10.1002/adrr.202500072
- 物理因素显式喂 LLM 改善操作（TAROS 2025）：eprints.whiterose.ac.uk/id/eprint/228303/1/TAROS_2025_Poster_110.pdf
- VLMPC（LLM 给目标约束、MPC 求解底层，RSS 2024）：roboticsproceedings.org/rss20/p106.pdf
- 基础模型操作综述（能力分层依据）：arxiv.org/html/2404.18201v7
- 本架构与 MPC 的关系：per-step propose+clip = horizon-1 MPC 骨架，
  h>1 优化留待有明确收益再上（§11.2）
