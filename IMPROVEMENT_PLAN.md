# DarwinBot 改进与发布 Plan

> 起草日期：2026-09-21
> 目标：把当前"特性散乱的个人项目"包装并改进为**有清晰叙事、有公开 benchmark、有可复现主实验**的可发布工作。

***

## 0. TL;DR

把 DarwinBot 定位为**工程落地方案**（不是科研方案）：

**Parameter-as-Policy as Engineering Interface + SOTA Portfolio + Parameter Accumulation**
（参数即策略作为工程接口 + SOTA 方法组合 + 参数库积累）

一句话：**CaP 让 LLM 写代码组合 primitives；DarwinBot 让 LLM 只出参数喂给专用原语模型。每个任务类借用最佳 SOTA（Pi0.5 / DW-DOB / RDP / CGP），DarwinBot 提供统一参数接口 + 力控安全网 + 失败重试 + 参数库积累 + 国产模型路由——大规模泛化靠 portfolio，部署积累靠参数库，不依赖单模型泛化。**

三个唯一卖点（USP）：

- **Specialized Primitive Models（含借用的 SOTA）**：每类任务用最佳 SOTA 借用（VLA for pick / DW-DOB for peg-in-hole / RDP for 精细 / DarwinBot 力控 for 接触安全），统一封装为 DarwinBot skill 接口。
- **Parameter-as-Policy（工程接口）**：LLM 不写代码、不直接出动作，只决定**调哪个原语 + 生成什么参数**（接触角度/力阈值/螺旋半径/抓取偏移/移动目标）。参数是物理量、可观测、可 git、可累积——**这是工程接口的卖点，不需要参数本身大规模泛化**。
- **Portfolio + Accumulation for Large-Scale Generalization**：大规模泛化靠**方法组合 + 参数库积累**（不靠单模型）；VLA 给你 80% 单次，DarwinBot 给 100% 完成（重试 + 组合 + 力控 termination）；参数库跨部署累积（rel_offset 已验证 5.5→3.0），即使单组参数不泛化，portfolio 仍兜底。

***

## 1. 现状自评：为什么现在发会被喷

诚实地列出会被攻击的点，便于预先回应（见 §9）。

### 1.1 工程化弱

- 感知模型（`mobile_sam.pt` / `yolo26n-depth.pt` / `checkpoint-rs.tar`）单进程 `import`，加新模型会卡死、不能跨进程隔离。
- 无 CI、无 pre-commit、契约测试薄（`tests/` 只有 quat/controller/register/calibrate 等单元测试，无 API 契约测试）。
- 文档结构是 `docs/01-07_*.md` 编号体系，但无 readthedocs、无统一构建。
- 用 `pip` 而非 `uv`，依赖锁定弱。

### 1.2 学术叙事弱

- 无公开 benchmark：只在自建 robopal 任务上跑，外界无法对比。
- 无主实验、无对照基线：现有成功率表（§13/14/15）是自己 vs 自己，没有 VLA baseline 对照。
- 任务面窄：PickAndPlace / MultiCubeStack / Drawer / DrawerBox / LockedCabinet / BimanualPickAndPlace，共 \~6 类；cap-x 是 39 任务横跨 3 仿真器。
- 无 fine-grained / contact-rich 任务（peg-in-hole / 装配 / 拧螺丝）——而这正是力控最能体现差异化的场景。

### 1.3 定位模糊

- 既像 cap-x（写代码 agent）又像 RPent（agent + memory）但都不如人家完整。
- README 标榜"不用 VLA / 不用 RL / 不用 WAM"，这是**反向定位**（我不是什么），但没有正向定位（我是什么）。
- 力控 / skill\_forge / MAP-Elites / 100% 重试各自为战，没有统一叙事。

### 1.4 与三方对比的弱势

- 工程完成度：cap-x > RPent > **DarwinBot** > VoxPoser
- 学术含金量：cap-x ≈ RPent > VoxPoser > **DarwinBot**
- 生态覆盖：cap-x > RPent > VoxPoser > **DarwinBot**

***

## 2. 定位重塑：核心叙事

### 2.1 主叙事

**Parameter-as-Policy: Specialized Primitive Models + Failure-Driven Parameter Learning**

> CaP（Code-as-Policy）让 LLM 生成**代码**去组合 perception/control primitives——但代码本身不带物理直觉，每次都从零写、不可累积、不可跨任务复用。
> DarwinBot 走另一条路：**把操作能力分摊到每个原语的专用模型**（pick / place / insert / locate / move 各有自己的小模型 + 力控参数集），LLM 只决定**调哪个原语 + 生成什么参数**。物理世界的规律（怎么抓、怎么插、怎么移动）用**参数**表示，参数从**失败经验**中学习并跨物体/任务复用。

**对立面（与 CaP 的根本差异）**：

| 维度 | Code-as-Policy (cap-x / VoxPoser) | **Parameter-as-Policy (DarwinBot)** |
|---|---|---|
| LLM 输出 | Python 代码（compose primitives） | **参数向量**（调哪个原语 + 参数） |
| 物理规律在哪 | 在 prompt / 代码里 | **在参数空间里** |
| 复用单位 | 代码片段 | **参数库（按原语 × 任务族 × 物体形状）** |
| 学习机制 | RL 后训练（cap-x CaP-RL） | **失败驱动的参数沉淀**（无 RL） |
| 模型形态 | 一个通用 LLM | **专用原语模型阵列**（每个原语一个） |
| GPU 依赖 | 高（VLA + RL） | **低（小模型 + 力控）** |

**为什么这条路有学术价值**：
1. **直接挑战 CaP 范式**：cap-x / VoxPoser 都让 LLM 生成代码，你是 LLM 生成参数——这是一个可对立的 paradigm，不是工程变种。
2. **物理规律的表示方式不同**：CaP 把物理规律塞进 prompt（LLM 内化），你把物理规律塞进参数库（外部化、可观测、可版本化）。
3. **可累积**：每次成功的参数沉淀为正样本，失败的参数沉淀为负样本——CaP 的代码每次重写，你的参数库跨 episode 累积。
4. **可解释**：参数（接触角度、力阈值、螺旋半径）是物理量，可读、可调；代码生成的轨迹是黑盒。

### 2.2 副叙事（学术包装）

**Failure-Driven Parameter Learning for Fine-Grained & Long-Horizon Manipulation**

> 精细操作（peg-in-hole / 装配）和长程操作（多步组合）的成败都**高度参数敏感**：
> - 精细：接触角度差 5°、力阈值差 0.5N，peg 就卡死或滑出。
> - 长程：每个子任务的参数组合决定整条 trajectory 成败。
> DarwinBot 把这种参数敏感性**显式化**——参数从失败中学习，跨物体迁移（rel_offset 归一化已验证 ↓45% 尝试次数）。

- "Fine-grained / contact-rich manipulation" 是公认研究方向。
- "Long-horizon manipulation" 是公认研究方向。
- 你把**两者用同一个"参数学习"框架统一**——这是叙事 contribution。

### 2.3 一句话表述（用于 README 顶部）

> **DarwinBot: Parameter-as-Policy for Manipulation**
> An LLM-driven manipulation agent that distributes capability across specialized primitive models (pick/place/insert/locate/move) and learns their parameters from failure experience — without VLA, without RL, without WAM. Physical regularities live in the parameter space, not in the prompt.

### 2.4 大规模泛化的真正机制（必须诚实说明）

**没有任何单一方法能大规模泛化到 contact-rich**。DarwinBot 的"大规模泛化"靠**三层兜底**，**不依赖单模型泛化**：

| 层 | 机制 | 已验证 |
|---|---|---|
| **L1 方法组合** | 每类任务用最佳 SOTA 借用（Pi0.5 / DW-DOB / RDP / CGP） | VLA 大规模粗操作泛化已证（Pi0.5）；其他单任务 |
| **L2 参数库积累** | rel_offset 归一化跨部署沉淀，成功参数库扩 | ✅ 5.5→3.0 跨物体已验证 |
| **L3 重试 + 力控 termination** | GraspNet top-K + 失败沉淀 + 力控安全网 | ✅ 100% 完成率已验证 |

**Parameter-as-Policy 在其中的位置**：是 LLM 和 primitives 之间的**工程接口**，不是科研贡献。
- 即使参数本身不泛化，**portfolio 兜底 L1，积累兜底 L2，重试兜底 L3**——这是工程落地，不是单点突破。
- 不需要参数泛化才能讲故事：**"portfolio + 积累 + 重试"三层兜底是 DarwinBot 的真实价值**。

**对比 VLA 单模型路线**：
- VLA 路线：单模型 + 大数据 + GPU 集群 → 大规模泛化
- DarwinBot 路线：portfolio + 积累 + 重试 + 边缘端 → 大规模泛化（不同实现机制）
- 两者**互补不冲突**：DarwinBot 在 portfolio 里可以包含 VLA 作为 L1 的一员。

***

## 3. 三个唯一卖点 (USPs)

### USP 1: Specialized Primitive Models（专用原语模型阵列）

**现状**：`darwin/skills/primitives/` 已分摊为 `control.py` / `motion.py` / `ik_servo.py` / `collision.py` / `libero.py`；`skills/perception/` 分摊为 `detect.py` / `segment.py` / `grasp.py`。每个原语有自己的小模型 + 力控参数集（guarded_move / impedance_push / spiral_search / servo_align）+ CARTIK + 关节阻抗 + 力传感器终止。

**对标**：

- cap-x：LLM 写代码组合 perception+control primitives，**primitives 自身没有专用模型**，靠 LLM 临时组装。
- RPent：动作原语是**冻结 VLA**（Pi0.5/RLDX-1），一个大模型包打天下，**不专用、不可单独调参**。
- VoxPoser：`move_to_pose` 一个端点位姿，无任何力觉闭环、无原语分层，自己代码里有 `CustomMoveArmThenGripper` hack 绕 zero division。

**叙事**："VLA 是通用大脑，DarwinBot 是专用小脑阵列——每个原语一个专用模型，小而精、力控可调、可独立部署、可单独升级。"
**实验锚**：在 peg-in-hole / 平滑抽屉任务上，VLA 失败率 vs DarwinBot 力控收敛后的失败率。

### USP 2: Parameter-as-Policy（vs Code-as-Policy）

**现状**：LLM 不写代码、不直接出动作，只决定**调哪个原语 + 生成什么参数**（接触角度 / 力阈值 / 螺旋半径 / 抓取偏移 `rel_offset` / 移动目标）；`darwin/evolution/skill_forge.py` 把成功参数沉淀为 `entry.py + SKILL.md`，下次直接 import。

**对标**：

- cap-x：LLM 生成 Python 代码，每次从零写、不可累积、不可跨任务复用；`scripts/skill_library_compilation/` 是**事后分析**编译，**DarwinBot 是 episode 内动态 forge**。
- RPent：memory 是 `frontmatter + Markdown body`，喂 LLM 当上下文，**没变成可执行代码、没变成参数库**。
- VoxPoser：LLM 写代码生成 3D 值地图，物理规律在 prompt 里。

**叙事**："物理规律不在 prompt 里、不在 VLA 权重里——在参数库里。参数是物理量，可观测、可调、可累积。"
**实验锚**：相同任务下，CaP-baseline（LLM 写代码）vs DarwinBot（LLM 出参数）的：完成率、平均尝试次数、LLM token 消耗、跨物体迁移率。

### USP 3: Portfolio + Accumulation for Large-Scale Generalization

**现状**：DarwinBot 已有 GraspNet top-K 候选轮换 + 6 轮重采样 + 失败沉淀跳过坏偏移；失败参数沉淀为 `failure` 记忆（带 `fail_phase`），成功参数沉淀为 `success` 记忆（带 `rel_offset` 归一化）；实测冷启动 100% / 带记忆 100%（平均尝试 3.4 次，↓45%）；跨物体迁移 PickAndPlace→Stack 已验证（5.5 → 3.0）。

**对标**：
- cap-x：CaP-RL 用 GRPO 后训练 LLM，需 GPU 集群 + Isaac Sim + VeRL，**重训练**。
- RPent：memory 是 frontmatter + Markdown，喂 LLM 当上下文，**没变成可执行代码、没变成参数库**。
- VoxPoser：无记忆、无进化。
- VLA 单模型路线：靠大数据 + GPU 集群做大规模泛化，**contact-rich 弱、不积累**。

**叙事**："DarwinBot 不卷单模型泛化——大规模泛化靠 portfolio（每类任务用最佳 SOTA）+ 参数库积累（跨部署沉淀）+ 重试（100% 兜底）。这是工程路线，不是科研路线。"

**三层兜底机制**（详细见 §2.4）：
- L1 方法组合：Pi0.5 / DW-DOB / RDP / CGP 各管一类
- L2 参数积累：rel_offset 跨部署沉淀（已验证 5.5→3.0）
- L3 重试 + 力控 termination：100% 完成（已验证）

**实验锚**：DarwinBot portfolio + 重试 vs 纯 VLA 单模型——完成率 + 平均尝试次数 + 部署门槛（GPU / 数据 / 国产化）。

### USP 4: SOTA Borrowing Matrix（SOTA 方法借用矩阵）

**核心定位**：DarwinBot 不发明新方法，**每类任务借用最佳 SOTA**，统一封装为 DarwinBot skill 接口。

| 任务类 | 借用谁（2025-2026 SOTA） | DarwinBot 工程包装 | 状态 |
|---|---|---|---|
| Pick-Place 粗操作 | **Pi0.5 / OpenVLA-OFT**（VLA 大规模泛化） | VLA 给轨迹 + GraspNet top-K + rel_offset 参数库 | 待集成 |
| Drawer / Cabinet | **Pi0.5** + DarwinBot 力控 | VLA 粗轨迹 + guarded_move 做接触 + 力控 termination | 待集成 |
| Peg-in-hole（工业公差） | **DW-DOB**（learning-free 零力控制，2026-01） | 力控 primitive + 接触收敛 + 参数库调接触角度 | 待集成 |
| 精细 insertion | **M2-ResiPolicy / CGP**（diffusion + tactile，2026） | 接入作为子 skill，DarwinBot 提供参数接口 + 重试 | 待集成 |
| 长程组合 | **VLA + DarwinBot 编排** | skill graph + LLM 调度 + skill_forge 沉淀 | 已部分实现 |
| 抓取位姿 | **GraspNet**（已集成） | top-K 候选轮换 + 失败沉淀 + rel_offset 归一化 | ✅ 已有 |
| 视觉感知 | **YOLO + MobileSAM**（已集成） | service 化（见 §4.1） | ✅ 已有 |

**叙事**："不重新发明轮子——每类任务用最佳 SOTA，DarwinBot 是组合器 + 安全网 + 经验积累层。"

**借用规则**：
- 每个 SOTA 借用必须**统一封装为 DarwinBot skill 接口**（接受参数 + 输出状态）。
- 借用 SOTA 的失败案例由 DarwinBot 重试 + 力控兜底（L3）。
- 借用 SOTA 的成功参数沉淀到 DarwinBot 参数库（L2）。
- 借用 SOTA 之间的协调由 DarwinBot LLM agent 调度（L1）。

***

## 4. 工程化升级（从 cap-x / RPent 抄）

按 ROI 排序，从高到低。

### 4.1 Perception Service 化（抄 cap-x）

**痛点**：现在 `mobile_sam.pt` / `yolo26n-depth.pt` / `checkpoint-rs.tar` 单进程 `import`，加新模型（SAM3 / OWL-ViT / PyRoKi）会卡死。

**改造**：

```
darwin/serving/
├── launch_servers.py        # 抄 capx/serving/launch_servers.py，YAML profile 自动起
├── sam_server.py            # mobile_sam.pt 包成 HTTP 服务
├── yolo_depth_server.py     # yolo26n-depth.pt 包成 HTTP 服务
├── graspnet_server.py       # checkpoint-rs.tar 包成 HTTP 服务
└── profiles/
    ├── default.yaml         # SAM + YOLO + GraspNet（当前栈）
    ├── full.yaml            # + 未来加 SAM3 / OWL-ViT
    └── minimal.yaml         # 仅 GT 真值（oracle eval 用）
```

`darwin/skills/perception/{detect,segment,grasp}.py` 改为 thin client，调 HTTP 而非直 import。

**好处**：模型可独立部署、跨进程隔离、未来加模型不破坏现有、可多 GPU 分配。

### 4.2 Skill Library Compilation 脚本（抄 cap-x）

**痛点**：现在 `skill_forge.py` 是 episode 内动态 forge，但缺少**事后系统化整理**——技能库散乱、无统计、无导出。

**改造**：新建 `scripts/skill_library_compilation/`，仿 cap-x：

- 分析 `data/episodes/*.json` eval 输出。
- 按任务族 + 动作类型聚类成功轨迹。
- 编译为 `darwin/skills/forged/<task_family>/<skill_name>/{entry.py, SKILL.md}`。
- 生成 `skill_library_stats.md`：每类技能成功率、被调用次数、覆盖任务。

**好处**：把动态 forge 的产出系统化、可统计、可导出，配合 §4.4 的契约测试做技能回归。

### 4.3 Dashboard + Interactive Mode（抄 RPent / cap-x）

**痛点**：现在调试靠 `videos/` 回放 + print，演示和调试效率低。

**改造**：`darwin/dashboard/`

- FastAPI 后端 + Vite/React 前端（直接抄 cap-x 的 `web-ui/` 结构）。
- 实时显示：agent 推理过程、相机画面、动作时间线、力觉曲线（这是你的独家——cap-x/RPent 都没有力觉可视化）。
- `--interactive` 终端模式：`you>` 提示符随时插话引导 agent。
- `--dashboard-language zh-cn` 中文界面。

**独家优势**：力觉曲线可视化是 cap-x/RPent 没有的（他们没力控），这是**演示时的差异化亮点**。

### 4.4 Unit Tests 契约 + pre-commit + readthedocs（抄 cap-x / RPent）

**痛点**：`tests/` 只有单元测试（quat/controller/register/calibrate），无 API 契约测试；无 pre-commit；文档无统一构建。

**改造**：

- `tests/contract/`：技能 API 契约测试（每个 primitive 的 input/output schema 强制校验，仿 RPent 的 `tests/unit_tests/rpent/tools/test_toolkit_contracts.py`）。
- `.pre-commit-config.yaml`：ruff + mypy + yaml 校验。
- `docs/` 改 readthedocs 结构（conf.py + index.rst），仿 RPent 的中英双语。
- `pyproject.toml` 加 `[project.optional-dependencies]` dev/test/docs 分组。

### 4.5 uv 替代 pip（抄 cap-x）

**痛点**：`pip install -e .` + requirements.txt 依赖锁定弱。

**改造**：迁移到 `uv` + `pyproject.toml` + `uv.lock`。`requirements.txt` 保留作 fallback。

***

## 5. 记忆系统升级（从 RPent 抄）

### 5.1 Evidence 合并 + Confidence 自动升级

**现状**：`ragbot/` 已有归一化偏移 + TTL + UCB，但无 evidence 合并、无 confidence 三档升级。

**改造**：在 `ragbot/memory/store.py` 加：

- `evidence.cells: list[str]`（验证单元 = task\_seed）
- `evidence.attempts: int`
- `evidence.solved_seeds / failed_seeds: list`
- `confidence: single-shot / probable / verified`，规则：`cells≥3 且 tasks≥2 → verified`。
- `_merge_evidence()`：同一记忆多次验证自动合并 cells 并升级 confidence。

### 5.2 Contradicted\_by 正负对照

**现状**：失败记忆靠 TTL（20 episodes）过期，**没有指向成功记忆**——失败原因不显式。

**改造**：失败记忆必须带 `contradicted_by: <success_memory_id>`，形成正负对照。反思时由 LLM 决策 add/update/delete（抄 EvoAgentX）。

### 5.3 HuggingFace Dataset 同步

**改造**：`ragbot/memory/sync.py`，仿 RPent 的 `MemoryManager.sync()`，把记忆库同步到 HF dataset，跨机器共享。

**好处**：未来公开 leaderboard 时，记忆库可作为可复现 artifact 发布。

***

## 6. Benchmark 设计（抄 cap-x 思路，但走差异化路线）

### 6.1 任务分级（不是单一成功率）

cap-x 是 S1-S4 × M1-M4（抽象级别 × 交互模式），DarwinBot 走**任务难度分级**：

| 级别                  | 任务族                              | 特征          | VLA baseline 预期   |
| ------------------- | -------------------------------- | ----------- | ----------------- |
| **L1 粗操作**          | PickAndPlace / CubeStack         | 自由空间运动 + 抓放 | 80-95%            |
| **L2 articulated**  | Drawer / Cabinet / LockedCabinet | 关节物体操作      | 60-80%            |
| **L3 contact-rich** | **Peg-in-hole / 装**配 / 平滑抽屉      | 接触丰富、需力觉    | **30-50%** ← 你的主场 |
| **L4 长程**           | 开抽屉+取物+放置                        | 多步 + 多物体    | 20-40%            |

**重点做 L3**——这是 VLA 失败率最高、你最有力控差异化的场景。需要新增 L3 任务集（peg-in-hole / 双臂对齐 / 螺纹旋入）。

### 6.2 对比组设计

每个任务跑四组：

1. **VLA-baseline**：Pi0.5 / RLDX-1 单次（无重试）
2. **VLA + Retry**：VLA 失败时重试（同等尝试预算）
3. **DarwinBot-NoForge**：力控 + 重试，但关闭 skill\_forge（消融）
4. **DarwinBot-Full**：力控 + skill\_forge + 重试 + MAP-Elites（完整版）

**核心指标**：完成率 + 平均尝试次数 + 平均 LLM token 消耗。

### 6.3 公开 leaderboard

哪怕小规模也做：在 `darwin/leaderboard/` 维护一个 markdown 表，列每任务四组的完成率。配合 §5.3 HF dataset 同步做可复现。

***

## 7. 主实验设计

### 7.1 主实验

**问题**：DarwinBot 在 contact-rich 任务上是否显著优于 VLA baseline？

**setup**：L1-L4 共 12 个任务 × 50 episodes × 4 组对照。
**指标**：完成率、平均尝试次数、平均 LLM token 消耗。
**预期**：L1-L2 各组接近，**L3 DarwinBot-Full 显著优于 VLA baseline**（这是主结论）。

### 7.2 消融实验

| 消融组      | 关掉什么           | 验证什么      |
| -------- | -------------- | --------- |
| NoForce  | 关力控 primitive  | USP1 的价值  |
| NoForge  | 关 skill\_forge | USP2 的价值  |
| NoRetry  | 关在线重试          | USP3 的价值  |
| NoMAP    | 关 MAP-Elites   | 多样性优化器的价值 |
| NoMemory | 关 RAG 记忆       | 经验复用的价值   |

### 7.3 跨物体迁移（你已有数据）

复用 §15.3 的 PickAndPlace→MultiCubeStack 迁移数据，包装为：
**"DarwinBot 的归一化偏移记忆可跨物体/任务迁移"**——平均尝试 5.5 → 3.0（↓45%）。

### 7.4 真机 demo

- Franka 单臂（cap-x 已有 bringup 文档可借鉴）
- 至少 2 个 L3 contact-rich 任务真机视频
- 进 `videos/` 作为发布物料

### 7.5 参数泛化实验（nice-to-have，不是成败判据）

**重要：参数泛化不是 DarwinBot 路线成立的判据**——大规模泛化靠 §2.4 的三层兜底（portfolio + 积累 + 重试），不靠单组参数泛化。

但参数泛化**如果成立**会让 DarwinBot 更强（L2 积累层更有效），所以仍跑这组实验**作为加分项**。

| 实验 | Setup | 成功判据 | 不成立时 |
|---|---|---|---|
| **物体泛化** | cube A 的 `rel_offset` → cube B/C/D | 完成率 ≥ 80% | 不影响主叙事，仅 L2 积累效率降 |
| **任务泛化** | pick 的接触参数 → place/insert 接触阶段 | 完成率 ≥ 60% | 不影响，L1 portfolio 兜底 |
| **机型泛化** | DualPanda 力控参数 → DianaTripleStack | 完成率 ≥ 70% | 不影响，per-robot 仍可发 |
| **场景泛化** | 同任务不同布局 | 完成率 ≥ 80% | 不影响，per-scene 仍可发 |

**报告规范**：每个泛化实验报告 mean ± std over 5 seeds，**显式标注"哪些泛化成立、哪些不成立"——不藏失败**。

**叙事策略**（任何结果都成立）：
- 若 ≥2/4 泛化成立 → 加分项，叙事强调"参数库跨部署积累，验证 DarwinBot 三层兜底中 L2 有效"。
- 若 ≤1/4 成立 → **不影响主叙事**，因为大规模泛化靠 L1 portfolio + L3 重试，**L2 不泛化 DarwinBot 仍能 100% 完成 + 大规模部署**。

**主实验仍是 §7.1**：DarwinBot portfolio + 重试 vs 纯 VLA 单模型的完成率 + 平均尝试次数 + 部署门槛对比——这才是 DarwinBot 的真实工程价值锚点。

***

## 8. 发布节奏

按 ROI 排序，每个 Phase 都可独立演示，不阻塞下一阶段。

### Phase 1: 工程化基线（2-3 周）

- [ ] §4.1 Perception Service 化（3 个 server + default profile）
- [ ] §4.4 Unit Tests 契约 + pre-commit + readthedocs
- [ ] §4.5 uv 迁移
- [ ] README 顶部叙事替换为 §2.3 一句话

**Phase 1 演示**：`uv run ... --profile default` 一条命令起所有服务 + 跑 PickAndPlace。

### Phase 2: 记忆升级 + 技能库编译（1-2 周）

- [ ] §5.1 Evidence 合并 + Confidence 升级
- [ ] §5.2 Contradicted\_by
- [ ] §4.2 Skill Library Compilation 脚本
- [ ] §5.3 HF dataset 同步（可选，后期）

**Phase 2 演示**：跑 20 episodes → 自动编译技能库 + 生成 stats.md。

### Phase 3: Dashboard + Benchmark + 主实验（2-3 周）

- [ ] §4.3 Dashboard + Interactive（含力觉曲线可视化）
- [ ] §6.1 任务分级（重点加 L3 contact-rich 任务）
- [ ] §6.2 四组对照实现（接入 Pi0.5 / RLDX-1）
- [ ] §7.1 主实验 + §7.2 消融

**Phase 3 演示**：`--dashboard` 启动 → 实时看 agent 推理 + 力觉曲线 + 对照表。

# Phase 4: 论文 + 真机 demo（2-3 周）

- [ ] §7.3 跨物体迁移实验
- [ ] §7.4 真机 Franka demo（2 个 L3 任务视频）
- [ ] arXiv 预印本（标题见 §9.1）
- [ ] 公开 leaderboard + HF dataset 发布

***

## 9. 防喷清单：可能的攻击点 + 预先回应

### 9.1 标题候选

- **Parameter-as-Policy: Distributing Manipulation Capability across Specialized Primitive Models with Failure-Driven Parameter Learning**
- 备选 1：From Code-as-Policy to Parameter-as-Policy: Learning Physical Regularities from Failure Experience
- 备选 2：Failure-Driven Parameter Learning for Fine-Grained and Long-Horizon Manipulation without VLA

### 9.2 预期攻击点

| 攻击 | 预先回应 |
|---|---|
| "Parameter-as-Policy 就是 skill library + hyperparameter tuning" | 关键差异：参数是**物理量**（接触角度/力阈值/rel_offset），不是 hyperparameter；且从**失败经验**学习而非 grid search；§7.5 泛化实验证明参数库可跨物体迁移，hyperparameter tuning 不可 |
| "和 Code-as-Policy 没本质区别，只是接口换了" | §2.1 对立面表：CaP 复用单位是"代码片段"（每次重写），DarwinBot 复用单位是"参数库"（跨 episode 累积）；CaP 物理规律在 prompt，DarwinBot 在参数空间——可观测、可版本化、可 git |
| "专用原语模型阵列就是 backward，VLA 才是未来" | 不是取代 VLA，是**互补**——VLA 给粗轨迹，专用原语做精细接触；§6.2 VLA+DarwinBot 混合组应该 > 纯 VLA 组，这是子结论 |
| "参数泛化是你嘴硬，没数据" | §7.5 四组泛化实验是核心，**报告 mean ± std over 5 seeds**，不藏失败；§7.5 叙事策略已写明降级路径，3/4 不成立就降级 |
| "100% 靠重试堆出来，作弊" | §7.2 NoRetry 消融展示无重试 baseline；VLA+Retry 同预算对照组——DarwinBot 靠力控 termination + skill_forge 复用经验赢，不是无脑重试 |
| "仿真里 100%，真机呢" | §7.4 真机 Franka demo 至少 2 个 L3 任务视频；真机不强求 100%，报告"达到 X% 所需平均尝试次数" |
| "任务太少，6 类 vs cap-x 39" | 主战场是 L3 contact-rich + 长程组合（§7.5 的精细 + 长程叙事），不卷广度，卷**参数学习深度**；§7.5 泛化实验比"任务多"更有方法论意义 |
| "没有 VLA 对比" | §6.2 四组对照直接接 Pi0.5/RLDX-1，硬数据说话 |
| "力控是 90 年代技术" | 不是新力控算法，是**把力控和 LLM agent 结合做参数学习**——LLM 决定何时调力控 primitive + 生成参数，力控 primitive 做安全网 + 参数从失败中学习，这个结合是新的 |
| "MAP-Elites 是 EvoAgentX 抄的" | 承认借鉴，但应用场景（操作技能参数库的 QD 优化）是新的；引用 EvoAgentX，不藏 |
| "skill_forge 是 AgentFactory 抄的" | 承认借鉴 AgentFactory，但 forge 的是**参数化原语**而非 LLM agent 本身；引用 AgentFactory，不藏 |
| "没有 NVIDIA/Stanford 团队，没说服力" | 个人项目不卷学术声望，卷**可复现 artifact**（HF dataset 参数库 + 真机视频 + 公开 leaderboard + §7.5 泛化数据） |
| "国产模型路由是工程不是研究" | 包装为 **"边缘端可部署的操作 Agent"**——不依赖 Claude/Codex 闭源 SDK，可在国产模型 + 本地 Ollama 上跑；部署级 contribution |

### 9.3 必须避免的坑

- **不要在 README 顶部和 cap-x/RPent 直接对比**——会被喷"碰瓷"。只在自己的 benchmark 里给硬数据。
- **不要标榜"通用框架"**——主叙事是"Parameter-as-Policy 范式 + 参数泛化判据"，不是"我又做了一个通用操作框架"。
- **不要藏借鉴**：cap-x 的 service 化、RPent 的 memory 元数据、EvoAgentX 的 MAP-Elites、AgentFactory 的 skill_forge 全部在 related work 里引用，不藏。
- **不要藏 §7.5 失败**：如果某组泛化不成立，**显式报告**——"诚实的 negative result"比"包装过的 positive result"更抗喷。
- **不要过度承诺真机**：真机 demo 限定 2 个 L3 任务，不要承诺多机型。
- **不要把"专用原语模型阵列"包装成反 VLA**：是互补层，不是替代；§6.2 混合组对照证明这点。

***

## 10. 资源盘点（你已有的资产）

| 资产                | 路径                                   | 在本 plan 中的角色                           |
| ----------------- | ------------------------------------ | -------------------------------------- |
| mobile\_sam.pt    | /home/lifd/Public/mobile\_sam.pt     | §4.1 SAM server 权重                     |
| yolo26n-depth.pt  | /home/lifd/Public/yolo26n-depth.pt   | §4.1 YOLO server 权重                    |
| checkpoint-rs.tar | /home/lifd/Public/checkpoint-rs.tar  | §4.1 GraspNet server 权重                |
| graspnet\_repo    | /home/lifd/Public/graspnet\_repo     | §4.1 GraspNet server 源码                |
| LIBERO            | /home/lifd/Public/LIBERO             | §6.1 L1/L2 任务源 + §6.2 VLA baseline 接入点 |
| datasets\_libero  | /home/lifd/Public/datasets\_libero   | §7.3 跨物体迁移实验数据                         |
| robopal           | /home/lifd/Public/darwin-bot/robopal | L3 contact-rich 自建任务平台                 |
| 具身小册的latex        | /home/lifd/Public/具身小册的latex         | 本 plan 整理为小册子"四个框架对照 + 改进案例"章节         |
| arkcli / ARK      | 系统 skill                             | §3 USP3 国产模型路由（GLM-4 / Doubao）         |

***

## 11. 与三个参照框架的最终对照（用于具身小册子）

| 维度 | VoxPoser | RPent | cap-x | **DarwinBot（本 plan 后）** |
|---|---|---|---|---|
| 定位 | 3D 值地图 LLM 代码生成 | Agent SDK 驾驭冻结 VLA | CaP benchmark + RL 改进 | **Parameter-as-Policy + 参数泛化判据** |
| LLM 输出 | 写代码生成值地图 | 调 VLA 出轨迹 | 写代码组合 primitives | **生成参数喂专用原语模型** |
| 能力分布 | 一个通用 LLM | 一个通用 VLA | 一个通用 LLM + 一个 RL 训练 | **专用原语模型阵列（pick/place/insert/locate/move）** |
| 物理规律在哪 | 在 prompt | 在 VLA 权重 | 在 prompt + RL | **在参数库**（外部化、可观测、可 git） |
| 复用单位 | 代码片段 | Markdown memory | 代码 + skill_library | **参数库（跨 episode 累积）** |
| 学习机制 | 无 | memory 蒸馏 | CaP-RL GRPO | **失败驱动参数沉淀**（无 RL） |
| 完成率 | 无重试 | 92.63% | 92.63% | **100%（仿真）** |
| 感知 | RLBench mask | SAM3+Molmo | SAM3+GraspNet+PyRoKi | **SAM+YOLO+GraspNet（service 化）** |
| 仿真 | RLBench 13 任务 | LIBERO/RoboCasa/RoboTwin | Robosuite+LIBERO+BEHAVIOR 39 任务 | **robopal + LIBERO（L3 重点自建）** |
| 真机 | 无 | Franka/Dual Franka | Franka bringup | Franka 单臂（2 个 L3 demo） |
| GPU 依赖 | 低 | 中 | **高（必 CUDA）** | **低（不用 VLA/RL）** |
| 工程化 | 极简 | Apache 2.0+测试+文档 | MIT+uv+完整 docs | **MIT+uv+契约测试+readthedocs（Phase 1 后）** |
| 学术 | CVPR/Stanford | arXiv+RLinf | arXiv+NVIDIA/Berkeley/Stanford/CMU | **arXiv 预印本（Phase 4）** |
| **核心 open question** | — | — | — | **§7.5 参数能否泛化？** |

**DarwinBot 的不可替代位**：唯一**把物理规律外部化为参数库 + 用失败驱动学习 + 不依赖 VLA/RL + 力控自研 + 国产模型可路由**的 framework。这是 cap-x/RPent/VoxPoser 都没有的组合——他们要么把物理规律塞进 LLM prompt（cap-x/VoxPoser），要么塞进 VLA 权重（RPent）。

***

## 12. 风险与决策点

### 12.0 风险评估：参数不泛化不再是 make-or-break

**根据 §2.4 的三层兜底机制**，参数泛化失败**不再让整条叙事崩盘**——大规模泛化靠 portfolio + 重试兜底。

但仍有以下真实风险：

| 风险 | 严重性 | 缓解 |
|---|---|---|
| 借用的 SOTA 集成成本高（Pi0.5 / DW-DOB / RDP 接入难） | 中 | USP 4 借用矩阵从 GraspNet 已集成的开始，Pi0.5 已有公开 SFT checkpoint，DW-DOB 是 learning-free 易接入 |
| Portfolio 调度决策错（LLM 选错原语） | 中 | LLM agent 调度有 retry 兜底；调度错误 = 失败 = 重试 |
| 借用 SOTA 之间参数不兼容（Pi0.5 出轨迹 vs DW-DOB 出力） | 中 | DarwinBot 统一参数接口（USP2）做转换层 |
| 真机部署 demo 失败 | 低 | §7.4 选力控最容易体现差异的 L3 任务 |

**§7.5 参数泛化实验**：nice-to-have，不是 make-or-break——见 §7.5。

### 12.1 第二风险：100% 是仿真靠重试换的，真机不一定

**缓解**：
- §7.4 真机 demo 选 2 个**力控最容易体现差异**的 L3 任务（peg-in-hole + 平滑抽屉）。
- 真机允许多次重试，记录"达到 X% 所需平均尝试次数"，不强求 100%。
- 真机视频放 README 顶部，让叙事先于数字。

### 12.2 第三风险：L3 contact-rich 任务要从零搭建

**缓解**：
- robopal 已有 MuJoCo + 力传感器，加 peg-in-hole 场景不难。
- 参考 LIBERO 的 `libero_spatial` / `libero_object` 加 contact-rich 变体。
- 至少 3 个 L3 任务即可，不贪多。

### 12.3 决策点：要不要接 VLA baseline

**建议**：接。不接 VLA 对比，"100% vs 92.63%" + "参数 vs 代码"两组对照都立不住。LIBERO 已有 Pi0.5 SFT checkpoint，接入成本可控。这正是 §6.2 四组对照的核心。

**额外好处**：§6.2 加一组 **VLA+DarwinBot 混合组**（VLA 给粗轨迹 + DarwinBot 专用原语做精细接触），如果混合组 > 纯 VLA 组，就证明 DarwinBot 是**互补层而非替代层**——这是更稳的叙事。

### 12.4 决策点：要不要做 CaP-RL 式的 GRPO 训练

**不做**。你没有 GPU 集群 + Isaac Sim + VeRL 基础设施，且这不是你的赛道。叙事是"**不用 RL 也能学到参数库**"——这是和 cap-x 的根本差异化。

***

## 13. 立即可做的第一周任务（最小可演示）

- [ ] README 顶部叙事替换为 §2.3 一句话（Parameter-as-Policy + Portfolio）
- [ ] §4.1 起一个 `sam_server.py` 把 `mobile_sam.pt` 包成 HTTP（其余后续）
- [ ] §4.4 加 `.pre-commit-config.yaml`（ruff + mypy）
- [ ] §6.1 在 robopal 加 1 个 L3 任务（peg-in-hole）作为新主战场
- [ ] §4.3 dashboard 起一个最简 FastAPI + 静态页（先不做 React，先做力觉曲线 + 参数库可视化）
- [ ] **§3 USP4 集成一个借用的 SOTA**（建议先 Pi0.5 SFT checkpoint 或 DW-DOB，作为 DarwinBot skill 接口的最小验证）
- [ ] §7.5 物体泛化实验设计草案（最便宜的实验：cube A→B/C/D，先在 PickAndPlace 上跑）

完成这七项即可演示 "Parameter-as-Policy + Portfolio on contact-rich tasks" 的最小闭环——证明 DarwinBot 能集成 SOTA + 力控 + 参数库 + 重试。

***

## 14. 结语

把 DarwinBot 定位为**工程落地方案**——不是科研方案、不卷单模型泛化。

核心叙事保留 **Parameter-as-Policy**（LLM 出参数 vs 出代码，是工程接口，是 DarwinBot 的灵魂），但**大规模泛化靠三层兜底**：
- **L1 方法组合**：每类任务借用最佳 SOTA（Pi0.5 / DW-DOB / RDP / CGP）
- **L2 参数库积累**：跨部署沉淀，已验证 5.5→3.0
- **L3 重试 + 力控 termination**：100% 完成，已验证

**Parameter-as-Policy 的真正价值**：不是"比 VLA 更泛化"，是**"参数是物理量、可观测、可 git、可累积"** 的工程接口——LLM 和 primitives 之间用参数对话，不用代码。

**最关键的一句话**：你不是"另一个 LLM 机器人框架"，你是 **"每类任务用最佳 SOTA + LLM 用参数调 primitives + 力控做安全网 + 失败重试到 100% + 跨部署积累参数库"** 的工程组合器——这是 VLA 单模型路线之外的可落地、可国产化、可边缘部署的另一条路。

不卷科研泛化，卷**工程部署的可靠性 + 可观测性 + 可积累性**——这是你的突围路径。
