# 12 · LIBERO 调试纪要（2026-09-30）

> 本文是一次实际排障的完整记录：**怎么定位、测到了什么、改了什么、哪些还没解决**。
> 相关：[08 规划与执行栈](08_libero_planning_stack.md)、[11 环境适配层](11_envs_libero_adapter.md)、[07 感知技能](07_perception.md)
>
> 记录数字均为当次实测值，不是估算。

## 0. 背景

`scripts/run_libero_agent.sh` 循环跑 spatial/object/goal 三套件（共 30 个任务），台账显示 **21 通过 / 9 失败**，且失败任务反复重试也不收敛。本纪要记录这 9 个的归因与修复。

**先记一个定位技巧**：任务编号**不能**按 `bddl_files/<suite>/` 的文件名排序推断（该目录里有 `tasks_info.txt` 会打乱，且实际顺序不同）。权威来源是 LIBERO 的
`libero/libero/benchmark/libero_suite_task_map.py`。据此才确认：

| 编号 | 任务 |
|---|---|
| `goal:7` | turn_on_the_stove |
| `goal:5` | push_the_plate_to_the_front_of_the_stove |
| `goal:9` | put_the_wine_bottle_on_the_rack |
| `object:0/1/6/7/9` | 抓 alphabet_soup / cream_cheese / butter / milk / orange_juice 放篮子 |
| `spatial:4` | 抓木柜上层抽屉里的黑碗 |

## 1. 三个失败分组（不是同一个病根）

| 组 | 任务 | 主导机制 |
|---|---|---|
| **A** | `goal:7` | `toggle` 一致失败，**`q` 恒 ≈ 0**（"关节卡死"），机制 `contact_blocked` |
| **B** | `goal:5`、`object:1/6/9` | `grasp` 返回 `no_candidate`（几何候选耗尽）→ 析取规划退回 `push_to` → `推入受阻/失去接触/reach_limit` |
| **C** | `object:7`(301.8s)、`spatial:4`(45s) | `attempt_timeout` |

只有 A 组是"能修好的 bug"，B/C 是能力/超时问题。

## 2. 发现一：`turnon` 目标取错端点（`goal:7` 的根因）

### 2.1 取证链

1. 台账里 `goal:7` 的 8 次 attempt × 4 轮重规划，**32 次 `toggle` 全部**报
   `关节卡死在 q=-0.002~0.027（目标 1.780）`——关节几乎没动过。
2. 追 `q_goal` 来源：`libero_skills.toggle` → `adapter.articulation_info` 的 `turnon_qpos`。
3. 追 LIBERO 侧判定：`libero_goal/turn_on_the_stove.bddl` 的 goal 是 `(And (Turnon flat_stove_1))`，谓词求值链是
   `base_predicates.TurnOn → ObjectState.turn_on → FlatStove.turn_on(qpos)`，而
   `FlatStove.turn_on` 的判据是 **`qpos >= min(default_turnon_ranges)`**，其值为 `[0.5, 2.1]` → **阈值 0.5**。
4. 而 `articulation_info._conservative(..., "max")` 取的是 **80% 侧**：
   `0.5 + 0.8×(2.1−0.5) = 1.780`。**1.78 rad ≈ 102°**，正好是 `toggle` docstring 里抱怨的"102° 级旋转手掌扫掠躲不开灶面外壳"那个角度。
5. 结论：**目标要求是判定阈值 0.5 的 3.6 倍**。之前的工作把"102° 扫不开"当成环境限制、去做分段重抓（治标），没有质疑目标本身。

### 2.2 修复与验证

`darwin/envs/libero_adapter.py`：`turnon_qpos` 由 `"max"` 侧改为 `"min"` 侧（与 `open_qpos` 的约定一致，因为 `turn_on` 的阈值在下界）。

| | 修复前 | 修复后 |
|---|---|---|
| 目标角 | 1.780 rad（102°）| **0.820 rad（47°）** |
| 关节实测 `q` | 最大 **0.027** | 0.069 → **0.418**（+）|
| 失败机制 | 恒 `contact_blocked`（关节卡死）| `reach_limit` / `slow_budget` |
| 结果 | 8 attempts × 4 轮全败 | **第 1 次尝试通过（63.3s）** |

"关节卡死"这个词在修复后的日志里**再未出现**；`q` 从"纹丝不动"变成能到 0.418+，说明拖动机制本身是好的，之前的瓶颈是目标超出物理可及范围。

### 2.3 回归风险核查（全量对拍）

写脚本从 LIBERO 源码解析 6 类物件的 `default_*_ranges` 与其 `is_open/is_close/turn_on/turn_off` 比较方向，逐一验证 agent 的语义目标是否满足判定：

- **12 个语义目标（4 语义 × 物件）全部满足**；
- `open_qpos` / `close_qpos` **取值未变**（本就取 range 内部点，两种比较方向都满足）；
- 只有 `turnon_qpos` 变化 → 影响面仅 `toggle`。

另核台账技能足迹：`toggle` **只出现在 `goal:7` 一个任务**里 → 该改动对已通过的 21 个任务零影响。

## 3. 发现二：挂钟超时让结果不可复现

### 3.1 问题

`attempt_timeout` 用挂钟 `SIGALRM` 在**任意字节码边界**截断 episode，而 `mem`（机制级失败记忆）跨 attempt 复用 → 第 1 次尝试"跑了多久、在哪儿被打断"取决于机器负载，于是下一次尝试的候选搜索起点不同，结果不同。

### 3.2 改法：步数为主判据，挂钟只兜底

| 判据 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 仿真步数（主）| `--attempt-steps` | 7000 | 确定性截断；`0` = 不限 |
| 挂钟（兜底）| `--attempt-timeout` | 300s | 只在"不走步但卡住"时才可能先触发 |

实现：`libero_adapter.AttemptStepLimit`（**继承 `BaseException`**，否则会被技能层的 `except Exception` 吞掉）+ `step()` 内计数 + runner 显式捕获并落 `fail = step_budget>Nsteps`。

### 3.3 标定数据（实测）

| 观测 | 值 |
|---|---|
| `spatial:4` 一次**完整自然失败**尝试 | **1605 步 / 88.2s** |
| 30s 挂钟 alarm 触发时的步数 | **0 步**（前 ~30s 全在环境构建 + GraspNet 感知，不走步）|
| 反推伺服段速率 | ≈ 32 步/秒 → 旧 300s ≈ 8500 步，故取 **7000** |

### 3.4 功能验证

```
--attempt-steps 300  →  [runner] attempt 1/1 FAIL (56.3s, 301步) step_budget>300steps
```

预算 300 → **恰好在第 301 步截断**，且 **56.3s 就触发**（挂钟 300s 远未到）→ 主判据确实先生效。

### 3.5 回退：挂钟兜底由 600s 收回 300s

步数判据上线后曾把挂钟放宽到 600s（希望确定性判据总是先生效）。实测发现该假设**对"不走步"的任务不成立**：

```
libero:libero_object:7  attempt 1~7 全部 FAIL (≈601.8s, ≈2100步) attempt_timeout>600s
```

7 次尝试都是**挂钟先触发**，而步数只走到 ~2100（≈3.3 步/秒，远低于伺服段的 32 步/秒）——说明该任务的 600s 大部分花在**感知/候选枚举**上（本仓库多处注释记"重枚举是分钟级"），步数判据对它完全不起作用。放宽到 600s 的直接后果是**每条失败尝试的代价翻倍**（`object:7`：8×600s≈80min/轮 → 8×300s≈40min/轮）。

故取回 **300s**：走步型任务仍由 7000 步主导，而这类"不走步"任务的代价回到原来的量级。**残余代价**是"走步但慢"的轨迹上挂钟可能先于步数预算触发（即 §4 的非确定性残留）。

## 4. 发现三：仍不可复现（未解决）

### 4.1 现象

同一任务、同一初始状态（`demo_0 states[0]`）、同一代码，单次尝试的步数实测到过 **6 个不同值**：
`1605 / 3697 / 1920 / 1653 / 1900 / 2239`，失败机制也随之在 `action_fail` 与 `ik_unreachable` 之间跳变。

### 4.2 取证：分歧起点在第一次抓取候选枚举

对比两次运行的完整日志，**第一处差异就是第一次 `grasp`**：

```
run A: [grasp] try=0 grasp_all FAIL reason=GraspNet 无候选        ← 一个候选都没给出
run B: [grasp] try=1 closure_gate FAIL ... / try=2 closure_marginal ok ... / try=3 exec_path FAIL ...
```

### 4.3 已排除（**别重复验证**）

| 假设 | 实验 | 结果 |
|---|---|---|
| 挂钟超时 | 改成步数判据 | ❌ 三次仍 `1653/1900/2239` |
| CPU 线程调度 | `OMP/MKL/OPENBLAS/NUMEXPR=1` | ❌ 三次 `1605/3697/1920`，全不同 |
| RRT 采样种子 | 读代码 | ❌ 本来就是固定 `plan_arm_path(seed=0)` |

### 4.4 已确认并修复的一个真 bug

`darwin/skills/perception/grasp.py` 的点云下采样调用**全局 `np.random`**，而全局 RNG **从未播种**：

```
164:  idxs = np.random.choice(len(points), num_point, replace=False)
310:  idx = np.random.choice(len(cloud), n_points, replace=False)
357:  idx = np.random.choice(len(cloud), n_points, replace=False)
```

已由 runner 在每 attempt 边界 `np.random.seed(attempt)` 修掉（步数从 1605 变 1653，证明**输入点云确实变了**）。

### 4.5 剩余方差与未验证项

- **定位**：播种后**输入**已确定，但输出仍变 → 剩余方差在 **GraspNet 前向本身**（`device = cuda:0`，CUDA 的 scatter/atomicAdd 类算子默认非确定）。
- **未验证**：关掉 CUDA 复测。注意单纯 `CUDA_VISIBLE_DEVICES=""` **不行**——robosuite 的 EGL 渲染会把空设备列表拿去 `int(x)` 解析而崩（`invalid literal for int() with base 10: ''`），必须同时处理 torch 侧 device 与 `MUJOCO_EGL_DEVICE_ID`。

## 5. 本次改动与验证状态

| # | 改动 | 位置 | 验证 |
|---|---|---|---|
| 1 | `turnon` 目标端点（102°→47°）| `envs/libero_adapter.py` | ✅ 实跑：`goal:7` 首次通过 |
| 2 | `toggle` 零进展升级为 θ 驱动 | `agents/libero_skills.py` | ⚪ 未被触发（防御性）|
| 3 | 失败信息带真实机制 | `agents/libero_runner.py` | ✅ 日志可见 |
| 4 | 替代分支失败一次即收尾 | `agents/libero_runner.py` | ✅ 轮数 4→2；`object:1/6` 耗时 −63%/−70% |
| 5 | 超时主判据改仿真步数 | `envs/libero_adapter.py` + `agents/libero_runner.py` | ✅ 精确截断且先于挂钟 |
| 6 | 感知下采样按 attempt 播种 | `agents/libero_runner.py` | ✅ 生效，但未换来可复现 |
| 7 | `run_task` 默认值与 CLI 对齐（300s / 7000 步）| `agents/libero_runner.py` | ✅ 一致性修复 |
| 8 | 挂钟兜底由 600s 收回 300s（依据见 §3.5）| `agents/libero_runner.py` | ✅ 代价减半 |

## 6. 结论与遗留

- **真修好的只有 `goal:7` 一个**（有质变证据链）。台账从 21/30 变 23/30，但其中 `object:0` 的成功路径只经过 `grasp + place_at`（未被本次改动触及）→ 属**噪声导致的运气通过**，不应计入修复成果。
- **B 组（`grasp` 候选耗尽 → 推不动/够不着）未解决**：`spatial:4` 是"抓不到 + 也推不到"的复合失败；`goal:5` 是推盘子受阻。需要从几何候选生成与推拨策略入手。
- **可复现性未达成**：噪声源已收敛到"GraspNet 的 GPU 前向"，但未验证关 CUDA 是否可解；即便可解，也需评估 CPU 推理的耗时代价。
- **纪律**：在解决可复现性之前，报告通过率必须附带说明——**单次通过 ≠ 稳定通过**。
