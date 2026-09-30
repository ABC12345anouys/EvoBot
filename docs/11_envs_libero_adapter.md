# 11 · 环境适配层设计（LIBERO）

> 对应代码：`darwin/envs/libero_adapter.py`
>
> 相关：[08 规划与执行栈](08_libero_planning_stack.md)（谁在调用它）、[10 物理机制与判别](10_physics_mechanisms.md)（失败机制从哪来）

## 1. 职责边界

`LiberoEnvAdapter` 是 **darwin 与 LIBERO/robosuite 之间的唯一边界**。它做三件事：

1. **把 LIBERO 的仿真包成"统一动作语义"**：向上只暴露 `servo_step(site, target, gripper, ...)` 这类与机型无关的调用，向下翻译成 robosuite 的 7 维 OSC 动作。
2. **提供 BDDL 语义查询**：`eval_subgoal` / `check_success` / `fixture_open|close` / `articulation_info`（region 名 → 驱动关节 + 各语义目标 qpos）。
3. **管理 episode 边界**：固定初始状态、自管 episode 长度（绕开 robosuite 的 horizon）、确定性截断计数。

## 2. 构造与初始状态

```python
def __init__(self, suite, task_idx, camera_size=128):
    info = _task_info(suite, task_idx)                 # 缓存：读一次 hdf5
    self._inner_env = OffScreenRenderEnv(bddl_file_name=info["bddl"], ...)
    self._inner_env.seed(0)
    self.mj_model = self._inner_env.sim.model
    self._state0 = info["state0"]                      # demo_0 的 states[0]
    self.steps = 0                                     # 本 attempt 步数（reset 清零）
    self.step_budget = None                            # None = 不限
```

- `_task_info()` 用 `lru_cache` 缓存，从 `benchmark.get_task(idx)` 取 BDDL 路径，并从数据集 hdf5 读 `demo_0` 的 `states[0]` 与动作条数；`libero_adapter.py:56-76`
- **关键设计决策：初始状态固定为 demo_0 的 `states[0]`**，`reset()` 每次都 `set_init_state(self._state0)` → 同一任务每次 attempt 从**同一状态**起步。
  ⚠️ 但这**不等于结果确定**——见 §6 与 [12 调试纪要](12_libero_debug_2026-09-30.md)。
- `mj_data` 是 **property 而非缓存字段**：robosuite 的 `binding_utils` 每次访问 `sim.data` 都返回新的 `MjData` 包装，缓存会得到 `time/qpos/ctrl` 全部冻结的过时快照；`libero_adapter.py:104-132`

## 3. `reset()` 与 `step()`

### 3.1 reset

```python
def reset(self):
    obs = self._inner_env.reset()
    if self._state0 is not None:
        obs = self._inner_env.set_init_state(self._state0)
    self.steps = 0                      # 步数预算按 attempt 计
    self._darwin_done = False           # episode 终止标记
    self.init_pos = {...grip_site...}    # home 原语用
```

`reset()` 与每次 attempt 都会调用：`init_pos` 因此始终指向**开局安全位**，即使技能中途回退快照也不会漂；`libero_adapter.py:583-598`

### 3.2 step：自管 episode 长度

```python
def step(self, action):
    if self._darwin_done: raise RuntimeError("episode_terminated")
    self.steps += 1                                     # 确定性截断计数
    if self.step_budget is not None and self.steps > self.step_budget:
        raise AttemptStepLimit(self.steps, self.step_budget)
    if self._inner_timestep() >= self._inner_horizon():  # robosuite 计数到点 → 回卷
        self._rewind_inner_timestep()
    obs, reward, done, info = self._inner_env.step(np.asarray(action, float))
    ...
    if done:
        self._rewind_inner_timestep(); done = False      # done 只当"到点"，不当终止
    return obs, reward, done, info
```

**为什么必须回卷**：robosuite 的 `horizon`（默认 1000）一到就 `done=True`，而 darwin 是**自管 episode 长度**（终态由 BDDL 谓词核验，而不是由步数决定）。实测单条 attempt 常超 1000 步（`carry` 单段就 500+，加 `home/above/lift` 很容易越界），所以到点即重置内层计数继续跑。`libero_adapter.py:603-643`

- `_rewind_inner_timestep()` 必须**沿 wrapper 链递归**到 `MujocoEnv` 本体：robosuite 的 wrapper 只有 `__getattr__`（读转发）而没有 `__setattr__`，直接对 wrapper 赋值会落在外层实例上；`libero_adapter.py:645-664`
- 若 `step` 抛 "executing action in terminated episode"，先回卷再重试一次，仍失败才置 `_darwin_done`；`libero_adapter.py:617-630`

### 3.3 确定性截断：`AttemptStepLimit`

```python
class AttemptStepLimit(BaseException):   # ← 故意继承 BaseException
    def __init__(self, steps, budget): ...
```

- **为什么继承 `BaseException`**：技能层有大量 `except Exception` 兜底（重规划/换候选/回退），预算耗尽必须**穿透**它们直达 runner，否则会被当成普通技能失败吞掉，丢掉"这条轨迹作废"的语义。
- **为什么用步数而不是挂钟**：让"同一初始状态 ⇒ 同一结果"成立——挂钟截断点随机器负载漂移。
- 预算由 runner 在每 attempt 设置（`--attempt-steps`，默认 7000）；`libero_adapter.py:55-69`、`libero_runner.py:244`

## 4. `servo_step()`：统一动作语义的落点

```python
def servo_step(self, site, target, gripper=0.0, k=5.0, vcap=1.0,
               target_rot=None, kr=5.0, actor="agent0") -> None:
    end = np.asarray(self.get_site_pos(site), float)
    delta = np.clip(k * (target - end), -vcap, vcap)      # 位置：一阶 P + 速度上限
    err = 0.5 * (cross(Rc[:,0],Rd[:,0]) + ... )           # 姿态：世界系轴角误差
    d_rot = np.clip(kr * err, -vcap, vcap)
    self.step([dx,dy,dz, dax,day,daz, g])                 # 交给底层 7 维 OSC
```

- gripper 语义统一为 `+1=闭合 / -1=张开 / 0=保持`，与 robosuite OSC 同号直传；
- 姿态伺服用"世界系轴角误差"（`0.5·Σ cross(current, desired)`），与 OSC 的 `goal_R = R_err @ current_R` 左乘约定一致；
- 通用技能只调 `servo_step`，**不需要知道底层是 7 维 OSC 还是别的**；`libero_adapter.py:704-733`
- `set_nullspace_posture(q_arm)` 单独设置 OSC 零空间姿态目标，消除"静态规划构型 ≠ 实际零空间构型"导致的擦碰；`libero_adapter.py:735-744`

## 5. BDDL 语义查询：`articulation_info()`

region/夹具名 → 驱动关节 + 各语义目标 qpos：

```python
cands = fx.joints                                     # fixture 的关节名单
matched = [j for j in cands if token 过滤 top/middle/bottom]   # 多抽屉按 token 区分
if not matched:  # 回退：模型里以夹具名为前缀的关节（如 stove 的 button）
    matched = [j for j in allj if j.startswith(f"{fixture_name}_")]
joint = matched[0]
```

`libero_adapter.py:323-380`

### 5.1 语义目标：取 range 的**内部点**

`_conservative(key, toward)` 取范围内部点（`toward="min"` → 20% 处，`"max"` → 80% 处），而不是取端点：**所有 LIBERO 判定阈值都在 range 端点**（`q < max(...)` 或 `q > min(...)`），内部点必然满足。

| 语义 | 取值 | 对应 LIBERO 判定 |
|---|---|---|
| `open_qpos` | 20% 处 | `is_open: qpos < max(open_ranges)`（**各物件方向不同**，见下）|
| `close_qpos` | 80% 处 | `is_close: qpos > min(close_ranges)` |
| `turnon_qpos` | **20% 处** | `FlatStove.turn_on: qpos >= min(turnon_ranges)` |
| `turnoff_qpos` | 20% 处 | `FlatStove.turn_off: qpos < max(turnoff_ranges)` |

> ⚠️ **曾经的 bug（2026-09-30 修复）**：`turnon_qpos` 原本取"80% 侧"，而 `turn_on` 的阈值在**下界**。flat_stove 的 `default_turnon_ranges = [0.5, 2.1]` 于是被算成 **1.78 rad ≈ 102°**——是判定阈值 0.5 的 3.6 倍，也是 `goal:7`（turn_on_the_stove）连续 32 次"关节卡死在 q≈0"的根因。改成 20% 侧后目标为 0.82 rad ≈ 47°，首次尝试即通过。完整取证见 [12 调试纪要](12_libero_debug_2026-09-30.md)。

注意 LIBERO 各物件的 `is_open` 方向**并不一致**（Microwave/WoodenCabinet 是 `q < max`，ShortCabinet/ShortFridge 是 `q > min`），"取 range 内部点"这一条同时满足两种写法，这是刻意选择。

## 6. 确定性与已知限制

| 项 | 状态 |
|---|---|
| 初始状态 | 固定为 `demo_0 states[0]`（每 attempt 相同）|
| 关节/相机处理 | `_inner_env.seed(0)`、`default_rng(0)` 局部随机源 → 确定 |
| 感知子采样 | **曾被忽略**：`skills/perception/grasp.py` 的 `np.random.choice` 用未播种的全局 RNG → 每次喂给 GraspNet 的点不同。现由 runner 按 attempt 播种（`np.random.seed(attempt)`）|
| 超时截断 | 主判据改为仿真步数（确定性），挂钟仅兜底 |
| **结果** | **仍不完全可复现**：同一任务同一初始状态实测出现过 6 个不同步数（1605/3697/1920/1653/1900/2239）。剩余方差定位在 GraspNet 的 GPU 前向（已排除挂钟超时、CPU 线程调度、RRT 种子）|
| `step()` 卡死风险 | 若某次调用在**不走步**的相位（感知/规划）卡住，步数判据不会触发，只能靠挂钟兜底 |

**推论**：报告基准通过率时，单次通过 ≠ 稳定通过；比较改动效果需要多次运行，或把判据改成"连续一致才算通过"。
