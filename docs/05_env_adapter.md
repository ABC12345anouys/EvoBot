# 05 · 环境适配层

> 对应代码：`darwin/envs/libero_adapter.py`
>
> 相连的部分：上层是《规划与执行栈》和《技能库》两篇里描述的执行链，下层是 LIBERO / robosuite 提供的仿真环境。

## 1. 这一层解决什么问题

项目用 **Agent + 可复用技能库** 的方式跑 LIBERO 机器人操作基准：Agent 负责把任务拆成有序子目标，技能库负责把每个子目标落到具体的机械臂动作上。

但 LIBERO 这侧自己有一套接口，和技能库需要的接口对不上：

- LIBERO 只接受一个 7 维的机械臂控制向量（末端三个方向的位移、三个方向的转动、加夹爪开合），没有"把夹爪移到某个位置"这种说法；
- 它用自己的一套数据描述任务是否完成，技能库需要的是按子目标逐条查询；
- 它给回合长度设了固定上限，到点就宣布本回合结束，而这里希望由上层自己决定跑多少步。

`LiberoEnvAdapter` 就是这两边之间唯一的适配层。它把 LIBERO 的仿真包成统一接口，让技能库不需要知道底层是 robosuite 还是别的仿真器；换后端时，执行链主体不用改。

下面几节分别说它对外暴露的接口、初始状态的处理、回合长度的接管、步数上限，以及环境的创建方式。

## 2. 对外提供的统一接口

### 2.1 动作：`servo_step()`

技能库描述动作只有一种方式——"让末端朝某个目标点伺服一步"：

```python
servo_step(site, target, gripper=0.0, k=5.0, vcap=1.0,
           target_rot=None, kr=5.0, actor="agent0")
```

- `site` 是末端参考点（LIBERO 里是 `gripper0_grip_site`），`target` 是想要的末端位置；
- 位移按一阶比例控制算：`delta = clip(k * (target - 当前位置), -vcap, vcap)`，`k` 是增益，`vcap` 限制单步移动上限；
- 给了 `target_rot` 就同时伺服姿态，用世界坐标系下的轴角误差，`kr` 是旋转增益；
- `gripper` 统一成 `+1=闭合 / -1=张开 / 0=保持`，方向和 robosuite 一致，直接传下去。

也就是说，通用技能只调 `servo_step`，由适配层把它翻译成底层需要的 7 维向量。技能代码里不会出现"第 5 维是绕 x 轴转动"这类细节。

另外提供一个 `set_nullspace_posture(q_arm)`，用来指定机械臂在任务空间之外那些自由度（零空间）的目标关节角，让实际构型跟着关节空间的规划走，减少不该有的擦碰。

### 2.2 读取位姿与关节角

| 想要什么 | 怎么取 |
|---|---|
| 末端位置 | `get_site_pos(site)` |
| 末端姿态（3×3 旋转矩阵） | `get_site_rot(site)` |
| 物体位置 | `get_body_pos(name)`，`name` 用任务里的物体名，内部会解析到实际的模型刚体 |
| 关节角 | `mj_data.qpos[...]`，关节地址从 `mj_model` 查 |

`mj_model` 是仿真模型句柄，`mj_data` 是仿真数据。注意 `mj_data` 是每次访问现取的，不能存成字段缓存下来——包装层每次调用都会返回一个新的对象，缓存会拿到一份不再更新的旧数据。

### 2.3 子目标是否达成

- `check_success()`：整个任务是否完成，对应 LIBERO 自己的完成判定。
- `eval_subgoal(predicate, obj, target)`：按任务定义里的同名规则求值，用于一条一条地检查子目标。
- `fixture_open(name)` / `fixture_close(name)`：柜门、抽屉这类可开合部件的开合状态。

任务定义文件（LIBERO 用 `.bddl` 格式描述任务）里有些条件只写了"打开"而没有单独列成一条规则，这类子目标由 `fixture_open` 负责判定。

### 2.4 关节类物体的语义目标：`articulation_info()`

开门、拉抽屉、按开关这类动作需要知道"该动哪个关节"和"动到多少才算成"。`articulation_info(name)` 一次给出这些信息：

```python
{
  "fixture":   夹具名,
  "joint":     驱动关节名,
  "qpos":      该关节的当前角度,
  "range":     关节的活动范围,
  "open_qpos":   打开的目标角度,
  "close_qpos":  关闭的目标角度,
  "turnon_qpos": 开启的目标角度,
  "turnoff_qpos":关闭的目标角度,
}
```

关节怎么找：先看夹具自己声明的关节名单，名字里带 `top` / `middle` / `bottom` 的用来区分上下多个抽屉；如果名单是空的，就退回"模型里以夹具名为前缀的关节"（比如灶台上那个按钮）。

目标角度不取范围的端点，而是取范围内部靠边的一点（`open` 取靠近下界的 20% 处，`close` 取靠近上界的 80% 处）。原因是 LIBERO 的判定阈值本身就压在范围端点上，取内部点能稳定满足条件，同时给物理漂移留出余量。`turnon` / `turnoff` 的方向和 `open` / `close` 不完全一样，按各自判定阈值所在的一侧取值。

## 3. 固定初始状态

同一任务每次尝试都从**同一条示范轨迹的初始状态**开始。具体做法是：从该任务的数据文件里读出第一条示范（`demo_0`）的第一个状态，`reset()` 时先做环境的常规复位，再用 `set_init_state()` 把状态设成这一帧。

这样做的意义是：所有对比实验的起点完全相同。改了一个参数、换了一种抓取方式之后，前后的差别来自改动本身，而不是来自随机的初始摆放。`reset()` 同时会记下开局的末端位置，供"回到安全位"这类动作使用。

需要回放示范动作时，`demo_actions(suite, idx)` 可以读出对应的动作序列。

## 4. 回合长度管理（回卷 horizon）

robosuite 给每个环境设了固定的回合长度（默认 1000 个控制步），计数到点就把 `done` 置为真。但这里的执行链是**自己控制步数**的：一条轨迹该跑多少步，由子目标是否达成决定，而不是由仿真器的计数器决定。技能里比较长的搬运段单独就可能跑 500 步以上，加上移动、下压、抬升，很容易超过 1000 步。

适配层的处理是：`step()` 时如果发现内层计数已经到上限，就把它清零再继续（回卷 horizon）。同时把内层给出的 `done` 当成"计数器到点了"，不当成本回合结束——回合是否结束由任务条件来判定。

清零时有个实现细节：robosuite 的包装层只转发属性读取，不转发属性写入，所以要对包装层逐层往里走，直到真正的环境对象再赋值，否则数字会写在外层壳上、起不到作用。

## 5. 确定性截断：`AttemptStepLimit`

每条轨迹有步数上限。超过上限时，抛出一个 `AttemptStepLimit` 异常：

- **它继承的是 `BaseException` 而不是 `Exception`**。技能层里有不少 `except Exception` 用来处理一般的技能失败（换抓取候选、重规划等），而"这条轨迹的步数用完了"是更高一层的信号，必须穿过这些处理直接到 runner，否则会被当成普通技能失败吞掉。
- **上限以仿真步数计，不以挂钟时间计**。挂钟会随机器负载变化，同一个初始状态在同一台机器上忙时闲时会在不同位置被截断；用步数则与负载无关，同一初始状态走到的是同一个地方。另外还有一个挂钟上限作为备用限制，只在进程卡住、步数根本没往前走的时候才会先触发。

上限由 runner 在每次尝试开始时设置（命令行参数 `--attempt-steps`，默认 7000）。为了让重试有变化，runner 还会按尝试序号给感知用到的随机数发生器设种子。

## 6. 环境创建与缓存

环境用 `env_id` 标识，格式是 `libero:<suite>:<idx>`，例如 `libero:libero_spatial:0`。`parse_env_id()` 负责解析。

有两层缓存：

- `_task_info(suite, idx)`：缓存任务的元信息（任务定义文件路径、数据文件路径、示范初始状态、示范动作条数），读一次就能反复用，用 `lru_cache` 限制在最近 8 个任务。
- `env_utils.get_env()`：按 `suite` 和 `idx` 缓存已经建好的环境，同一个任务重复跑不会反复加载模型。

基准侧由 `darwin/benchmarks/__init__.py` 的 `get_libero_benchmark(suite, task_idx)` 动态生成任务条目（LIBERO 的任务集太大，不放进静态表里），其中带上环境 id、任务语言描述、目标信息等，供上层使用。

## 7. 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| 仿真步长 | 0.002 s | robosuite 默认，适配层不改 |
| 控制频率 | 20 Hz | 每个动作推进 0.05 s |
| 动作维度 | 7 | 末端位移 ×3、末端转动 ×3、夹爪开合 ×1 |
| 内层回合长度 | 1000 控制步 | 到点回卷，不作为回合结束 |
| 相机分辨率 | 128 px（默认） | 构造时可传 `camera_size` 修改 |
| 环境随机种子 | `seed(0)` | 构造时固定 |
| 感知随机种子 | 按尝试序号设置 | 同一次尝试内结果一致，不同尝试仍有差异 |
| 单条轨迹步数上限 | 7000 | runner 参数 `--attempt-steps`，0 表示不限 |
| 单条轨迹挂钟上限 | 300 s | runner 参数 `--attempt-timeout`，备用限制 |
| 伺服默认增益 / 速度上限 | `k=5.0` / `vcap=1.0` | `servo_step` 默认值 |
| 姿态伺服默认增益 | `kr=5.0` | 给了 `target_rot` 时生效 |
| 任务元信息缓存 | 最近 8 个任务 | `_task_info` 的 `lru_cache` |

## 8. 最小用法示例

```python
import numpy as np

from darwin.envs.libero_adapter import LiberoEnvAdapter

# 建立环境：任务集 + 任务序号
env = LiberoEnvAdapter("libero_spatial", 0)
env.reset()

site = "gripper0_grip_site"
start = env.get_site_pos(site)
print("末端起点:", start)

# 让末端往下走 5cm，夹爪张开（真实使用时会循环下发，直到到位）
target = start + np.array([0.0, 0.0, -0.05])
env.servo_step(site, target, gripper=-1.0)

# 读关节角：地址在 mj_model 里查，数值在 mj_data 里读
jid = env.mj_model.joint_name2id("robot0_joint1")
q = float(env.mj_data.qpos[env.mj_model.jnt_qposadr[jid]])
print("关节角:", q)

# 查询某个柜门的开合状态与驱动关节
print(env.articulation_info("wooden_cabinet_1_top_region"))
print("任务是否完成:", env.check_success())

env.close()
```

如果环境已经由 runner 建好，用 `darwin.agents.env_utils.get_env("libero:libero_spatial:0", "libero_panda")` 直接取缓存实例即可，不要重复创建。

实际执行一条轨迹时，步数上限要显式设置：

```python
env.reset()
env.step_budget = 7000          # None 表示不限
# …技能循环调用 env.servo_step(...)…
```
