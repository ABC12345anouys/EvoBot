# DarwinBot 改造实施计划（对齐 robopal 真实 API）

> 配套文档：`../darwin-bot架构设计.md`。本计划把架构设计落到当前这份 robopal 0.4.1 代码的**真实接口**上，保证每一步可运行、可验证、可回滚。

## 0. 现状基线（已完成 ✅）

- 专用 conda 环境 `darwin`（Python 3.10），`pip install -e . -r requirements-extra.txt` 已完成。
  - 注意：本机 `~/.local/bin/pip` 会劫持 `pip`，安装/运行一律使用 `/home/lifd/anaconda3/envs/darwin/bin/python -m pip` 与该解释器绝对路径。
- 无头链路已跑通并稳定复现（连跑 2 次）：`tests/smoke_headless.py`
  - PickAndPlace-v1（Panda，单臂，含 EGL 离屏出图）、Drawer-v1（Diana，长程）、BimanualReach-v0（双臂）均可 reset + step。
- 两个**服务器环境坑**及既定解法（所有 darwin 入口统一处理）：
  1. 远程 X11 `DISPLAY` 会让 robopal 导入链（间接经 GL/X 初始化）偶发**无限阻塞**。无头运行时在 `import robopal` 前 `os.environ.pop("DISPLAY")`。
  2. 多卡机器 EGL 可能默认选到显存占满的 GPU 而卡死。默认 `CUDA_VISIBLE_DEVICES=0`，可用 `DARWIN_GPU` 覆盖。
  - 统一封装到 `darwin/runtime.py`，禁止每个脚本各写一遍。
- `pynput` 在 darwin 环境**不要安装**（遥操作才需要，且其 X 后端会阻塞）；键盘/手柄仅在本地有显示的机器上用。

## 1. 设计原则

1. **robopal 零侵入**：不修改 `robopal/` 源码（已验证的基线不动），所有能力放在新顶层包 `darwin/`，通过 wrapper/adapter 调用。确需改动时，优先向 robopal 提 PR 而非本地 fork。
2. **环境即接口**：以 `ManipulateEnv` 的 4 维动作 `[dx,dy,dz,gripper]`（velocity 模式，scale=0.1）与 `(obs, reward, terminated, truncated, info)` 五元组为唯一控制契约；atomic 与 eef 两种模式最终都编译成这 4 维动作。
3. **结构化输出**：LLM 规划结果用 JSON Schema 强约束，禁止自由文本解析；无工具调用能力的模型走 `chat_json`。
4. **每阶段可独立演示**：每阶段交付一个 `scripts/` 入口 + 一个无头验证命令 + 明确验收指标。
5. **成本与延迟可控**：规划用强模型、eef 高频控制用快模型，全部可配置、可 mock（离线/无 key 也能跑通流程）。
6. **进化防退化**：新技能/新 prompt 必须 replay 验证 + benchmark 回归通过才合入（产物走 git 分支）。

## 2. robopal 真实 API 对接表（实现时直接照此调用）

| 能力 | 真实接口 | 位置 |
|---|---|---|
| 构建环境 | `robopal.make(env_name, robot=..., render_mode=None, control_freq=20)`；`robot` 可传注册名(str)或机器人类 | `envs/base.py:21` |
| reset | `obs, info = env.reset(seed=None, options=None)` | `manipulation_tasks/robot_manipulate.py:144` |
| 单臂 step | `obs,r,term,trunc,info = env.step(np.ndarray(4))`；`a[:3]` 末端笛卡尔速度输入，`a[3]` 夹爪[-1,1] | `robot_manipulate.py:73` |
| 双臂 step | `env.step({agent: ndarray(4)})`，`env.agents==['agent0','agent1']`，obs 为 dict | `bimanual_tasks/*` |
| 末端绝对位姿 IK | `env.controller` 为 `CartesianIKController`；底层接受 3 维(仅位置)或 7 维(pos+quat) | `controllers/task_ik_controller.py:39` |
| 正运动学 | `p, quat = env.controller.forward_kinematics(robot.get_arm_qpos(agent))`；`RobotEnv.get_end_abs_pos/quat(agent)` | `envs/robot.py:125` |
| 物体位姿读 | `env.get_body_pos(name)` / `get_body_quat(name)`；site：`get_site_pos(name)` | `envs/base.py:231/313` |
| 物体位姿写 | `env.set_object_pose('green_block:joint', pose7)`；`env.set_site_pos('goal_site', xyz3)` | `envs/base.py:188/194` |
| 关节读 | `robot.get_arm_qpos/qvel(agent)` | `robots/base.py:188` |
| 夹爪直接控制 | `robot.end[agent].open()/close()/apply_action(ctrl)`；PandaHand 范围[0,255]，Rethink[-0.01,0.02] | `robots/grippers.py` |
| 状态存取 | `env.save_state()/get_state()/load_state(state)`（用于 verifier 快照与技能 replay） | `envs/base.py:373` |
| 离屏渲染 | 构造 `render_mode=None, is_render_camera_offscreen=True`；`renderer.image_renderer.update_scene(data,camera=); .render()` 得 RGB | `commons/renderers.py:187` |
| episode 结束 | `truncated` 在 `_timestep>=max_episode_steps(=50)` 为真；成功判据 `info['is_success']`（PickAndPlace 阈值 0.02） | `demo_pick_place.py:96` |

关键约束（实现时务必遵守）：
- `Drawer-v1 / BimanualReach-v0` 等环境的 `__init__` **不接受** `is_render_camera_offscreen` 等 kwargs（签名比 PickAndPlace 窄），wrapper 构造参数要按环境能力裁剪，或统一走 `render_mode=None` + 直接调 `image_renderer`。
- 环境每实例化一次会重新拼接并覆盖 `robopal/assets/robot.xml`（`xml_splice`），并发评测需串行或隔离工作目录。
- 当前 darwin 环境为 numpy 2.2.6；新增代码避免 `np.float`/`np.bool` 等已删除别名。

## 3. 目标目录结构（在 robopal 旁新增）

```
darwin-bot/
├── robopal/                 # 不动
├── darwin/
│   ├── runtime.py           # 【最先做】DISPLAY/GPU/无头环境统一引导
│   ├── config.py            # yaml + 环境变量加载（LLM key、模型名、路径）
│   ├── llm/                 # base/openai_compat/anthropic/ollama/router + mock
│   ├── perception/          # scene_encoder / grounding
│   ├── agents/
│   │   ├── planner/         # atomic_planner / eef_planner / mode_router
│   │   ├── executor.py      # 技能/EEF 指令 -> robopal 4维动作的状态机
│   │   └── verifier.py      # 后置条件（位姿阈值）+ 失败重试
│   ├── skills/              # primitives / registry（带前置后置条件 + replay）
│   ├── memory/              # store(sqlite) / reflector / retrieval
│   ├── evolution/           # skill_forge / failure_miner / prompt_evo / loop
│   ├── benchmarks/          # suite + envs/*.yaml 任务卡
│   └── real2sim/            # urdf2mjcf / calibrate / retarget（最后）
├── configs/                 # llm.yaml env.yaml memory.yaml evolution.yaml
├── scripts/                 # run_agent.py / run_benchmark.py / run_evolution.py
└── tests/                   # 已新增 smoke_headless.py；后续每阶段补单测
```

## 4. 分阶段任务（与架构文档 §9 对应，每阶段独立可演示）

### P0 — 工程骨架与运行时（0.5 天）
- 新增 `darwin/runtime.py`：`setup_headless(gpu=None)` 统一处理 `MUJOCO_GL=egl`、`CUDA_VISIBLE_DEVICES`、移除 `DISPLAY`。
- `darwin/config.py` + `configs/*.yaml`：LLM provider/base_url/model/api_key（读 env，不入库）、控制频率、动作 scale、路径。
- `darwin/envs/adapter.py`：封装 `make_env(env_id, render=False)`，屏蔽各环境 `__init__` 签名差异；统一 `reset/step/render_rgb/close`。
- 验收：`python -c "import darwin; from darwin.envs.adapter import make_env"` 不卡、不弹窗；PickAndPlace 跑 50 步。

### P1 — LLM 层 + 场景编码 + atomic 模式跑通 PickAndPlace（核心里程碑，2–3 天）
- `llm/base.py`：只暴露 `chat(messages)` / `chat_tools(messages, tools)` / `chat_json(messages, schema)`。
- `llm/openai_compat.py`：一套客户端覆盖 DeepSeek/通义/Kimi/GLM（OpenAI 兼容协议）；`llm/mock.py`：规则/模板假模型，保证无 key 可端到端演示。
- `perception/scene_encoder.py`：用 `get_body_pos/quat`、`get_site_pos`、`robot.get_end_xpos/quat` 生成文本场景图（物体/目标/末端位姿 + 工作空间边界），可选挂离屏 RGB。
- `skills/primitives.py` + `registry.py`：先实现 `move_to / open_gripper / close_gripper / pick / place / push`，每个技能声明 `preconditions/postconditions` 与参数 schema，内部产出 robopal 4 维动作序列（velocity 模式）或调用 controller IK。
- `agents/planner/atomic_planner.py`：任务+场景图+技能清单 → 受约束 JSON 指令序列；`agents/executor.py` 逐条执行并维护状态机；`agents/verifier.py` 用位姿阈值判定后置条件。
- `scripts/run_agent.py --env PickAndPlace-v1 --planner atomic --llm mock|deepseek`。
- 验收：mock 模型下机械臂能完成「移动到块上方→下降→闭合→抬起」的脚本化闭环；真实模型在配置 key 后可运行；输出逐步观测日志与渲染帧。

### P2 — executor 重试 + verifier 强化（1 天）
- 失败分类：未到达 / 夹取失败 / 超步数；指数退避式重规划子任务；`env.save_state/load_state` 支持回退到最近检查点。
- 验收：人为制造 1–2 次扰动时任务仍能恢复；记录失败类型计数。

### P3 — eef 模式 + 模式路由（1–2 天）
- `planner/eef_planner.py`：观测 → `chat_json` 输出 4Hz 的 `[dx,dy,dz,gripper]` 增量（直接对齐 env 4 维动作）或绝对位姿（走 controller 7 维 IK）。
- `planner/mode_router.py`：任务特征(长程/精度/历史成功率) → atomic/eef；支持 eef 打底、verifier 连续失败升级 atomic。
- 验收：同一 PickAndPlace，atomic/eef 两路都能跑；路由可手动强制。

### P4 — Benchmark suite + 第一版对比报告（1–2 天）
- `benchmarks/suite.py`：`run(env_id, policy, n_episodes) -> success_rate/avg_steps/失败类型分布`，串行创建环境（规避 robot.xml 覆盖）。
- `benchmarks/envs/*.yaml` 任务卡：随机化范围、目标、评分、难度；覆盖 PickAndPlace/Drawer/Cabinet/Bimanual*，加「开抽屉再取物」组合任务。
- `scripts/run_benchmark.py`，产出 JSON + Markdown 报告（atomic vs eef vs 混合）。
- 验收：无头批量评测出报告，渲染图/轨迹可选存盘。

### P5 — 三层记忆 + reflector（2 天）
- `memory/store.py`：SQLite 存 episodic/semantic/procedural；向量检索先用 numpy 余弦（embedding 可插拔：LLM embedding 或本地哈希），不引重依赖。
- `memory/reflector.py`：episode 结束生成摘要与归因；`retrieval.py` 按任务相似度 top-k 注入 planner prompt。
- 验收：有记忆 vs 无记忆在同一任务集上对比；经验确被注入 prompt。

### P6 — 自进化闭环 + 防退化（2–3 天）
- `evolution/{skill_forge,failure_miner,prompt_evo,loop}.py`：采样→执行→反思→产出技能/避坑/prompt 变体。
- 防退化：新技能必须在仿真 replay 成功才入 registry；产物写 git 分支，P4 benchmark 全量回归通过方可合 main。
- `scripts/run_evolution.py` 支持限时挂机与指标落盘。

### P7 — Real2Sim（按需，最后）
- `real2sim/urdf2mjcf.py` 先导入一台真机 URDF → MJCF 并比对质量/惯量；`calibrate.py` 手眼标定；`retarget.py` 接 eef 轨迹数据。

## 5. 测试与 CI 约定

- 每个 P 阶段在 `tests/` 增加无头测试，命令模板：
  `MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 /home/lifd/anaconda3/envs/darwin/bin/python tests/xxx.py`
- 测试默认 `llm=mock`，真实 LLM 测试用 `@pytest.mark.live` 标记，本地手动跑。
- 不向仓库提交任何 key；`data/`、轨迹、渲染输出加入 `.gitignore`。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| X11/EGL 在服务器阻塞或选错卡 | `runtime.setup_headless()` 统一处理（已验证） |
| LLM 输出不稳定 | 强制 JSON Schema + 重试 + mock 兜底；解析失败按 verifier 失败处理 |
| 各环境构造签名不一致 | adapter 按 env_id 白名单传参；新增环境先补适配 |
| 速度动作积分漂移 | atomic 用绝对位姿经 IK 落地；eef 限制单步增量并由 verifier 纠偏 |
| numpy 2.x / 依赖兼容 | darwin 环境锁定；CI 固定该解释器 |
| 进化产生退化技能 | replay 验证 + benchmark 回归 + git 分支隔离 |
| API key 泄露 | 仅环境变量/本地配置；日志脱敏；定期轮换令牌 |

## 7. 近期立即可做（建议从这里开工）

1. 建 `darwin/` 骨架 + `runtime.py` + `envs/adapter.py`（P0）。
2. 实现 mock LLM + 4 个基础技能 + atomic executor，先在 PickAndPlace 上闭环（P1 的无 key 版本）。
3. 接入你的 DeepSeek/通义等 OpenAI 兼容 key，跑通真实规划。

## 8. 纯控制器任务验证结论（2026-09-17 附录）

### 8.1 成功率总表（判据全部采用官方 info flag，无出生点过滤）

| 任务 | 成功率 | 判据 | 策略脚本 | 视频 |
|---|---|---|---|---|
| PickAndPlace-v1 | 5/5 | 抬离 0.52m + 终点<2cm | `scripts/controller_success.py` | `PickAndPlace-v1-success.mp4` |
| MultiCubeStack-v1 | 2/3 | 逐块 is_success | `scripts/check_multi_stack.py` | `MultiCubeStack-v1.mp4` |
| BimanualReach-v0 | 3/3 | 双 eef 入域 | `scripts/check_bimanual_reach.py` | `BimanualReach-v0.mp4` |
| BimanualPickAndPlace-v0 | 3/3 | is_success | `scripts/check_bimanual_pick_place.py` | `BimanualPickAndPlace-v0.mp4` |
| BimanualTransport-v0 | 3/3 | is_success | `scripts/check_bimanual_transport.py` | `BimanualTransport-v0.mp4` |
| Drawer-v1 | 3/3 | is_success（dist<2cm） | `scripts/check_cabinet_family.py` | `Drawer-v1-ep{1,2,3}.mp4` |
| DrawerBox-v1 | 3/3 | is_drawer+is_place | 同上 | `DrawerBox-v1-ep{1,2,3}.mp4` |
| LockedCabinet-v1 | 3/3 | is_unlock+is_door | 同上 | `LockedCabinet-v1-ep{1,2,3}.mp4` |

运行模板：`MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 [REC=1] [TASKS=<名>] python scripts/check_cabinet_family.py`
（REC=1 录像至 `data/videos/<task>-ep<N>.mp4`，失败局自动删除；TASKS 支持子串过滤分任务跑防墙钟超时。）

### 8.2 柜体族三任务攻关记录（修正 8.3 旧 Drawer 根因结论）

旧结论"腕姿跟踪无法收敛、需改控制器内核"**被推翻**——根因是抓取/拉取力学，CARTIK 完全可用。实证机制链：

1. **夹爪间隙**：env 归一化闭合 −0.01 → 指尖条间隙 2.65cm 夹不住 2cm 把手。补丁：闭合指令发 −0.020833（机械全闭合，actuator ctrlrange 下限）→ 间隙 1.45cm，对 2cm 把手 5.5mm 过盈楔紧（kp=1000 ≈11N/指，μ≈1）。
2. **强楔窗口**：抓取点 z=0.437（eef 落 0.4335-0.4385 窗）时指杆中段对把手形成 2mm 过盈，静态楔紧 ~20N > 抽屉 frictionloss 10N。**descend 必须慢降**（k=1.5 + lag<0.015 严门控）：k=2 快降会让 desired 过冲到 eef 下方 27mm，close 后无钩载必打滑。
3. **钩载机制**：close 后 eef 被把手卡阻压低（0.468 目标 → eef 停 0.435），desired−eef 的 z 差自动形成恒定向上拉力，把手钩死在指杆楔角。拉取期 desired_z 再自由积分（+0.0015/步，z lag 最多 87mm）持续加钩——**z lag <30mm 会滑脱，50-90mm 稳定**。
4. **相对蠕进拉取**：target=当前 eef+4mm·dir，desired ~0.8mm/步匀速前移，滞后自然突破摩擦。门控冻结版力恒低于 frictionloss 突破阈值 → 死锁，弃用。
5. **y 伺服**：close 接触反馈会把 desired 沿 y 泄放最多 17mm（把手厚仅 2cm）——pull 的 step_fn 加 `−0.003 if desired_y>3mm else 0` 把 desired_y 拉回把手中心，DrawerBox 必需。
6. **工作空间边界**：DianaDrawerCube 臂在 z=0.462 目标时逆解收敛不到 x=0.62（desired 卡 0.583），z=0.437 时可达 0.610——**降目标 z 可显著改善逆解收敛域**。
7. **LockedCabinet 门**：横梁解锁用 gate 模式（t≈40 即成）；门把手 C 方杆在固定 yaw 夹爪中随门旋转会挤出楔缝 → **多次抓取循环**（4 次 × 300 步，铰链 frictionloss=2 锁角、进度累积 0.56→0.62→突破 0.96 rad 判据）。切向方向 = cross(z, 把手−铰链)，符号朝 opened site。
8. **墙钟/预算**：模拟 ~12 步/秒；门 ep ~1036 步 ≈ 85s。滑脱早退检测（armed 后 lag<8mm 连续 10 步）避免死等。

### 8.3 通用经验

- `MujocoEnv.close()` 会 `os._exit(0)`，多 episode 脚本必须 monkeypatch（`MujocoEnv.close = lambda self: self.renderer.close()`）。
- `env.is_randomize_end = False` 须在 make 后赋值（构造器不透传该参）。
- P 速度控制的 desired 积分在 eef 被反顶时会漂移 → 一切推进带 lag 门控；lag 必须用 reset 时一次性捕获的基座平移 off 计算（每步重算会恒为 0）。
- close 阶段全冻结 desired（vel=0），楔紧靠阻抗自稳；z 伺服配 desired 积分是失控积分器（desired 漂 25cm 把手臂拽走）。
- 已应用 robopal 补丁：`RethinkGripper.xml` 指面摩擦 0→1（真实夹爪内侧为橡胶，合理化）。
