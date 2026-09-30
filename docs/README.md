# EvoBot 模块设计文档

本目录包含 EvoBot 各核心模块的设计思路文档。

## 文档索引

| 编号 | 文档 | 对应代码 | 核心内容 |
|------|------|----------|----------|
| 01 | [RAG 记忆层](01_rag_memory.md) | `ragbot/memory/` | 归一化经验、失败 TTL、语义检索、UCB 排序、evidence 置信度升级 |
| 02 | [智能体与执行器](02_agent_runner.md) | `darwin/agents/` | 五步自进化闭环、去分发架构、候选生成、多尝试重试 |
| 03 | [技能框架](03_skills_framework.md) | `darwin/skills/` | 三层技能体系、SkillSpec 元数据、控制/运动/感知原语 |
| 04 | [进化系统](04_evolution_system.md) | `darwin/evolution/` | MAP-Elites 质量多样性、skill_forge 成功轨迹沉淀 |
| 05 | [机器人后端抽象](05_robot_backend.md) | `darwin/robot/` | RobotBackend ABC、YAML Profile、机型解耦、mplib 集成 |
| 06 | [LLM 集成](06_llm_integration.md) | `darwin/llm/` | OpenAI 兼容协议、function calling 风格、规则回退 |
| 07 | [感知技能](07_perception.md) | `darwin/skills/perception/` | YOLO/SAM/GraspNet 双模式、点云采样、RAG 接口；**LIBERO/MuJoCo 真值采样路径、两条感知路径的区别、随机源全景、GPU 前向与不可复现性** |
| 08 | [LIBERO 规划与执行栈](08_libero_planning_stack.md) | `darwin/agents/task_spec.py`、`libero_planner.py`、`libero_runner.py` | BDDL→有序子目标（稳定拓扑排序）、谓词→技能映射、析取规划、机制驱动重规划、attempt 闭环与双判据超时、台账 |
| 09 | [LIBERO 技能库](09_libero_skills.md) | `darwin/agents/libero_skills.py` | `serve`/`goto` 底座与统计判据、grasp 的候选枚举与 13 道门、`mem` 记忆字典、`articulate`/`toggle` 的流形跟随（θ vs q 驱动）、调参常量全表 |
| 10 | [物理机制判别与退路](10_physics_mechanisms.md) | `darwin/physics/` | 10 种失败机制及判别依据、Θ 割与几何指令两类退路、关节/流形数学、机制消费方接线图 |
| 11 | [环境适配层（LIBERO）](11_envs_libero_adapter.md) | `darwin/envs/libero_adapter.py` | 固定初始状态、自管 episode 长度（horizon 回卷）、`servo_step` 统一动作语义、`articulation_info` 语义目标、`AttemptStepLimit` 确定性截断 |
| 12 | [LIBERO 调试纪要 2026-09-30](12_libero_debug_2026-09-30.md) | — | 一次实际排障的全记录：三组失败归因、`turnon` 目标端点 bug 的取证链、步数判据标定、不可复现性排查与**已排除清单**、遗留问题 |

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                      进化系统 (04)                        │
│   MAP-Elites 搜索  ←→  skill_forge 成功轨迹沉淀          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   智能体与执行器 (02)                      │
│   perceive → retrieve → decide → execute → reflect      │
└──────┬───────────┬──────────────┬──────────────┬────────┘
       │           │              │              │
┌──────▼──┐ ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│ 感知(07) │ │ RAG记忆(01) │ │ 技能(03) │ │ LLM(06)     │
│ YOLO/SAM │ │ 经验库+TTL  │ │ 原语+策略│ │ 规划+回退   │
│ GraspNet │ │ 语义检索    │ │ forged   │ │             │
└─────────┘ └─────────────┘ └────┬─────┘ └─────────────┘
                                 │
                          ┌──────▼──────┐
                          │ 机器人后端(05)│
                          │ Backend ABC │
                          │ + YAML Profile│
                          └─────────────┘
```

### 另一条栈：LIBERO 基准（08–12）

上面的 01–06 描述的是 **robopal 栈**（LLM 决策 + RAG 记忆）。LIBERO 基准跑的是**另一套并列的栈**，关键取舍是"不靠 LLM 猜子目标，把 BDDL 当形式化规格直接解析"：

```
BDDL ──► task_spec ──► libero_planner ──► libero_skills ──► envs/libero_adapter ──► LIBERO/robosuite
(08)         │              │                  │(09)              │(11)
             │              │                  └── 失败机制 ──► physics 判别与退路 (10)
             └──────────────┴──► libero_runner：attempt 闭环 / 双判据超时 / 台账 / 断点续跑
```

实测记录与踩坑汇总见 [12 调试纪要](12_libero_debug_2026-09-30.md)。

## 阅读建议

- **想理解核心创新**：先读 [01 RAG 记忆层](01_rag_memory.md) 和 [04 进化系统](04_evolution_system.md)
- **想理解运行流程**：读 [02 智能体与执行器](02_agent_runner.md)
- **想扩展新机型**：读 [05 机器人后端抽象](05_robot_backend.md)
- **想理解技能体系**：读 [03 技能框架](03_skills_framework.md)
- **要在 LIBERO 基准上复现/排障**：依次读 [08 规划与执行栈](08_libero_planning_stack.md) → [09 技能库](09_libero_skills.md) → [10 物理机制](10_physics_mechanisms.md) → [11 环境适配层](11_envs_libero_adapter.md)，排障手法与实测数据见 [12 调试纪要](12_libero_debug_2026-09-30.md)
- **注意**：LIBERO 基准的结果**目前不完全可复现**（噪声来自 GraspNet 的 GPU 前向，已排除挂钟超时/CPU 线程/RRT 种子）。报告通过率时请看 [12](12_libero_debug_2026-09-30.md) 的说明——**单次通过 ≠ 稳定通过**。
