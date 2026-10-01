# LIBERO 实验栈设计文档

本目录只保留 **LIBERO 基准栈**（BDDL 形式化规格 → 确定性执行）的设计文档。

> 2026-10-01：与 LIBERO 无关的部分（robopal 栈：LLM 决策 + RAG 记忆 + 进化）连同其代码（`robopal/`、`ragbot/`）已从仓库移除；原 01–06、13–16 号文档随之下线（历史仍可在 git 里取回）。

## 文档索引

| 编号 | 文档 | 对应代码 | 核心内容 |
|------|------|----------|----------|
| 07 | [感知技能](07_perception.md) | `darwin/skills/perception/` | YOLO/SAM/GraspNet 双模式、点云采样；**LIBERO/MuJoCo 真值采样路径、两条感知路径的区别、随机源全景、GPU 前向与不可复现性** |
| 08 | [LIBERO 规划与执行栈](08_libero_planning_stack.md) | `darwin/agents/task_spec.py`、`libero_planner.py`、`libero_runner.py` | BDDL→有序子目标（稳定拓扑排序）、谓词→技能映射、析取规划、机制驱动重规划、attempt 闭环与双判据超时、台账 |
| 09 | [LIBERO 技能库](09_libero_skills.md) | `darwin/agents/libero_skills.py` | `serve`/`goto` 底座与统计判据、grasp 的候选枚举与 13 道门、`mem` 记忆字典、`articulate`/`toggle` 的流形跟随（θ vs q 驱动）、调参常量全表 |
| 10 | [物理机制判别与退路](10_physics_mechanisms.md) | `darwin/physics/` | 10 种失败机制及判别依据、Θ 割与几何指令两类退路、关节/流形数学、机制消费方接线图 |
| 11 | [环境适配层（LIBERO）](11_envs_libero_adapter.md) | `darwin/envs/libero_adapter.py` | 固定初始状态、自管 episode 长度（horizon 回卷）、`servo_step` 统一动作语义、`articulation_info` 语义目标、`AttemptStepLimit` 确定性截断 |
| 12 | [LIBERO 调试纪要 2026-09-30](12_libero_debug_2026-09-30.md) | — | 一次实际排障的全记录：三组失败归因、`turnon` 目标端点 bug 的取证链、步数判据标定、不可复现性排查与**已排除清单**、遗留问题 |

## 栈结构

关键取舍：**不靠 LLM 猜子目标，把 BDDL 当形式化规格直接解析**。

```
BDDL ──► task_spec ──► libero_planner ──► libero_skills ──► envs/libero_adapter ──► LIBERO/robosuite
(08)         │              │                  │(09)              │(11)
             │              │                  └── 失败机制 ──► physics 判别与退路 (10)
             └──────────────┴──► libero_runner：attempt 闭环 / 双判据超时 / 台账 / 断点续跑
```

## 阅读建议

- **想跑起来**：读 [08 规划与执行栈](08_libero_planning_stack.md)；入口是 `scripts/verify_30.sh`（30 任务全量回归）或 `scripts/ipc_learn.py`（单任务 + 反思学习）。
- **想理解技能**：读 [09 技能库](09_libero_skills.md)。
- **想加新技能/新任务**：读 [08](08_libero_planning_stack.md)（谓词→技能映射）与 [09](09_libero_skills.md)（技能实现约定）。
- **排障**：读 [10 物理机制](10_physics_mechanisms.md) → [12 调试纪要](12_libero_debug_2026-09-30.md)；取证脚本在 `scripts/dev_probes/`。
- **注意**：结果**目前不完全可复现**（噪声来自 GraspNet 的 GPU 前向，已排除挂钟超时/CPU 线程/RRT 种子）。报告通过率时请看 [12](12_libero_debug_2026-09-30.md)——**单次通过 ≠ 稳定通过**。
