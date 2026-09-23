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
| 07 | [感知技能](07_perception.md) | `darwin/skills/perception/` | YOLO/SAM/GraspNet 双模式、点云采样、RAG 接口 |

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

## 阅读建议

- **想理解核心创新**：先读 [01 RAG 记忆层](01_rag_memory.md) 和 [04 进化系统](04_evolution_system.md)
- **想理解运行流程**：读 [02 智能体与执行器](02_agent_runner.md)
- **想扩展新机型**：读 [05 机器人后端抽象](05_robot_backend.md)
- **想理解技能体系**：读 [03 技能框架](03_skills_framework.md)
