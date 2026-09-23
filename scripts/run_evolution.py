"""端到端进化：EvolutionLoop + MAP-Elites + DynamicEpisodeRunner 真实运行。

改造点（P0-1）：从旧版 EpisodeRunner 切换到 DynamicEpisodeRunner，
接入 SkillCreator（新任务自动生成方法）+ ExperienceStore（跨进程经验累积）
+ 条件驱动规划 + forged skill 注册表。

流程：
1. benchmark 定义任务分布（darwin/benchmarks）
2. DynamicEpisodeRunner 作为 agent_runner：条件驱动 observe→select→execute
3. MAP-Elites 按 (attempts × steps) 分箱维护多样 cfg
4. 成功 → skill_forge 沉淀 forged skill（自动进 registry）；失败 → RAG + ExperienceStore
5. 经验库 logs/experience/*.jsonl 跨进程累积，反馈到 SkillCreator 防重复

运行（darwin 环境）：
    CUDA_VISIBLE_DEVICES=0 python scripts/run_evolution.py                 # pickplace 3 轮
    CUDA_VISIBLE_DEVICES=0 python scripts/run_evolution.py stack_grasp_red 2
    CUDA_VISIBLE_DEVICES=0 python scripts/run_evolution.py pickplace 2 norecord
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from darwin.benchmarks import get_benchmark
from darwin.evolution import EvolutionLoop
from darwin.memory import RAGMemory
from darwin.agents import ManipulationAgent
from darwin.agents.runner_dynamic import DynamicEpisodeRunner
from darwin.agents.experience_store import ExperienceStore
from darwin.agents.chain_registry import default_registry

RECORD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "videos", "evolution_loop")


def main():
    bench_name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else "pickplace"
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3

    entry = get_benchmark(bench_name)
    rag = RAGMemory()
    agent = ManipulationAgent(rag=rag, task_name=entry["task_name"], seed="me")

    # 条件驱动 runner：注入 registry（含内置+YAML+forged）+ 跨进程经验库
    registry = default_registry()
    experience = ExperienceStore()   # 自动加载 logs/experience/*.jsonl 历史

    if "norecord" not in sys.argv[1:]:
        os.makedirs(RECORD_DIR, exist_ok=True)
    runner = DynamicEpisodeRunner(
        entry, rag=rag, agent=agent, max_attempts=6,
        record_dir=(None if "norecord" in sys.argv[1:] else RECORD_DIR),
        seed="me", verbose=True,
        registry=registry, experience=experience,
    )

    loop = EvolutionLoop(
        benchmark=entry, memory_store=None,
        search_space=entry["search_space"],
        feature_dimensions=entry["feature_dimensions"],
        feature_ranges=entry["feature_ranges"],
        n_iterations=n_iter,
    )

    print("=" * 62)
    print(f"EvolutionLoop 端到端 | benchmark={bench_name} | iterations={n_iter}")
    print(f"registry: {[m['name'] for m in registry.list_methods()]}")
    print(f"经验库: {experience.summary()}")
    print(f"初始记忆: {rag.n_failure} fail / {rag.n_success} succ | skills: {list(agent.skills.keys())}")
    print("=" * 62)

    result = loop.run(runner.run)

    print("\n===== MAP-Elites 归档 =====")
    archive_info = result["history"]
    print(f"迭代数: {len(archive_info)} | 归档 cell 数: {result['archive_size']}")
    for h in archive_info:
        tag = "accept" if h["accepted"] else "reject"
        print(f"  step{h['step']} [{h['source']}] cell={h['cell']} "
              f"fitness={h['fitness']:.3f} {tag} | cfg={json.dumps(h['cfg'])}")

    print("\n===== forged 技能 =====")
    from darwin.skills.forged import FORGED_DIR
    if FORGED_DIR.exists():
        for d in FORGED_DIR.iterdir():
            if d.is_dir():
                print(f"  {d.name}")

    print("\n===== 记忆 =====")
    print(f"RAG 经验: {rag.n_failure} fail / {rag.n_success} succ")
    print(f"统计表: {json.dumps(rag.stats, ensure_ascii=False)}")
    print(f"ExperienceStore: {json.dumps(experience.summary(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
