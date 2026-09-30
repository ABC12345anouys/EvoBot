"""探针：验证面对新任务（无匹配 method）时 SkillCreator 自动生成方法。

场景：只向 registry 注册 OpenDrawerMethod，跑 drawer_place。
place 条件（BodyNearSite cart）无匹配 → 触发 SkillCreator：
  1. LLM 不可用 → 规则回退生成 auto_body_near_site.yaml
  2. DeclarativeMethod 校验通过 → register
  3. 重试 select 成功 → 执行该链
结束后检查：
  - methods/auto_body_near_site.yaml 已生成
  - experience store 有该方法的记录
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from darwin.benchmarks import get_benchmark
from darwin.agents.runner_dynamic import DynamicEpisodeRunner
from darwin.agents.chain_registry import ChainRegistry, OpenDrawerMethod, METHODS_DIR
from darwin.agents.experience_store import ExperienceStore
from darwin.memory import RAGMemory

# 1. 构造一个"只有开抽屉方法"的精简 registry → place 条件无匹配
reg = ChainRegistry()
reg.register(OpenDrawerMethod())

exp = ExperienceStore(root=tempfile.mkdtemp(prefix="probe_exp_"))

entry = dict(get_benchmark("drawer_place"))
entry["safe_home_qpos"] = [-0.614, -0.2586, -0.7121, 2.2855,
                           0.2737, -0.6727, -0.2971]
entry["obstacles"] = ["cupboard", "drawer"]

log_dir = tempfile.mkdtemp(prefix="probe_sk_logs_")
rag = RAGMemory(root=tempfile.mkdtemp(prefix="probe_rag_"))
runner = DynamicEpisodeRunner(entry, rag=rag, max_attempts=2, verbose=True,
                              log_dir=log_dir, sample_every=20,
                              registry=reg, experience=exp)

# 确认初始 registry 没有 place 方法
print(f"[probe] 初始 registry: {[m['name'] for m in reg.list_methods()]}")

auto_yaml = METHODS_DIR / "auto_body_near_site.yaml"
if auto_yaml.exists():
    auto_yaml.unlink()
    print(f"[probe] 清理旧的 {auto_yaml.name}")

cfg = {"hover": 0.12, "k_descend": 2.0, "lift_height": 0.52, "jit": 0.0}
res = runner.run(cfg)

print("\n[probe] ===== result =====")
print(f"success={res['success']} metrics={res['metrics']}")

# 2. 验证自动生成的 YAML 方法存在且被注册
print(f"[probe] registry 现在: {[m['name'] for m in reg.list_methods()]}")
print(f"[probe] auto YAML 存在: {auto_yaml.exists()}  path={auto_yaml}")
if auto_yaml.exists():
    print("--- auto_body_near_site.yaml ---")
    print(auto_yaml.read_text())

# 3. 验证经验库有记录
print("[probe] experience summary:")
import json
print(json.dumps(exp.summary(), ensure_ascii=False, indent=2))

assert auto_yaml.exists(), "自动生成方法未写入磁盘"
assert any(m["name"] == "auto_body_near_site" for m in reg.list_methods()), \
    "自动生成方法未注册到 registry"
print("\n[probe] OK: 新任务触发 SkillCreator 自动生成方法并注册 ✓")
