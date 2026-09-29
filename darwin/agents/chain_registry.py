"""ChainRegistry：调用链方法的 list+map 动态注册表。

解决的问题：runner_dynamic 的 METHOD_REGISTRY 是写死的 Python 列表，新增/调整
方法要改代码。本模块提供：

- list + map 双层结构：_order(list) 保存优先级顺序，_items(dict) 按 name O(1)
  定位，支持运行时 register / unregister / reorder / 同名替换。
- 声明式方法 DeclarativeMethod：从 YAML dict 定义"能匹配哪类条件 + skill 步骤
  模板"，参数里的 $变量 在规划当下从 env/cond/cfg 现场解析（白名单取值，无 eval）。
  因此 Codex/Claude Code 或用户新增方法 = 往 methods/ 目录放一个 YAML，不改 Python。
- 来源标记 source：builtin(Python 底座) / yaml:<路径>(声明式，可增删改)。
  同名时 yaml 可 replace 内置方法，实现"动态改执行的函数"。

YAML 方法 schema：
    name: open_drawer
    description: 可选说明
    match: {kind: joint_ge}            # 或 {kind: body_near_site, flavor: cart}
    include_home: false                # 链是否自带 home（默认 false，引擎补）
    steps:
      - skill: move_to
        params:
          target: ["$site:drawer.x", 0.0, "$safe_z"]
          gripper: 0.0
      - skill: pull_drawer
        params: {joint_name: "$cond.joint", target_qpos: "$cond.target_qpos"}

模板变量（$ 前缀，仅这些白名单，支持 .x/.y/.z 取分量）：
    $safe_z                  安全走廊高度（collision.SAFE_Z）
    $grasp_pt[.x|.y|.z]      当前抓取候选点
    $site:<name>[.axis]      env.get_site_pos(name) 现场坐标（把手/目标点）
    $goal_site[.axis]        cond.site 的现场坐标（BodyNearSite 条件）
    $cond.<attr>             条件对象属性（joint/target_qpos/body/site/...）
    $cfg.<key>               搜索空间配置（hover/k_descend/lift_height/...）
    $entry.<key>             benchmark entry 字段
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..skills.primitives.collision import SAFE_Z
from .objectives import GoalCond
from .methods import (
    Method, PlanContext,
    OpenDrawerMethod, GraspLiftMethod, TransferCartMethod,
    TransferPoseMethod, TransferPoseCartMethod,
    LiberoPushMethod, IKLiberoTransferMethod,
    LiberoArticulateMethod, ReplayDemoMethod,
)

# 声明式方法目录（仓库内持久化的动态方法；examples/ 子目录不自动加载）
METHODS_DIR = Path(__file__).resolve().parent / "methods"

# $cfg.* 缺失时的兜底默认值（与 PlanContext 属性一致）
_CFG_DEFAULTS: Dict[str, Any] = {
    "hover": 0.12,
    "k_descend": 2.0,
    "lift_height": 0.52,
    "stiffness": 100.0,
    "damping": 40.0,
    "spiral_radius": 0.004,
}


# ============================================================
# 模板变量解析（白名单，禁止任意表达式）
# ============================================================

_AXIS = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


class TemplateError(ValueError):
    """YAML 方法模板解析失败（未知变量/字段/类型）。"""


def _axis_of(vec, token: str):
    """取 .x/.y/.z 分量；无后缀返回完整 list。"""
    if "." not in token:
        return [float(v) for v in np.asarray(vec, float).reshape(-1)]
    suffix = token.rsplit(".", 1)[1]
    if suffix not in _AXIS:
        raise TemplateError(f"未知分量 '.{suffix}'（仅支持 .x/.y/.z）: ${token}")
    return float(np.asarray(vec, float).reshape(3)[_AXIS[suffix]])


def resolve_var(token: str, ctx: PlanContext, cond: GoalCond):
    """解析单个 $变量（token 不含前导 $）。"""
    env = ctx.env
    if token == "safe_z":
        return float(SAFE_Z)
    if token == "grasp_pt" or token.startswith("grasp_pt."):
        return _axis_of(ctx.grasp_pt, token)
    if token == "goal_site" or token.startswith("goal_site."):
        site_name = getattr(cond, "site", None)
        if not site_name:
            raise TemplateError(f"${token} 要求条件带有 site 属性: {cond.describe()}")
        return _axis_of(env.get_site_pos(site_name), token)
    if token.startswith("site:"):
        rest = token[len("site:"):]
        name, _, axis = rest.partition(".")
        if not name:
            raise TemplateError("$site: 后必须给出 site 名")
        vec = env.get_site_pos(name)
        return _axis_of(vec, rest) if axis else [float(v) for v in np.asarray(vec, float)]
    if token.startswith("cond."):
        attr = token.split(".", 1)[1]
        if not hasattr(cond, attr):
            raise TemplateError(f"条件 {cond.kind} 没有属性 '{attr}'")
        return getattr(cond, attr)
    if token.startswith("cfg."):
        key = token.split(".", 1)[1]
        val = ctx.cfg.get(key)
        if val is None:
            # cfg 缺失时用合理默认值兜底，避免 None 传入 skill 导致崩溃
            val = _CFG_DEFAULTS.get(key)
        return val
    if token.startswith("entry."):
        return ctx.entry.get(token.split(".", 1)[1])
    raise TemplateError(f"未知模板变量 ${token}（白名单见 chain_registry 文档字符串）")


def _render(value, ctx: PlanContext, cond: GoalCond):
    """递归渲染 params：$整串替换保留原始类型；list/dict 逐元素；其余原样。"""
    if isinstance(value, str) and value.startswith("$"):
        return resolve_var(value[1:], ctx, cond)
    if isinstance(value, list):
        return [_render(v, ctx, cond) for v in value]
    if isinstance(value, dict):
        return {k: _render(v, ctx, cond) for k, v in value.items()}
    return value


# ============================================================
# 声明式方法
# ============================================================

class DeclarativeMethod(Method):
    """YAML 定义的方法：match 声明匹配条件，steps 为 skill 模板序列。"""

    def __init__(self, spec: Dict[str, Any], source: str = "yaml:<dict>") -> None:
        self.spec = spec
        self.source = source
        self.name = str(spec["name"])
        self.description = str(spec.get("description", ""))
        match = spec.get("match") or {}
        self.match_kind = match.get("kind")
        self.match_flavor = match.get("flavor")
        if not self.match_kind:
            raise TemplateError(f"方法 {self.name} 缺少 match.kind")
        self.raw_steps = spec.get("steps") or []
        if not isinstance(self.raw_steps, list) or not self.raw_steps:
            raise TemplateError(f"方法 {self.name} 缺少非空 steps 列表")
        for i, s in enumerate(self.raw_steps):
            if "skill" not in s:
                raise TemplateError(f"方法 {self.name} steps[{i}] 缺少 skill 字段")
        self.includes_home = bool(spec.get("include_home", False))
        self.flavor = self.match_flavor
        self.next = spec.get("next")  # 方法编排：执行后优先选的下一方法

    def can_achieve(self, cond: GoalCond, entry: Dict[str, Any]) -> bool:
        if cond.kind != self.match_kind:
            return False
        if self.match_flavor is not None \
                and getattr(cond, "flavor", None) != self.match_flavor:
            return False
        return True

    def make_steps(self, ctx: PlanContext, cond: GoalCond) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for s in self.raw_steps:
            # cmn（actor/grip_site）自动注入，YAML 只写业务参数
            params = dict(ctx.cmn)
            params.update(_render(s.get("params") or {}, ctx, cond))
            steps.append({"action": s["skill"], "params": params})
        return steps

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "match": {"kind": self.match_kind,
                                             "flavor": self.match_flavor},
                "source": self.source, "include_home": self.includes_home,
                "description": self.description, "steps": self.raw_steps}


# ============================================================
# list + map 注册表
# ============================================================

class ChainRegistry:
    """方法注册表：_order(list) 定优先级，_items(map) O(1) 定位增删改查。

    select 按 _order 顺序返回第一个 can_achieve 的方法；后注册的同名方法
    replace 旧方法（可用于 YAML 覆盖内置 Python 方法）。
    """

    def __init__(self) -> None:
        self._order: List[str] = []
        self._items: Dict[str, Method] = {}

    # ---- 增 ----

    def register(self, method: Method, *, before: Optional[str] = None,
                 after: Optional[str] = None, replace: bool = True) -> None:
        name = method.name
        if name in self._items and not replace:
            raise ValueError(f"方法 {name} 已存在（replace=False）")
        if name not in self._order:
            if before is not None:
                if before not in self._order:
                    raise KeyError(f"定位锚点 {before} 不存在")
                self._order.insert(self._order.index(before), name)
            elif after is not None:
                if after not in self._order:
                    raise KeyError(f"定位锚点 {after} 不存在")
                self._order.insert(self._order.index(after) + 1, name)
            else:
                self._order.append(name)
        self._items[name] = method

    def register_yaml_file(self, path, *, before=None, after=None) -> DeclarativeMethod:
        import yaml
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise TemplateError(f"{path}: 顶层必须是 YAML 映射")
        m = DeclarativeMethod(spec, source=f"yaml:{path}")
        self.register(m, before=before, after=after)
        return m

    def load_dir(self, directory=METHODS_DIR) -> List[str]:
        """扫描目录下 *.yaml（跳过 examples/ 与 *.disabled）。

        返回加载的方法名列表；目录不存在视为空（不报错）。
        """
        loaded: List[str] = []
        d = Path(directory)
        if not d.is_dir():
            return loaded
        for p in sorted(d.glob("*.yaml")):
            if ".disabled" in p.name:
                continue
            loaded.append(self.register_yaml_file(p).name)
        return loaded

    # ---- 删 ----

    def unregister(self, name: str) -> None:
        if name not in self._items:
            raise KeyError(f"方法 {name} 不存在")
        self._order.remove(name)
        del self._items[name]

    # ---- 查 ----

    def get(self, name: str) -> Method:
        if name not in self._items:
            raise KeyError(f"方法 {name} 不存在，现有: {self._order}")
        return self._items[name]

    def names(self) -> List[str]:
        return list(self._order)

    def list_methods(self) -> List[Dict[str, Any]]:
        out = []
        for i, n in enumerate(self._order):
            m = self._items[n]
            match_kind = getattr(m, "match_kind", None)
            if match_kind is None:
                # 内置 Python 方法：用能匹配的条件类型反推不出，标注 python
                match_kind = f"python:{type(m).__name__}"
            out.append({
                "order": i, "name": n, "kind": match_kind,
                "flavor": getattr(m, "flavor", None),
                "source": getattr(m, "source", "builtin"),
                "description": getattr(m, "spec", {}).get("description", "")
                        if isinstance(getattr(m, "spec", None), dict) else "",
            })
        return out

    def select(self, cond: GoalCond, entry: Dict[str, Any],
               exclude: Optional[set] = None) -> Method:
        """按优先级顺序匹配第一个能达成 cond 的方法（exclude 中的跳过）。"""
        exclude = exclude or set()
        for n in self._order:
            if n in exclude:
                continue
            m = self._items[n]
            try:
                if m.can_achieve(cond, entry):
                    return m
            except Exception:
                continue
        raise ValueError(f"没有可达成目标条件的方法: {cond.describe()}；"
                         f"已注册: {self._order}")

    # ---- 改（顺序）----

    def reorder(self, names: List[str]) -> None:
        """按给定 name 序列重排（必须是当前名称集合的一个排列）。"""
        if sorted(names) != sorted(self._order):
            raise ValueError("reorder 必须给出全部方法名且不重不漏")
        self._order = list(names)

    def reorder_by_experience(self, experience, task: str) -> List[str]:
        """按经验库中该 task 的成功率重排方法（成功多的排前面）。

        rate：成功=1.0；无记录=0.5；失败时按失败模式分级——
        goal_not_reached（链走通只差终态）=0.3 高于 skill 级失败=0.1，
        避免全失败方法无区分度导致排序打转。返回新顺序。
        """
        def rate(name: str) -> float:
            stats = experience.summary().get(task, {}).get(name, {})
            s = stats.get("success", 0)
            f = stats.get("failure", 0)
            if s > 0:
                return 1.0
            if s + f == 0:
                return 0.5  # 无记录：排中间（比全失败好，比有成功差）
            if stats.get("last_fail") == "goal_not_reached":
                return 0.3  # 链基本走通，只差终态
            return 0.1

        new_order = sorted(self._order, key=lambda n: rate(n), reverse=True)
        self._order = new_order
        return new_order

    def render(self, name: str, ctx: PlanContext, cond: GoalCond) -> List[Dict[str, Any]]:
        """预览某方法在给定上下文下渲染出的 skill 链（供 CLI/调试）。"""
        return self.get(name).make_steps(ctx, cond)


# ============================================================
# 默认注册表：内置 Python 方法（稳定底座）+ methods/*.yaml（动态覆盖层）
# ============================================================

_DEFAULT: Optional[ChainRegistry] = None


def default_registry() -> ChainRegistry:
    """进程级单例：内置 4 个 Python 方法，再叠加 methods/ 目录 YAML。

    YAML 与内置同名时 register(replace=True) 直接覆盖执行函数，
    实现"不动 Python 源码动态改链"。
    """
    global _DEFAULT
    if _DEFAULT is not None:
        return _DEFAULT
    reg = ChainRegistry()
    for m in (OpenDrawerMethod(), GraspLiftMethod(),
              TransferCartMethod(), TransferPoseMethod(),
              TransferPoseCartMethod(),
              LiberoPushMethod(), IKLiberoTransferMethod(),
              LiberoArticulateMethod(),
              ReplayDemoMethod()):
        reg.register(m)
    reg.load_dir()
    load_forged(reg)   # 成功轨迹沉淀的技能也进 registry，可被 select 选中
    _DEFAULT = reg
    return reg


def reset_default_registry() -> ChainRegistry:
    """测试/探针用：丢弃单例重建（含重新扫描 YAML 目录）。"""
    global _DEFAULT
    _DEFAULT = None
    return default_registry()


# ============================================================
# ForgedSkillMethod：成功轨迹沉淀的技能包装为 Method，进 registry 可被选中
# ============================================================

class _CaptureRegistry:
    """捕获型 registry：run() 调用 execute 时不真正执行，而是记录成 step。"""

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []

    def execute(self, name: str, env=None, **kwargs):
        self.steps.append({"action": name, "params": kwargs})
        return {"success": True}


class ForgedSkillMethod(Method):
    """把 skills/forged/<name>/entry.py 的 run() 包装成 Method。

    can_achieve: entry["task_name"] 在 forged skill 的 evidence.tasks 里；
    make_steps: 用捕获型 registry 调 run()，把 execute 调用序列转成 steps。

    架构要点：forged 链内嵌的是**锻造时 episode 的绝对坐标**（entry.py 快照），
    而新 episode 的物体位置/抽屉开度都变了 —— 直接重放必然抓空。因此捕获后
    做锚点重投影，把"轨迹快照"升级为"参数化模板"：
      1. 链中带 body 的 descend/move_above（抓取段）→ point 重投影为该 body
         当前位置；其前序同坐标的无 body 步骤（move_above 悬停）同步重投影。
      2. 其余 3 维数值 point/target/goal → 在 entry 声明的 site/body 锚点中
         最近匹配（<0.06m）后重投影（如 place goal ↔ cube_goal_site）。
      3. 链中第一个抓取段点既未重投影也无锚点可匹配 → 链几何已过期，
         raise 让 runner exclude 换内置方法，而不是拿着陈旧坐标去执行。
    """

    # 抓取段动作：point 语义 = 抓取目标（对应 body 位置），可安全用 body 重投影
    _GRASP_ACTS = ("descend", "move_above")
    _POINT_KEYS = ("point", "target", "goal")

    def __init__(self, name: str, run_fn, tasks: List[str]) -> None:
        self.name = f"forged_{name}"
        self._run = run_fn
        self._tasks = set(tasks)

    def can_achieve(self, cond: GoalCond, entry: Dict[str, Any]) -> bool:
        return entry.get("task_name", "") in self._tasks

    # ---- 锚点重投影 ----

    def _anchors(self, ctx: PlanContext, steps: List[Dict[str, Any]]):
        """构造当前观察下的锚点集合 [(name, kind, pos)]。

        来源：entry 中 *_site 字段（drawer_site/cube_goal_site/...）、
        entry["body"] 主物体、链中显式引用的其他 body。
        """
        env, entry = ctx.env, ctx.entry
        anchors = []
        for k, v in entry.items():
            if k.endswith("_site") and isinstance(v, str):
                try:
                    anchors.append((v, "site", np.asarray(env.get_site_pos(v), float)))
                except Exception:
                    pass
        body = entry.get("body")
        if body:
            try:
                anchors.append((str(body), "body", np.asarray(env.get_body_pos(body), float)))
            except Exception:
                pass
        for s in steps:
            b = (s.get("params") or {}).get("body")
            if b and b != body:
                try:
                    anchors.append((str(b), "body", np.asarray(env.get_body_pos(b), float)))
                except Exception:
                    pass
        return anchors

    def _nearest(self, p, anchors):
        best, bd = None, 1e9
        for a in anchors:
            d = float(np.linalg.norm(np.asarray(p, float) - a[2]))
            if d < bd:
                best, bd = a, d
        return best, bd

    def _reproject(self, steps: List[Dict[str, Any]], ctx: PlanContext):
        """就地重投影捕获链中的过期坐标，返回 (重投影点数, 首个抓取段是否有效)。"""
        env, entry = ctx.env, ctx.entry
        anchors = self._anchors(ctx, steps)

        def body_pos(b):
            try:
                return np.asarray(env.get_body_pos(b), float)
            except Exception:
                return None

        def set_pt(params, key, val):
            params[key] = [float(x) for x in np.asarray(val, float).reshape(3)]

        first_grasp_ok = False
        n_refit = 0
        # 1) 抓取段：带 body 的 descend/move_above，point ← body 当前位置
        #    记录 (旧point, 新point) 对：传播匹配用旧值判断"两步是否同目标"
        refit_pairs: List[tuple] = []
        for s in steps:
            params = s.get("params") or {}
            if s.get("action") in self._GRASP_ACTS and params.get("body"):
                bp = body_pos(params["body"])
                if bp is not None and "point" in params:
                    old = np.asarray(params["point"], float).copy()
                    set_pt(params, "point", bp)
                    refit_pairs.append((old, np.asarray(params["point"], float)))
                    n_refit += 1
                    first_grasp_ok = True
        # 2) 传播：前序无 body 的悬停/趋近步骤，其 point 与某个已重投影步骤的
        #    原始 point 同坐标（<0.02，即原链中两步指向同一目标）→ 同步重投影
        for s in steps:
            params = s.get("params") or {}
            if params.get("body") or params.get("point") is None:
                continue
            p = np.asarray(params["point"], float)
            for old, new in refit_pairs:
                if np.linalg.norm(p - old) < 0.02:
                    set_pt(params, "point", new)
                    n_refit += 1
                    if s.get("action") in self._GRASP_ACTS:
                        first_grasp_ok = True
                    break
        # 3) 其余 3 维数值点 → entry 声明锚点最近匹配（如 place goal ↔ goal site）
        first_seen = False
        for s in steps:
            params = s.get("params") or {}
            act = s.get("action")
            is_grasp = act in self._GRASP_ACTS and "point" in params
            for key in self._POINT_KEYS:
                v = params.get(key)
                if not (isinstance(v, (list, tuple)) and len(v) == 3
                        and all(isinstance(x, (int, float)) for x in v)):
                    continue
                if params.get("body") and key == "point" and act in self._GRASP_ACTS:
                    continue  # 已在 1) 处理
                a, d = self._nearest(v, anchors)
                if a is not None and d < 0.06:
                    set_pt(params, key, a[2])
                    n_refit += 1
                    if is_grasp:
                        first_grasp_ok = True
            if is_grasp and not first_seen:
                first_seen = True
                # 首个抓取段点：重投影后应贴近某锚点，否则链几何过期
                _, d = self._nearest(params["point"], anchors)
                if d > 0.06:
                    raise RuntimeError(
                        f"forged skill {self.name} 链几何过期：首个抓取点 "
                        f"{np.round(np.asarray(params['point'], float), 3)} "
                        f"距最近锚点 {d:.3f}m > 0.06（锻造时布局与当前环境不符），拒绝执行")
        return n_refit, first_grasp_ok

    def make_steps(self, ctx: PlanContext, cond: GoalCond) -> List[Dict[str, Any]]:
        cap = _CaptureRegistry()
        try:
            self._run(env=ctx.env, registry=cap)
        except Exception as e:
            print(f"[forged:{self.name}] run 捕获失败: {e}")
        # 防御：捕获步骤过少说明 forged skill 在当前环境失效（如不同 env/robot
        # 上内部静默失败），抛错让 runner 换方法，而不是拿着残链去执行。
        if len(cap.steps) < 2:
            raise RuntimeError(
                f"forged skill {self.name} 在当前环境只捕获到 {len(cap.steps)} 步，"
                f"疑似环境不匹配（tasks={sorted(self._tasks)}）")
        steps = [{"action": s["action"], "params": dict(s["params"])}
                 for s in cap.steps]
        try:
            _, _ = self._reproject(steps, ctx)
        except RuntimeError:
            raise
        except Exception as e:  # 重投影自身异常不阻断执行（保守保留原链）
            print(f"[forged:{self.name}] 重投影失败（保留原链）: {e}")
        return steps


def load_forged(reg: ChainRegistry) -> int:
    """扫描 skills/forged/，把每个成功轨迹技能包装成 ForgedSkillMethod 注册。

    返回注册数量。forged 方法优先级低于内置/YAML（append 到末尾）。
    """
    try:
        from ..skills.forged import load_forged_skills, FORGED_DIR
    except Exception:
        return 0
    if not FORGED_DIR.exists():
        return 0
    forged = load_forged_skills()
    n = 0
    for name, fs in forged.items():
        tasks = list(fs.spec.evidence.tasks) if fs.spec.evidence else []
        if not tasks:
            # 无 task 证据时按 applies_when 文本粗略匹配 task_name
            tasks = [name]
        m = ForgedSkillMethod(name, fs._run, tasks)
        try:
            reg.register(m, replace=True)
            n += 1
        except Exception:
            pass
    return n


__all__ = ["ChainRegistry", "DeclarativeMethod", "TemplateError",
           "default_registry", "reset_default_registry", "METHODS_DIR",
           "resolve_var", "ForgedSkillMethod", "load_forged"]
