"""策略注册表：每个决策点 (step) 可插拔多个实现 (impl)。

- per-env YAML 可用 `<step>_policy: <impl>` 声明用哪个实现（如
  `grasp_pose_policy: knn`），未声明走默认实现，全部缺失走 "rule"。
- 提案超出 ParamSpace 被裁剪并记录（champion-challenger 评估的输入）。

见 docs/architecture_step_planner.md §2/§8。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .context import StepContext
from .spaces import ParamSpace


class StepPolicy(Protocol):
    name: str
    def propose(self, ctx: StepContext,
                param_space: Optional[ParamSpace] = None) -> Dict[str, Any]: ...


_REGISTRY: Dict[Tuple[str, str], StepPolicy] = {}
_DEFAULTS: Dict[str, str] = {}


def register_policy(step: str, impl: str, policy: StepPolicy,
                    default: bool = False) -> None:
    _REGISTRY[(step, impl)] = policy
    if default or step not in _DEFAULTS:
        _DEFAULTS[step] = impl


def available(step: str) -> List[str]:
    return [impl for (s, impl) in _REGISTRY if s == step]


def propose(step: str, ctx: StepContext,
            param_space: Optional[ParamSpace] = None) -> Dict[str, Any]:
    """按 per-env 声明 → 默认 → rule 的顺序取策略；提案裁剪进可行域。"""
    impl = ctx.cfg.get(f"{step}_policy") or _DEFAULTS.get(step) or "rule"
    policy = _REGISTRY.get((step, impl)) or _REGISTRY.get((step, "rule"))
    if policy is None:
        raise KeyError(f"no policy registered for step={step!r}")
    out = dict(policy.propose(ctx, param_space))
    if param_space is not None:
        out, clipped = param_space.clip(out)
        if clipped:
            out["_clipped"] = clipped     # champion-challenger 评估信号
    return out


def propose_all(step: str, ctx: StepContext,
                param_space: Optional[ParamSpace] = None
                ) -> Dict[str, Dict[str, Any]]:
    """全部已注册实现各提一案 → {impl_name: proposal}（向后兼容 propose）。

    单实现异常不中断其余提案（调用方合并 candidates 后统一排序）。
    """
    out: Dict[str, Dict[str, Any]] = {}
    for impl in available(step) or [_DEFAULTS.get(step, "rule")]:
        policy = _REGISTRY.get((step, impl))
        if policy is None:
            continue
        try:
            proposal = dict(policy.propose(ctx, param_space))
        except Exception as e:    # proposer 崩不拖垮决策链
            print(f"[registry] propose_all {step}/{impl} failed: {e!r}")
            continue
        if param_space is not None:
            proposal, clipped = param_space.clip(proposal)
            if clipped:
                proposal["_clipped"] = clipped
        out[impl] = proposal
    return out


# ---- 内置实现导入（放最后避免循环 import）----
from . import rules as _rules  # noqa: E402,F401
