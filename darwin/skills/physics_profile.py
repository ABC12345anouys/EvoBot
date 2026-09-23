"""PhysicsProfile：把"物理世界的硬限制"从代码常量变成可学习参数。

与 skill_config 的分工：
- skill_config（ik_servo.<env>.yaml）= **控制参数**：怎么动（增益/速度/超时…）
- physics_profile（physics.<env>.yaml）= **物理事实/阈值**：世界允许什么
  （多大力才算夹住、物体不动多久判滑脱、多近判碰撞…）

可辨识性原则（只收可直接观测的量，不做质量/摩擦反演）：
- 每个参数都有 sim 中可测的对应观测量；
- 参数在**写入侧**有界（PHYS_SPEC clip），消费侧不 clamp；
- 只接受带标签样本（verified success / 明确 reason 的失败），
  且达到最小样本数才更新，存疑样本只记录不更新；
- 观测量存环形列表（每类 50 个），参数变化留 history。

sim_worker 只读；agent_learner 调 observe() 更新并原子落盘。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# key: (类型, 默认值, 最小, 最大, 说明)
PHYS_SPEC: Dict[str, Tuple[type, Any, float, float, str]] = {
    "slip_rise_m": (float, 0.006, 0.002, 0.020,
                    "lift 段物体上升低于此值判定没抓住（滑脱/空夹）m"),
    "grip_contact_min_n": (float, 0.0, 0.0, 20.0,
                           "抓持最小接触力 N；0=只测不判（观测足够后自动启用）"),
    "place_fast_xy_m": (float, 0.025, 0.015, 0.028,
                        "place 快判成功的 xy 阈值 m（必须 < BDDL 0.03 硬定义）"),
    "collision_abort_m": (float, 0.004, 0.002, 0.010,
                          "障碍 signed distance 低于此值中止 m"),
    "safe_vcap_m": (float, 0.05, 0.01, 0.20,
                    "默认每步末端速度上限 m"),
    "place_descend_xy_m": (float, 0.02, 0.008, 0.03,
                           "进容器竖直下放前物体质心须对中到 region 的 xy 半径 m；"
                           "实测最窄容器 desk_caddy 后格 x 半尺寸 0.0278，取 0.02 兜底"),
    "place_descend_clear_m": (float, 0.09, 0.05, 0.14,
                              "进容器水平对中阶段相对 region 中心的安全高度 m；"
                              "须高于篮/格沿（basket 判定盒 z 顶 +0.06）"),
    "carry_z_cap_m": (float, 1.0, 0.9, 1.2,
                      "carry/place 高度上限 m：须高于最高容器口沿+物体半高"
                      "（微波炉 rim1.03+mug 半高 0.05+悬深 0.06≈1.15），"
                      "但不得超 OSC 可达极限（实测 1.2 起跑飞）"),
    "place_release_comp_m": (float, 0.0, 0.0, 1.0,
                             "On 放置释放高度补偿系数：1=补 half_h+悬深（穿入夹持"
                             "的大物体必需，10_04 mug 陷 plate 教训），0=旧口径"),
}

# 自动更新所需最小样本数；不足只累积观测
MIN_SAMPLES = {"grip_contact_min_n": 8, "slip_rise_m": 3, "place_fast_xy_m": 5}
# place_fast_xy_m 的硬上限：永远不得越过 BDDL On 谓词的 0.03m 定义
_PLACE_XY_HARDMAX = 0.028
_OBS_MAXLEN = 50

# observe(summary) 可识别的观测量键 → 落盘统计桶（带成功/失败标签语义）
_OBS_KEYS = (
    "grip_force_n",            # 抓持段手指-物体接触力（成功/失败都记，更新只用成功）
    "lift_fail_rise_m",        # lift_no_grip 失败时物体实际上升量
    "place_xy_success_m",      # 成功放置的终态 xy 距离
    "clearance_min_m",         # 成功轨迹的最小障碍净空（v1 只累积不自动更新）
)

_HEADER = (
    "# 物理世界参数文件（agent_learner 自动维护，可手工编辑）\n"
    "# params: 当前生效物理阈值，范围/含义见 darwin/skills/physics_profile.py PHYS_SPEC\n"
    "# stats:  带标签观测量（环形保留最近 50 个）\n"
    "# active: 某参数是否已被观测数据激活（如 grip_contact_min_n 初期只测不判）\n"
    "# history:每次参数变化的溯源记录\n"
)


def phys_defaults() -> Dict[str, Any]:
    return {k: v[1] for k, v in PHYS_SPEC.items()}


def phys_get(env: Any, key: str, default: Optional[float] = None) -> Any:
    """原语侧统一入口：env 挂了 profile 就读，否则回退默认/给定值。"""
    if default is None:
        default = PHYS_SPEC[key][1]
    prof = getattr(env, "_darwin_physics", None)
    if prof is None:
        return default
    try:
        return prof.get(key, default)
    except Exception:
        return default


def phys_active(env: Any, key: str) -> bool:
    """该物理判定是否已被观测激活（未激活时只测量、不中止）。"""
    prof = getattr(env, "_darwin_physics", None)
    if prof is None:
        return False
    try:
        return bool(prof.active.get(key, False))
    except Exception:
        return False


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


class PhysicsProfile:
    """加载 / 观测更新 / 落盘一份 physics.<env>.yaml。"""

    def __init__(self, env: str, params: Dict[str, Any],
                 stats: Dict[str, List[float]], active: Dict[str, bool],
                 history: List[Dict[str, Any]], path: Path) -> None:
        self.env = env
        self.params = params
        self.stats = stats
        self.active = active
        self.history = history
        self.path = path

    # ---- 加载 ----
    @classmethod
    def load(cls, env: str) -> "PhysicsProfile":
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        path = CONFIGS_DIR / f"physics.{env}.yaml"
        params = phys_defaults()
        stats: Dict[str, List[float]] = {k: [] for k in _OBS_KEYS}
        active: Dict[str, bool] = {}
        history: List[Dict[str, Any]] = []
        data = cls._read(path)
        if data:
            params.update(cls._coerce(data.get("params", {})))
            for k, v in (data.get("stats") or {}).items():
                if k in stats and isinstance(v, list):
                    stats[k] = [float(x) for x in v][-_OBS_MAXLEN:]
            for k, v in (data.get("active") or {}).items():
                active[k] = bool(v)
            history = list(data.get("history", []) or [])
        return cls(env, params, stats, active, history, path)

    def get(self, key: str, default: Optional[float] = None) -> Any:
        return self.params.get(key, PHYS_SPEC[key][1] if key in PHYS_SPEC else default)

    # ---- 观测更新（agent 侧，带标签）----
    def observe(self, summary: Dict[str, Any], *, success: bool,
                reason: str = "", source: str = "") -> Dict[str, Any]:
        """消费一次 attempt 的物理测量摘要，返回实际变化的参数。

        summary 可能含（均可选，只接受有限数值）：
            grip_contact_force_n, lift_rise_m, place_xy_m, min_clearance_m
        标签规则（保守，宁可不学）：
        - grip_force_n：力桶成功/失败都记，自动更新只用 verified success；
        - lift_fail_rise_m：只在 reason=lift_no_grip 的失败记录；
        - place_xy_success_m：只在 success 且提供 place_xy_m 时记录；
        - clearance_min_m：只在 success 记录，v1 不自动改 abort。
        """
        changed: Dict[str, Any] = {}

        def num(k: str) -> Optional[float]:
            v = summary.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v)
            return None

        f_grip = num("grip_contact_force_n")
        if f_grip is not None and f_grip >= 0:
            self._push("grip_force_n", f_grip)
            if success:
                self._maybe_update_grip(changed, source)

        rise = num("lift_rise_m")
        if rise is not None and not success and "lift_no_grip" in str(reason):
            self._push("lift_fail_rise_m", rise)
            self._maybe_update_slip(changed, source)

        pxy = num("place_xy_m")
        if pxy is not None and success:
            self._push("place_xy_success_m", pxy)
            self._maybe_update_place_xy(changed, source)

        clr = num("min_clearance_m")
        if clr is not None and success:
            self._push("clearance_min_m", clr)

        return changed

    def _maybe_update_grip(self, changed: Dict[str, Any], source: str) -> None:
        succ = list(self.stats["grip_force_n"])
        if len(succ) < MIN_SAMPLES["grip_contact_min_n"]:
            return
        # 成功抓取力分布的 10 分位打对折 = "至少需要这么多"，低于 1N 不启用
        # 保守策略：p5（更低的成功力）×0.3 = 只有很轻的抓取才算异常
        # p10*0.5 太激进——偏心抓取（碗沿/瓶侧）接触力天然较低但能成功抓取
        p5 = _quantile(succ, 0.05)
        cand = self._clip("grip_contact_min_n", p5 * 0.3)
        if cand < 1.5:
            return
        old = float(self.params.get("grip_contact_min_n", 0.0))
        # 只在显著更有把握时上调，避免逐样本抖动
        if cand > old + 0.5:
            self.params["grip_contact_min_n"] = cand
            self.active["grip_contact_min_n"] = True
            changed["grip_contact_min_n"] = cand
            self._log_change("grip_contact_min_n", old, cand,
                             f"n={len(succ)} p5={p5:.2f}", source)

    def _maybe_update_slip(self, changed: Dict[str, Any], source: str) -> None:
        fails = self.stats["lift_fail_rise_m"]
        if len(fails) < MIN_SAMPLES["slip_rise_m"]:
            return
        # 失败上升量的 90 分位 ×1.5 安全系数：超过这么多还没提起来才算异常
        p90 = _quantile(fails, 0.90)
        cand = self._clip("slip_rise_m", p90 * 1.5)
        old = float(self.params.get("slip_rise_m", PHYS_SPEC["slip_rise_m"][1]))
        if abs(cand - old) > 0.0005:
            self.params["slip_rise_m"] = cand
            changed["slip_rise_m"] = cand
            self._log_change("slip_rise_m", old, cand,
                             f"n={len(fails)} p90={p90:.4f}", source)

    def _maybe_update_place_xy(self, changed: Dict[str, Any], source: str) -> None:
        succ = self.stats["place_xy_success_m"]
        if len(succ) < MIN_SAMPLES["place_fast_xy_m"]:
            return
        p90 = _quantile(succ, 0.90)
        cand = self._clip("place_fast_xy_m", min(p90 * 1.2, _PLACE_XY_HARDMAX))
        old = float(self.params.get("place_fast_xy_m",
                                    PHYS_SPEC["place_fast_xy_m"][1]))
        if abs(cand - old) > 0.0005:
            self.params["place_fast_xy_m"] = cand
            changed["place_fast_xy_m"] = cand
            self._log_change("place_fast_xy_m", old, cand,
                             f"n={len(succ)} p90={p90:.4f}", source)

    # ---- 落盘 ----
    def save(self) -> None:
        payload = {
            "params": dict(self.params),
            "active": dict(self.active),
            "stats": {k: list(v)[-_OBS_MAXLEN:] for k, v in self.stats.items()},
            "history": self.history[-200:],
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(_HEADER)
                yaml.safe_dump(payload, f, allow_unicode=True,
                               sort_keys=False, default_flow_style=False)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ---- 内部 ----
    def _push(self, bucket: str, value: float) -> None:
        lst = self.stats.setdefault(bucket, [])
        lst.append(float(value))
        if len(lst) > _OBS_MAXLEN:
            del lst[:-_OBS_MAXLEN]

    def _log_change(self, key: str, old: float, new: float,
                    evidence: str, source: str) -> None:
        import time
        self.history.append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "param": key, "old": round(old, 5), "new": round(new, 5),
            "evidence": evidence, "source": source,
        })

    @staticmethod
    def _clip(key: str, value: Any) -> Any:
        typ, default, lo, hi, _ = PHYS_SPEC[key]
        try:
            v = typ(value)
        except (TypeError, ValueError):
            v = default
        return min(hi, max(lo, v))

    @classmethod
    def _coerce(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (raw or {}).items():
            if k in PHYS_SPEC:
                out[k] = cls._clip(k, v)
        return out

    @staticmethod
    def _read(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
