"""抓取模型对比 MVP（LIBERO spatial 场景，多模型同一物理管线）。

设计
----
- 模型只负责"出候选"：场景点云（虚拟相机系）→ GraspGroup (K,17)。
- 后续完全复用 darwin.agents.libero_skills.grasp 的现有管线
  （站点 IK/碰撞过滤、pregrasp、RRT、伺服接近、闭爪、抬升验证），
  各模型条件完全一致。
- 模型通过运行时 monkeypatch 注入，不改动任何主链路文件：
    * S.graspnet_scene_candidates → 当前后端候选
    * analytic_rim_pinch / analytic_side_grip → None（关解析兜底，
      保证测的是模型本身，而不是"模型不行就几何兜底"的混合体）
- 同一 (task,object) reset 后采一次场景点云，所有模型共用。
- GSNet / EconomicGrasp 与 graspnet-baseline 仓库共享 models/、
  dataset/、utils/、pointnet2 等顶层包名，故用 sys.path+sys.modules
  隔离导入；导入后各模型类靠自身模块 globals 工作，互不影响。

用法：
  python scripts/bench_grasp_models.py --models baseline gsnet economic \
      --tasks 0 3 6 --top-k 8 --out logs/grasp_bench
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "8")
# ME 链接 openblas（faiss 自带）；建议运行时 shell 里也 export
_FAISS_LIBS = "/home/lifd/.local/lib/python3.10/site-packages/faiss_cpu.libs"
if os.path.isdir(_FAISS_LIBS):
    os.environ["LD_LIBRARY_PATH"] = (
        _FAISS_LIBS + ":" + os.environ.get("LD_LIBRARY_PATH", ""))

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, "/home/lifd/Public/LIBERO")

import numpy as np  # noqa: E402

from darwin.agents import libero_skills as S  # noqa: E402
from darwin.envs.libero_adapter import LiberoEnvAdapter  # noqa: E402
from darwin.skills.perception.grasp import (  # noqa: E402
    GraspPoseSkill, _load_graspnet, sample_scene_point_cloud)

MODELS_ROOT = Path("/home/lifd/Public/grasp_models")
F_FRAME = S.GRASPNET_FRAME  # diag(1,-1,-1)：世界系 → 虚拟相机系基变换


# ============================================================
# 同名顶层包隔离导入
# ============================================================

_FAMILY_TOPS = {
    "models", "dataset", "utils", "pointnet2", "pointnet2_utils",
    "pointnet2_modules", "pytorch_utils", "loss_utils", "label_generation",
    "data_utils", "batch_utils", "knn", "knn_pytorch", "knn_modules",
    "collision_detector", "graspnet", "inplace_abn",
}
_FAMILY_PATH_MARKS = ("graspnet_repo", "grasp_models", "/tmp/gsnet_shims")


def family_import(paths, modname):
    """在只含 paths 的 sys.path 视图下导入 modname，导入后还原现场。

    模型类/函数的 globals 仍指向各自模块对象，因此还原后照常工作。
    """
    import types
    saved_path = sys.path[:]
    saved_argv = sys.argv[:]
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                  if k.split(".")[0] in _FAMILY_TOPS}
    # 推理代码不实际使用 open3d，但部分模块顶层有无意义 import
    o3d_stub = None
    if "open3d" not in sys.modules:
        o3d_stub = types.ModuleType("open3d")
        sys.modules["open3d"] = o3d_stub
    # economic 的 utils/arguments.py 在 import 期 parse_args()
    sys.argv = ["x", "--dataset_root", "/tmp", "--camera", "realsense"]
    sys.path[:] = [p for p in sys.path
                   if not any(m in p for m in _FAMILY_PATH_MARKS)]
    for p in reversed([str(x) for x in paths]):
        sys.path.insert(0, p)
    importlib.invalidate_caches()
    try:
        mod = importlib.import_module(modname)
    finally:
        sys.path[:] = saved_path
        sys.argv = saved_argv
        if o3d_stub is not None:
            sys.modules.pop("open3d", None)
        for k in list(sys.modules):
            if k.split(".")[0] in _FAMILY_TOPS:
                sys.modules.pop(k, None)
        sys.modules.update(saved_mods)
    return mod


_ME_PATCHED = False


def _patch_me_sparse_quantize():
    """新版 ME 的 sparse_quantize 对整型 torch.Tensor 也强制 floor
    （_vml_cpu 不支持），economic forward 直接传入 sparse_collate 的
    整型坐标。整型张量转 numpy 分支（旧 ME 行为）再转回。"""
    global _ME_PATCHED
    if _ME_PATCHED:
        return
    import MinkowskiEngine as ME
    import torch
    orig = ME.utils.sparse_quantize

    def _sq(coordinates, features=None, **kw):
        if (isinstance(coordinates, torch.Tensor)
                and not torch.is_floating_point(coordinates)):
            c_np = np.ascontiguousarray(coordinates.int().cpu().numpy())
            f_np = None if features is None else np.ascontiguousarray(
                features.detach().cpu().numpy())

            def conv(x):
                if isinstance(x, np.ndarray):
                    t = torch.from_numpy(x)
                    return t.long() if x.ndim == 1 else t
                return x

            out = orig(c_np, f_np, **kw)
            return (tuple(conv(x) for x in out) if isinstance(out, tuple)
                    else conv(out))
        return orig(coordinates, features, **kw)

    ME.utils.sparse_quantize = _sq
    _ME_PATCHED = True


def _ensure_knn_shim():
    """graspness 顶层 import knn_pytorch（torch1.11 删了 THC/THC.h，老
    CUDA 扩展编不过）；推理 is_training=False 路径不调 knn，用
    torch.cdist 的同名 shim 仅满足导入。"""
    shim = Path("/tmp/gsnet_shims/knn")
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "__init__.py").write_text("")
    (shim / "knn_modules.py").write_text(
        "import torch\n"
        "def knn(ref, query, k=1):\n"
        "    # ref:(B,3,N) query:(B,3,M) -> (B,k,M)，1-based\n"
        "    d = torch.cdist(ref.transpose(1, 2).float(),\n"
        "                    query.transpose(1, 2).float())\n"
        "    return torch.topk(d, k, dim=1, largest=False).indices + 1\n")
    return shim.parent


# ============================================================
# 模型后端：predict_gg(pc_cam (N,3) f32, top_k) -> GraspGroup (K,17)
# ============================================================

class _Backend:
    name = "base"

    @staticmethod
    def _resample(pc, n, seed=0):
        rng = np.random.RandomState(seed)
        if len(pc) >= n:
            return pc[rng.choice(len(pc), n, replace=False)].astype(np.float32)
        idx = np.concatenate(
            [np.arange(len(pc)),
             rng.choice(len(pc), n - len(pc), replace=True)])
        return np.ascontiguousarray(pc[idx], np.float32)

    @staticmethod
    def _topk(gg, top_k):
        if len(gg) == 0:
            return gg
        order = np.argsort(gg[:, 0])[::-1][:top_k]
        return np.ascontiguousarray(gg[order])


class BaselineBackend(_Backend):
    """GraspNet baseline（darwin 现有集成，checkpoint-rs.tar）。"""
    name = "baseline"

    def __init__(self):
        loaded = _load_graspnet()
        if loaded is None:
            raise RuntimeError("GraspNet baseline 权重/代码不可用")
        self.net, self.pred_decode, self.device = loaded

    def predict_gg(self, pc_cam, top_k):
        import torch
        cloud = self._resample(pc_cam, 20000)
        ep = {"point_clouds": torch.from_numpy(cloud[None]).to(self.device)}
        with torch.no_grad():
            ep = self.net(ep)
            gg = self.pred_decode(ep)[0].detach().cpu().numpy()
        return self._topk(gg, top_k)


class GSNetBackend(_Backend):
    """GSNet / graspness_unofficial（MinkUNet14D + graspable FPS/PVS）。"""
    name = "gsnet"

    def __init__(self, ckpt: Path):
        _patch_me_sparse_quantize()
        import torch
        root = MODELS_ROOT / "gsnet_graspness"
        paths = [_ensure_knn_shim(), root, root / "utils", root / "pointnet2"]
        graspnet_mod = family_import(paths, "models.graspnet")
        dataset_mod = family_import(paths, "dataset.graspnet_dataset")
        self._torch = torch
        self._pred_decode = graspnet_mod.pred_decode
        self._collate = dataset_mod.minkowski_collate_fn
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        net = graspnet_mod.GraspNet(seed_feat_dim=512, is_training=False)
        ck = torch.load(str(ckpt), map_location="cpu")
        net.load_state_dict(ck["model_state_dict"])
        net.to(self.device).eval()
        self.net = net

    def predict_gg(self, pc_cam, top_k):
        torch = self._torch
        cloud = self._resample(pc_cam, 15000)
        data = {"point_clouds": cloud,
                "coors": np.ascontiguousarray(cloud / 0.005),
                "feats": np.ones_like(cloud)}
        batch = self._collate([data])
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        with torch.no_grad():
            gg = self._pred_decode(self.net(batch))[0].detach().cpu().numpy()
        return self._topk(gg, top_k)


class EconomicBackend(_Backend):
    """EconomicGrasp（经济监督 + focal grasp head）。"""
    name = "economic"

    def __init__(self, ckpt: Path):
        _patch_me_sparse_quantize()
        import torch
        root = MODELS_ROOT / "economicgrasp"
        paths = [root, root / "libs" / "pointnet2"]
        mod = family_import(paths, "models.economicgrasp")
        self._torch = torch
        self._pred_decode = mod.pred_decode
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        net = mod.economicgrasp(seed_feat_dim=512, is_training=False)
        ck = torch.load(str(ckpt), map_location="cpu")
        net.load_state_dict(ck["model_state_dict"])
        net.to(self.device).eval()
        self.net = net

    def predict_gg(self, pc_cam, top_k):
        torch = self._torch
        cloud = self._resample(pc_cam, 20000)
        batch = {
            "point_clouds": torch.from_numpy(cloud[None]).to(self.device),
            "coordinates_for_voxel":
                [torch.from_numpy(np.ascontiguousarray(cloud / 0.005))],
        }
        with torch.no_grad():
            gg = self._pred_decode(self.net(batch))[0].detach().cpu().numpy()
        return self._topk(gg, top_k)


CKPTS = {
    "baseline": None,
    "gsnet": MODELS_ROOT / "gsnet_graspness/checkpoints/checkp_realsense.tar",
    "economic": MODELS_ROOT / "economicgrasp/checkpoints/"
                           "economicgrasp_realsense.tar",
}


# ============================================================
# GraspGroup（虚拟相机系）→ 世界系 site 候选
# （与 libero_skills.graspnet_scene_candidates 相同变换）
# ============================================================

def gg_to_site_cands(gg, c0, hand, top_k):
    out = []
    for g in gg[:top_k]:
        # GraspGroup 17 列：score,width,height,depth,rot(9)=4:13,
        # center(3)=13:16,obj_id
        Rcam = np.asarray(g[4:13], float).reshape(3, 3)
        Rg = F_FRAME @ Rcam
        Rs = np.empty((3, 3), float)
        Rs[:, 0] = Rg[:, 2]
        Rs[:, 1] = -Rg[:, 1]
        Rs[:, 2] = Rg[:, 0]
        center = np.asarray(g[13:16], float) @ F_FRAME.T + c0
        depth = float(g[3])
        site = center + (depth - hand["b_depth"]) * Rs[:, 2]
        out.append({"R": Rs, "site": site, "center": center,
                    # GraspNet width 是指尖全宽；S.grasp 的门限与 a_open
                    # （指尖半间距）比较，这里统一换算成半宽（量纲一致；
                    # side_grip 分支用的是 width/2 同款约定）
                    "width": float(g[1]) / 2.0,
                    "score": float(g[0])})
    return out


# ============================================================
# 场景工具
# ============================================================

def dynamic_object_names(adapter):
    """有 free joint 且未被 weld 的 BDDL 物体（物理上可抓起）。"""
    import mujoco
    m = adapter.mj_model
    welded = set()
    for i in range(m.neq):
        if int(m.eq_type[i]) == mujoco.mjtEq.mjEQ_WELD:
            welded.add(int(m.eq_obj1id[i]))
            welded.add(int(m.eq_obj2id[i]))
    names = []
    for name in adapter.object_names:
        rid = m.body_name2id(adapter._resolve_body(name))
        bid, has_free = rid, False
        while bid != 0:
            jadr, jnum = int(m.body_jntadr[bid]), int(m.body_jntnum[bid])
            for j in range(jadr, jadr + jnum):
                if int(m.jnt_type[j]) == 0:
                    has_free = True
            bid = int(m.body_parentid[bid])
        if has_free and rid not in welded:
            names.append(name)
    return names


def make_candidate_provider(cands, hand, assoc_radius=0.035,
                            max_assoc=48):
    """按目标物做通用几何关联（物体无关，不偏向任何模型）：
      1) 半宽 ≤ 指尖半间距（与 S.grasp 门限一致）；
      2) 抓取中心 xy 到物体 AABB 边缘的距离 ≤ assoc_radius（中心在
         bbox 内时距离为 0），允许网络中心有统一的固定容差；
      3) site z 不低于物体底面；
    随后按模型自身 score 降序取前 max_assoc 个，限制 IK 总耗时。
    最终可达性/碰撞全部由下游物理链路裁决。"""
    a_open = float(hand["a_open"])

    def _provider(adapter, obj, *, top_k=64):
        b0 = adapter.object_bounds(obj)
        cx, cy = float(b0["center"][0]), float(b0["center"][1])
        hx, hy = float(b0["half_x"]), float(b0["half_y"])
        z_floor = float(b0["z_bottom"]) - 0.005
        out = []
        for c in cands:
            dx = max(abs(float(c["center"][0]) - cx) - hx, 0.0)
            dy = max(abs(float(c["center"][1]) - cy) - hy, 0.0)
            if (float(c["width"]) <= a_open
                    and float(c["site"][2]) >= z_floor
                    and dx <= assoc_radius and dy <= assoc_radius):
                out.append(c)
        out.sort(key=lambda c: -float(c["score"]))
        return out[:max_assoc]

    return _provider


hand_cached = None


def install_instrumentation():
    """无侵入埋点：包装各阶段函数（S.grasp 经模块全局名调用），
    记录本 trial 最远到达阶段与各阶段通过次数。返回 stats 字典。"""
    stats = {"site_ok": 0, "pregrasp_ok": 0, "clear_ok": 0, "plan_ok": 0,
             "exec_ok": 0, "approach_ok": 0, "closed": 0, "lift_ok": 0,
             "max_stage": 0}

    def wrap(name, ok_fn):
        orig = getattr(S, name)

        def w(*a, **k):
            r = orig(*a, **k)
            stage, good = ok_fn(r)
            if good:
                stats[stage] += 1
                rank = {"site_ok": 1, "pregrasp_ok": 2, "clear_ok": 3,
                        "plan_ok": 4, "exec_ok": 5, "approach_ok": 6,
                        "closed": 7, "lift_ok": 8}[stage]
                stats["max_stage"] = max(stats["max_stage"], rank)
            return r
        setattr(S, name, w)

    wrap("filter_site_config", lambda r: ("site_ok", r is not None))
    wrap("pregrasp_point", lambda r: ("pregrasp_ok", r is not None))
    wrap("_approach_clear", lambda r: ("clear_ok", bool(r)))
    wrap("plan_arm_path", lambda r: ("plan_ok", r is not None))
    wrap("execute_arm_path", lambda r: ("exec_ok", r is None))
    wrap("_approach_to_grasp", lambda r: ("approach_ok", r is None))
    wrap("_move_gripper", lambda r: ("closed", True))
    wrap("_lift_until_free", lambda r: ("lift_ok", bool(r[1])))
    return stats


STAGE_NAMES = {0: "none", 1: "site_ik", 2: "pregrasp", 3: "approach_clear",
               4: "plan", 5: "exec_path", 6: "servo_approach",
               7: "closed", 8: "lift_free", 9: "success"}


def install_backend(backend, cloud_world, top_k):
    """推理 + monkeypatch：把当前模型候选注入 S.grasp。"""
    c0 = cloud_world.mean(axis=0)
    pc_cam = np.asarray((cloud_world - c0) @ F_FRAME, np.float32)
    gg = backend.predict_gg(pc_cam, top_k)
    cands = gg_to_site_cands(gg, c0, hand_cached, top_k)
    S.graspnet_scene_candidates = make_candidate_provider(cands, hand_cached)
    S.analytic_rim_pinch = lambda *a, **k: None
    S.analytic_side_grip = lambda *a, **k: None
    return len(cands)


# ============================================================
# 主流程
# ============================================================

def main():
    global hand_cached
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["baseline", "gsnet", "economic"])
    ap.add_argument("--tasks", nargs="+", type=int, default=[0, 3, 6])
    ap.add_argument("--objects", nargs="+",
                    default=["akita_black_bowl_1", "cookies_1",
                             "glazed_rim_porcelain_ramekin_1", "plate_1"])
    ap.add_argument("--top-k", type=int, default=1024,
                    help="送入关联/物理过滤的候选池（按 score 降序）；"
                         "统一放大以保证各模型低分窄宽度候选也有机会，"
                         "门限/管线完全一致；实际求 IK 的候选另有上限")
    ap.add_argument("--out", default="logs/grasp_bench")
    args = ap.parse_args()

    out_dir = Path(_REPO) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_cls = {"baseline": BaselineBackend,
                   "gsnet": GSNetBackend,
                   "economic": EconomicBackend}
    backends = {}
    for name in args.models:
        t0 = time.time()
        try:
            b = (backend_cls[name]() if CKPTS[name] is None
                 else backend_cls[name](CKPTS[name]))
            backends[name] = b
            print(f"[load] {name} OK ({time.time()-t0:.0f}s)", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[load] {name} FAIL", flush=True)

    rows = []
    env = None
    cur_task = None
    stats = install_instrumentation()
    for task in args.tasks:
        for obj in args.objects:
            if cur_task != task:
                if env is not None:
                    env.close()
                env = LiberoEnvAdapter("libero_spatial", task)
                cur_task = task
            env.reset()
            if obj not in env.object_names:
                print(f"[skip] task{task} 无物体 {obj}", flush=True)
                continue
            if hand_cached is None:
                hand_cached = S.measure_hand(env)
            np.random.seed(0)
            cloud = sample_scene_point_cloud(env, 20000)
            if len(cloud) < 100:
                print(f"[skip] task{task}/{obj} 点云为空", flush=True)
                continue
            for name in args.models:
                b = backends.get(name)
                if b is None:
                    rows.append(dict(task=task, object=obj, model=name,
                                     success=0, reason="load_failed",
                                     seconds=0.0, n_cands=0, tried=0,
                                     site_ok=0, max_stage=0, stage="none"))
                    continue
                env.reset()  # 同确定性初态，各模型起点一致
                # 统一 RNG 起点：三模型面对同一场景状态、同一规划随机性
                np.random.seed(1000 + task * 100
                               + args.objects.index(obj))
                t0 = time.time()
                stats.update(site_ok=0, pregrasp_ok=0, clear_ok=0,
                             plan_ok=0, exec_ok=0, approach_ok=0,
                             closed=0, lift_ok=0, max_stage=0)
                tried = 0
                try:
                    n_cand = install_backend(b, cloud, args.top_k)
                    r = S.grasp(env, obj, mem={})
                    ok = bool(r.get("success"))
                    reason = "ok" if ok else str(r.get("reason", "?"))
                    tried = int((r.get("measures") or {}).get(
                        "n_tried", 0))
                    if ok:
                        stats["max_stage"] = 9
                except Exception:
                    import traceback
                    traceback.print_exc()
                    ok, reason, n_cand = False, "EXC", 0
                dt = time.time() - t0
                stage = STAGE_NAMES[stats["max_stage"]]
                print(f"task{task} {obj:32s} {name:9s} "
                      f"{'SUCCESS' if ok else 'fail':7s} "
                      f"{dt:6.1f}s pool={n_cand} tried={tried} "
                      f"stage={stage} site_ok={stats['site_ok']} "
                      f"{reason}", flush=True)
                rows.append(dict(task=task, object=obj, model=name,
                                 success=int(ok), reason=reason,
                                 seconds=round(dt, 1), n_cands=n_cand,
                                 tried=tried,
                                 site_ok=stats["site_ok"],
                                 max_stage=stats["max_stage"],
                                 stage=stage))
            _write(out_dir, rows)
    if env is not None:
        env.close()
    _summary(out_dir, rows)


def _write(out_dir, rows):
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "object", "model",
                                          "success", "reason", "seconds",
                                          "n_cands", "tried", "site_ok",
                                          "max_stage", "stage"])
        w.writeheader()
        w.writerows(rows)


def _summary(out_dir, rows):
    by_model = {}
    for r in rows:
        d = by_model.setdefault(r["model"],
                                {"n": 0, "ok": 0, "t": 0.0, "site": 0,
                                 "closed": 0, "lift": 0})
        d["n"] += 1
        d["ok"] += int(r["success"])
        d["t"] += float(r["seconds"])
        d["site"] += int(r.get("site_ok", 0))
        d["closed"] += int(r.get("max_stage", 0) >= 7)
        d["lift"] += int(r.get("max_stage", 0) >= 8)
    lines = ["model,success_rate,success/total,avg_seconds,"
             "trials_reach_close,trials_reach_lift,total_site_ik_passes"]
    print("\n===== 对比结果 =====")
    for name, d in sorted(by_model.items(),
                          key=lambda kv: -kv[1]["ok"] / max(kv[1]["n"], 1)):
        n = max(d["n"], 1)
        rate = d["ok"] / n
        lines.append(f"{name},{rate:.2%},{d['ok']}/{d['n']},"
                     f"{d['t']/n:.1f},{d['closed']},{d['lift']},"
                     f"{d['site']}")
        print(f"{name:9s} 成功率 {rate:6.1%}  ({d['ok']}/{d['n']})  "
              f"avg {d['t']/n:.1f}s  闭爪到达 {d['closed']}  "
              f"抬升 {d['lift']}  站点可行 {d['site']}")
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    print(f"\n明细: {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
