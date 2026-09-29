"""可插拔抓取模型后端（GraspNet baseline / GSNet）。

统一接口: backend.predict_gg(pc_cam, top_k) -> GraspGroup 17 列数组
[score, width, height, depth, rot(9)=4:13, center(3)=13:16, obj_id]
选择: DARWIN_GRASP_BACKEND=gsnet|graspnet，默认 gsnet。
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS_ROOT = Path(os.environ.get("DARWIN_GRASP_MODELS_ROOT",
                                   "/home/lifd/Public/grasp_models"))
GRASPNET_ROOT = Path(os.environ.get("DARWIN_GRASPNET_ROOT",
                                     "/home/lifd/Public/graspnet_repo/graspnet-baseline"))
DEFAULT_CHECKPOINT = Path(os.environ.get("DARWIN_GRASPNET_CKPT",
                                          "/home/lifd/Public/checkpoint-rs.tar"))
GSNET_CHECKPOINT = MODELS_ROOT / "gsnet_graspness/checkpoints/checkp_realsense.tar"

_FAMILY_TOPS = {
    "models", "dataset", "utils", "pointnet2", "pointnet2_utils",
    "pointnet2_modules", "pytorch_utils", "loss_utils", "label_generation",
    "data_utils", "batch_utils", "knn", "knn_pytorch", "knn_modules",
    "collision_detector", "graspnet", "inplace_abn",
}
_FAMILY_PATH_MARKS = ("graspnet_repo", "grasp_models", "/tmp/gsnet_shims")


def _family_import(paths, modname):
    saved_path = sys.path[:]
    saved_argv = sys.argv[:]
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                  if k.split(".")[0] in _FAMILY_TOPS}
    o3d_stub = None
    if "open3d" not in sys.modules:
        o3d_stub = types.ModuleType("open3d")
        sys.modules["open3d"] = o3d_stub
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
    shim = Path("/tmp/gsnet_shims/knn")
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "__init__.py").write_text("")
    (shim / "knn_modules.py").write_text(
        "import torch\n"
        "def knn(ref, query, k=1):\n"
        "    d = torch.cdist(ref.transpose(1, 2).float(),\n"
        "                    query.transpose(1, 2).float())\n"
        "    return torch.topk(d, k, dim=1, largest=False).indices + 1\n")
    return shim.parent


def _resample(pc, n, seed=0):
    rng = np.random.RandomState(seed)
    if len(pc) >= n:
        return pc[rng.choice(len(pc), n, replace=False)].astype(np.float32)
    idx = np.concatenate(
        [np.arange(len(pc)),
         rng.choice(len(pc), n - len(pc), replace=True)])
    return np.ascontiguousarray(pc[idx], np.float32)


def _topk(gg, top_k):
    if len(gg) == 0:
        return gg
    order = np.argsort(gg[:, 0])[::-1][:top_k]
    return np.ascontiguousarray(gg[order])


class GSNetBackend:
    """GSNet / graspness（MinkUNet14D + graspable FPS/PVS）。"""
    name = "gsnet"

    def __init__(self):
        _patch_me_sparse_quantize()
        import torch
        root = MODELS_ROOT / "gsnet_graspness"
        paths = [_ensure_knn_shim(), root, root / "utils", root / "pointnet2"]
        graspnet_mod = _family_import(paths, "models.graspnet")
        dataset_mod = _family_import(paths, "dataset.graspnet_dataset")
        self._torch = torch
        self._pred_decode = graspnet_mod.pred_decode
        self._collate = dataset_mod.minkowski_collate_fn
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        net = graspnet_mod.GraspNet(seed_feat_dim=512, is_training=False)
        ck = torch.load(str(GSNET_CHECKPOINT), map_location="cpu")
        net.load_state_dict(ck["model_state_dict"])
        net.to(self.device).eval()
        self.net = net

    def predict_gg(self, pc_cam, top_k):
        torch = self._torch
        cloud = _resample(pc_cam, 15000)
        data = {"point_clouds": cloud,
                "coors": np.ascontiguousarray(cloud / 0.005),
                "feats": np.ones_like(cloud)}
        batch = self._collate([data])
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        with torch.no_grad():
            gg = self._pred_decode(self.net(batch))[0].detach().cpu().numpy()
        return _topk(gg, top_k)


class GraspNetBackend:
    """GraspNet baseline（checkpoint-rs.tar）。"""
    name = "graspnet"

    def __init__(self):
        import sys, torch
        sys.path.insert(0, str(GRASPNET_ROOT / "models"))
        sys.path.insert(0, str(GRASPNET_ROOT / "utils"))
        from graspnet import GraspNet, pred_decode
        net = GraspNet(
            input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        ckpt = torch.load(str(DEFAULT_CHECKPOINT), map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        self.net, self.pred_decode, self.device = net, pred_decode, device

    def predict_gg(self, pc_cam, top_k):
        import torch
        cloud = _resample(pc_cam, 20000)
        ep = {"point_clouds": torch.from_numpy(cloud[None]).to(self.device)}
        with torch.no_grad():
            ep = self.net(ep)
            gg = self.pred_decode(ep)[0].detach().cpu().numpy()
        return _topk(gg, top_k)


_BACKENDS = {"gsnet": GSNetBackend, "graspnet": GraspNetBackend}


@lru_cache(maxsize=1)
def load_backend():
    pref = os.environ.get("DARWIN_GRASP_BACKEND", "gsnet")
    order = [pref] + [k for k in _BACKENDS if k != pref]
    errs = []
    for name in order:
        try:
            be = _BACKENDS[name]()
            if name != pref:
                # 首选后端静默失败是隐蔽陷阱（曾致"以为在用 gsnet，
                # 实际全程 graspnet"）：回退必须带原因可见。
                print(f"[grasp] 警告: 首选后端 {pref} 加载失败"
                      f"（{'; '.join(errs)}），回退到 {name}")
            print(f"[grasp] 抓取后端: {name}")
            return be
        except Exception as e:
            errs.append(f"{name}: {e}")
    print(f"[grasp] 无可用抓取后端（{'；'.join(errs)}），走解析求解")
    return None
