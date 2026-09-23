"""统一下载工具：所有模型权重/资源下载优先走 Gitee 镜像。

使用方式：
    from darwin.utils.download import ensure_weight, gitee_download
    path = ensure_weight("yolo26n-depth.pt")  # 自动从 Gitee 下载到本地缓存

Gitee 镜像源（公开可匿名访问）：
- ultralytics 模型：https://gitee.com/ultralytics/ultralytics 或 release 附件
- 通用策略：先查本地缓存 → 再试 Gitee → 最后 fallback 原始 URL
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

# 默认缓存目录
CACHE_DIR = Path(os.environ.get("DARWIN_WEIGHTS_CACHE",
                                str(Path.home() / ".cache" / "darwin" / "weights")))

# Gitee 镜像映射表（name -> gitee 直链/gitee release）
GITEE_MIRRORS = {
    # YOLO26 检测模型（HuggingFace 镜像，Gitee ultralytics/assets 仓库为空）
    "yolo26n.pt": "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26n.pt",
    "yolo26n-seg.pt": "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26n-seg.pt",
    "yolo26n-depth.pt": "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26n-depth.pt",
    "mobile_sam.pt": "https://gitee.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt",
    "sam_b.pt": "https://gitee.com/ultralytics/assets/releases/download/v8.4.0/sam_b.pt",
    "FastSAM-s.pt": "https://gitee.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt",
    # GraspNet 权重（从百度网盘下载，已存于 /home/lifd/Public/checkpoint-rs.tar）
    "checkpoint-rs.tar": None,  # 无公开直链，本地已下载
}

# 备用原始 URL（Gitee 失败时用）
FALLBACK_URLS = {
    "yolo26n-depth.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-depth.pt",
    "yolo26n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolo26n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
    "mobile_sam.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt",
}

def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def gitee_download(name: str, dest: Optional[Path] = None,
                   md5: Optional[str] = None) -> Path:
    """从 Gitee 镜像下载权重到本地缓存。
    Args:
        name: 权重文件名
        dest: 目标路径，默认 CACHE_DIR/name
        md5: 期望 MD5（可选，校验用）
    Returns:
        下载后的本地路径
    """
    dest = Path(dest or CACHE_DIR / name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 1. 本地已存在且校验通过
    if dest.exists():
        if md5 is None or _file_md5(dest) == md5:
            return dest
    urls = []
    if name in GITEE_MIRRORS:
        urls.append(GITEE_MIRRORS[name])
    if name in FALLBACK_URLS:
        urls.append(FALLBACK_URLS[name])
    if not urls:
        raise FileNotFoundError(f"未找到权重 {name} 的下载源，请手动下载放到 {dest}")
    last_err = None
    for url in urls:
        try:
            print(f"[download] {name} <- {url}", flush=True)
            _download_url(url, dest)
            if md5 and _file_md5(dest) != md5:
                raise ValueError(f"MD5 校验失败: {name}")
            return dest
        except Exception as e:
            last_err = e
            if dest.exists():
                dest.unlink()
            print(f"[download] 失败: {e}，尝试下一个源...", flush=True)
            continue

    raise RuntimeError(f"下载 {name} 失败，所有源均不可用: {last_err}")


def _download_url(url: str, dest: Path) -> None:
    """下载单个 URL，优先用 wget（支持重试），失败用 urllib。"""
    # 优先 wget -c（断点续传 + 重试）
    try:
        subprocess.run(
            ["wget", "-q", "--tries=3", "--timeout=30", "-O", str(dest), url],
            check=True, capture_output=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # fallback: urllib
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(dest)


def ensure_weight(name: str, local_path: Optional[str] = None,
                  md5: Optional[str] = None) -> Path:
    """确保权重文件存在：优先用用户指定的本地路径，否则从 Gitee 下载。

    这是感知技能加载权重的统一入口。
    """
    # 1. 用户指定的本地路径
    if local_path and Path(local_path).exists():
        p = Path(local_path)
        if md5 is None or _file_md5(p) == md5:
            return p

    # 2. 公共目录 /home/lifd/Public
    public = Path("/home/lifd/Public") / name
    if public.exists():
        if md5 is None or _file_md5(public) == md5:
            return public

    # 3. 缓存目录
    cached = CACHE_DIR / name
    if cached.exists():
        if md5 is None or _file_md5(cached) == md5:
            return cached

    # 4. 从 Gitee 下载
    return gitee_download(name, dest=cached, md5=md5)
