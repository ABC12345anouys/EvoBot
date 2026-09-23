#!/usr/bin/env python3
"""JARVIS 风格科技视频：drawer_place 任务全流程 + Agent 调度 / 感知 / Skill / RAG 可视化。

用法：
    python scripts/make_jarvis_video.py
    python scripts/make_jarvis_video.py --out videos/xxx.mp4 --fps 30

输出 1920x1080 MP4：
  - 主画面：MuJoCo 物理仿真（阴影/反光/缓慢环绕机位 + 3D 发光标记 + TCP 轨迹）
  - 右上：视觉传感器 RGB / 深度热成像
  - 右中：点云 + 抓取位姿
  - 右下：RAG 记忆核心（检索条目 / UCB / 成功率统计 / 新经验写入）
  - 底部：Agent Skill 调度链（10 个原语，实时状态）
"""
from __future__ import annotations

import os
# 无头渲染引导：必须先于 robopal/mujoco 导入
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
# ============================================================
# 配色 / 布局 / 字体
# ============================================================
W, H = 1920, 1080
BG = (4, 10, 18)
PANEL_BG = (7, 17, 29)
PANEL_EDGE = (0, 96, 122)
CYAN = (0, 229, 255)
CYAN_DIM = (0, 110, 140)
AMBER = (255, 179, 0)
GREEN = (0, 255, 136)
RED = (255, 64, 92)
WHITE = (214, 236, 246)
GREY = (88, 112, 132)
DARK = (10, 22, 36)
SIM_RECT = (20, 84, 1280, 720)
RGB_RECT = (1320, 84, 580, 300)
DEPTH_RECT = (1320, 396, 282, 190)
PCL_RECT = (1618, 396, 282, 190)
RAG_RECT = (1320, 598, 580, 206)
SKILL_RECT = (20, 828, 1880, 232)
_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_FONTS: Dict[Tuple[int, bool], Any] = {}

def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONTS:
        from PIL import ImageFont
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
        _FONTS[key] = ImageFont.truetype(f"{_FONT_DIR}/{name}", size)
    return _FONTS[key]

# ============================================================
# 遥测状态（执行流程 → HUD 的数据桥）
# ============================================================

# (plan action, HUD 标签, 阶段组 1=抽屉 2=放置)
SKILL_DEFS = [
    ("home", "HOME", 1),
    ("move_to", "APPROACH", 1),
    ("pull_drawer", "PULL DRAWER", 1),
    ("move_above", "ALIGN", 2),
    ("descend", "DESCEND", 2),
    ("close_gripper", "GRASP", 2),
    ("lift", "LIFT", 2),
    ("move_to_xy_top", "TRANSPORT", 2),
    ("place", "PLACE", 2),
    ("open_gripper", "RELEASE", 2),
]

PHASE_TEXT = {
    "PERCEIVE": ("PHASE 01 // PERCEPTION", "wrist RGB-D scan -> 3D reconstruct -> GraspNet 6-DoF"),
    "RAG": ("PHASE 02 // RAG RETRIEVAL", "semantic memory lookup / failure filter / UCB ranking"),
    "PLAN": ("PHASE 03 // DECISION", "skill chain synthesis"),
    "EXECUTE": ("PHASE 04 // EXECUTION", "agent dispatching primitive skills"),
    "VERIFY": ("PHASE 05 // VERIFICATION", "physics-level task validation + memory consolidation"),
}
# 腕部相机（安装于末端法兰，垂直俯视）
WRIST_DISTANCE = 0.28   # 光轴距离 lookat（m）
WRIST_LIFT = 0.02       # 光心相对 TCP 的抬高（m）

def _fmt_lat(t: Optional[float]) -> str:
    if t is None:
        return ""
    if t < 1.0:
        return f"{t * 1000:.1f} ms"
    return f"{t:.2f} s"

class Telemetry:
    def __init__(self) -> None:
        self.phase = "PERCEIVE"
        self.frame = 0
        self.sim_steps = 0
        self.t0 = time.time()
        # skill 链
        self.states: List[str] = ["wait"] * len(SKILL_DEFS)  # wait/queue/run/done/fail
        self.current = -1
        self.skill_detail = "awaiting dispatch"
        self.result_detail = ""
        # 感知
        self.feat: Dict[str, Any] = {}
        self.cloud: Optional[np.ndarray] = None
        self.scan = 0.0                      # 点云扫描进度 0..1
        self.grasp_pt: Optional[np.ndarray] = None
        self.grasp_R: Optional[np.ndarray] = None
        self.grasp_w: float = 0.0
        self.grasp_score: float = 0.0
        self.grasp_source: str = ""
        self.candidates: List[Dict[str, Any]] = []
        # 阶段延迟（秒）/ 单 skill 延迟
        self.lat: Dict[str, float] = {}
        self.skill_ms: List[Optional[float]] = [None] * len(SKILL_DEFS)
        self.skill_live: float = 0.0
        self.perc_detail: str = ""
        # RAG
        self.rag_records: List[Dict[str, Any]] = []
        self.rag_succ = 0
        self.rag_fail = 0
        self.cold_start = True
        self.memory_flash = 0.0
        # 实时物理量
        self.tcp = np.zeros(3)
        self.drawer_qpos = 0.0
        self.block_dist = None
        self.trail: deque = deque(maxlen=24)
        # 碰撞监控（机械臂 vs 柜子/抽屉，米）
        self.coll_dist: float = 1.0
        self.coll_pair: str = ""
        # 结果
        self.success: Optional[bool] = None
        self.metrics: Dict[str, float] = {}


# ============================================================
# JARVIS HUD
# ============================================================

def _brackets(d, x, y, w, h, color=CYAN, L=16, width=2):
    d.line([(x, y + L), (x, y), (x + L, y)], fill=color, width=width)
    d.line([(x + w - L, y), (x + w, y), (x + w, y + L)], fill=color, width=width)
    d.line([(x + w, y + h - L), (x + w, y + h), (x + w - L, y + h)], fill=color, width=width)
    d.line([(x, y + h - L), (x, y + h), (x + L, y + h)], fill=color, width=width)


def _panel(d, rect, title: str):
    x, y, w, h = rect
    d.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=PANEL_BG,
                        outline=PANEL_EDGE, width=1)
    _brackets(d, x + 2, y + 2, w - 4, h - 4)
    # 标题条
    d.rectangle([x + 8, y + 8, x + 14, y + 26], fill=CYAN)
    d.text((x + 22, y + 8), title, font=font(15, True), fill=CYAN)
    d.line([(x + 8, y + 32), (x + w - 8, y + 32)], fill=PANEL_EDGE, width=1)


def _glow_text(d, xy, text, fnt, fill=WHITE, glow=(0, 70, 90), anchor=None):
    d.text(xy, text, font=fnt, fill=fill, stroke_width=2, stroke_fill=glow, anchor=anchor)


def _build_base():
    """预渲染静态底图：网格、面板框、标题。"""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    # 背景网格
    for gx in range(0, W, 48):
        d.line([(gx, 0), (gx, H)], fill=(12, 26, 40, 255), width=1)
    for gy in range(0, H, 48):
        d.line([(0, gy), (W, gy)], fill=(12, 26, 40, 255), width=1)
    # 顶栏
    d.rectangle([0, 0, W, 64], fill=(6, 15, 26, 255))
    d.line([(0, 64), (W, 64)], fill=CYAN_DIM, width=1)
    d.rectangle([24, 18, 30, 46], fill=CYAN)
    _glow_text(d, (40, 12), "DARWIN-BOT", font(26, True), fill=CYAN)
    d.text((208, 22), "SELF-EVOLVING MANIPULATION AGENT", font=font(15), fill=GREY)
    # 面板
    _panel(d, SIM_RECT, "MUJOCO // PHYSICS SIMULATION  —  DrawerBox-v1 / DianaDrawerCube")
    _panel(d, RGB_RECT, "WRIST CAM // EE RGB-D  -  ZENITH 90deg")
    _panel(d, DEPTH_RECT, "WRIST RANGE // DEPTH")
    _panel(d, PCL_RECT, "POINT CLOUD // GRASP POSE")
    _panel(d, RAG_RECT, "RAG // MEMORY CORE")
    _panel(d, SKILL_RECT, "AGENT // SKILL SCHEDULER")
    return img


def _build_scanlines():
    from PIL import Image, ImageDraw
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for y in range(0, H, 3):
        d.line([(0, y), (W, y)], fill=(0, 0, 0, 26), width=1)
    return layer


def _build_vignette():
    from PIL import Image, ImageDraw
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([6, 70, W - 6, H - 6], radius=14, fill=180)
    # PIL 无模糊则用同心框近似暗角
    return layer


def _build_sweep():
    """仿真画面上的扫描光带（80x720 垂直渐变条）。"""
    from PIL import Image
    strip = np.zeros((720, 90, 4), dtype=np.uint8)
    for i in range(90):
        a = int(70 * (1 - abs(i - 45) / 45))
        strip[:, i] = (0, 229, 255, a)
    return Image.fromarray(strip, "RGBA")


class JarvisHUD:
    def __init__(self) -> None:
        from PIL import Image
        self.base = _build_base().convert("RGBA")
        self.scanlines = _build_scanlines()
        self.sweep = _build_sweep()
        # 深度缓存（每 3 帧渲染一次）
        self._depth_cache = None

    # ---------- 主仿真 ----------

    def _draw_sim(self, canvas, frame_rgb, tele: Telemetry):
        from PIL import Image, ImageDraw
        x, y, w, h = SIM_RECT
        sim = Image.fromarray(frame_rgb, "RGB").resize((w - 16, h - 44), Image.BILINEAR)
        canvas.paste(sim, (x + 8, y + 36))
        d = ImageDraw.Draw(canvas)
        # 扫描光带
        sx = x + 8 + (tele.frame * 4) % (w + 120) - 60
        canvas.alpha_composite(self.sweep, (int(sx), y + 36))
        # 中心瞄准环
        cx, cy = x + w // 2, y + 36 + (h - 44) // 2
        pulse = 14 + 4 * math.sin(tele.frame * 0.12)
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], outline=(*CYAN, 150), width=1)
        d.ellipse([cx - pulse, cy - pulse, cx + pulse, cy + pulse], outline=(*CYAN, 70), width=1)
        d.line([(cx - 40, cy), (cx - 12, cy)], fill=(*CYAN, 180), width=1)
        d.line([(cx + 12, cy), (cx + 40, cy)], fill=(*CYAN, 180), width=1)
        d.line([(cx, cy - 40), (cx, cy - 12)], fill=(*CYAN, 180), width=1)
        d.line([(cx, cy + 12), (cx, cy + 40)], fill=(*CYAN, 180), width=1)
        # 相位横幅
        title, sub = PHASE_TEXT.get(tele.phase, ("", ""))
        if tele.phase == "EXECUTE" and 0 <= tele.current < len(SKILL_DEFS):
            title = f"PHASE 04 // EXECUTION  >>  {SKILL_DEFS[tele.current][1]}"
        if title:
            bw = min(w - 36, max(11 * len(title) + 80, 7 * len(sub) + 150))
            d.rounded_rectangle([x + 18, y + 46, x + 18 + bw, y + 102], radius=4,
                                fill=(4, 12, 22, 210), outline=(*CYAN, 120))
            _glow_text(d, (x + 32, y + 52), title, font(18, True), fill=CYAN)
            d.text((x + 32, y + 80), sub, font=font(12), fill=(*WHITE, 200))
            latv = tele.lat.get(tele.phase)
            if tele.phase == "EXECUTE" and tele.current >= 0 and latv is None:
                latv = tele.skill_live
            if latv is not None:
                d.text((x + 18 + bw - 14, y + 78), f"LAT {_fmt_lat(latv)}",
                       font=font(13, True), fill=AMBER, anchor="ra")
        # 底部数据条
        d.rounded_rectangle([x + 18, y + h - 56, x + w - 18, y + h - 12], radius=4,
                            fill=(4, 12, 22, 205), outline=(*CYAN_DIM, 160))
        tcp = tele.tcp
        d.text((x + 30, y + h - 48),
               f"TCP  x:{tcp[0]:+.3f} y:{tcp[1]:+.3f} z:{tcp[2]:+.3f}",
               font=font(13), fill=WHITE)
        # 抽屉开度条
        bx = x + 400
        d.text((bx, y + h - 48), "DRAWER", font=font(13), fill=GREY)
        d.rectangle([bx + 66, y + h - 44, bx + 206, y + h - 34], outline=(*CYAN_DIM, 200), width=1)
        frac = min(1.0, tele.drawer_qpos / 0.14)
        col = GREEN if frac > 0.57 else AMBER
        d.rectangle([bx + 68, y + h - 42, bx + 68 + int(136 * frac), y + h - 36], fill=col)
        d.text((bx + 212, y + h - 48), f"{tele.drawer_qpos:.3f}m", font=font(13), fill=col)
        # 放置误差
        if tele.block_dist is not None:
            dx = bx + 360
            dc = GREEN if tele.block_dist < 0.05 else AMBER
            d.text((dx, y + h - 48), f"PLACE ERROR  {tele.block_dist*1000:05.1f} mm",
                   font=font(13), fill=dc)
        # 碰撞距离（机械臂 vs 柜子）—— 碰撞时红色闪烁
        cdx = bx + 600
        cd = tele.coll_dist
        if cd <= 0.0:
            ccol, ctag = RED, "COLLISION"
            if (tele.frame // 8) % 2 == 0:
                ccol = (255, 80, 80, 255)
        elif cd < 0.03:
            ccol, ctag = AMBER, "NEAR"
        else:
            ccol, ctag = GREEN, "CLEAR"
        d.text((cdx, y + h - 48), f"{ctag}  {cd*1000:5.1f} mm",
               font=font(13, True), fill=ccol)

    # ---------- 感知面板 ----------

    def _draw_eye(self, canvas, rgb: np.ndarray, depth: Optional[np.ndarray], tele: Telemetry):
        from PIL import Image, ImageDraw
        import cv2
        x, y, w, h = RGB_RECT
        img = Image.fromarray(rgb, "RGB").resize((w - 16, 260), Image.BILINEAR).convert("RGBA")
        # 扫描线 + 扫描动画
        dd = ImageDraw.Draw(img)
        sy = int((tele.frame * 5) % 280)
        dd.rectangle([0, sy, w - 16, sy + 3], fill=(*CYAN, 90))
        for yy in range(0, 260, 4):
            dd.line([(0, yy), (w - 16, yy)], fill=(0, 0, 0, 18))
        canvas.alpha_composite(img, (x + 8, y + 36))
        d = ImageDraw.Draw(canvas)
        blink = (tele.frame // 24) % 2 == 0
        if tele.phase == "EXECUTE" and blink:
            d.ellipse([x + w - 92, y + 12, x + w - 82, y + 22], fill=RED)
            d.text((x + w - 78, y + 11), "REC", font=font(12, True), fill=RED)
        # 深度
        if depth is None:
            depth = self._depth_cache
        else:
            self._depth_cache = depth
        dx, dy, dw, dh = DEPTH_RECT
        if depth is not None:
            dn = np.clip(depth, 0.0, 2.5) / 2.5
            heat = (255 * (1 - dn)).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            himg = Image.fromarray(heat, "RGB").resize((dw - 16, dh - 44), Image.BILINEAR)
            canvas.paste(himg, (dx + 8, dy + 36))
        d2 = ImageDraw.Draw(canvas)
        d2.text((dx + 12, dy + dh - 22), "NEAR 0.0m", font=font(10), fill=GREY)
        d2.text((dx + dw - 78, dy + dh - 22), "FAR 2.5m", font=font(10), fill=GREY)

    # ---------- 点云面板 ----------

    def _draw_pcl(self, canvas, tele: Telemetry):
        from PIL import ImageDraw
        x, y, w, h = PCL_RECT
        d = ImageDraw.Draw(canvas)
        pad = 14
        x0, y0 = x + pad, y + 40
        pw, ph = w - 2 * pad, h - 62
        # 工作空间映射 x∈[0.20,0.64] y∈[-0.24,0.24]（俯视）
        xmin, xmax, ymin, ymax = 0.20, 0.64, -0.24, 0.24

        def mapp(p):
            u = x0 + (p[0] - xmin) / (xmax - xmin) * pw
            v = y0 + (p[1] - ymin) / (ymax - ymin) * ph
            return u, v

        # 栅格
        for i in range(5):
            gx = x0 + i * pw / 4
            d.line([(gx, y0), (gx, y0 + ph)], fill=(*PANEL_EDGE, 120), width=1)
            gy = y0 + i * ph / 4
            d.line([(x0, gy), (x0 + pw, gy)], fill=(*PANEL_EDGE, 120), width=1)
        # 点云（扫描揭示）
        if tele.cloud is not None and len(tele.cloud):
            pts = tele.cloud
            reveal = xmin + (xmax - xmin) * max(0.02, tele.scan)
            shown = pts[pts[:, 0] <= reveal]
            for p in shown[::2]:
                u, v = mapp(p)
                if x0 <= u <= x0 + pw and y0 <= v <= y0 + ph:
                    zn = float(np.clip((p[2] - 0.42) / 0.14, 0, 1))
                    zb = int(150 + 105 * zn)
                    ui, vi = int(u), int(v)
                    # 3x3 青色光晕 + 高亮中心
                    d.rectangle([ui - 1, vi - 1, ui + 1, vi + 1],
                                fill=(26, max(50, zb - 90), 215, 210))
                    d.point((ui, vi), fill=(170, 244, 255, 255))
        # top-K 候选（暗点）
        for cd in tele.candidates:
            try:
                cu, cv = mapp(np.asarray(cd.get("position", [0, 0, 0]), float))
                if x0 <= cu <= x0 + pw and y0 <= cv <= y0 + ph:
                    d.ellipse([cu - 2, cv - 2, cu + 2, cv + 2], fill=(*AMBER, 90))
            except Exception:
                pass
        # 最优抓取：脉冲十字 + approach 箭头 + 夹爪开合宽度
        if tele.grasp_pt is not None:
            u, v = mapp(tele.grasp_pt)
            r = 8 + 2 * math.sin(tele.frame * 0.2)
            d.ellipse([u - r, v - r, u + r, v + r], outline=AMBER, width=2)
            d.line([(u - r - 4, v), (u - 2, v)], fill=AMBER, width=2)
            d.line([(u + 2, v), (u + r + 4, v)], fill=AMBER, width=2)
            d.line([(u, v - r - 4), (u, v - 2)], fill=AMBER, width=2)
            d.line([(u, v + 2), (u, v + r + 4)], fill=AMBER, width=2)
            if tele.grasp_R is not None:
                Rm = np.asarray(tele.grasp_R, float).reshape(3, 3)
                # GraspNet 约定: col0=approach, col1=closing；几何 fallback 反之
                if tele.grasp_source == "graspnet":
                    app, cls = Rm[:, 0], Rm[:, 1]
                else:
                    app, cls = Rm[:, 2], Rm[:, 0]
                gp = np.asarray(tele.grasp_pt, float)
                if np.linalg.norm(app[:2]) > 0.05:
                    a2 = app[:2] / np.linalg.norm(app[:2])
                    ex, ey = mapp(gp + app * 0.055)
                    d.line([(u, v), (ex, ey)], fill=(255, 230, 160, 255), width=2)
                    d.polygon([(ex + 5 * a2[0], ey + 5 * a2[1]),
                               (ex - 3 * a2[0] + 3 * a2[1], ey - 3 * a2[1] - 3 * a2[0]),
                               (ex - 3 * a2[0] - 3 * a2[1], ey - 3 * a2[1] + 3 * a2[0])],
                              fill=(255, 230, 160, 255))
                hw = min(float(tele.grasp_w) / 2, 0.05)
                j1 = mapp(gp + cls * hw)
                j2 = mapp(gp - cls * hw)
                d.line([j1, j2], fill=AMBER, width=3)
                d.ellipse([j1[0] - 3, j1[1] - 3, j1[0] + 3, j1[1] + 3], fill=AMBER)
                d.ellipse([j2[0] - 3, j2[1] - 3, j2[0] + 3, j2[1] + 3], fill=AMBER)
        # 目标点
        for gpos, gc in (("cube_goal", GREEN),):
            try:
                gu, gv = mapp(tele.goal_xyz)
                d.rectangle([gu - 5, gv - 5, gu + 5, gv + 5], outline=gc, width=2)
            except Exception:
                pass
        if tele.grasp_source:
            tag = "GRASPNET 6-DOF" if tele.grasp_source == "graspnet" else "GEOM-PCA"
            d.text((x0 + pw - 185, y0 + ph + 4),
                   f"{tag}  S{tele.grasp_score:.2f} W{tele.grasp_w * 1000:.0f}mm",
                   font=font(10, True), fill=AMBER)

    # ---------- Skill 调度链 ----------

    def _draw_skills(self, canvas, tele: Telemetry):
        from PIL import ImageDraw
        x, y, w, h = SKILL_RECT
        d = ImageDraw.Draw(canvas)
        n = len(SKILL_DEFS)
        chip_w, gap = 168, 13
        total = n * chip_w + (n - 1) * gap
        cx = x + (w - total) // 2
        chip_y = y + 78
        chip_h = 86
        for i, ((act, label, grp), state) in enumerate(zip(SKILL_DEFS, tele.states)):
            bx = cx + i * (chip_w + gap)
            # 组背景
            grp_col = (0, 140, 170) if grp == 1 else (150, 110, 0)
            if state == "wait":
                fill, edge, tcol = (10, 20, 32, 255), (30, 52, 68, 255), (70, 92, 108, 255)
            elif state == "queue":
                fill, edge, tcol = (10, 28, 42, 255), (*CYAN_DIM, 255), (*WHITE, 230)
            elif state == "run":
                pulse = int(40 + 40 * (1 + math.sin(tele.frame * 0.25)))
                fill = (0 + 6, 60 + pulse // 3, 76 + pulse // 3, 255)
                edge, tcol = AMBER, (255, 240, 210, 255)
            elif state == "done":
                fill, edge, tcol = (6, 40, 30, 255), GREEN, (180, 255, 220, 255)
            else:
                fill, edge, tcol = (44, 12, 22, 255), RED, (255, 200, 210, 255)
            d.rounded_rectangle([bx, chip_y, bx + chip_w, chip_y + chip_h], radius=5,
                                fill=fill, outline=edge, width=2 if state in ("run", "done", "fail") else 1)
            # 编号 + 名称
            d.text((bx + 10, chip_y + 8), f"{i + 1:02d}", font=font(12, True), fill=grp_col if state != "wait" else (40, 60, 72))
            d.text((bx + 10, chip_y + 30), label, font=font(16, True), fill=tcol)
            d.text((bx + 10, chip_y + 58), act, font=font(10), fill=(90, 110, 124, 220) if state != "wait" else (50, 66, 78, 220))
            # 状态标
            if state == "run":
                d.text((bx + chip_w - 46, chip_y + 8), "EXEC", font=font(10, True), fill=AMBER)
            elif state == "done":
                d.text((bx + chip_w - 30, chip_y + 6), "✓", font=font(16, True), fill=GREEN)
            if state == "done" and tele.skill_ms[i] is not None:
                d.text((bx + chip_w - 40, chip_y + 9), _fmt_lat(tele.skill_ms[i]),
                       font=font(10, True), fill=(140, 220, 180, 255), anchor="ra")
            elif state == "fail":
                d.text((bx + chip_w - 30, chip_y + 6), "✕", font=font(16, True), fill=RED)
            # 连接箭头
            if i < n - 1:
                ax = bx + chip_w
                ay = chip_y + chip_h // 2
                d.line([(ax + 2, ay), (ax + gap - 4, ay)], fill=(*CYAN_DIM, 200), width=2)
                d.polygon([(ax + gap - 4, ay - 4), (ax + gap - 1, ay),
                           (ax + gap - 4, ay + 4)], fill=(*CYAN_DIM, 220))
        # 流水线延迟条（面板标题行右侧）
        pipe = [("PERCEPTION", "PERCEIVE"), ("RAG", "RAG"), ("PLAN", "PLAN"),
                ("EXEC", "EXECUTE"), ("VERIFY", "VERIFY")]
        seg_w = []
        for label, key in pipe:
            if key in tele.lat:
                val = _fmt_lat(tele.lat[key])
                seg_w.append((label, val,
                              d.textlength(label, font=font(11, True)),
                              d.textlength(val, font=font(11, True))))
        gap = 26
        total_w = sum(wl + wv + 8 for _, _, wl, wv in seg_w) + gap * max(0, len(seg_w) - 1)
        px = x + w - 24 - total_w
        for i, (label, val, wl, wv) in enumerate(seg_w):
            d.text((px, y + 11), label, font=font(11, True), fill=GREY)
            d.text((px + wl + 8, y + 11), val, font=font(11, True), fill=AMBER)
            px += wl + wv + 8
            if i < len(seg_w) - 1:
                d.text((px + 5, y + 11), "›", font=font(13, True), fill=CYAN_DIM)
                px += gap
        # 阶段组括号
        d.rounded_rectangle([cx - 10, chip_y - 22, cx + 3 * chip_w + 2 * gap + 10, chip_y - 4],
                            radius=3, outline=(0, 140, 170, 200), width=1)
        d.text((cx, chip_y - 20), "STAGE 01 // OPEN DRAWER", font=font(11, True), fill=(0, 190, 220))
        x2 = cx + 3 * (chip_w + gap)
        d.rounded_rectangle([x2 - 10, chip_y - 22, cx + total + 10, chip_y - 4],
                            radius=3, outline=(190, 140, 0, 200), width=1)
        d.text((x2, chip_y - 20), "STAGE 02 // PICK & PLACE INTO DRAWER", font=font(11, True),
               fill=(255, 190, 40))
        # 底部调度日志
        d.text((x + 20, y + h - 44), f"DISPATCH >> {tele.skill_detail}", font=font(13, True),
               fill=AMBER if tele.phase == "EXECUTE" else GREY)
        if tele.result_detail:
            d.text((x + 20, y + h - 24), tele.result_detail, font=font(12), fill=GREEN)

    # ---------- RAG 面板 ----------

    def _draw_rag(self, canvas, tele: Telemetry):
        from PIL import ImageDraw
        x, y, w, h = RAG_RECT
        d = ImageDraw.Draw(canvas)
        list_w = w - 180
        recs = tele.rag_records[:3]
        if not recs:
            d.text((x + 16, y + 48), "NO EXPERIENCE IN STORE", font=font(14, True), fill=GREY)
            d.text((x + 16, y + 74), "cold start -> UCB exploration engaged",
                   font=font(12), fill=AMBER)
        for i, r in enumerate(recs):
            ry = y + 42 + i * 52
            kind_raw = str(r.get("kind", "?")).lower()
            kind = {"strategy": "SUCCESS", "failure": "FAILURE"}.get(
                kind_raw, kind_raw.upper())
            kc = GREEN if "SUCC" in kind else (RED if "FAIL" in kind else CYAN)
            d.rounded_rectangle([x + 12, ry, x + list_w - 6, ry + 46], radius=3,
                                fill=(10, 24, 38, 255), outline=(*kc, 140), width=1)
            d.rectangle([x + 12, ry, x + 16, ry + 46], fill=kc)
            d.text((x + 24, ry + 4), f"{kind}  ·  {r.get('shape', '?')} / {r.get('task', '?')} / {r.get('score_band', '?')}",
                   font=font(11, True), fill=kc)
            off = r.get("rel_offset", [0, 0, 0])
            off_s = "[" + " ".join(f"{float(v):+.2f}" for v in off) + "]"
            d.text((x + 24, ry + 20), f"offset{off_s}", font=font(10), fill=WHITE)
            d.text((x + 24, ry + 33), str(r.get("exp", ""))[:52], font=font(9), fill=GREY)
        # 右侧统计
        sx = x + list_w + 6
        d.line([(sx, y + 40), (sx, y + h - 12)], fill=PANEL_EDGE, width=1)
        d.text((sx + 12, y + 44), "SUCCESS", font=font(11), fill=GREY)
        _glow_text(d, (sx + 12, y + 60), f"{tele.rag_succ:02d}", font(30, True), fill=GREEN)
        d.text((sx + 82, y + 44), "FAILURE", font=font(11), fill=GREY)
        _glow_text(d, (sx + 82, y + 60), f"{tele.rag_fail:02d}", font(30, True), fill=RED)
        # UCB 条
        d.text((sx + 12, y + 104), "UCB EXPLORE", font=font(11), fill=GREY)
        bar_x, bar_y = sx + 12, y + 124
        d.rectangle([bar_x, bar_y, bar_x + 150, bar_y + 8], outline=(*CYAN_DIM, 200), width=1)
        ucb = 0.55 if tele.cold_start else 0.2
        d.rectangle([bar_x + 2, bar_y + 2, bar_x + int(146 * ucb), bar_y + 6], fill=CYAN)
        d.text((sx + 12, y + 138), f"TTL {20}  ·  d<0.5 match", font=font(10), fill=GREY)
        # 新经验写入闪烁
        if tele.memory_flash > 0:
            a = int(255 * min(1.0, tele.memory_flash))
            d.rounded_rectangle([sx + 8, y + 160, sx + 166, y + 192], radius=4,
                                fill=(0, 60, 40, a), outline=GREEN, width=2)
            d.text((sx + 18, y + 168), "NEW EXPERIENCE", font=font(11, True), fill=(180, 255, 220, a))
            d.text((sx + 18, y + 182), "WRITTEN TO RAG", font=font(10), fill=(180, 255, 220, a))

    # ---------- 顶栏动态 ----------

    def _draw_topbar(self, canvas, tele: Telemetry):
        from PIL import ImageDraw
        d = ImageDraw.Draw(canvas)
        t = tele.sim_steps * 0.05
        d.text((W - 640, 22), f"TASK :: DRAWER_PLACE", font=font(14, True), fill=WHITE)
        d.text((W - 430, 22), f"SIM T+{int(t)//60:02d}:{t%60:05.2f}", font=font(14), fill=CYAN)
        d.text((W - 250, 22), f"FRAME {tele.frame:06d}", font=font(14), fill=GREY)
        d.ellipse([W - 120, 24, W - 108, 36], fill=GREEN)
        d.text((W - 102, 21), "ONLINE", font=font(13, True), fill=GREEN)

    # ---------- 角落装饰环 ----------

    def _draw_corner_rings(self, canvas, tele: Telemetry):
        from PIL import ImageDraw
        d = ImageDraw.Draw(canvas)
        for (cx, cy, a0) in ((1860, 90, 0), (84, 1010, 180)):
            r = 34
            for k, rr in enumerate((r, r - 10)):
                start = a0 + tele.frame * (1.2 if k == 0 else -2.0)
                d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start, start + 240,
                      fill=(*CYAN, 120), width=1)

    # ---------- 片头 / 片尾全屏 ----------

    # ---------- 合成 ----------

    def render(self, main_rgb, eye_rgb, eye_depth, tele: Telemetry):
        from PIL import Image
        canvas = self.base.copy()
        self._draw_sim(canvas, main_rgb, tele)
        self._draw_eye(canvas, eye_rgb, eye_depth, tele)
        self._draw_pcl(canvas, tele)
        self._draw_skills(canvas, tele)
        self._draw_rag(canvas, tele)
        self._draw_topbar(canvas, tele)
        self._draw_corner_rings(canvas, tele)
        canvas = Image.alpha_composite(canvas, self.scanlines)
        import cv2
        return cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


# ============================================================
# 录制器：三路 MuJoCo 渲染 + 3D 发光标记 + env.step 包装
# ============================================================

class SceneRecorder:
    def __init__(self, env, path: str, tele: Telemetry, hud: JarvisHUD,
                 grip_site: str, fps: int = 30) -> None:
        import cv2
        import mujoco
        self.env, self.tele, self.hud = env, tele, hud
        self.grip_site = grip_site
        self.cv2, self.mujoco = cv2, mujoco
        model = env.mj_model
        # 扩大离屏 framebuffer（模型 XML 默认 640x480）
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 1280)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
        # 电影级光照：开阴影
        model.light_castshadow[:] = 1
        self.main_r = mujoco.Renderer(model, 720, 1280)
        self.eye_r = mujoco.Renderer(model, 360, 640)
        self.cam = mujoco.MjvCamera()
        self.eye_cam = mujoco.MjvCamera()
        self.eye_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.41, 0.0, 0.45]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.52, 134.0, -23.0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        assert self.writer.isOpened(), "VideoWriter 打开失败"
        self.path = path
        self._depth_f = -99
        # 碰撞检测：复用 skill 层 CollisionMonitor（含 mesh 过滤，避免
        # mj_geomDistance 对 mesh 伪 0.0 的假阳性）。惰性构建——本构造先于
        # env.obstacle_bodies 注册，首次 emit 时 obstacles 已就绪。
        self._monitor = None

    def _wrist_pose(self):
        """腕部相机：光心在 TCP 正上方 WRIST_LIFT，垂直俯视。"""
        tcp = np.asarray(self.env.get_site_pos(self.grip_site), float)
        self.eye_cam.lookat[:] = tcp + np.array([0.0, 0.0,
                                                 -(WRIST_DISTANCE - WRIST_LIFT)])
        self.eye_cam.distance = WRIST_DISTANCE
        self.eye_cam.elevation = -90.0
        self.eye_cam.azimuth = 90.0

    def capture_wrist(self) -> Dict[str, Any]:
        """抓拍一帧腕部 RGB-D，并返回深度反投影所需的相机几何。"""
        self._wrist_pose()
        self.eye_r.update_scene(self.env.mj_data, camera=self.eye_cam)
        self._beautify(self.eye_r.scene)
        rgb = self.eye_r.render().copy()
        self.eye_r.enable_depth_rendering()
        self.eye_r.update_scene(self.env.mj_data, camera=self.eye_cam)
        depth = self.eye_r.render().copy()
        self.eye_r.disable_depth_rendering()
        c = self.eye_r.scene.camera[0]
        pos = np.array(c.pos, float)
        fwd = np.array(c.forward, float)
        up = np.array(c.up, float)
        right = np.cross(fwd, up)
        fovy = float(self.env.mj_model.vis.global_.fovy)
        return {"rgb": rgb, "depth": depth, "pos": pos, "fwd": fwd,
                "up": up, "right": right, "fovy": fovy}

    def _beautify(self, scn):
        f = self.mujoco.mjtRndFlag
        scn.flags[f.mjRND_SHADOW] = 1
        scn.flags[f.mjRND_REFLECTION] = 1
        scn.flags[f.mjRND_FOG] = 0

    def _sphere(self, scn, pos, rgba, r):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        self.mujoco.mjv_initGeom(
            g, self.mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0.0, 0.0]),
            np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, float))
        scn.ngeom += 1

    def _box(self, scn, pos, rgba, size):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        self.mujoco.mjv_initGeom(
            g, self.mujoco.mjtGeom.mjGEOM_BOX, np.asarray(size, float),
            np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, float))
        scn.ngeom += 1

    def collision_dist(self) -> tuple:
        """机械臂 link 与柜子/抽屉的最小 signed 距离 + 接触对名。"""
        if self._monitor is None:
            from darwin.skills.primitives.collision import CollisionMonitor
            self._monitor = CollisionMonitor(self.env, actor="agent0")
        return self._monitor.clearance()

    def _markers(self, scn, tele: Telemetry):
        env = self.env
        pulse = 0.006 + 0.002 * math.sin(tele.frame * 0.2)
        # 物体中心（青）
        center = env.get_body_pos("green_block")
        self._sphere(scn, center, (0.0, 0.9, 1.0, 0.18), 0.022)
        self._sphere(scn, center, (0.0, 0.9, 1.0, 0.95), pulse)
        # 抓取点 + GraspNet 6-DoF 位姿（琥珀）
        if tele.grasp_pt is not None:
            gp = np.asarray(tele.grasp_pt, float)
            self._sphere(scn, gp, (1.0, 0.7, 0.0, 0.2), 0.02)
            self._sphere(scn, gp, (1.0, 0.7, 0.0, 1.0), 0.007)
            for cd in tele.candidates[:5]:
                self._sphere(scn, cd["position"], (1.0, 0.7, 0.0, 0.35), 0.005)
            if tele.grasp_R is not None:
                Rm = np.asarray(tele.grasp_R, float).reshape(3, 3)
                if tele.grasp_source == "graspnet":
                    app, cls = Rm[:, 0], Rm[:, 1]
                else:
                    app, cls = Rm[:, 2], Rm[:, 0]
                # approach 轴珠串
                for k in range(1, 4):
                    self._sphere(scn, gp + app * (0.018 * k),
                                 (1.0, 0.85, 0.3, 0.95), 0.004)
                # 夹爪两指端（按预测开合宽度）
                hw = min(float(tele.grasp_w) / 2, 0.05)
                self._sphere(scn, gp + cls * hw, (1.0, 0.9, 0.3, 1.0), 0.006)
                self._sphere(scn, gp - cls * hw, (1.0, 0.9, 0.3, 1.0), 0.006)
        # cube_goal（绿框 + 心）
        goal = env.get_site_pos("cube_goal")
        self._box(scn, goal, (0.0, 1.0, 0.53, 0.25), (0.03, 0.03, 0.03))
        self._sphere(scn, goal, (0.0, 1.0, 0.53, 0.9), 0.006)
        # 抽屉把手（蓝）
        self._sphere(scn, env.get_site_pos("drawer"), (0.2, 0.5, 1.0, 0.9), 0.006)
        # TCP 轨迹尾迹
        trail = list(tele.trail)
        for i, p in enumerate(trail):
            a = 0.08 + 0.5 * i / max(1, len(trail))
            self._sphere(scn, p, (0.0, 0.9, 1.0, a), 0.004)

    def _render_views(self, tele: Telemetry):
        env = self.env
        # 缓慢环绕
        self.cam.azimuth = 134.0 + 6.0 * math.sin(tele.frame * 0.012)
        self.cam.elevation = -23.0 + 2.0 * math.sin(tele.frame * 0.02)
        self.main_r.update_scene(env.mj_data, camera=self.cam)
        self._beautify(self.main_r.scene)
        self._markers(self.main_r.scene, tele)
        main = self.main_r.render()
        # 腕部相机（末端垂直俯视，随 TCP 运动）
        self._wrist_pose()
        self.eye_r.update_scene(env.mj_data, camera=self.eye_cam)
        self._beautify(self.eye_r.scene)
        eye = self.eye_r.render()
        depth = None
        if tele.frame - self._depth_f >= 3:
            self.eye_r.enable_depth_rendering()
            self.eye_r.update_scene(env.mj_data, camera=self.eye_cam)
            depth = self.eye_r.render().copy()
            self.eye_r.disable_depth_rendering()
            self._depth_f = tele.frame
        return main, eye, depth

    def emit(self, tele: Telemetry):
        main, eye, depth = self._render_views(tele)
        tele.coll_dist, tele.coll_pair = self.collision_dist()
        frame = self.hud.render(main, eye, depth, tele)
        self.writer.write(frame)
        tele.frame += 1

    def close(self):
        self.writer.release()
        self.main_r.close()
        self.eye_r.close()


# ============================================================
# 主流程
# ============================================================

def _hold(rec: SceneRecorder, tele: Telemetry, n: int):
    """静态保持 n 帧（HUD 动画照常跑）。"""
    for _ in range(n):
        rec.emit(tele)


def _drawer_qpos(env) -> float:
    return float(np.asarray(env.mj_data.joint("drawer:joint").qpos).reshape(-1)[0])


def _depth_to_world(depth: np.ndarray, geom: Dict[str, Any]) -> np.ndarray:
    """腕部深度图 -> 世界坐标点云（深度反投影的唯一坐标边界）。"""
    h, w = depth.shape
    f = (h / 2) / math.tan(math.radians(geom["fovy"]) / 2)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xn = (u - w / 2) / f
    yn = (h / 2 - v) / f
    rays = (geom["right"][None, None, :] * xn[..., None]
            + geom["up"][None, None, :] * yn[..., None]
            + geom["fwd"][None, None, :])
    pts = geom["pos"][None, None, :] + depth[..., None] * rays
    return pts.reshape(-1, 3)


def produce(out_path: str, fps: int = 30) -> str:
    from darwin.benchmarks import get_benchmark
    from darwin.agents.runner import EpisodeRunner, _get_env
    from darwin.memory import RAGMemory
    from darwin.skills.perception.grasp import (
        GraspPoseSkill, object_features, rel_offset_of, score_band_of,
        sample_object_point_cloud, _load_graspnet)

    entry = get_benchmark("drawer_place")
    env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))
    # 视频可复现：关闭 reset 时 IK 随机末端（否则每进程 home 位姿随机、回程不稳定）
    env.is_randomize_end = False
    hud = JarvisHUD()
    cmn = {"actor": entry["actor"], "grip_site": entry["grip_site"]}

    # GraspNet 预载 + CUDA 预热（首次推理约 30s，不计入视频阶段延迟）
    print("[grasp] preload GraspNet + CUDA warmup ...", flush=True)
    if _load_graspnet() is not None:
        warm = (np.random.rand(20000, 3) * 0.04
                + np.array([0.4, -0.2, 0.44])).astype(np.float32)
        GraspPoseSkill().execute(point_cloud=warm, top_k=1)
        print("[grasp] warmup done", flush=True)
    grasp_skill = GraspPoseSkill()

    import tempfile
    video_rag_root = tempfile.mkdtemp(prefix="darwin_jarvis_rag_")
    print(f"[rag] 视频隔离经验库: {video_rag_root}", flush=True)

    for take in range(4):
        tele = Telemetry()
        rag = RAGMemory(root=video_rag_root)
        runner = EpisodeRunner(entry, rag=rag, verbose=False)
        rec = SceneRecorder(env, out_path, tele, hud,
                            grip_site=entry["grip_site"], fps=fps)
        env.reset()
        # 安全 home 位姿：原 DianaDrawerCube init_qpos 让夹爪贴着柜子(1mm)，
        # 改用远离柜子的位姿（TCP≈[0.55,-0.04,0.61]，与柜子间隙>30mm）
        import mujoco as _mj
        SAFE_HOME_QPOS = np.array(
            [-0.614, -0.2586, -0.7121, 2.2855, 0.2737, -0.6727, -0.2971])
        env.mj_data.qpos[:7] = SAFE_HOME_QPOS
        env.mj_data.qvel[:7] = 0.0
        _mj.mj_forward(env.mj_model, env.mj_data)
        env.init_pos[entry["actor"]] = np.array(env.get_site_pos(entry["grip_site"]))
        # 障碍 body 注册：skill 层 CollisionMonitor 据此做碰撞监测 + 走廊避障
        env.obstacle_bodies = ("cupboard", "drawer")

        # ========== PHASE 01 // PERCEIVE ==========
        tele.phase = "PERCEIVE"
        feat = object_features(env, entry["body"])
        center = np.array(env.get_body_pos(entry["body"]), float)
        feat["center"] = center.tolist()
        tele.goal_xyz = np.array(env.get_site_pos("cube_goal"), float)
        size = np.asarray(feat["size"], float)

        clock = {"t": time.time()}

        def wrapped_step(action):
            orig_step(action)
            tele.sim_steps += 1
            tele.tcp = env.get_site_pos(entry["grip_site"])
            tele.drawer_qpos = _drawer_qpos(env)
            tele.trail.append(np.array(tele.tcp, float))
            bp = env.get_body_pos(entry["body"])
            tele.block_dist = float(np.linalg.norm(bp - env.get_site_pos("cube_goal")))
            tele.skill_live = time.time() - clock["t"]
            rec.emit(tele)

        orig_step = env.step
        env.step = wrapped_step

        # 1) 末端移动到扫描位姿（腕部相机由此垂直对准方块）
        scan_tgt = [float(center[0] + 0.03), float(center[1] + 0.04), 0.60]
        tele.skill_detail = (f"move_to(scan_pose=[{scan_tgt[0]:.2f},{scan_tgt[1]:.2f},"
                             f"{scan_tgt[2]:.2f}]) :: wrist RGB-d acquisition")
        clock["t"] = time.time()
        runner.agent.skills["move_to"].execute(
            env=env, target=scan_tgt, gripper=0.0, tol=0.025, timeout=160, **cmn)

        # 2) 腕部 RGB-D 抓拍 -> 深度反投影重建物体点云（计时）
        t0 = time.perf_counter()
        cap = rec.capture_wrist()
        P = _depth_to_world(cap["depth"], cap)
        dd = cap["depth"].reshape(-1)
        r_xy = max(float(max(size[:2]) * 2.4), 0.045)
        m_xy = np.linalg.norm(P[:, :2] - center[:2], axis=1) < r_xy
        m_z = ((P[:, 2] > center[2] - size[2] - 0.006)
               & (P[:, 2] < center[2] + size[2] + 0.04))
        obj = P[m_xy & m_z & (dd < 1.3)]
        recon_ms = (time.perf_counter() - t0) * 1000
        cloud_source = "depth"
        if len(obj) < 1000:  # 重建点太少：真值表面采样兜底
            obj = sample_object_point_cloud(env, entry["body"], n_points=20000)
            cloud_source = "gt-sample"
        tele.cloud = obj.astype(np.float32)

        # 3) GraspNet 6-DoF 抓取位姿（计时，空预测时逐级回退）
        from darwin.skills.perception.grasp import _points_to_grasp_geometry

        def _safe_grasp(cloud):
            try:
                r = grasp_skill.execute(point_cloud=cloud, top_k=6)
                if r.get("success") and r.get("position") is not None:
                    return r
            except Exception as e:  # noqa: BLE001
                print(f"[grasp] retry after: {e}", flush=True)
            return {}

        t1 = time.perf_counter()
        gres = _safe_grasp(tele.cloud)
        if not gres:  # 真值采样云再试一次 GraspNet
            gt_cloud = sample_object_point_cloud(env, entry["body"], n_points=20000)
            gres = _safe_grasp(gt_cloud)
            if gres:
                tele.cloud = gt_cloud.astype(np.float32)
                cloud_source = "gt-sample"
        if not gres:  # 几何 PCA 兜底
            gres = _points_to_grasp_geometry(np.asarray(tele.cloud, np.float32))
        grasp_ms = (time.perf_counter() - t1) * 1000
        print(f"[perceive] cloud={cloud_source} n={len(tele.cloud)} "
              f"recon={recon_ms:.1f}ms grasp={grasp_ms:.1f}ms "
              f"src={gres.get('source')}", flush=True)
        tele.lat["PERCEIVE"] = (recon_ms + grasp_ms) / 1000.0

        cand: Dict[str, Any]
        raw_cands = [c for c in gres.get("candidates", []) if "position" in c]
        if gres.get("source") == "graspnet" and raw_cands:
            # 孤立物体云会混入侧向/仰视候选：仅保留落在包络内的俯视抓取
            pick = None
            for c in raw_cands:
                p0 = np.asarray(c["position"], float)
                R0 = np.asarray(c["rotation_matrix"], float).reshape(3, 3)
                dxy0 = float(np.linalg.norm(p0[:2] - center[:2]))
                dz0 = float(p0[2] - center[2])
                # GraspNet col0 = 物体->夹爪方向：俯视有效抓取要求其朝上 (+z)
                if (dxy0 < float(np.max(size[:2])) * 1.4 + 0.006
                        and abs(dz0) < float(size[2]) + 0.025
                        and R0[2, 0] > 0.4):
                    if pick is None or float(c.get("score", 0)) > float(pick.get("score", 0)):
                        pick = c
            cands = raw_cands[:6]
            if pick is not None:
                p0 = np.asarray(pick["position"], float)
                R0 = np.asarray(pick["rotation_matrix"], float).reshape(3, 3)
                # 可执行化：xy 夹回物体包络、z 取已验证的物体半高、强制俯视
                off = np.clip(p0[:2] - center[:2], -size[:2] * 0.6, size[:2] * 0.6)
                gpos = np.array([center[0] + off[0], center[1] + off[1], center[2]])
                cls_h = R0[:, 1].copy()
                cls_h[2] = 0.0
                if np.linalg.norm(cls_h) < 0.1:
                    cls_h = np.array([1.0, 0.0, 0.0])
                cls_h /= np.linalg.norm(cls_h)
                # 可视化约定 col0=物体->夹爪(朝上)，col1=closing(水平)
                gapp = np.array([0.0, 0.0, 1.0])
                gR = np.stack([gapp, cls_h, np.cross(gapp, cls_h)], axis=1)
                gw = float(np.clip(pick.get("width", 0.06), 0.03, 0.07))
                gscore = float(pick.get("score", 0.5))
                gsrc = "graspnet"
            else:
                print("[grasp] 候选均不满足俯视几何约束 -> 中心兜底", flush=True)
                gpos, gR, gw, gscore, gsrc = center, None, 0.06, 1.05, "geometry"
        elif gres.get("success"):
            # PCA 几何兜底（col2=approach 约定）
            gpos = np.asarray(gres["position"], float)
            gR = np.asarray(gres["rotation_matrix"], float).reshape(3, 3)
            gw = float(gres.get("width", 0.06))
            gscore = float(gres.get("score", 0.5))
            gsrc = str(gres.get("source", "geometry"))
            cands = []
        else:
            gpos, gR, gw, gscore, gsrc, cands = center, None, 0.06, 1.05, "geometry", []
        cand = {"position": np.asarray(gpos).tolist(), "score": float(gscore),
                "rel_offset": rel_offset_of(np.asarray(gpos, float), center, feat["size"]),
                "score_band": score_band_of(float(gscore))}
        tele.skill_detail = (f"reconstruct {recon_ms:.1f}ms ({len(tele.cloud)} pts) + "
                             f"{gsrc or 'fallback'} {grasp_ms:.1f}ms")

        # 扫描结束：一次回到 home（CARTIK 从 home 构型出发后续最稳定）
        home_xy = np.asarray(env.init_pos[entry["actor"]], float)[:2]
        for h_try in range(1, 4):
            runner.agent.skills["home"].execute(env=env, timeout=130, **cmn)
            tcp_xy = np.asarray(env.get_site_pos(entry["grip_site"]), float)[:2]
            if float(np.linalg.norm(tcp_xy - home_xy)) < 0.03:
                break
        tele.skill_detail = (f"reconstruct {recon_ms:.1f}ms ({len(tele.cloud)} pts) + "
                             f"{gsrc or 'fallback'} {grasp_ms:.1f}ms")
        # 点云揭示 + 抓取位姿发布（紧凑，避免长时间静止）
        REVEAL = 16
        for f in range(REVEAL):
            tele.scan = (f + 1) / REVEAL
            _hold(rec, tele, 1)
        tele.grasp_pt, tele.grasp_R = gpos, gR
        tele.grasp_w, tele.grasp_score = gw, gscore
        tele.grasp_source, tele.candidates = gsrc, cands
        _hold(rec, tele, 6)

        # ========== PHASE 02 // RAG ==========
        tele.phase = "RAG"
        t0 = time.perf_counter()
        recs = rag.retrieve(f"drawer_place {feat['shape']} grasp", feat, k=3)
        tele.lat["RAG"] = time.perf_counter() - t0
        tele.rag_records = recs
        matched = [r for r in recs if r.get("task") == "drawer_place"]
        tele.cold_start = len(matched) == 0
        stats = rag.stats
        tele.rag_succ = sum(int(v.get("succ", 0)) for v in stats.values())
        tele.rag_fail = sum(int(v.get("fail", 0)) for v in stats.values())
        _hold(rec, tele, int(1.2 * fps))

        # ========== PHASE 03 // PLAN ==========
        tele.phase = "PLAN"
        cfg = {"hover": 0.12, "k_descend": 2.0, "lift_height": 0.52}
        t0 = time.perf_counter()
        plan = runner._plan_for(cand, cfg, None, env)
        tele.lat["PLAN"] = time.perf_counter() - t0
        action_to_idx = {a: i for i, (a, _, _) in enumerate(SKILL_DEFS)}
        for idx in [action_to_idx[p["action"]] for p in plan]:
            tele.states[idx] = "queue"
            tele.skill_detail = f"chain[{idx:02d}] :: {SKILL_DEFS[idx][1]} queued"
            _hold(rec, tele, 2)
        _hold(rec, tele, int(0.5 * fps))

        # ========== PHASE 04 // EXECUTE ==========
        tele.phase = "EXECUTE"
        traj: List[Dict[str, Any]] = []
        ok_all = True
        fail_idx = -1
        t_exec = time.perf_counter()
        for i, p in enumerate(plan):
            idx = action_to_idx[p["action"]]
            tele.current = idx
            tele.states[idx] = "run"
            tele.skill_detail = f"{p['action']}({_brief_params(p['params'])})"
            tele.result_detail = ""
            clock["t"] = time.time()
            tele.skill_live = 0.0
            skill = runner.agent.skills[p["action"]]
            boost = {"home": 150, "move_to": 220, "move_above": 200,
                     "descend": 200, "lift": 200, "move_to_xy_top": 200,
                     "place": 380}
            ex_params = dict(p["params"])
            if p["action"] in boost:
                ex_params["timeout"] = boost[p["action"]]
            # 避障已下沉到 skill 层（collision.py：走廊规划 + 每步碰撞监测）
            try:
                result = skill.execute(env=env, **ex_params)
            except TypeError:
                try:
                    result = skill.execute(env=env, **p["params"])
                except Exception as e:  # noqa: BLE001
                    result = {"success": False, "error": str(e)}
            except Exception as e:  # noqa: BLE001
                result = {"success": False, "error": str(e)}
            dt_skill = time.time() - clock["t"]
            tele.skill_ms[idx] = dt_skill
            tele.lat["EXECUTE"] = time.perf_counter() - t_exec
            traj.append({"action": p["action"], "params": p["params"], "result": result,
                         "step_idx": i})
            if result.get("success"):
                tele.states[idx] = "done"
                extra = f" in {result.get('steps', '?')} steps" if "steps" in result else ""
                tele.result_detail = f"[{p['action']}] success{extra}"
            else:
                tele.states[idx] = "fail"
                tele.result_detail = f"[{p['action']}] FAIL :: {result.get('reason') or result.get('error')}"
                ok_all = False
                fail_idx = idx
                break
        tele.current = -1
        env.step = orig_step

        # ========== PHASE 05 // VERIFY ==========
        drawer_qpos = _drawer_qpos(env)
        block_dist = float(np.linalg.norm(
            env.get_body_pos(entry["body"]) - env.get_site_pos("cube_goal")))
        verified = ok_all and drawer_qpos > 0.08 and block_dist < 0.05
        tele.phase = "VERIFY"
        tele.success = verified
        tele.metrics = {"drawer_qpos": drawer_qpos, "block_dist": block_dist,
                        "fitness": (1.0 - 0.02) if verified else -0.12}
        if not verified:
            _hold(rec, tele, int(1.2 * fps))
            rec.close()
            Path(out_path).unlink(missing_ok=True)
            _fail = traj[-1]["result"] if traj else {}
            print(f"[take {take + 1}] 验证失败 ok={ok_all} drawer={drawer_qpos:.3f} "
                  f"dist={block_dist:.3f} fail_idx={fail_idx} "
                  f"reason={_fail.get('reason')} clear={_fail.get('min_clearance')} "
                  f"pair={_fail.get('coll_pair')} end={_fail.get('end')} "
                  f"body_z={_fail.get('body_z')}，重试…", flush=True)
            continue

        # 经验沉淀（计时：写入 + tick/decay + 再检索）
        t0 = time.perf_counter()
        runner.agent.reflect(entry["task_name"], feat, traj, True,
                             grasp_pt=cand["position"])
        rag.tick()
        rag.decay()
        new_recs = rag.retrieve(f"drawer_place {feat['shape']} grasp", feat, k=3)
        tele.lat["VERIFY"] = time.perf_counter() - t0
        tele.rag_records = new_recs
        stats = rag.stats
        tele.rag_succ = sum(int(v.get("succ", 0)) for v in stats.values())
        tele.rag_fail = sum(int(v.get("fail", 0)) for v in stats.values())
        tele.cold_start = False
        for f in range(int(2.2 * fps)):
            tele.memory_flash = max(0.0, 1.0 - f / (2.2 * fps))
            _hold(rec, tele, 1)

        rec.close()
        print(f"[OK] 视频已生成: {out_path}  ({tele.frame} 帧 @ {fps}fps, "
              f"drawer={drawer_qpos:.3f} dist={block_dist*1000:.1f}mm)", flush=True)
        return out_path

    raise RuntimeError("多次尝试仍未成功完成 drawer_place，请检查仿真环境。")


def _brief_params(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if k in ("actor", "grip_site"):
            continue
        if isinstance(v, list) and len(v) == 3:
            parts.append(f"{k}=[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="videos/darwin_drawer_place_jarvis.mp4")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    produce(args.out, fps=args.fps)


if __name__ == "__main__":
    main()
