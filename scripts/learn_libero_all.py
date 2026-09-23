#!/usr/bin/env python
"""LIBERO 全任务批量学习一键入口（队列 spatial→object→goal→10→90）。

例：
  # 冒烟两个任务（台账/跳过/resume/profile/哨兵路径全走一遍）
  python scripts/learn_libero_all.py \\
      --limit libero_spatial:0,libero_object:0 --max-hours 0.5

  # 后台全量：墙钟 8 小时、每任务最多 8 attempt、规则反思
  nohup python scripts/learn_libero_all.py --max-hours 8 \\
      > logs/learn_all.out 2>&1 &

  # 带 LLM 兜底分解/反思 + token 预算
  python scripts/learn_libero_all.py --llm --llm-token-budget 500000

断点续跑：直接重跑即可，台账中 learned 自动跳过（--force 强制重学）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from darwin.ipc.batch_learner import main

if __name__ == "__main__":
    raise SystemExit(main())
