---
scope: global
kind: infra
confidence: single-shot
evidence:
  cells:
  - grasp_stats
  attempts: 1
applies_when: 统计成功率表 (shape|task|band)
table:
  box|pickplace|mid:
    succ: 1
    fail: 0
---
(shape, task, score_band) → 成功/失败计数
