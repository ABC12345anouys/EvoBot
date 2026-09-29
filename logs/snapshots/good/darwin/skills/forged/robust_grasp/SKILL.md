---
name: robust_grasp
description: Robust grasp with GraspNet score fallback
entry_file: entry.py
kind: strategy
confidence: verified
applies_when: grasping objects where GraspNet score may be low
evidence:
  cells:
  - pickplace_sA
  - pickplace_sC
  - stack_sB
  tasks:
  - pickplace
  - stack
  attempts: 3
  solved_seeds:
  - pickplace_sA
  - pickplace_sC
  - stack_sB
  failed_seeds: []
---
