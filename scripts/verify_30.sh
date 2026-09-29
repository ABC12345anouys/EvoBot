#!/usr/bin/env bash
# 30 任务全量回归（验收脚本）：spatial+goal+object 各任务 N 次尝试，串行。
# 用法: bash scripts/verify_30.sh [attempts] [tag]   默认 8 次，标签 verify
# 结果: /tmp/<tag>/ 下每任务日志 + 屏幕逐任务 PASS/FAIL 与累计
ATTEMPTS=${1:-8}
TAG=${2:-verify}
mkdir -p /tmp/$TAG
cd "$(dirname "$0")/.."
pass=0; total=0
for suite in libero_spatial libero_goal libero_object; do
  for i in 0 1 2 3 4 5 6 7 8 9; do
    task=$suite:$i
    t=${suite}_${i}
    rm -f /tmp/$TAG/$t.sock
    PYTHONPATH=$PWD:/home/lifd/Public/LIBERO MUJOCO_GL=egl \
      timeout 1800 /home/lifd/anaconda3/envs/darwin/bin/python -u scripts/ipc_learn.py \
      "$task" --max-attempts $ATTEMPTS --sock /tmp/$TAG/$t.sock \
      > /tmp/$TAG/$t.log 2>&1
    rc=$?
    total=$((total+1)); [ $rc -eq 0 ] && pass=$((pass+1))
    echo "$([ $rc -eq 0 ] && echo PASS || echo FAIL) $task ($(date +%H:%M:%S)) 累计 $pass/$total"
  done
done
echo "ALL DONE: $pass/$total"
