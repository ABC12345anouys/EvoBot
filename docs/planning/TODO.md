# 剩余工作清单与实验方法
更新时间: 2026-09-23. 基线:非90 四件套台账 31/40(spatial 10/10、goal 10/10、object 10/10、libero_10 1/10)
09-23 全量基线（30 任务 × 5 次尝试）：**18/30**——spatial 6/10（败 :4/6/7/9）、goal 4/10（败 :2/3/4/5/8/9）、object 8/10（败 :6/:8，对比 09-22 的 10/10 是回归）。
**归因更正**：此前记录"碗任务候选偏心 0.36~0.5m"是误读——那列 xy_dist 是**失败时物体到目标的距离**，不是抓取点偏移（标定脚本实测 grasp_pt 偏移 P50≈0.004~0.038m）。碗任务真实失败模式是 F=0 夹空（descend/接触侧），与 object:6(butter)/:8(chocolate_pudding) 的 grip_failed 同型。object:6/:8 高度疑似 09-22"中心候选置顶"改动回归（末次成功快照本身也 5 连败），已恢复快照并跑 DARWIN_CENTER_THRESH=999 对照（后台）。goal:8/9 place 振荡已修滞环；goal:3/4 是 carry 撞墙型。
09-23 已处理：① place 规则加 ±8mm 滞环（reflection._xy_side + agent_learner._last_place_dir，xy 在 0.022~0.038 之间维持上次判定，冒烟验证 place_k 不再往返）；② 复测失败的 10 个任务（spatial 4/6/7/9、goal 2/3/4/5/8/9）+ object 6/8 已恢复各自末次成功快照（contact_stop_band 标定值保留，streak≥4 自动回退也已豁免该参数）。Phase 1 感知层 mesh 真值顶点采样（修 F=0 夹空的根因：mesh 物体点云是 AABB 盒壳、填满容器空腔，GraspNet 拿不到真实沿口/外壁）；Phase 2 规划层 AABB 绕障 + stall fail-fast（治 carry/above 撞墙磨 368 步）——已完成。

09-23 下午攻坚（18/30 → 21/30 确认，Round3 新增 goal:3、goal:8、spatial:1 复测通过）：
- **反思死亡螺旋修复**（`reflection.py` 硬守卫，全部有实证）：k_descend 硬顶 6.0（历史成功分布 90% ≤5.4；spatial:2 被 streak 加速推到 8.14 后指尖压碗沿 F=48N 提不动）；hover 地板 0.06（漂到 0.03 后 TCP 贴顶擦碰把物体推离）；容器 z_delta 地板 0.04、ratio 地板 0.62（goal_not_reached 降到 0.52 后插指进不了碗壁 8/8 夹空 F=0）。到地板后改调释放/放置，不再继续压。
- **cfg 重置法**：从 episode 日志提取每任务"末次 PASS 完整 cfg"与当前 YAML diff，把 9 个失败任务重置回历史最佳（spatial:2 → 当场 PASS）。比让反思在 8 次尝试里自己摸回来快得多。
- **z_top 几何根因三连修**（`libero_adapter.object_bounds`，探针逐 geom 解剖实证）：① mesh 视觉 geom 的 xpos 是局部原点不是表面（butter 框架悬在顶上方 17mm）→ mesh 用真值顶点经姿态变换求 z 范围；② **碰撞盒带旋转**时 pos+size[2] 严重失真（butter 碰撞盒长轴横放，旧公式 z_top=0.047、角点真值 0.0174）→ box 8 角点变换、cylinder 用旋转公式。object:6/8 因此从"抓取点悬物体上方 3cm 夹空 F=0 8/8"恢复几何正确（09-21 的"工作高度"其实来自 GraspNet 候选 z，z-override 加入后该信息丢失）；③ 容器 descend 接触软停带收紧 0.015（`methods.py`，band 0.05 会把"指被沿口挡住停在 z_goal 上 3~4cm"误判到达，高位闭合夹空）。
- **容器方向抽签修复**（`runner_dynamic`）：投票方向受点云随机采样抖动，同场景 [-0.22,-0.98] 夹得住 F=16、[-0.59,-0.80] 夹空 F=0 → 方向 snap 到最近 45°（圆碗任意方向几何等价，snap 后 straddle 裕度决定性）+ 顶点下采样固定种子（点云/候选可复现，逐 attempt 探索仍由 jit 负责）+ 偏置落点 lateral 净空 <2cm 时换方向（治柜沿楔止，spatial:4）。
- **Round4/5 教训**：后台批跑必须 disable_timeout（600s 默认超时会连坐杀 sim）；批跑期间改代码会污染后续任务（每任务新进程 import 当前代码）。
- 探针方法：`get_libero_benchmark` + `env_utils.get_env` 直接建 env（与批跑并行无害，EGL 多 context 共存），打印 bounds/geom/候选/逐步指位+接触力，两轮探针定位了全部三个根因。脚本在 /tmp/probe_*.py。

09-23 晚间攻坚（21/30 → 23/30 确认，Round6 新增 object:6、object:8）：
- **object:6/:8 修复验证 PASS**：z_top 角点变换修后 butter z_top=0.0177（旧公式 0.0468 悬顶 3cm），两任务 8 attempts 内通过。
- **straddle 几何结论**（探针+推导，akita_black_bowl）：夹爪两指沿世界 Y 固定分离 80mm；偏置方向**纯 Y 最优**——off=0.7·half_y=39.4mm 时内指径向≈0.6mm 落碗腔、外指 79.4mm 远超碗沿 56.3mm，straddle 决定性；**X 向最差**（双指都在 56.1mm≈碗沿半径上蹭沿磨停，spatial:2 Round6 回归根因，45° snap 恰好把投票方向 snap 到 X 向）；45° 对角内指入腔 30.4mm 过深。
- **容器分支重写**（`runner_dynamic`）：废弃 vote+45° snap，固定纯 Y 偏置（`center ± 0.7·half_y·[0,∓1]`），符号按 ±Y 两侧障碍净空（<2cm 为楔止风险）翻向，两侧都不足才退回 12 方向搜最大净空。探针对照实证：纯 Y 点 hold-close F=8.45N、press-close（生产代码方式）F=8.70N **都夹住**——CloseGripperSkill 下压 6mm 不是夹空元凶，不改。
- **扁平物包边规则**（`runner_dynamic` 实心分支）：thickness<0.015 时 grasp_z 从 z_top−0.012 改为 z_top−0.002（指尖包盘沿，修 goal:5 薄盘 lift F=0 8/8）。
- **goal:2 修复**：PARAM_SPEC 新增 `place_vcap`（默认回落 vcap），place 下落段独立限速；goal:2 YAML 设 0.05（原 vcap=1.0，place 起始 3~7 步内酒瓶滑脱 8/8）。
- **探针环境坑**：同一进程两次 `get_env`（先 close 再建）会挂死（26min 无 GPU 活动）；探针脚本必须单 env 单进程。
- 架构愿景文档已写：`docs/architecture_step_planner.md`（步级快照重试 + 可插拔小模型 policy 接口 + Phase A–D 分期）。**已开工**：§13 把 24/30 后剩余 6 个失败任务逐一映射到架构机制并提炼 4 条修订（静默 fallback 计数/预算参数约束声明/Phase B 提升最高优先级/approach 升一级 step）。Phase A 骨架落地：`darwin/policies/`（context/spaces/registry/rules + retry），grasp_pose 规则从 runner_dynamic 搬入注册表（行为不变，runner 改调 propose）；Phase B 落地：`_execute_chunk` 每步 save_physics 快照，失败 restore+换参原地重试 ≤5（retry.py 累积幂次序列，5 组参数两两不同），白名单 move_above/descend/lift/carry/place，close_gripper 等无参可换的步直接向上传播。
- Round7 批跑：spatial:2/4/9、goal:2/4/5 重攻 + goal:3/8 回归抽查（/tmp/r7_batch.sh）。

09-23 深夜 Round9/9b/10（23/30 → 24/30，Round9 新增 spatial:2；当前剩 spatial:4/9、goal:2/3/4/5 共 6 个）：
- **Round9+**：spatial:2 根因是 YAML z_delta=0.042 污染（重置 0.03 + ratio 0.7 当场 PASS，ik_servo.libero_spatial_2/4.yaml params 已改）。
- **Phase B 重试序列改非累积**（`policies/retry.py`）：5 次重试用 base 参数的绝对倍数（timeout ×1.5/2/3/4.5=180/240/360/540 步），不再幂等叠加——r9 的 270/405 是旧累积序列产物。r9b 证实旧序列全灭（spatial:4/9 五次重试全触发仍 FAIL）。
- **r9b 关键发现**（goal:3/4，进程级 import 时间戳比对）：探针用干净参数（k=5.0、jit=0、hover=0.06）descend 40~90 步成功；真实运行同款 grasp_pt（episode jsonl attempt_start 有记录）却 8/8 xy_drift/stalled 交替。机制=**几何余量边缘运行**：contact_stop_band 容器收紧到 0.015 后，OSC 深 z 到达极限（指尖停在目标上方 8~11mm，F=0）距 15mm 上限只剩 3~4mm，jit=0.005~0.01（5~10mm 横向抖动，runner_dynamic:476-480 对非记忆候选全步生效）直接把余量吃光；失败在 stalled/xy_drift 间交替 = 不同 attempt 抖动方向不同。处置：goal:3/4/5、spatial:4 的 jit 归零（容器 straddle 是精密几何，抖动是为扁平实心物设计的）。goal:2/8/9 本来就是 0 未动。
- **F=0 到达极限接受准则**（`ik_servo.py` descend）：双窗口停滞时读 `env.contact_force_on_body(body)`——F≈0 说明指尖在自由空间，停滞是 OSC 可达极限而非几何阻挡（spatial:4 实证停沿口上方 22mm F=0），若 z 在 reach_limit_band=0.035 内且 xy 收敛则判成功（reason=reach_limit）；F>0 仍走原 contact_stop/fail 路径。三条件缺一不可。对慢速型（0.3mm/步×60 步=18mm>stall_progress 5mm）不误判。
- **place 族成为最大失败簇**：spatial:9（place slipped）、goal:2（酒瓶 place slipped 8/8，place_vcap=0.05 未解决）、goal:8（r9b 回归 place_timeout，resume 机制本身工作正常）——下一步单独立项攻 place 滑脱/超时。
- r10 批跑（/tmp/r10_batch.sh）：jit 归零 + F=0 准则 + 新重试序列，7 任务全量（6 败 + goal:8 回归）。
## 一、当前状态与真实差距
台账分数是历史学习结果；今晚用"每任务 3~5 次尝试"冒烟复测，发现部分任务复现不稳定。台账 ≠ 稳定通过，下表是真实状态：
| 套件 | 台账 | 复测稳定 | 说明 |
|---|---|---|---|
| libero_spatial | 10/10 | 5/10 | :2/:4/:5/:6/:7/:9 复测失败（全是 akita_black_bowl 容器抓取） |
| libero_goal | 10/10 | 4/10 | :2/:3/:4/:5/:6/:8/:9 复测失败 |
| libero_object | 10/10 | 10/10 | :9/:7/:4/:1 均实测修复后通过 |
| libero_10 | 1/10 | 1/10 | 剩余 :0 :1 :2 :4 :5 :6 :7 :8 :9 |
注意：spatial/goal 当初学习时普遍用了 8 次以上尝试（自动反思逐轮调参），冒烟只给 3~5 次，失败未必代表退化；但 goal 套件恢复参数后 5 次仍全败，需要正式重攻。

## 二、未完成事项（按优先级）
### 1. 归因实验未完成（半小时内可做）
问题：9 月 22 日加了"实心物体几何中心候选置顶"改动（`runner_dynamic.py`，厚度阈值 0.06），需要确认它对 spatial/goal 无回归。
已准备好开关：环境变量 `DARWIN_CENTER_THRESH`（默认 0.06，设 999 等价关闭中心候选、恢复旧 GraspNet 逻辑）。
实验设计：
- 对 goal:5（plate）、goal:6（cream_cheese）分别用 `DARWIN_CENTER_THRESH=999` 和默认值各跑 5 attempts
- 对比成功轮数：两者无显著差异 → 改动无回归，失败是历史脆弱性；有差异 → 回滚或调阈值

### 2. 重攻 13 个复测失败任务（spatial 6 个 + goal 7 个）

先用足 8 次尝试（与当初学习条件对齐），仍失败再按失败类型修：
| 任务 | 物体 | 末次失败 | 修复方向 |
|---|---|---|---|
| spatial:2/4/5/6/7/9 | akita_black_bowl | descend_stalled / lift_no_grip | 容器偏心分支，与中心候选无关；查 descend 卡住原因（k_descend 被反思改乱后已恢复） |
| goal:2/9 | wine_bottle | carry move_timeout / place_unstable | 同 object:7/4 的进篮问题：跨距大、carry_vcap 不够，改 per-task YAML `carry_vcap: 0.10~0.15`；place_unstable 是挤墙污染次生问题，carry 修好后自然消失 |
| goal:3/4/8 | akita_black_bowl | timeout / grip_failed | 同 spatial 碗 |
| goal:5 | plate_1 | lift_no_grip（F=0） | 微波炉顶高位 descend 接触软停过早（end_z≈0.97 没降到沿口），查 contact_stop 阈值对高位薄物的适配 |
| goal:6 | cream_cheese | lift_no_grip | 扁平物（厚 4.1cm），走偏心分支但没蹭到边；增大 `jit` 或恢复当初成功 cfg 后重试 |

### 3. libero_10 剩余 9 个任务
:4 进度最深（Fix7 修复多 chunk place 跑偏已验证，第一个 mug 的 On 判据通过），剩余三个瓶颈：
- porcelain_mug place 不稳定：TCP 降不到 z_target（place_unstable），base 碰桌沿 clear_mm=-24
- white_yellow_mug_1 抓不住：lift_no_grip
- chunk1 place goal 疑似跑偏 44cm：查 `libero_adapter.py` 的 `support_point(plate_2)` 实现
其余 :0 :1 :2 :5 :6 :7 :8 :9 还没单独分析过失败类型，先各跑一轮收集 fail_phase 再分类。

### 4. 沉淀项（可选）

- 把"失败类型 → 调参路径"经验（lift_no_grip→抓取候选、move_timeout→carry_vcap、place_unstable→先修上游污染）写成 skill 或扩展 `adapt_cfg` 规则
- `resume_unsupported`：carry 超时后 resume 被拒（goal:2/9 实测 resume 一次都不生效），查 sim_worker 判定条件是否过严

## 三、实验运行方法

### 环境

```bash
# python 与环境变量（所有命令共用）
PY=/home/lifd/anaconda3/envs/darwin/bin/python
export PYTHONPATH=/home/lifd/Public/darwin-bot:/home/lifd/Public/LIBERO
export MUJOCO_GL=egl
cd /home/lifd/Public/darwin-bot
```

### 单任务快速验证（最常用）

```bash
rm -f /tmp/quick.sock && PYTHONPATH=/home/lifd/Public/darwin-bot:/home/lifd/Public/LIBERO \
MUJOCO_GL=egl /home/lifd/anaconda3/envs/darwin/bin/python -u scripts/ipc_learn.py \
  libero_object:9 --max-attempts 3 --sock /tmp/quick.sock 2>&1 | tail -20
```

- `-u` 无缓冲必须加；不要用 `| tail` 时还想要实时输出（会缓冲到结束），要看实时就重定向文件再 `tail -f`
- `--sock` 用独立名字，避免与其他运行中的实例冲突
- 成功判据：日志出现 `verified=True ... bddl=True`，且 YAML `learned: true`

### 交互模式（边跑边改参数，推荐攻坚用）

```bash
rm -f /tmp/quick.sock && PYTHONPATH=/home/lifd/Public/darwin-bot:/home/lifd/Public/LIBERO \
MUJOCO_GL=egl /home/lifd/anaconda3/envs/darwin/bin/python -u scripts/ipc_learn.py \
  libero_10:4 --max-attempts 8 --interactive --sock /tmp/quick.sock
```

失败后进程停下打印遥测与 YAML 路径 → 手动改 `darwin/skills/configs/ik_servo.<task>.yaml` 保存 → 终端回车 → 重读 YAML 发 resume（回退续跑，省 2-3 分钟）。`resumable=false` 时回车=全 reset，`q`=退出。

### 批量复测（对照实验/回归检查）

```bash
cat > /tmp/regtest/run.sh <<'SH'
cd /home/lifd/Public/darwin-bot
for task in libero_spatial:2 libero_spatial:4 libero_goal:5; do   # 按需改列表
  tag=$(echo $task | tr ':' '_' | tr '.' '_')
  rm -f /tmp/regtest/$tag.sock
  PYTHONPATH=/home/lifd/Public/darwin-bot:/home/lifd/Public/LIBERO MUJOCO_GL=egl \
    timeout 700 /home/lifd/anaconda3/envs/darwin/bin/python -u scripts/ipc_learn.py \
    "$task" --max-attempts 5 --sock /tmp/regtest/$tag.sock > /tmp/regtest/$tag.log 2>&1
  echo "$([ $? -eq 0 ] && echo PASS || echo FAIL) $task ($(date +%H:%M:%S))"
done
SH
mkdir -p /tmp/regtest && bash /tmp/regtest/run.sh
```

- 并发注意：不要同时跑两个 sim 实例抢 GPU（EGL），除非显式分开 `--sock` 且确认显存够
- 对照实验加环境变量前缀：`DARWIN_CENTER_THRESH=999` 关中心候选，默认 0.06 开

### 调参（per-task YAML，不共用）

配置在 `darwin/skills/configs/ik_servo.<suite>_<idx>.yaml`，脚本改参：

```bash
PYTHONPATH=/home/lifd/Public/darwin-bot /home/lifd/anaconda3/envs/darwin/bin/python - <<'PY'
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_goal_2")
s.update_params({"carry_vcap": 0.12})   # 只改这个任务的
s.save()
PY
```

常用参数速查：
- `carry_vcap`：carry 水平巡航速度上限。move_timeout 且物体卡在目标附近 → 提到 0.10~0.15（跨距 36cm 用 0.10，48cm+ 用 0.15）
- `timeout_scale`：所有 skill timeout 缩放（被 clip 到 3.0）
- `jit`：抓取点逐 attempt 抖动幅度（偏心试探用，扁平物可加大）
- `k_descend`：下降伺服增益（被反思改乱后恢复 10.0 或按成功快照恢复）
- `resume_rollback_steps`：resume 回退物理步数（默认 10）

恢复被反思改乱的参数：从 YAML `history` 里找末次 `success=True` 记录的 `cfg` 快照，`update_params` 写回（本文写作时 13 个任务已全部恢复过一轮）。

### 排查失败时看哪里

1. 终端/日志：`grep -E "✘|attempt .* 结果|skill_end" /tmp/xxx.log`——先看 `fail_phase`（grip_failed / timeout / place_failed / goal_not_reached）
2. Episode 明细：`logs/episodes/<task>_<ts>.jsonl`，抓 `attempt_start`（grasp_pt/body）、`skill_end`（end/tcp/reason/接触力 F）、`sample`（TCP 轨迹）
3. 几何判据：接触力 F=0 → 夹空（候选点偏心或 descend 没到位）；carry 卡住 → 看物体 z 是否低于障碍顶（撞墙）；place 跑偏 → 查 support_point/object_bounds center
4. BDDL 真值：以日志里 `condition satisfied` 为准，不要用 `_place_telemetry` 的 z_delta 直接推断 On 判据（mug/plate 的 body frame 参考面不同）

## 四、本次会话已落盘的改动（防重复劳动）

- `runner_dynamic.py`：实心物体中心候选置顶（厚度≥`DARWIN_CENTER_THRESH` 默认 0.06，可 env 关闭）；Fix7 多 chunk body0_arg 传 None（place 不再跑偏 23cm）
- `agent_learner.py` / `ipc_learn.py`：`--interactive` 交互模式 + `--max-resume`
- YAML 已改：`libero_object_9`（carry_vcap 0.10）、`libero_object_7`/`libero_object_4`（0.15）、`libero_object_1`（恢复定型参数）、13 个 spatial/goal 任务恢复末次成功快照
- object:9 六连过、object:7/:4 一次过、object:1 恢复过——object 套件在当前代码下确认稳定
- `scripts/calibrate_from_logs.py`（新增，只读）：从 logs/episodes 遥测反推参数推荐（`--task` 过滤、`--out` 落盘）。已跑全量，报告在 `logs/calibration_report.md`：多数 libero_10 任务 descend 失败时 TCP 停在目标上方 0.05~0.18m，建议 contact_stop_band 提到 0.06~0.12；carry 超时分"卡住/慢"两类（spatial:2/9 失败 carry 是撞墙型，提 vcap 无效）
- `darwin/agents/reflection.py` + `darwin/ipc/agent_learner.py`：adapt_cfg 新增 `anchor`（末次成功快照）漂移裁剪——单次反思任一参数只能落在 anchor ±50% 邻域（或 _DRIFT_FLOOR 下限）内，跨档学习须经成功逐档 ratchet，根治 streak 加速把参数改乱；runner_dynamic 不传 anchor，行为不变
- 已按标定报告写入 contact_stop_band（备份在 /tmp/darwin_configs_bak_20260923，回滚 = 拷回）：spatial_4→0.066、spatial_6→0.108、goal_3/4→0.12、goal_9→0.118。注意 goal_5 的 descend 残余 gap ≤0.042（TCP 其实到位了），它的 lift_no_grip 不是 band 问题；spatial_2 日志里无 descend 失败，均保持 0.05 未动
- 09-23 复测结果与归因见文首更新记录；锚定裁剪工作正常（spatial_4 的 k_descend 被限制在快照邻域内），但 goal:8/9 暴露 place 规则冲突（telemetry xy 阈值两侧规则方向相反，resume 循环内振荡）——**已修**：reflection._xy_side ±8mm 滞环 + agent_learner._last_place_dir 提供上次判定方向，冒烟验证往返消失；streak≥4 回退与快照恢复均豁免 contact_stop_band（标定值不被冲掉）
