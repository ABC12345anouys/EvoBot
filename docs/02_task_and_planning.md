# 02 · 任务定义与规划

> 代码位置：`darwin/agents/task_spec.py`、`darwin/agents/libero_planner.py`、`darwin/agents/libero_runner.py`、`scripts/gen_task_specs.py`

这个项目用「Agent + 可复用技能库」的方式跑 LIBERO 机器人操作基准。规划层负责把一份任务定义变成**有序子目标**，再把每个子目标变成**技能序列**，交给技能库去执行。

LIBERO 每个任务都自带一份官方任务定义文件，其中写清了「最后要达到什么状态」，例如"碗在盘子上""抽屉是打开的""炉灶是点着的"。规划层不靠模型去猜这些目标，而是把这份定义解析成结构化数据，再决定执行顺序。

## 一、在整体流程中的位置

```
任务定义文件（LIBERO 官方 .bddl）
   │  task_spec.build_formal()      物化成结构化任务定义
   ▼
结构化任务定义（谓词表 + 候选前置 + 词表 + 参考序）
   │  scripts/gen_task_specs.py     离线调用 LLM 做一次「谓词级分解」
   ▼
校验 task_spec.validate_order() → 冻结成 YAML
   │  产物：darwin/skills/configs/task_specs/<suite>_<idx>.yaml
   ▼
运行期：task_spec.load_or_parse() 只读该 YAML（不调 LLM、零 token）
   │  libero_planner.plan_subgoal() 子目标 → 技能序列
   ▼
技能库执行（grasp / place_at / articulate / toggle / push_to …）
   │  失败 → 返回失败机制 → libero_planner.replan_hint() → 换参数或换候选
   ▼
libero_runner 的 attempt 闭环（重试、进度记录、断点续跑）
```

关键取舍是：**LLM 只在离线阶段出场一次，产物入库冻结；运行期完全确定，也不花 token。** 这样同一任务每次跑出来的计划完全相同，出现问题时能分清「是计划错了还是执行错了」。

## 二、完整流程

1. **读取任务定义。** `task_spec.py` 只解析任务定义里的目标段（`(:goal ...)`），把 `On` / `In` / `Open` / `Close` / `Turnon` / `Turnoff` 这些谓词（描述物体状态的小短句）连同参数一起取出来；大小写按官方文件的实际写法处理。

2. **物化成结构化任务定义**，由 `task_spec.py` 的 `build_formal()` 完成。它给每条谓词分配一个稳定编号（`p0`、`p1`…），列出允许插入的前置项、涉及的对象与目标名、夹具属性（是否能开合、是否是炉面），并附上一份确定性参考顺序。

3. **离线调用 LLM 做分解。** `scripts/gen_task_specs.py` 把上面这份结构化定义发给模型，让它只做一件事：排列执行顺序，并决定要不要插入「先开抽屉」这类前置动作。

4. **校验模型输出。** `task_spec.validate_order()` 是一个纯函数（给定输入必定给出相同结果，方便写单元测试），逐项检查模型回复是否合法。

5. **冻结成文件。** 通过校验的结果写入 `darwin/skills/configs/task_specs/<suite>_<idx>.yaml` 并提交进仓库。文件名里的 `suite` 是任务集（`libero_spatial` / `libero_object` / `libero_goal`），`idx` 是任务序号。

6. **运行期只读文件。** `task_spec.load_or_parse()` 直接读这份 YAML，**不再调用 LLM**。文件不存在或格式有问题时，退回确定性解析。

7. **子目标变成技能。** 运行时 `libero_runner.run_task()` 拿到子目标列表，逐个交给 `libero_planner.plan_subgoal()` 映射成技能序列并执行。

## 三、结构化任务定义里有什么

`build_formal()` 产出的是模型和校验器共同依赖的一份自描述数据，主要字段：

| 字段 | 含义 |
|---|---|
| `items` | 全部子目标谓词，每项带稳定编号 `id`（`p0`…）。**不可增删**，是任务的全部要求 |
| `implicit_candidates` | 允许插入的隐式前置项，例如放东西进抽屉前先 `Open`；编号形如 `open:<夹具名>` |
| `vocabulary` | 出现的对象名、目标名，以及每个夹具是否能开合、是否是炉面 |
| `skills` | 谓词类别与技能类别的对应说明，只做声明，不参与执行 |
| `unsupported` | 已识别但当前不执行的谓词（如 `NextTo`） |
| `reference_order` | 确定性排序给出的参考顺序，用于和模型结果做比对、也作为备用路径 |

## 四、冻结后的任务定义文件实例

下面这份是「开抽屉并把碗放进去」这个任务的真实产物（为便于阅读省略了部分重复字段）：

```yaml
language: open the top drawer and put the bowl inside
source: llm                     # llm | deterministic，标明这份定义是怎么来的

formal:                         # 结构化任务定义（模型与校验器都读这一份）
  items:
    - {id: p0, predicate: In, kind: place,
       object: akita_black_bowl_1, target: wooden_cabinet_1_top_region,
       implicit: false}
  implicit_candidates:          # 允许插入的前置项
    - {id: 'open:wooden_cabinet_1', predicate: Open, kind: articulate,
       target: wooden_cabinet_1_top_region, implicit: true}
  vocabulary:
    objects: [akita_black_bowl_1]
    targets: [wooden_cabinet_1_top_region]
    fixtures:
      wooden_cabinet_1: {articulated: true, cook: false}
  reference_order:              # 确定性排序给出的参考顺序
    - {predicate: Open, kind: articulate, target: wooden_cabinet_1_top_region, implicit: true}
    - {predicate: In, kind: place, object: akita_black_bowl_1, target: wooden_cabinet_1_top_region, implicit: false}

llm:                            # 这次分解的原始记录，便于回溯
  ok: true
  model: deepseek-v4-pro
  order: [p0]
  add: ['open:wooden_cabinet_1']
  rationale: 碗放入柜子顶部抽屉前必须先打开该抽屉。

subgoals:                       # 最终产物：执行器直接消费
  - {predicate: Open, kind: articulate, target: wooden_cabinet_1_top_region, implicit: true}
  - {predicate: In, kind: place, object: akita_black_bowl_1, target: wooden_cabinet_1_top_region, implicit: false}
```

几个要点：

- `subgoals` 才是执行器看到的东西，`formal` 和 `llm` 是留档：前者说明依据是什么，后者说明这次顺序是谁、按什么理由排的。
- `implicit: true` 表示这一步不是任务定义里显式写的要求，而是为完成显式要求补上的前置动作。
- `source` 为 `deterministic` 时，表示这份文件是确定性解析的结果，不是模型分解的产物。

## 五、LLM 的权限边界

模型能做的只有两件事：

1. **排列顺序。** 输出 `order`，必须恰好是 `items` 里全部编号的一个排列，不重不漏。
2. **挑前置项。** 输出 `add`，只能从 `implicit_candidates` 里原样照抄，不需要就留空。

模型不能做的：

- 不能新增、删除或改写任何谓词名、对象名、目标名；
- 不能自己选择用哪个技能；
- 不能给技能参数。

`validate_order()` 会逐条检查编号是否已知、是否重复、是否漏项、`add` 是否只取自候选项。任何一项不通过，或者模型不可用（缺少密钥、网络异常、返回内容解析不了），`decompose_spec()` 就退回 `parse_bddl_spec()` 的确定性解析，并把原因写进文件里的 `llm.error` 字段。**生成脚本因此永远能产出一份可用定义，失败也有迹可查。**

## 六、确定性排序：备用路径与校验基准

确定性排序由 `task_spec.py` 的 `_order_subgoals()` 实现，做法是「依赖关系 + 原文顺序」的稳定排序（拓扑排序，即先满足依赖、再照作者书写顺序排列）：

- **依赖边只针对放置类子目标建立**：同一个夹具上，`Open` 排在「把东西放进去」之前，「放进去」排在 `Close` 之前；如果放的目标是炉面，`Turnon` 也排在它之前。
- **隐式前置自动补齐**：需要放进抽屉、柜子、微波炉，而任务定义里没写 `Open` 时，自动补一个 `implicit=True` 的 `Open`，并让它排在对应的放置动作之前。
- **排序规则**：每一轮从当前没有未满足依赖的节点里，取原文序号最小的那个，因此尽量保持任务定义原本的书写顺序。
- **不执行的谓词**（如 `NextTo`）不参与排序和依赖，只被记录，排在最后。

它有三个用途：模型不可用时的**备用路径**；模型输出的**校验基准**；以及写进 YAML 的 `reference_order`，用来对比「模型排的顺序和确定性顺序是否一致」。

## 七、运行时：子目标 → 技能序列

`libero_planner.plan_subgoal()` 只做「谓词类别 → 技能」的映射，不针对具体任务写特判：

| 子目标类别 | 技能序列 |
|---|---|
| `place`（`On` / `In`） | `grasp(物体)` → `place_at(物体, 目标, 谓词)` |
| `articulate`（`Open` / `Close`） | `articulate(目标, 方向)` |
| `toggle`（`Turnon` / `Turnoff`） | `toggle(目标, 方向)` |
| `unsupported`（如 `NextTo`） | 空序列，执行器记录后跳过 |

执行前会先用 `subgoal_satisfied()` 检查子目标是否已经满足（判定标准与 LIBERO 自己的成功判定保持一致），已满足的直接跳过，所以重试时从当前世界状态继续，而不是从头重跑。

当某个放置类子目标反复抓取失败，并且失败原因明确是「没有可用的抓取候选」（属于几何条件限制，而不是执行失误），`plan_subgoal_alternative()` 会改用物理上的另一种做法：`push_to(obj, target, predicate)`，也就是推或拨物体到位。

## 八、失败处理

技能失败时返回的不是一个布尔值，而是**失败原因（机制）**，例如 `grasp_miss`（没夹住）、`contact_blocked`（被挡住）、`ik_unreachable`（够不到）、`no_candidate`（没有候选）。执行器据此重试，`libero_planner.replan_hint()` 把机制映射成下一次要用的参数：

- `grasp_miss` / `ik_unreachable`：换下一个几何抓取候选；
- 其他机制：认为世界已经被刚才的动作改变了，用新的感知结果重跑同一个子目标。

重试在同一轨迹内进行，子目标前面失败次数越多，换的候选越靠后。如果某次替换成 `push_to` 后仍然失败，就直接收尾并把真实失败原因上报，不再空转剩余重试轮次。

## 九、重新生成任务定义

任务定义文件是冻结的快照。任务定义文件本身变了，需要重新生成：

```bash
# 全部 30 个任务（libero_spatial / libero_object / libero_goal 各 10 个）
python scripts/gen_task_specs.py

# 指定任务
python scripts/gen_task_specs.py --tasks libero_goal:7

# 不调模型，只写确定性规格
python scripts/gen_task_specs.py --no-llm

# 覆盖已存在的文件
python scripts/gen_task_specs.py --model deepseek-v4-pro --force
```

脚本参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--suites` | `libero_spatial,libero_object,libero_goal` | 要生成哪些任务集 |
| `--tasks` | 空 | 显式任务列表（`suite:idx`，逗号分隔），优先于 `--suites` |
| `--model` | 环境变量 `OPENAI_MODEL`，否则 `deepseek-v4-pro` | 使用的模型 |
| `--base-url` | 环境变量 `OPENAI_BASE_URL`，否则 `https://api.deepseek.com` | 模型接口地址 |
| `--api-key-env` | `DEEPSEEK_API_KEY` | 从哪个环境变量读密钥 |
| `--temperature` | `0.0` | 采样温度；分解任务取 0，尽量稳定 |
| `--no-llm` | 关 | 不调模型，只写确定性规格 |
| `--force` | 关 | 覆盖已存在的 YAML |

脚本逐条打印每个任务的来源（`llm` 还是 `deterministic`）、是否与参考顺序不同，以及最终的子目标序列，末尾汇总成功、不一致和回退的数量。

## 十、关键参数

运行期重试与规划相关的参数（以代码默认值为准）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--max-attempts` | `8` | 单个任务最多重试多少次，每次从同一个固定初始状态重来 |
| `--attempt-steps` | `7000` | 单条轨迹的仿真步数上限，确定性截断，是主要的时间控制方式 |
| `--attempt-timeout` | `300` | 单条轨迹的挂钟秒数上限，只在进程不走步却卡住时才可能先触发 |
| `run_episode` 的 `max_rounds` | `4` | 单条轨迹内部最多做几轮重规划 |

模型调用侧还有两个固定取值：分解时 `temperature=0.0`、`max_tokens=800`。
