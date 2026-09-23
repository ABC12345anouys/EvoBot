# 06 · LLM 集成设计

> 对应代码：`darwin/llm/client.py`、`darwin/agents/agent.py`（decide 方法）

## 1. 设计目标

LLM 集成层为 Agent 提供**可选的决策能力**。设计目标：

1. **零硬编码密钥**：所有配置来自环境变量，支持火山方舟 Ark 和 OpenAI 兼容端点。
2. **优雅降级**：未配置 LLM 时 `available()` 返回 False，Agent 自动回退规则规划，链路仍可端到端运行。
3. **function calling 风格**：LLM 通过 `<action>` / `<params>` 标签选择技能并传参，与 SkillSpec 元数据对接。

## 2. LLMClient

### 2.1 配置优先级

```python
class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=120):
        if api_key is None and os.environ.get("ARK_API_KEY"):
            # 火山方舟 Ark
            base_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            api_key = os.environ["ARK_API_KEY"]
            model = os.environ.get("ARK_MODEL")
        else:
            # OpenAI 兼容
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = os.environ.get("OPENAI_API_KEY")
            model = os.environ.get("OPENAI_MODEL")
```

优先级：构造参数 > `ARK_*` 环境变量 > `OPENAI_*` 环境变量。

### 2.2 可用性检测

```python
def available(self) -> bool:
    return bool(self.api_key and self.model)
```

只有 API Key 和模型名都配置了才可用。Agent 初始化时：

```python
self.llm = llm if llm is not None else (LLMClient() if LLMClient().available() else None)
```

### 2.3 chat 接口

```python
def chat(self, messages, temperature=0.2, max_tokens=1024) -> str:
    req = urllib.request.Request(
        f"{self.base_url}/chat/completions",
        data=json.dumps({"model": self.model, "messages": messages,
                         "temperature": temperature, "max_tokens": max_tokens}),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]
```

- 使用标准库 `urllib`，不依赖 `openai` / `httpx` 等第三方包
- 失败抛异常，由 Agent 捕获并回退规则规划

## 3. Agent 决策流程

### 3.1 三级回退

```python
def decide(self, task, feat, rag_ctx, plan=None):
    if plan is not None:
        return plan(feat, rag_ctx) if callable(plan) else plan   # 1. 外部传入
    if self.llm is None:
        return self._default_plan(task, feat, rag_ctx)            # 3. 规则兜底
    # 2. LLM 决策
    ...
```

### 3.2 LLM 系统提示词

```python
sys_prompt = (
    f"你是机械臂操作规划器。任务：{task}\n"
    f"物体特征：{json.dumps(feat)}\n"
    f"历史经验（RAG）：{json.dumps(rag_ctx['context'])[:2000]}\n"
    f"可用技能：\n{self._skill_specs_text()}\n\n"
    "每轮只输出一个动作，格式：\n"
    "<action>skill_name</action>\n<params>{\"参数名\": 值}</params>\n"
    f"完成后输出 <action>finish</action><params>{{\"success\": true/false}}</params>"
)
```

注入内容：
- **任务描述**：自然语言任务
- **物体特征**：shape / size / center
- **RAG 上下文**：`rag.retrieve()` 返回的相关经验（截断 2000 字符）
- **可用技能列表**：所有技能的 name + description + parameters

### 3.3 多轮对话

```python
messages = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"请规划完成 {task} 的动作序列。"}]
actions = []
for _ in range(self.max_turns):
    resp = self.llm.chat(messages)
    messages.append({"role": "assistant", "content": resp})
    act = _parse_action(resp)
    if act is None:
        messages.append({"role": "user", "content": "请严格按格式输出。"})
        continue
    if act["action"] == self.FINISH:
        break
    if act["action"] not in self.skills:
        messages.append({"role": "user", "content": f"技能 {act['action']} 不存在。"})
        continue
    actions.append(act)
    messages.append({"role": "user", "content": "继续下一个动作，或 finish。"})
return actions or self._default_plan(task, feat, rag_ctx)
```

- 最多 `max_turns`（默认 30）轮
- 解析失败 / 技能不存在 → 反馈给 LLM 重试
- 输出 `finish` → 结束
- LLM 全程无有效输出 → 回退规则规划

### 3.4 动作解析

```python
def _parse_action(response):
    m = re.search(r"<action>\s*(.*?)\s*</action>", response, re.DOTALL)
    if not m:
        return None
    name = m.group(1).strip()
    params = {}
    pm = re.search(r"<params>\s*(.*)\s*</params>", response, re.DOTALL)
    if pm:
        try:
            params = json.loads(pm.group(1).strip())
        except json.JSONDecodeError:
            # 容错：截取最外层 {}
            s = pm.group(1)
            lo, hi = s.find("{"), s.rfind("}")
            if lo != -1 and hi > lo:
                params = json.loads(s[lo:hi + 1])
    return {"action": name, "params": params}
```

## 4. 规则兜底规划

`_default_plan` 提供 pick / place 的规则动作序列：

```python
def _default_plan(self, task, feat, rag_ctx):
    cand = (rag_ctx.get("ranked_candidates") or [{}])[0]
    grasp_pt = center + rel_offset * size
    plan = [home, move_above(grasp_pt), descend(grasp_pt), close_gripper, lift]
    if "place" in task:
        plan += [move_to_xy_top(goal), place(goal), open_gripper]
    return plan
```

即使无 LLM，Agent 也能完成基本的 pick / pickplace 任务，保证进化循环可运行。

## 5. 与 RAG 的协同

LLM 决策前，RAG 经验已通过 `retrieve()` 注入 prompt：
- `best_success`：最佳成功抓取偏移（直接告诉 LLM"上次这里成功了"）
- `context`：语义相关的历史经验（成功/失败描述）
- `ranked_candidates`：UCB 重排后的候选点

LLM 可基于这些信息选择更优的抓取点和动作序列，减少盲目探索。

## 6. 无 LLM 模式的价值

即使完全不配置 LLM，EvoBot 仍可运行：
- `EpisodeRunner` 用 `_plan_for` 生成规则计划
- `ManipulationAgent.decide` 走 `_default_plan`
- RAG 记忆仍正常工作（候选过滤、偏移复用、避坑）
- `skill_forge` 仍可沉淀成功轨迹

这确保了**进化循环不依赖 LLM 可用性**，可在无网络/无 API Key 的环境下持续运行。
