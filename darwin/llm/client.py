"""LLM 客户端：OpenAI 兼容协议（支持火山方舟 Ark / OpenAI / 任意兼容端点）。

配置全部来自环境变量，无硬编码密钥：
- ARK_API_KEY / ARK_BASE_URL(默认 https://ark.cn-beijing.volces.com/api/v3) / ARK_MODEL
- 或 OPENAI_API_KEY / OPENAI_BASE_URL(默认 https://api.openai.com/v1) / OPENAI_MODEL

未配置时 available() 返回 False，Planner 自动回退 OfflinePlanner（链路仍可端到端验证）。
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


class LLMClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: int = 120) -> None:
        if api_key is None and os.environ.get("ARK_API_KEY"):
            base_url = base_url or os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            api_key = os.environ["ARK_API_KEY"]
            model = model or os.environ.get("ARK_MODEL")
        else:
            base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
            model = model or os.environ.get("OPENAI_MODEL")
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2,
             max_tokens: int = 1024) -> str:
        """OpenAI 兼容 /chat/completions 调用，返回首条回复文本。失败抛异常。"""
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        # 累计 token 用量（批学三重预算用；无 usage 字段时按字符粗估）
        try:
            u = data.get("usage") or {}
            n = int(u.get("total_tokens") or 0)
            if n <= 0:
                n = (len(json.dumps(messages, ensure_ascii=False))
                     + len(data["choices"][0]["message"].get("content") or "")) // 4
            _USAGE["total"] += n
        except Exception:
            pass
        return data["choices"][0]["message"]["content"]


# 进程级累计用量（agent_learner 结束时打印供批学器读取）
_USAGE = {"total": 0}


def total_tokens_used() -> int:
    return int(_USAGE["total"])
