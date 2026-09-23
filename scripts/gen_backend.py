#!/usr/bin/env python
"""LLM 生成器骨架:为新机型生成 RobotBackend 适配器(Profile YAML + 可选 Backend 子类)。

设计:**开发时工具**(非运行时)。读 prompt 模板 + 机型 SDK/URDF 内容,组装成 LLM 输入,
调 darwin.llm.client 生成两文件,人 review 后落盘。运行时纯 Python 多态,不依赖 LLM。

MVP 状态(用户选"留 prompt 模板不做实链路"):
- ✅ 参数解析 + prompt 模板加载 + SDK 内容拼装
- ✅ 打印组装好的 LLM 输入(可手动喂给任意 LLM)
- ✅ 若 LLM 可用(ARK_API_KEY/ARK_MODEL 或 OPENAI_* 已配置)则自动调用并落盘
- ⚠️  生成的代码必须跑"自检清单"(见 llm_prompts/gen_backend.md)后人 review

用法:
    python scripts/gen_backend.py --name panda --sdk-path /path/to/panda_sdk [--docs-url URL]
    python scripts/gen_backend.py --name rm75 --docs-url https://...  # 无本地 SDK,靠文档
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "scripts" / "llm_prompts" / "gen_backend.md"
PROFILES_DIR = REPO_ROOT / "darwin" / "assets" / "profiles"
BACKEND_DIR = REPO_ROOT / "darwin" / "robot"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def collect_sdk_context(sdk_path: str, docs_url: str) -> str:
    """收集机型 SDK/URDF 内容作为 LLM 上下文。"""
    parts = []
    if sdk_path:
        sp = Path(sdk_path)
        if not sp.exists():
            print(f"[warn] sdk-path 不存在: {sp}", file=sys.stderr)
        else:
            # 收集 URDF / 关键 Python 源(限制大小避免 prompt 爆炸)
            urdfs = list(sp.rglob("*.urdf"))[:5]
            for u in urdfs:
                try:
                    text = u.read_text(encoding="utf-8", errors="ignore")
                    if len(text) < 200000:
                        parts.append(f"=== URDF: {u.name} ===\n{text}\n")
                except Exception:
                    pass
            # SDK Python 源(找 joint/link/eef 关键词)
            for py in sp.rglob("*.py"):
                try:
                    text = py.read_text(encoding="utf-8", errors="ignore")
                    if any(k in text.lower() for k in ("joint", "link", "eef", "gripper", "urdf")):
                        if len(text) < 50000:
                            parts.append(f"=== {py.relative_to(sp)} ===\n{text[:5000]}\n")
                except Exception:
                    pass
                if len(parts) > 30:
                    break
    if docs_url:
        parts.append(f"=== 文档 URL(需 LLM 自行 fetch)==\n{docs_url}\n")
    if not parts:
        parts.append("[warn] 无 SDK/文档上下文,LLM 需完全凭训练知识生成,务必 review")
    return "\n".join(parts)


def assemble_prompt(name: str, sdk_ctx: str) -> str:
    prompt = load_prompt()
    return (
        f"{prompt}\n\n"
        f"---\n## 本次任务\n\n"
        f"机型名: `{name}`\n\n"
        f"## SDK / 文档上下文\n\n{sdk_ctx}\n\n"
        f"## 开始生成\n\n"
        f"按 prompt 指示输出 `<name>.yaml`(name={name})和(若需)`{name}_backend.py`。"
    )


def call_llm(prompt: str) -> str:
    """调 darwin.llm.client 生成。未配置环境变量时抛,由调用方处理。"""
    sys.path.insert(0, str(REPO_ROOT))
    from darwin.llm.client import LLMClient
    client = LLMClient()
    if not client.available():
        raise RuntimeError("LLM 未配置(需 ARK_API_KEY/ARK_MODEL 或 OPENAI_*);用 --print 自取 prompt")
    messages = [{"role": "user", "content": prompt}]
    return client.chat(messages, temperature=0.1, max_tokens=4096)


def main():
    ap = argparse.ArgumentParser(description="为新机型生成 RobotBackend 适配器")
    ap.add_argument("--name", required=True, help="机型 profile 名(如 panda / rm75)")
    ap.add_argument("--sdk-path", default="", help="机型 SDK/URDF 本地目录")
    ap.add_argument("--docs-url", default="", help="机型文档 URL(LLM 自行 fetch)")
    ap.add_argument("--print", action="store_true", help="只打印组装好的 prompt,不调 LLM")
    ap.add_argument("--write", action="store_true", help="调 LLM 并落盘生成的文件(需 review)")
    args = ap.parse_args()

    sdk_ctx = collect_sdk_context(args.sdk_path, args.docs_url)
    prompt = assemble_prompt(args.name, sdk_ctx)

    if args.print or not args.write:
        print("=" * 70)
        print("组装好的 LLM 输入(可手动喂给任意 LLM):")
        print("=" * 70)
        print(prompt)
        if not args.write:
            print("\n[--write 调 LLM 自动生成并落盘;--print 只输出 prompt]")
            return

    try:
        out = call_llm(prompt)
    except RuntimeError as e:
        print(f"[info] {e}", file=sys.stderr)
        print("[info] 改用 --print 自取 prompt 手动生成,或配置 LLM 环境变量后 --write。")
        return

    out_path = REPO_ROOT / f"gen_backend_{args.name}_output.md"
    out_path.write_text(out, encoding="utf-8")
    print(f"[ok] LLM 输出写到 {out_path}")
    print(f"[next] review 后把 yaml 段落存到 {PROFILES_DIR}/{args.name}.yaml,"
          f"把 python 段落(若有)存到 {BACKEND_DIR}/{args.name}_backend.py")
    print(f"[verify] 跑自检清单(见 scripts/llm_prompts/gen_backend.md)")


if __name__ == "__main__":
    main()
