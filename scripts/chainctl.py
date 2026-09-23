#!/usr/bin/env python3
"""chainctl：调用链方法注册表的增删改查 CLI（给人和 Codex/Claude Code 用）。

用法：
    python scripts/chainctl.py list
    python scripts/chainctl.py show open_drawer
    python scripts/chainctl.py validate path/to/method.yaml
    python scripts/chainctl.py add path/to/method.yaml [--before open_drawer]
    python scripts/chainctl.py remove <name>      # 仅 YAML 方法；内置方法请用同名 YAML 覆盖
    python scripts/chainctl.py disable <name>     # YAML 文件改名 .disabled（重启进程生效）
    python scripts/chainctl.py enable  <name>

动态方法目录：darwin/agents/methods/*.yaml（examples/ 不自动加载）。
注意：注册表在进程启动时扫描目录，add/remove 后新启动的 runner 进程生效。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _yaml_path_of(method) -> Path:
    src = getattr(method, "source", "")
    if not src.startswith("yaml:"):
        raise ValueError(f"{method.name} 是内置 Python 方法，不能直接删/禁用；"
                         f"请用同名 YAML 方法覆盖（add），或改 darwin/agents/runner_dynamic.py")
    return Path(src.split("yaml:", 1)[1])


def cmd_list(args) -> int:
    from darwin.agents.chain_registry import default_registry
    reg = default_registry()
    print(f"{'#':>2}  {'name':15s} {'match':28s} {'source'}")
    print("-" * 80)
    for m in reg.list_methods():
        match = m["kind"] + (f"/{m['flavor']}" if m["flavor"] else "")
        src = m["source"].replace("yaml:", "")
        flag = "" if src == "builtin" else "  <- 动态"
        print(f"{m['order']:>2}  {m['name']:15s} {match:28s} {src}{flag}")
    print("-" * 80)
    print(f"{len(reg.names())} 个方法；select 按上表顺序匹配第一个 can_achieve 的方法")
    return 0


def cmd_show(args) -> int:
    import yaml
    from darwin.agents.chain_registry import default_registry
    m = default_registry().get(args.name)
    if isinstance(getattr(m, "spec", None), dict):
        print(yaml.safe_dump(m.to_dict(), allow_unicode=True, sort_keys=False))
    else:
        print(f"name: {m.name}  (builtin Python: {type(m).__module__}.{type(m).__name__})")
        print("内置方法不可直接查看 YAML；其步骤定义见 runner_dynamic.py 对应 Method 类")
    return 0


def cmd_validate(args) -> int:
    from darwin.agents.chain_registry import DeclarativeMethod, TemplateError
    import yaml
    try:
        spec = yaml.safe_load(Path(args.path).read_text(encoding="utf-8"))
        m = DeclarativeMethod(spec, source=f"yaml:{args.path}")
    except (TemplateError, KeyError, yaml.YAMLError) as e:
        print(f"[INVALID] {args.path}: {e}")
        return 1
    print(f"[OK] {args.path}: name={m.name} match.kind={m.match_kind} "
          f"flavor={m.match_flavor} steps={len(m.raw_steps)}")
    for i, s in enumerate(m.raw_steps):
        print(f"  {i}. {s['skill']}  {s.get('params', {})}")
    return 0


def cmd_add(args) -> int:
    from darwin.agents.chain_registry import METHODS_DIR, DeclarativeMethod, TemplateError
    import yaml
    src = Path(args.path).resolve()
    if not src.is_file():
        print(f"[ERR] 文件不存在: {src}")
        return 1
    try:
        spec = yaml.safe_load(src.read_text(encoding="utf-8"))
        m = DeclarativeMethod(spec, source=f"yaml:{src}")
    except (TemplateError, KeyError, yaml.YAMLError) as e:
        print(f"[INVALID] {e}")
        return 1
    METHODS_DIR.mkdir(parents=True, exist_ok=True)
    dst = METHODS_DIR / f"{m.name}.yaml"
    if dst.exists() and not args.force:
        print(f"[ERR] {dst.name} 已存在；加 --force 覆盖")
        return 1
    shutil.copyfile(src, dst)
    # 定位锚点提示（实际顺序由文件加载顺序 + register 默认追加决定；
    # 需要精确 before/after 时在 YAML 同级暂不支持，先复制再用 list 核对，
    # 或在 runner 代码里显式 register_yaml_file(before=...)）
    print(f"[OK] 已安装 {m.name} -> {dst}")
    if args.before:
        print(f"[NOTE] --before {args.before}：默认按文件名字母序加载，"
              f"如需严格插在该方法之前，请在测试代码中用 "
              f"registry.register_yaml_file(path, before='{args.before}')")
    print("[NEXT] 新进程生效；验证: python scripts/probe_dynamic_runner.py")
    return 0


def cmd_remove(args) -> int:
    from darwin.agents.chain_registry import default_registry
    reg = default_registry()
    try:
        m = reg.get(args.name)
        path = _yaml_path_of(m)
    except (KeyError, ValueError) as e:
        print(f"[ERR] {e}")
        return 1
    path.unlink()
    print(f"[OK] 已删除 {path}（新进程生效）")
    return 0


def _toggle(name: str, disable: bool) -> int:
    # 直接按文件名切换（禁用后新进程中该名回退 builtin，不能依赖注册表 source）
    from darwin.agents.chain_registry import METHODS_DIR
    on = METHODS_DIR / f"{name}.yaml"
    off = METHODS_DIR / f"{name}.yaml.disabled"
    if disable:
        if not on.exists():
            print(f"[ERR] 未找到活动方法文件: {on}（内置方法无需禁用，用同名 YAML 覆盖即可）")
            return 1
        on.rename(off)
        print(f"[OK] 已禁用 -> {off.name}（新进程生效，该名回退内置 Python 方法）")
    else:
        if not off.exists():
            print(f"[ERR] 未找到禁用文件: {off}")
            return 1
        off.rename(on)
        print(f"[OK] 已启用 -> {on.name}（新进程生效）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sp = sub.add_parser("show"); sp.add_argument("name")
    sp = sub.add_parser("validate"); sp.add_argument("path")
    sp = sub.add_parser("add"); sp.add_argument("path")
    sp.add_argument("--before", default=None)
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("remove"); sp.add_argument("name")
    sp = sub.add_parser("disable"); sp.add_argument("name")
    sp = sub.add_parser("enable"); sp.add_argument("name")
    args = ap.parse_args()
    return {
        "list": cmd_list, "show": cmd_show, "validate": cmd_validate,
        "add": cmd_add, "remove": cmd_remove,
        "disable": lambda a: _toggle(a.name, True),
        "enable": lambda a: _toggle(a.name, False),
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
