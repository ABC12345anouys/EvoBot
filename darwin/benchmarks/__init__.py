"""darwin.benchmarks：任务分布定义（EvolutionLoop 的 benchmark 数据源）。

每个条目包含：
- env/robot/body/actor/grip_site/bimanual/mode：环境与任务参数
- task_desc：自然语言任务描述（供 LLM planner / RAG 语义检索）
- search_space：MAP-Elites 搜索空间（cfg 变量，会传给 agent_runner）
- feature_dimensions/feature_ranges：MAP-Elites 分箱维度
- fitness_key：适应度字段
"""
from __future__ import annotations

# MAP-Elites 公共配置：按 attempts × steps 分箱维护多样技能库
_FEATURES = {
    "feature_dimensions": ["attempts", "steps"],
    "feature_ranges": {"attempts": (0, 8), "steps": (0, 1500)},
    "fitness_key": "score",
}

COMMON_SEARCH_SPACE = {
    "hover": [0.10, 0.12, 0.15],          # 悬停高度
    "k_descend": [1.5, 2.0, 2.5],         # 下降 P 增益
    "lift_height": [0.50, 0.52, 0.55],    # 抬升目标高度
    "jit": [0.0, 0.005, 0.01],            # 抓取点抖动幅度
}

# insert 任务专用搜索空间：力控参数（CARTIMP 刚度/阻尼/螺旋半径）
INSERT_SEARCH_SPACE = {
    "hover": [0.10, 0.12, 0.15],
    "lift_height": [0.50, 0.52, 0.55],
    "stiffness": [60.0, 100.0, 150.0],    # CARTIMP 刚度（推进方向更软）
    "damping": [20.0, 40.0, 60.0],        # CARTIMP 阻尼
    "spiral_radius": [0.003, 0.004, 0.006],  # 螺旋搜索半径
}

BENCHMARKS = {
    "pickplace": {
        "env_id": "BimanualPickAndPlace-v0", "robot": "DualPandaPickAndPlace",
        "bimanual": True, "body": "green_block", "task_name": "pickplace",
        "actor": "agent1", "grip_site": "1_grip_site", "home_pos": None,
        "mode": "full",
        "bounds": {
            "agent0": {"max": [0.65, 0.65, 0.65], "min": [-0.10, -0.30, 0.02]},
            "agent1": {"max": [0.65, 0.40, 0.65], "min": [0.25, -0.10, 0.02]},
        },
        "task_desc": "pick up the green block and place it at the goal site (bimanual panda, dual arm)",
        "search_space": COMMON_SEARCH_SPACE, **_FEATURES,
    },
    "stack_grasp_red": {
        "env_id": "MultiCubeStack-v1", "robot": "DianaTripleStack",
        "bimanual": False, "body": "red_block", "task_name": "any",
        "actor": "agent0", "grip_site": "0_grip_site", "home_pos": [0.35, 0.0, 0.40],
        "mode": "grasp", "bounds": None,
        "task_desc": "grasp the red block on the table (single diana arm, lift = success)",
        "search_space": COMMON_SEARCH_SPACE, **_FEATURES,
    },
    "stack_grasp_blue": {
        "env_id": "MultiCubeStack-v1", "robot": "DianaTripleStack",
        "bimanual": False, "body": "blue_block", "task_name": "any",
        "actor": "agent0", "grip_site": "0_grip_site", "home_pos": [0.35, 0.0, 0.40],
        "mode": "grasp", "bounds": None,
        "task_desc": "grasp the blue block on the table (single diana arm, lift = success)",
        "search_space": COMMON_SEARCH_SPACE, **_FEATURES,
    },
    "peg_in_hole": {
        "env_id": "InsertEnv", "robot": "DianaTripleStack", "controller": "CARTIK",
        "bimanual": False, "body": "red_block", "task_name": "peg_in_hole",
        "actor": "agent0", "grip_site": "0_grip_site", "home_pos": [0.35, 0.0, 0.40],
        # 注：当前 robot.xml 中 red_block 是 4cm 立方体、red_goal 是空中目标点（无孔），
        # 实际为 cube 放置任务，故用 mode=full（transfer_cart + place）。
        # 若未来换成真实 peg/hole 模型，再切回 mode=insert 启用力控插装链。
        "mode": "full", "goal_site": "red_goal", "bounds": None,
        "task_desc": "pick red block and place it at red_goal site (diana + CARTIK)",
        "search_space": INSERT_SEARCH_SPACE, **_FEATURES,
    },
    "drawer_place": {
        "env_id": "DrawerBox-v1", "robot": "DianaDrawerCube",
        "bimanual": False, "body": "green_block", "task_name": "drawer_place",
        "actor": "agent0", "grip_site": "0_grip_site", "home_pos": None,
        "mode": "drawer",
        "drawer_site": "drawer", "drawer_goal_site": "drawer_goal",
        "cube_goal_site": "cube_goal", "bounds": None,
        "task_desc": "open the drawer by pulling the handle, then pick up the green block and place it inside the drawer (diana single arm, CARTIK)",
        "search_space": COMMON_SEARCH_SPACE, **_FEATURES,
    },
}


def get_benchmark(name: str) -> dict:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark: {name}, available: {list(BENCHMARKS)}")
    return BENCHMARKS[name]


def get_libero_benchmark(suite: str, task_idx: int) -> dict:
    """构造 LIBERO 任务 entry（动态工厂，任务集不进静态 BENCHMARKS 表）。

    body 取 BDDL 中声明的第一个对象的 mujoco body 名（<bddl_obj>_main，
    robosuite 命名约定），供感知链/logger 使用；回放方法本身不依赖抓取候选。
    env 由 LiberoEnvAdapter 在首次 get_env 时创建。
    """
    import re
    from ..envs.libero_adapter import _task_info, _ensure_libero_path
    from ..agents.libero_tasks import parse_bddl_goal
    from ..agents.task_spec import load_or_parse

    _ensure_libero_path()
    info = _task_info(suite, int(task_idx))
    bddl_path = info["bddl"]
    bddl_text = open(bddl_path, encoding="utf-8").read()
    spec = load_or_parse(suite, int(task_idx), bddl_path,
                         getattr(info["task"], "language", ""))
    seg = re.search(r"\(:objects(.*?)\)", bddl_text, re.S)
    obj_names: list = []
    if seg:
        for line in seg.group(1).splitlines():
            if "-" in line:
                obj_names.extend(line.split("-", 1)[0].split())
    goal = parse_bddl_goal(bddl_path)
    # IK 抓放链的主物体 = goal 谓词中的被抓物（BDDL 名，适配器负责解析
    # 装配根/几何子树）；无 goal 信息时退回 BDDL 首对象
    main_obj = (goal.get("object") if goal and goal["kind"] == "place"
                else (obj_names[0] if obj_names else ""))
    body = main_obj
    return {
        "env_id": f"libero:{suite}:{task_idx}", "robot": "libero_panda",
        "bimanual": False, "body": body,
        "task_name": f"{suite}_{int(task_idx):02d}",
        "actor": "agent0", "grip_site": "gripper0_grip_site", "home_pos": None,
        "mode": "libero", "bounds": None,
        "libero_suite": suite, "libero_task_idx": int(task_idx),
        "libero_language": info["task"].language,
        # BDDL 结构化目标（单谓词兼容字段）+ 全谓词有序形式化 spec
        "libero_goal": goal,
        "libero_spec": spec,
        "task_desc": f"[LIBERO/{suite}] {info['task'].language}",
        "search_space": COMMON_SEARCH_SPACE, **_FEATURES,
    }


__all__ = ["BENCHMARKS", "get_benchmark", "get_libero_benchmark"]
