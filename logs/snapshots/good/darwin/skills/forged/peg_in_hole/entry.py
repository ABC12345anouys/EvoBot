"""自动生成技能: peg_in_hole（skill_forge 抽象，成功轨迹重放模板）。"""
def run(env=None, registry=None):
    """由成功轨迹抽象而来：按验证过的动作序列重放（坐标为该次成功观测值）。"""
    result = registry.execute('home', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('pose_move_above', env=env, **{'point': [0.5, -0.1, 0.46], 'hover': 0.12, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('pose_descend', env=env, **{'point': [0.5, -0.1, 0.46], 'body': 'red_block', 'k': 2.0, 'stop_above': 0.0, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('pose_close_gripper', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('pose_lift', env=env, **{'height': 0.52, 'body': 'red_block', 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('pose_place', env=env, **{'goal': [0.4, 0.0, 0.44], 'body': 'red_block', 'tol': 0.05, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('pose_open_gripper', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    return result
