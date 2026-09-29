"""自动生成技能: drawer_place（skill_forge 抽象，成功轨迹重放模板）。"""
def run(env=None, registry=None):
    """由成功轨迹抽象而来：按验证过的动作序列重放（坐标为该次成功观测值）。"""
    result = registry.execute('home', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('move_above', env=env, **{'point': [0.6000000000000001, 0.0, 0.46799999999999997], 'hover': 0.15, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('descend', env=env, **{'point': [0.6000000000000001, 0.0, 0.46799999999999997], 'actor': 'agent0', 'grip_site': '0_grip_site', 'k': 1.5})
    result = registry.execute('close_gripper', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('move_to', env=env, **{'target': [0.4600000000000001, 0.0, 0.47], 'gripper': -1.0, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('open_gripper', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('move_above', env=env, **{'point': [0.3844689263735242, -0.14782745159351088, 0.44], 'hover': 0.15, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('descend', env=env, **{'point': [0.3844689263735242, -0.14782745159351088, 0.44], 'body': 'green_block', 'actor': 'agent0', 'grip_site': '0_grip_site', 'k': 1.5})
    result = registry.execute('close_gripper', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('lift', env=env, **{'height': 0.55, 'body': 'green_block', 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('move_to_xy_top', env=env, **{'target': [0.59, 0.0, 0.478], 'height': 0.55, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('place', env=env, **{'goal': [0.59, 0.0, 0.478], 'body': 'green_block', 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('open_gripper', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    return result
