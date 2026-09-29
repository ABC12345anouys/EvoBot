"""自动生成技能: any（skill_forge 抽象，成功轨迹重放模板）。"""
def run(env=None, registry=None):
    """由成功轨迹抽象而来：按验证过的动作序列重放（坐标为该次成功观测值）。"""
    result = registry.execute('home', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('move_above', env=env, **{'point': [0.3550186834520122, 0.01532807706724229, 0.46], 'hover': 0.12, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('descend', env=env, **{'point': [0.3550186834520122, 0.01532807706724229, 0.46], 'body': 'red_block', 'k': 2.0, 'stop_above': 0.0, 'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('close_gripper', env=env, **{'actor': 'agent0', 'grip_site': '0_grip_site'})
    result = registry.execute('lift', env=env, **{'height': 0.52, 'body': 'red_block', 'actor': 'agent0', 'grip_site': '0_grip_site'})
    return result
