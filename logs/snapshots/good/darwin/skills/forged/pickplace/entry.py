"""自动生成技能: pickplace（skill_forge 抽象，成功轨迹重放模板）。"""
def run(env=None, registry=None):
    """由成功轨迹抽象而来：按验证过的动作序列重放（坐标为该次成功观测值）。"""
    result = registry.execute('home', env=env, **{'actor': 'agent1', 'grip_site': '1_grip_site'})
    result = registry.execute('move_above', env=env, **{'point': [0.3583053853017242, -0.09938862197180531, 0.46], 'hover': 0.12, 'actor': 'agent1', 'grip_site': '1_grip_site'})
    result = registry.execute('descend', env=env, **{'point': [0.3583053853017242, -0.09938862197180531, 0.46], 'body': 'green_block', 'actor': 'agent1', 'grip_site': '1_grip_site', 'k': 2.5})
    result = registry.execute('close_gripper', env=env, **{'actor': 'agent1', 'grip_site': '1_grip_site'})
    result = registry.execute('lift', env=env, **{'height': 0.55, 'body': 'green_block', 'actor': 'agent1', 'grip_site': '1_grip_site'})
    result = registry.execute('move_to_xy_top', env=env, **{'target': [0.495205897286759, -0.08824229243386675], 'height': 0.7416447849469858, 'actor': 'agent1', 'grip_site': '1_grip_site'})
    result = registry.execute('place', env=env, **{'goal': [0.495205897286759, -0.08824229243386675, 0.6216447849469858], 'body': 'green_block', 'actor': 'agent1', 'grip_site': '1_grip_site'})
    result = registry.execute('open_gripper', env=env, **{'actor': 'agent1', 'grip_site': '1_grip_site'})
    return result
