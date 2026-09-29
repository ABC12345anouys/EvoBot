"""自动生成技能: libero_goal_07（skill_forge 抽象，成功轨迹重放模板）。"""
def run(env=None, registry=None):
    """由成功轨迹抽象而来：按验证过的动作序列重放（坐标为该次成功观测值）。"""
    result = registry.execute('home', env=env, **{'timeout': 90, 'actor': 'agent0', 'grip_site': 'gripper0_grip_site'})
    result = registry.execute('articulate', env=env, **{'joint_name': 'flat_stove_1_button', 'target_qpos': 1.3, 'tol': 0.35, 'actor': 'agent0', 'grip_site': 'gripper0_grip_site'})
    return result
