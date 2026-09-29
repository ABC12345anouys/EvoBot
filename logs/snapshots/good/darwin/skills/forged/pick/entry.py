"""自动生成技能: pick（skill_forge 抽象，成功轨迹重放模板）。"""
def run(env=None, registry=None):
    """由成功轨迹抽象而来：按验证过的动作序列重放（坐标为该次成功观测值）。"""
    result = registry.execute('home', env=env, **{'actor': 'agent0'})
    return result
