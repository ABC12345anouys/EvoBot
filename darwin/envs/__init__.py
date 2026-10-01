"""darwin.envs：环境适配层（LIBERO / robosuite）。

适配器按需导入——`libero_adapter` 需要 LIBERO 在 `PYTHONPATH` 上，
因此不在包导入时一次性加载。
"""
