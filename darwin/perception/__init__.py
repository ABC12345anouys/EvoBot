"""darwin.perception：RoboDojo 3 相机 RGB-D 场景感知。

纯感知决策链（不依赖真值 object_states）：
    obs['vision'] ──rgbd_fusion.RgbdFusion.obs_to_cloud──▶ world 融合点云
                  ──scene_graph.SceneGraphBuilder.build──▶ SceneGraph{instances, table_z}
                  ──reachability.ReachabilityModel──▶ 臂分配 / 腿级可达预检
"""
from .rgbd_fusion import RgbdFusion, backproject, voxel_downsample
from .scene_graph import (Instance, SceneGraph, SceneGraphBuilder,
                          PerceptionFail)
from .reachability import ReachabilityModel

__all__ = [
    "RgbdFusion", "backproject", "voxel_downsample",
    "Instance", "SceneGraph", "SceneGraphBuilder", "PerceptionFail",
    "ReachabilityModel",
]
