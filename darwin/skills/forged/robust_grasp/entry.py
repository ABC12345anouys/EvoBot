"""forged skill: robust_grasp - GraspNet score 低时回退到物体中心。"""
import numpy as np


def run(env=None, body_name="green_block", score_threshold=99.0, **kwargs):
    """GraspNet 预测抓取位姿，若 score 低于阈值则回退到物体中心。"""
    from darwin.skills.perception.grasp import grasp_from_env
    gt_pos = np.array(env.get_body_pos(body_name), dtype=float)
    rg = grasp_from_env(env, body_name)
    if rg.get("success"):
        grasp_pt = np.array(rg["position"], dtype=float)
        score = rg.get("score", 0)
        if score >= score_threshold:
            return {"success": True, "position": grasp_pt.tolist(),
                    "source": "graspnet", "score": float(score)}
        # score 低，回退到物体中心
        return {"success": True, "position": gt_pos.tolist(),
                "source": "fallback_center", "score": float(score),
                "reason": f"score {score:.2f} < {score_threshold}"}
    return {"success": True, "position": gt_pos.tolist(),
            "source": "fallback_center", "reason": "graspnet failed"}
