"""ragbot：机械臂操作经验记忆层（RAG）。

定位：给 LLM(GPT/deepseek 等) 操控机械臂当记忆层，减少 token 消耗、提高准确率。
- 执行前：检索历史成功/失败经验，注入 prompt，跳过已知坑、复用已知好 offset。
- 执行后：把本轮结果(成功/失败 + rel_offset)写回，越用越准。

调用方用自己的感知(3D 相机点云 / GraspNet / 几何 PCA)算 shape/size/center/抓取点，
用自己的控制(ROS moveit / libfranka / 自研)执行，中间的"经验记忆"用 ragbot。

公开 API（pip install ragbot 后即可使用）：

    from ragbot import RAGMemory, MemoryStore, merge_evidence, size_band_of

    rag = RAGMemory()                         # 经验库（默认 ragbot/memory/experience/）
    # 执行前
    cands = rag.rank_candidates("pick", feat={"shape":"box","size":[.05,.05,.05]},
                                candidates=[{"score_band":"mid"}, {"score_band":"small"}])
    best = rag.best_success("pick", feat)
    rel = rag.is_failed("pick", feat, rel_offset=[0.1, 0.0, 0.0])
    # 执行后
    rag.add_success("pick", feat, rel_offset, score_band="mid", attempts=2, exp="...")
    rag.add_failure("pick", feat, rel_offset, score_band="mid", fail_phase="align", exp="...")
    rag.tick()

设计要点：
- 归一化经验：rel_offset = (grasp_point - object_center) / object_half_size
- 跨物体泛化：Euclidean distance < 0.5 in normalized space
- 失败记忆 TTL=20 episodes，tick()/decay() 推进时钟
- 语义检索：TF-IDF 余弦（默认）；RAGBOT_EMBED=1 启用 sentence-transformers
- 落盘：YAML frontmatter .md，可 git 追踪、可人读

只暴露 memory 子包；感知/控制/执行/任务编排/LLM 等代码保留在仓库但不作为公开 API。
"""
from .memory import RAGMemory, MemoryStore, merge_evidence, size_band_of

__all__ = ["RAGMemory", "MemoryStore", "merge_evidence", "size_band_of"]
__version__ = "0.1.0"
