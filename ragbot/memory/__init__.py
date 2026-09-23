"""ragbot.memory：机械臂操作经验记忆层（RAG）。

公开 API：
- RAGMemory：泛化抓取经验库（MemoryStore 后端 + TTL 过期 + 语义检索 + UCB 重排）
- MemoryStore：YAML frontmatter 记忆存储（evidence 合并 + 自动置信度升级）
- merge_evidence / size_band_of：辅助函数

设计要点（详见 ragbot/memory/rag.py 顶部文档串）：
- 归一化抓取经验：rel_offset = (grasp_point - object_center) / object_half_size
- 跨物体泛化：Euclidean distance < OFFSET_TOL(0.5) in normalized space
- 失败记忆 TTL=20 episodes，tick()/decay() 推进时钟
- 落盘：memory/experience/*.md（YAML frontmatter）
- 语义检索：默认 TF-IDF 余弦；RAGBOT_EMBED=1 启用 sentence-transformers
"""
from .store import MemoryStore, merge_evidence
from .rag import RAGMemory, size_band_of

__all__ = ["MemoryStore", "merge_evidence", "RAGMemory", "size_band_of"]
