"""
配置模块 - GitHub公开版本
作用：所有系统参数的统一配置中心
优先级：.env文件 > 默认值
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==================== 模型配置 ====================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
DEVICE = os.getenv("DEVICE", "cpu")

# ==================== 知识库配置 ====================

# 获取项目根目录（假设此文件在 rag_system/ 下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_BASE_DIR = os.path.abspath(os.getenv("KNOWLEDGE_BASE_DIR", os.path.join(PROJECT_ROOT, "data", "knowledge_base")))
CHROMA_DB_PATH = os.path.abspath(os.getenv("CHROMA_DB_PATH", os.path.join(PROJECT_ROOT, "data", "chroma_db")))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")

# 支持的文件格式
SUPPORTED_FORMATS = [".pdf", ".docx", ".md", ".txt"]

# ==================== 检索配置 ====================

USE_MIXED_RETRIEVAL = os.getenv("USE_MIXED_RETRIEVAL", "True").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "True").lower() == "true"
USE_CONTEXT_COMPRESSION = os.getenv("USE_CONTEXT_COMPRESSION", "False").lower() == "true"

TOP_K_WIDE = int(os.getenv("TOP_K_WIDE", "10"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# ==================== 对话配置 ====================

USE_MULTI_TURN = os.getenv("USE_MULTI_TURN", "True").lower() == "true"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ==================== Streamlit配置 ====================

PAGE_TITLE = os.getenv("PAGE_TITLE", "完整RAG问答系统")
PAGE_ICON = os.getenv("PAGE_ICON", "📚")
LAYOUT = os.getenv("LAYOUT", "wide")


def get_config_summary():
    """获取配置摘要（用于调试）"""
    return {
        "模型配置": {
            "嵌入模型": EMBEDDING_MODEL,
            "重排序模型": RERANKER_MODEL,
            "大模型": LLM_MODEL,
            "设备": DEVICE,
        },
        "知识库配置": {
            "知识库路径": KNOWLEDGE_BASE_DIR,
            "向量库路径": CHROMA_DB_PATH,
            "分块大小": CHUNK_SIZE,
            "分块重叠": CHUNK_OVERLAP,
        },
        "检索配置": {
            "混合检索": USE_MIXED_RETRIEVAL,
            "重排序": USE_RERANKER,
            "TOP_K_FINAL": TOP_K_FINAL,
            "向量权重": VECTOR_WEIGHT,
            "BM25权重": BM25_WEIGHT,
        },
        "对话配置": {
            "多轮对话": USE_MULTI_TURN,
            "温度": LLM_TEMPERATURE,
        },
    }