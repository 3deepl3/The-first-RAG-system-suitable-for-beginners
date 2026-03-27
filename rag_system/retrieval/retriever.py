"""
检索器构建模块
作用：整合向量检索、BM25检索、重排序等功能
"""

from config import (
    USE_MIXED_RETRIEVAL,
    USE_RERANKER,
    TOP_K_WIDE,
    VECTOR_WEIGHT,
    BM25_WEIGHT
)


class RetrieverBuilder:
    """
    检索器构建器类
    负责构建最终的检索器
    """
    
    def __init__(self, vector_db, bm25_retriever=None, reranker=None):
        """
        初始化检索器构建器
        
        参数：
            vector_db: 向量数据库对象
            bm25_retriever: BM25检索器对象
            reranker: 重排序模型对象
        """
        self.vector_db = vector_db
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
    
    def build_retriever(self):
        """
        构建最终检索器
        
        检索链路：
            用户问题 → 向量检索 → (混合检索) → (重排序) → 最终结果
        
        返回：
            Retriever: 最终的检索器对象
        """
        # 步骤1：创建基础向量检索器
        vector_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": TOP_K_WIDE}
        )
        
        # 步骤2：混合检索（向量 + BM25）
        if USE_MIXED_RETRIEVAL and self.bm25_retriever is not None:
            base_retriever = self._build_ensemble_retriever(
                vector_retriever, self.bm25_retriever
            )
        else:
            base_retriever = vector_retriever
        
        # 步骤3：重排序
        if USE_RERANKER and self.reranker is not None:
            final_retriever = self._build_rerank_retriever(
                base_retriever
            )
        else:
            final_retriever = base_retriever
        
        return final_retriever
    
    def _build_ensemble_retriever(self, vector_retriever, bm25_retriever):
        """
        构建混合检索器（向量 + BM25）
        
        参数：
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
        
        返回：
            EnsembleRetriever: 混合检索器
        """
        # 兼容新旧版本 LangChain
        try:
            from langchain.retrievers import EnsembleRetriever
        except ImportError:
            from langchain_community.retrievers import EnsembleRetriever
        
        # 创建集成检索器（混合检索）
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[VECTOR_WEIGHT, BM25_WEIGHT]
        )
    
    def _build_rerank_retriever(self, base_retriever):
        """
        构建重排序检索器
        
        参数：
            base_retriever: 基础检索器
        
        返回：
            ContextualCompressionRetriever: 重排序检索器
        """
        # 兼容新旧版本 LangChain
        try:
            from langchain.retrievers import ContextualCompressionRetriever
        except ImportError:
            from langchain_community.retrievers import ContextualCompressionRetriever
        
        # 创建上下文压缩检索器（包含重排序功能）
        return ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=base_retriever
        )


# 便捷函数
def build_retriever(vector_db, bm25_retriever=None, reranker=None):
    """
    便捷函数：构建检索器
    
    参数：
        vector_db: 向量数据库对象
        bm25_retriever: BM25检索器对象
        reranker: 重排序模型对象
    
    返回：
        Retriever: 检索器对象
    """
    builder = RetrieverBuilder(vector_db, bm25_retriever, reranker)
    return builder.build_retriever()
