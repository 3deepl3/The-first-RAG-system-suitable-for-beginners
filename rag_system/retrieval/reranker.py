"""
重排序模块
作用：对检索结果进行精细排序，提升最终检索精度
"""

from config import TOP_K_FINAL, DEVICE


class Reranker:
    """
    重排序器类
    负责对检索结果进行精细排序
    """
    
    def __init__(self, model_name=None, top_n=TOP_K_FINAL, device=DEVICE):
        """
        初始化重排序器
        
        参数：
            model_name (str): 重排序模型名称
            top_n (int): 最终保留的文档数量
            device (str): 运行设备
        """
        from config import RERANKER_MODEL
        
        self.model_name = model_name or RERANKER_MODEL
        self.top_n = top_n
        self.device = device
        self.model = None
        self.cross_encoder_reranker = None
    
    def load_model(self):
        """
        加载重排序模型
        
        返回：
            CrossEncoderReranker: 重排序检索器对象
        """
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        
        # 创建交叉编码器模型
        self.model = HuggingFaceCrossEncoder(
            model_name=self.model_name,
            model_kwargs={"device": self.device}
        )
        
        # 创建重排序器
        try:
            # 新版本 LangChain 的导入路径
            from langchain_community.retrievers.document_compressors import CrossEncoderReranker as CommunityReranker
            self.cross_encoder_reranker = CommunityReranker(
                model=self.model,
                top_n=self.top_n
            )
        except ImportError:
            # 兼容旧版本
            self.cross_encoder_reranker = CrossEncoderReranker(
                model=self.model,
                top_n=self.top_n
            )
        
        return self.cross_encoder_reranker
    
    def get_reranker(self):
        """
        获取重排序器
        
        返回：
            CrossEncoderReranker: 重排序检索器对象
        """
        if self.cross_encoder_reranker is None:
            return self.load_model()
        return self.cross_encoder_reranker


# 便捷函数
def create_reranker(model_name=None, top_n=TOP_K_FINAL):
    """
    便捷函数：创建重排序器
    
    参数：
        model_name (str): 模型名称
        top_n (int): 最终保留的文档数量
    
    返回：
        CrossEncoderReranker: 重排序检索器对象
    """
    reranker = Reranker(model_name=model_name, top_n=top_n)
    return reranker.get_reranker()
