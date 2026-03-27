"""
文档分块处理模块
作用：将长文档切分成较小的块，适合向量化存储和检索
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY


class DocumentChunker:
    """
    文档分块器类
    负责将长文档切分成小块
    """
    
    def __init__(self, chunk_size=CHUNK_SIZE, 
                 chunk_overlap=CHUNK_OVERLAP, 
                 strategy=CHUNK_STRATEGY):
        """
        初始化分块器
        
        参数：
            chunk_size (int): 分块大小（字符数）
            chunk_overlap (int): 分块重叠大小（字符数）
            strategy (str): 分块策略（recursive/markdown）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def split_documents(self, documents):
        """
        对文档列表进行分块
        
        参数：
            documents (list): Document对象列表
        
        返回：
            list: 分块后的Document对象列表
        """
        if self.strategy == "markdown":
            return self._split_by_markdown(documents)
        else:
            return self._split_by_recursive(documents)
    
    def _split_by_recursive(self, documents):
        """
        使用递归策略分块（默认）
        
        优点：通用性强，保证语义完整性
        适用：各种文档类型
        
        工作原理：
            1. 按段落切分（\n\n）
            2. 按行切分（\n）
            3. 按句子切分（。！？）
            4. 保留原始元数据
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # 创建递归文本分割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        # 对文档列表进行分块
        return splitter.split_documents(documents)
    
    def _split_by_markdown(self, documents):
        """
        使用Markdown标题策略分块
        
        优点：保持文档结构，标题与正文绑定
        适用：技术文档、API文档、教程等
        
        工作原理：
            1. 基于标题结构切分
            2. 对超长块进行二次递归切分
            3. 保留原始元数据
        """
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
            MarkdownHeaderTextSplitter
        )
        
        # 创建Markdown标题分割器
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "一级标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
                ("####", "四级标题"),
            ]
        )
        
        # 初始化分块列表
        split_docs = []
        
        # 遍历每个文档进行分块
        for doc in documents:
            # 使用标题分割器切分文档
            chunks = header_splitter.split_text(doc.page_content)
            
            # 为每个分块更新元数据（保留原始文档的元数据）
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            
            # 将分块添加到结果列表
            split_docs.extend(chunks)
        
        # 对超长块进行二次递归切分
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        return recursive_splitter.split_documents(split_docs)


# 便捷函数
def split_documents(documents):
    """
    便捷函数：对文档进行分块
    
    参数：
        documents (list): Document对象列表
    
    返回：
        list: 分块后的Document对象列表
    """
    chunker = DocumentChunker()
    return chunker.split_documents(documents)
