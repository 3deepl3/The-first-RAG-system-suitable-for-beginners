"""
知识库管理模块
作用：管理文档的加载、分块、向量化、入库等离线流程
"""

import os
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_DB_PATH,
    SUPPORTED_FORMATS,
    TOP_K_WIDE,
    USE_MIXED_RETRIEVAL
)
from utils.file_utils import (
    get_file_md5,
    get_file_name,
    is_supported_format,
    ensure_directory_exists
)
from .document_loader import DocumentLoader
from .chunker import DocumentChunker


class KnowledgeBase:
    """
    知识库管理类
    负责文档的加载、分块、向量化、入库等操作
    """
    
    def __init__(self, embedding_model, vector_db_path=CHROMA_DB_PATH):
        """
        初始化知识库
        
        参数：
            embedding_model: 嵌入模型对象
            vector_db_path (str): 向量库存储路径
        """
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.vector_db = None
        self.all_docs = None
        self.bm25_retriever = None
        
        # 初始化组件
        self.document_loader = DocumentLoader()
        self.document_chunker = DocumentChunker()
    
    def load_vector_db(self):
        """
        加载或创建向量数据库
        
        返回：
            tuple: (向量库对象, 全量文档数据)
        """
        # 检查向量库目录是否存在且不为空
        if os.path.exists(self.vector_db_path) and len(os.listdir(self.vector_db_path)) > 0:
            # 向量库已存在，直接加载
            self.vector_db = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embedding_model
            )
            self.all_docs = self.vector_db.get()
        else:
            # 向量库不存在，创建新的空向量库
            self.vector_db = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embedding_model
            )
            self.all_docs = {"documents": [], "metadatas": []}
        
        # 初始化BM25检索器
        self._init_bm25_retriever()
        
        return self.vector_db, self.all_docs
    
    def _init_bm25_retriever(self):
        """
        初始化BM25关键词检索器
        """
        if not self.all_docs["documents"]:
            self.bm25_retriever = None
            return
        
        # 从向量库数据中恢复Document对象
        bm25_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(self.all_docs["documents"], self.all_docs["metadatas"])
        ]
        
        # 创建BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        self.bm25_retriever.k = TOP_K_WIDE
    
    def update_knowledge_base(self):
        """
        增量更新知识库
        只处理新增或修改的文件
        
        返回：
            tuple: (新增的分块列表, 结果消息)
        """
        # 步骤1：扫描知识库文件夹
        target_files = self._scan_knowledge_folder()
        
        if not target_files:
            return [], "知识库文件夹中没有找到支持的文档"
        
        # 步骤2：筛选需要更新的文件
        pending_files = self._filter_files_to_update(target_files)
        
        if not pending_files:
            return [], "知识库已是最新，没有新增/修改的文档"
        
        # 步骤3：批量处理文件
        all_chunks, success_count, fail_count, fail_files = self._process_files(pending_files)
        
        # 步骤4：存入向量库
        if all_chunks:
            self.vector_db.add_documents(all_chunks)
        
        # 步骤5：重新初始化BM25检索器
        self._init_bm25_retriever()
        
        # 步骤6：生成结果消息
        result_msg = (
            f"知识库更新完成！\n"
            f"成功处理：{success_count}个文件\n"
            f"失败：{fail_count}个文件\n"
            f"新增分块：{len(all_chunks)}个"
        )
        
        if fail_files:
            result_msg += f"\n失败文件：\n" + "\n".join(fail_files)
        
        return all_chunks, result_msg
    
    def _scan_knowledge_folder(self):
        """
        扫描知识库文件夹，获取所有支持格式的文件
        
        返回：
            list: 文件路径列表
        """
        # 确保知识库文件夹存在
        ensure_directory_exists(KNOWLEDGE_BASE_DIR)
        
        target_files = []
        
        # 遍历知识库文件夹及其子文件夹
        for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                if is_supported_format(file_path, SUPPORTED_FORMATS):
                    target_files.append(file_path)
        
        return target_files
    
    def _filter_files_to_update(self, target_files):
        """
        筛选需要更新的文件（新增或修改）
        
        参数：
            target_files (list): 所有目标文件路径列表
        
        返回：
            list: 需要更新的文件路径列表
        """
        # 提取已有文档的MD5列表
        existing_md5s = list(set([
            meta.get("file_md5", "")
            for meta in self.all_docs["metadatas"]
        ]))
        
        # 筛选需要更新的文件
        pending_files = []
        for file_path in target_files:
            file_md5 = get_file_md5(file_path)
            if file_md5 not in existing_md5s:
                pending_files.append(file_path)
        
        return pending_files
    
    def _process_files(self, file_paths):
        """
        批量处理文件
        
        参数：
            file_paths (list): 文件路径列表
        
        返回：
            tuple: (所有分块, 成功数, 失败数, 失败文件列表)
        """
        all_chunks = []
        success_count = 0
        fail_count = 0
        fail_files = []
        
        for file_path in file_paths:
            try:
                # 加载文档
                docs = self.document_loader.load_document(file_path)
                
                # 分块处理
                chunks = self.document_chunker.split_documents(docs)
                
                # 添加到总列表
                all_chunks.extend(chunks)
                
                success_count += 1
            except Exception as e:
                fail_count += 1
                fail_files.append(f"{get_file_name(file_path)}：{str(e)}")
                continue
        
        return all_chunks, success_count, fail_count, fail_files
    
    def get_vector_db(self):
        """
        获取向量数据库对象
        
        返回：
            Chroma: 向量数据库对象
        """
        return self.vector_db
    
    def get_bm25_retriever(self):
        """
        获取BM25检索器
        
        返回：
            BM25Retriever: BM25检索器对象
        """
        return self.bm25_retriever
    
    def get_stats(self):
        """
        获取知识库统计信息
        
        返回：
            dict: 统计信息
        """
        if not self.all_docs:
            return {
                "document_count": 0,
                "chunk_count": 0,
                "unique_files": 0
            }
        
        unique_files = set([
            meta.get("file_name", "")
            for meta in self.all_docs["metadatas"]
        ])
        
        return {
            "document_count": len(unique_files),
            "chunk_count": len(self.all_docs["documents"]),
            "unique_files": len(unique_files)
        }
