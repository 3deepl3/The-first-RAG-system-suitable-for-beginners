"""
文档加载器模块
作用：根据文件格式选择对应的加载器，提取文本内容和元数据
"""

import os
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,        # PDF文档加载器
    Docx2txtLoader,     # Word文档加载器
    TextLoader,         # 纯文本加载器
    UnstructuredMarkdownLoader  # Markdown加载器
)
from utils.file_utils import (
    get_file_md5,
    get_file_extension,
    get_file_name
)
from config import SUPPORTED_FORMATS


class DocumentLoader:
    """
    文档加载器类
    负责加载各种格式的文档
    """
    
    def __init__(self, supported_formats=None):
        """
        初始化文档加载器
        
        参数：
            supported_formats (list): 支持的文件格式列表
        """
        self.supported_formats = supported_formats or SUPPORTED_FORMATS
    
    def load_document(self, file_path):
        """
        加载单个文档
        
        参数：
            file_path (str): 文件路径
        
        返回：
            list: Document对象列表，每个对象包含页面内容和元数据
        
        异常：
            ValueError: 当文件格式不支持时抛出
        """
        # 检查文件格式
        if not self._is_supported(file_path):
            raise ValueError(
                f"不支持的文件格式：{get_file_extension(file_path)}。"
                f"支持的格式：{', '.join(self.supported_formats)}"
            )
        
        # 根据文件类型选择对应的加载器
        loader = self._get_loader(file_path)
        
        # 加载文档
        documents = loader.load()
        
        # 添加自定义元数据
        self._add_metadata(documents, file_path)
        
        return documents
    
    def _is_supported(self, file_path):
        """
        检查文件格式是否支持
        
        参数：
            file_path (str): 文件路径
        
        返回：
            bool: 是否支持
        """
        return get_file_extension(file_path) in self.supported_formats
    
    def _get_loader(self, file_path):
        """
        根据文件类型获取对应的加载器
        
        参数：
            file_path (str): 文件路径
        
        返回：
            loader: 文档加载器对象
        """
        file_ext = get_file_extension(file_path)
        
        if file_ext == ".pdf":
            return PyPDFLoader(file_path)
        elif file_ext == ".docx":
            return Docx2txtLoader(file_path)
        elif file_ext == ".md":
            return UnstructuredMarkdownLoader(file_path)
        elif file_ext == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式：{file_ext}")
    
    def _add_metadata(self, documents, file_path):
        """
        为每个文档添加自定义元数据
        
        参数：
            documents (list): Document对象列表
            file_path (str): 文件路径
        """
        # 计算文件的MD5值（用于增量更新）
        file_md5 = get_file_md5(file_path)
        
        # 获取文件名
        file_name = get_file_name(file_path)
        
        # 获取当前时间
        update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 为每个文档添加元数据
        for doc in documents:
            doc.metadata["file_name"] = file_name
            doc.metadata["file_md5"] = file_md5
            doc.metadata["update_time"] = update_time
    
    def load_documents(self, file_paths):
        """
        批量加载多个文档
        
        参数：
            file_paths (list): 文件路径列表
        
        返回：
            list: 所有文档的Document对象列表
        
        说明：
            即使某个文件加载失败，也会继续加载其他文件
        """
        all_documents = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                failed_files.append({
                    "file": file_path,
                    "error": str(e)
                })
        
        return all_documents, failed_files


# 便捷函数：用于快速加载单个文档
def load_document(file_path):
    """
    便捷函数：加载单个文档
    
    参数：
        file_path (str): 文件路径
    
    返回：
        list: Document对象列表
    """
    loader = DocumentLoader()
    return loader.load_document(file_path)


# 便捷函数：用于批量加载文档
def load_documents(file_paths):
    """
    便捷函数：批量加载文档
    
    参数：
        file_paths (list): 文件路径列表
    
    返回：
        tuple: (所有文档列表, 失败文件列表)
    """
    loader = DocumentLoader()
    return loader.load_documents(file_paths)
