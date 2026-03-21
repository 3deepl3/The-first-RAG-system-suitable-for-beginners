"""
完整RAG文档问答系统
基于 LangChain + Chroma + Ollama 全本地部署

作者：RAG学习实践
日期：2026-03-20
功能：整合离线知识库构建与在线问答全链路，支持混合检索、重排序、多轮对话
"""

# ====================== 第一部分：全局配置中心 ======================
# 说明：所有系统参数都在这里统一配置，方便根据实际需求调整

import os
import hashlib
import streamlit as st
from datetime import datetime

# ---------------------- 模型配置 ----------------------
# 说明：选择适合你硬件的模型配置

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文嵌入模型，CPU友好，效果好
# 作用：将文本转换为向量，用于语义相似度检索
# 其他选项："BAAI/bge-base-zh-v1.5"（更高精度，更大体积）

RERANKER_MODEL = "BAAI/bge-reranker-base"  # 中文重排序模型
# 作用：对检索结果进行精细排序，提升最终检索精度
# 工作原理：将查询和文档输入模型，计算它们的相关性得分

LLM_MODEL = "qwen2.5:3b"  # 本地大模型，通过 Ollama 运行
# 作用：基于检索到的上下文生成最终答案
# 注意：需要先通过 `ollama pull qwen2.5:0.5b` 下载模型

DEVICE = "cpu"  # 设备选择
# 作用：指定模型运行的硬件设备
# 选项："cpu"（CPU运行，最通用）、"cuda"（NVIDIA显卡，速度快）、"mps"（Mac M系列芯片）

# ---------------------- 离线知识库配置 ----------------------
# 说明：控制文档处理和向量库构建的参数

KNOWLEDGE_BASE_DIR =r"D:\Vscodeproject\lstm股票预测\knowledge_base"  # 知识库文件夹路径
# 作用：存放你的PDF、Word等文档的地方
# 使用：将文档复制到此文件夹，系统会自动读取

CHROMA_DB_PATH = r"D:\Vscodeproject\lstm股票预测\chroma_full_db"  # 向量库持久化路径
# 作用：Chroma向量数据库的存储位置
# 说明：向量库会保存在磁盘上，下次启动可直接加载，无需重新构建

CHUNK_SIZE = 500  # 分块大小（字符数）
# 作用：将长文档切分成小块的长度
# 经验值：300-500 是常用范围，太小会丢失上下文，太大会降低检索精度

CHUNK_OVERLAP = 50  # 分块重叠大小（字符数）
# 作用：相邻分块之间保留的重叠内容长度
# 目的：避免关键信息在分块边界被切断，保证语义连贯性

CHUNK_STRATEGY = "recursive"  # 分块策略
# 选项：
#   "recursive" - 递归分块，按段落、句子等层级切分，通用性强
#   "markdown" - 基于Markdown标题结构分块，适合技术文档

SUPPORTED_FORMATS = [".pdf", ".docx", ".md", ".txt"]  # 支持的文档格式
# 作用：系统只会处理这些格式的文件
# 扩展：可以添加其他格式，但需要对应的加载器

# ---------------------- 在线问答配置 ----------------------
# 说明：控制检索策略和生成效果的参数

USE_MIXED_RETRIEVAL = True  # 混合检索开关
# 作用：开启后同时使用向量检索和BM25关键词检索
# 原理：向量检索擅长理解语义，BM25擅长精确匹配专业术语
# 权重：由 VECTOR_WEIGHT 和 BM25_WEIGHT 控制

USE_RERANKER = True  # 重排序开关
# 作用：对初步检索结果进行精细排序，只保留最相关的内容
# 效果：可以显著提升最终答案的准确性
# 性能：会稍微增加响应时间

USE_CONTEXT_COMPRESSION = False  # 上下文压缩开关
# 作用：压缩检索到的文档内容，减少token消耗
# 场景：当文档很长但只有少部分相关时有用

USE_MULTI_TURN = True  # 多轮对话开关
# 作用：支持连续对话，系统会记住对话历史
# 功能：自动处理指代词（如"它"、"这个"），保持对话连贯

TOP_K_WIDE = 10  # 宽召回数量
# 作用：初步检索返回的文档数量
# 说明：先召回更多候选，再通过重排序精简，提高命中率

TOP_K_FINAL = 3  # 最终召回数量
# 作用：最终喂给大模型的文档数量
# 原则：数量要控制，太多会引入噪声，太少可能信息不足

VECTOR_WEIGHT = 0.7  # 向量检索权重
# 作用：混合检索中向量检索结果的权重

BM25_WEIGHT = 0.3  # BM25检索权重
# 作用：混合检索中BM25检索结果的权重
# 注意：两者权重之和应该为1.0

LLM_TEMPERATURE = 0.1  # 大模型温度参数
# 作用：控制生成答案的随机性和创造性
# 取值范围：0.0（完全确定，适合问答）到 1.0（更有创造性）
# 说明：问答场景建议用低温度，减少幻觉


# ====================== 第二部分：工具函数模块 ======================
# 说明：提供通用的辅助功能，如文件哈希、系统检查等

def get_file_md5(file_path):
    """
    计算文件的MD5哈希值
    
    作用：用于判断文件是否发生变化（新增或修改）
    应用场景：知识库增量更新，避免重复处理未修改的文件
    
    参数：
        file_path (str): 文件路径
    
    返回：
        str: 文件的MD5哈希值（32位十六进制字符串）
    
    工作原理：
        1. 打开文件（二进制模式）
        2. 逐块读取文件内容（每次4096字节）
        3. 将每块内容更新到MD5哈希对象中
        4. 计算最终的哈希值
    """
    # 创建MD5哈希对象
    hash_md5 = hashlib.md5()
    
    # 打开文件，以二进制模式读取
    with open(file_path, "rb") as f:
        # 使用迭代器逐块读取，避免大文件占用过多内存
        # lambda: f.read(4096) - 读取4096字节的匿名函数
        # b'' - 迭代终止标志，当读到空字节时停止
        for chunk in iter(lambda: f.read(4096), b""):
            # 将读取到的块更新到哈希对象中
            hash_md5.update(chunk)
    
    # 返回十六进制格式的哈希值（32位字符串）
    return hash_md5.hexdigest()


def check_ollama_running():
    """
    检查Ollama服务是否正常运行，模型是否可用
    
    作用：系统启动时的健康检查，确保RAG系统可以正常工作
    
    返回：
        tuple: (是否成功, 消息)
            - 成功： (True, "Ollama运行正常，模型可用")
            - 失败： (False, "错误详情...")
    
    工作流程：
        1. 导入Ollama库
        2. 创建Ollama实例，指定模型
        3. 发送测试请求（"你好"）
        4. 根据返回结果判断状态
    """
    try:
        # 导入LangChain的Ollama封装
        from langchain_community.llms import Ollama
        
        # 创建Ollama实例
        # model: 使用的模型名称（从全局配置读取）
        llm = Ollama(model=LLM_MODEL)
        
        # 发送测试请求，验证服务是否正常
        # invoke() - 调用大模型生成文本
        llm.invoke("你好")
        
        # 如果没有抛出异常，说明服务正常
        return True, "Ollama运行正常，模型可用"
    
    except Exception as e:
        # 捕获异常，返回错误信息
        return False, f"Ollama检查失败：{str(e)}，请确保Ollama已启动，模型已下载"


# ====================== 第三部分：离线知识库管理模块 ======================
# 说明：负责文档的加载、处理、分块、向量化、入库等离线流程

@st.cache_resource
def init_embeddings():
    """
    初始化嵌入模型（全局缓存）
    
    作用：加载文本嵌入模型，用于将文本转换为向量
    缓存机制：使用Streamlit的缓存，避免重复加载，提升性能
    
    返回：
        HuggingFaceEmbeddings: 嵌入模型对象
    
    工作原理：
        1. 导入HuggingFace的嵌入模型库
        2. 创建嵌入模型实例
        3. 配置设备（CPU/GPU/MPS）
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # 创建嵌入模型实例
    # model_name: 模型名称（从全局配置读取）
    # model_kwargs: 模型参数字典
    #   - device: 运行设备（CPU/GPU/MPS）
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )


@st.cache_resource
def init_vector_db(_embeddings):
    """
    初始化或加载向量数据库（全局缓存）
    
    作用：从磁盘加载已有的向量库，或创建新的空向量库
    缓存机制：使用Streamlit的缓存，避免重复加载
    
    参数：
        _embeddings: 嵌入模型对象（用于向量化）
    
    返回：
        tuple: (向量库对象, 全量文档数据)
            - 向量库对象: Chroma实例，用于检索
            - 全量文档数据: 包含所有文档和元数据的字典
    
    工作流程：
        1. 检查向量库路径是否存在且不为空
        2. 存在则加载已有向量库
        3. 不存在则创建新的空向量库
    """
    from langchain_chroma import Chroma
    
    # 检查向量库目录是否存在且不为空
    # os.path.exists() - 判断路径是否存在
    # os.listdir() - 列出目录下的所有文件
    if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
        # 向量库已存在，直接加载
        # persist_directory: 向量库持久化路径
        # embedding_function: 嵌入模型，用于向量化
        vector_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=_embeddings
        )
        
        # 获取向量库中的所有文档
        # get() - 获取所有文档和元数据
        all_docs = vector_db.get()
        
        return vector_db, all_docs
    
    else:
        # 向量库不存在，创建新的空向量库
        vector_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=_embeddings
        )
        
        # 返回空文档列表
        # documents: 文档内容列表（初始化为空）
        # metadatas: 元数据列表（初始化为空）
        return vector_db, {"documents": [], "metadatas": []}


def load_and_parse_documents(file_path):
    """
    加载并解析文档
    
    作用：根据文件格式选择对应的加载器，提取文本内容和元数据
    
    参数：
        file_path (str): 文件路径
    
    返回：
        list: Document对象列表，每个对象包含页面内容和元数据
    
    工作流程：
        1. 根据文件扩展名选择加载器
        2. 使用对应的加载器加载文档
        3. 为每个文档添加自定义元数据（文件名、MD5、更新时间）
    
    支持的格式：
        - PDF: 使用PyPDFLoader
        - Word (.docx): 使用Docx2txtLoader
        - Markdown (.md): 使用UnstructuredMarkdownLoader
        - 纯文本 (.txt): 使用TextLoader
    """
    from langchain_community.document_loaders import (
        PyPDFLoader,        # PDF文档加载器
        Docx2txtLoader,     # Word文档加载器
        TextLoader,         # 纯文本加载器
        UnstructuredMarkdownLoader  # Markdown加载器
    )
    
    # 获取文件扩展名（转换为小写）
    # os.path.splitext() - 分离文件名和扩展名，返回 (name, ext)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 根据文件类型选择对应的加载器
    if file_ext == ".pdf":
        # PDF文档加载器
        loader = PyPDFLoader(file_path)
    
    elif file_ext == ".docx":
        # Word文档加载器
        loader = Docx2txtLoader(file_path)
    
    elif file_ext == ".md":
        # Markdown文档加载器
        loader = TextLoader(file_path,encoding="utf-8", autodetect_encoding=True)
    
    elif file_ext == ".txt":
        # 纯文本加载器
        # encoding="utf-8" - 指定编码为UTF-8，支持中文
        loader = TextLoader(file_path, encoding="utf-8")
    
    else:
        # 不支持的文件格式，抛出异常
        raise ValueError(f"不支持的文件格式：{file_ext}")
    
    # 加载文档，自动提取文本和基本元数据（如页码）
    # load() - 返回Document对象列表
    # Document对象结构：
    #   - page_content: 文档文本内容
    #   - metadata: 元数据字典（页码、来源等）
    documents = loader.load()
    
    # 计算文件的MD5值（用于增量更新）
    file_md5 = get_file_md5(file_path)
    
    # 获取文件名（不包含路径）
    # os.path.basename() - 提取文件名部分
    file_name = os.path.basename(file_path)
    
    # 为每个文档添加自定义元数据
    # 遍历所有文档对象
    for doc in documents:
        # 在metadata字典中添加文件名
        # metadata: Document对象的元数据属性（字典类型）
        doc.metadata["file_name"] = file_name
        
        # 添加文件MD5，用于判断文件是否修改
        doc.metadata["file_md5"] = file_md5
        
        # 添加更新时间
        # datetime.now() - 获取当前时间
        # strftime() - 格式化时间为字符串
        doc.metadata["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 返回处理后的文档列表
    return documents


def split_documents(documents):
    """
    文档分块处理
    
    作用：将长文档切分成较小的块，适合向量化存储和检索
    
    参数：
        documents (list): Document对象列表
    
    返回：
        list: 分块后的Document对象列表
    
    工作原理：
        1. 根据分块策略选择切分方式
        2. Markdown策略：基于标题结构切分
        3. Recursive策略：按段落、句子递归切分
        4. 保留原始元数据
    
    两种策略对比：
        - Markdown策略：适合有清晰结构的技术文档，保持标题与正文关联
        - Recursive策略：通用性强，适合各种文档类型
    """
    from langchain_classic.text_splitter import (
        RecursiveCharacterTextSplitter,   # 递归文本分割器
        MarkdownHeaderTextSplitter        # Markdown标题分割器
    )
    
    # 根据全局配置选择分块策略
    if CHUNK_STRATEGY == "markdown":
        """
        Markdown标题分块策略
        优点：保持文档结构，标题与正文绑定
        适用：技术文档、API文档、教程等
        """
        # 创建Markdown标题分割器
        # headers_to_split_on: 定义标题层级和对应的标签
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "一级标题"),      # # 对应一级标题
                ("##", "二级标题"),     # ## 对应二级标题
                ("###", "三级标题"),    # ### 对应三级标题
                ("####", "四级标题"),   # #### 对应四级标题
            ]
        )
        
        # 初始化分块列表
        # list() - 创建空列表，用于存储分块结果
        split_docs = []
        
        # 遍历每个文档进行分块
        for doc in documents:
            # 使用标题分割器切分文档
            # split_text() - 返回按标题切分的Document列表
            chunks = header_splitter.split_text(doc.page_content)
            
            # 为每个分块更新元数据（保留原始文档的元数据）
            # chunk.metadata.update() - 将原文档的元数据合并到分块中
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            
            # 将分块添加到结果列表
            # extend() - 将列表中的所有元素添加到当前列表
            split_docs.extend(chunks)
        
        # 对超长块进行二次递归切分
        # 原因：某些章节可能过长，需要进一步切分
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # separators: 分割符号优先级列表（从高到低）
            # 优先用\n\n（段落），其次\n（行），再。！？句子
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        # 对初次分块结果进行二次切分
        # split_documents() - 对Document列表进行分割
        final_docs = recursive_splitter.split_documents(split_docs)
    
    else:
        """
        递归分块策略（默认）
        优点：通用性强，保证语义完整性
        适用：各种文档类型
        """
        # 创建递归文本分割器
        # chunk_size: 每个分块的目标大小
        # chunk_overlap: 相邻分块之间的重叠大小
        # separators: 分割符号优先级列表
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        # 直接对文档列表进行分块
        final_docs = splitter.split_documents(documents)
    
    # 返回最终的分块结果
    return final_docs


def build_or_update_knowledge_base(_embeddings, _vector_db, _all_docs):
    """
    构建或更新知识库（增量更新）
    
    作用：扫描知识库文件夹，只处理新增或修改的文件，避免重复处理
    
    参数：
        _embeddings: 嵌入模型对象
        _vector_db: 向量库对象
        _all_docs: 全量文档数据（用于判断哪些文件需要更新）
    
    返回：
        tuple: (新增的分块列表, 结果消息)
    
    工作流程：
        1. 扫描知识库文件夹，获取所有支持格式的文件
        2. 对比文件MD5，筛选出需要更新的文件
        3. 加载、分块、向量化、入库
        4. 返回处理结果
    
    优化点：
        - 增量更新：只处理变化的文件，提升效率
        - MD5对比：准确判断文件是否修改
    """
    # ========== 步骤1：扫描知识库文件夹 ==========
    
    # 检查知识库文件夹是否存在
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        # 不存在则创建
        # os.makedirs() - 创建目录（包括所有必要的父目录）
        os.makedirs(KNOWLEDGE_BASE_DIR)
        return [], "知识库文件夹已创建，请放入文档后再更新"
    
    # 初始化目标文件列表
    # list() - 创建空列表
    target_files = []
    
    # 遍历知识库文件夹及其子文件夹
    # os.walk() - 递归遍历目录树
    # 返回：(当前目录路径, 子目录列表, 文件列表)
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        # 遍历当前目录下的所有文件
        for file in files:
            # 获取文件扩展名（转换为小写）
            file_ext = os.path.splitext(file)[1].lower()
            
            # 检查文件格式是否支持
            # in - 判断元素是否在列表中
            if file_ext in SUPPORTED_FORMATS:
                # 构建完整文件路径（目录 + 文件名）
                # os.path.join() - 智能拼接路径，适应不同操作系统
                file_path = os.path.join(root, file)
                
                # 将文件路径添加到目标列表
                target_files.append(file_path)
    
    # 检查是否找到支持格式的文件
    if not target_files:
        return [], "知识库文件夹中没有找到支持的文档，请放入PDF/Word/Markdown/TXT文件"
    
    # ========== 步骤2：筛选需要更新的文件 ==========
    
    # 提取已有文档的MD5列表
    # list(set()) - 去重，将列表转换为集合再转回列表，移除重复值
    existing_md5s = list(set([
        # 遍历元数据列表
        meta.get("file_md5", "")  # 获取file_md5，不存在则返回空字符串
        for meta in _all_docs["metadatas"]
    ]))
    
    # 初始化待处理文件列表
    pending_files = []
    
    # 遍历所有目标文件，判断是否需要更新
    for file_path in target_files:
        # 计算文件的MD5值
        file_md5 = get_file_md5(file_path)
        
        # 检查MD5是否已存在于向量库中
        # not in - 判断元素是否不在列表中
        if file_md5 not in existing_md5s:
            # MD5不存在，说明是新增或修改的文件
            pending_files.append(file_path)
    
    # 检查是否有需要更新的文件
    if not pending_files:
        return [], "知识库已是最新，没有新增/修改的文档"
    
    # ========== 步骤3：批量处理待更新的文件 ==========
    
    # 初始化统计变量
    # int() - 定义整数变量
    all_chunks = []       # 存储所有分块
    success_count = 0     # 成功处理的文件数量
    fail_count = 0        # 处理失败的文件数量
    fail_files = []       # 失败文件列表（存储错误信息）
    
    # 遍历待处理文件
    for file_path in pending_files:
        try:
            # --- 子步骤3.1：加载解析文档 ---
            docs = load_and_parse_documents(file_path)
            
            # --- 子步骤3.2：文档分块 ---
            chunks = split_documents(docs)
            
            # 将分块添加到总列表
            # extend() - 将一个列表的所有元素添加到另一个列表
            all_chunks.extend(chunks)
            
            # 成功计数加1
            # += - 自增运算符，等价于 success_count = success_count + 1
            success_count += 1
        
        except Exception as e:
            # 捕获异常，记录失败信息
            fail_count += 1
            fail_files.append(f"{os.path.basename(file_path)}：{str(e)}")
            continue  # 继续处理下一个文件
    
    # ========== 步骤4：存入向量库 ==========
    
    # 检查是否有分块需要存储
    if all_chunks:
        # 将所有分块批量添加到向量库
        # add_documents() - 批量添加文档，自动进行向量化
        _vector_db.add_documents(all_chunks)
    
    # ========== 步骤5：生成结果消息 ==========
    
    # 构建结果消息字符串
    # f-string - 格式化字符串，用 {} 包裹变量
    result_msg = (
        f"知识库更新完成！\n"
        f"成功处理：{success_count}个文件\n"
        f"失败：{fail_count}个文件\n"
        f"新增分块：{len(all_chunks)}个"
    )
    
    # 如果有失败文件，添加到消息中
    if fail_files:
        result_msg += f"\n失败文件：\n" + "\n".join(fail_files)
    
    # 返回结果
    return all_chunks, result_msg


def init_bm25_retriever(_all_docs):
    """
    初始化BM25关键词检索器
    
    作用：创建BM25检索器，用于混合检索中的关键词匹配
    
    参数：
        _all_docs: 全量文档数据
    
    返回：
        BM25Retriever: BM25检索器对象，如果文档为空则返回None
    
    工作原理：
        1. 从向量库数据中恢复Document对象
        2. 构建BM25倒排索引
        3. 返回检索器
    
    应用场景：
        - 精确匹配专业术语（如"错误码E-2049"）
        - 配合向量检索，形成混合检索
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    # 检查是否有文档
    if not _all_docs["documents"]:
        return None
    
    # 从向量库数据中恢复Document对象
    # zip() - 将两个列表打包成元组对，用于并行遍历
    # 例如：zip([1,2], [a,b]) 生成 (1,a), (2,b)
    bm25_docs = [
        # 创建Document对象
        Document(
            page_content=doc,   # 文档内容
            metadata=meta       # 元数据
        )
        # 遍历文档内容和元数据，同时迭代
        for doc, meta in zip(_all_docs["documents"], _all_docs["metadatas"])
    ]
    
    # 创建BM25检索器
    # from_documents() - 从文档列表创建检索器
    # k: 检索时返回的文档数量
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = TOP_K_WIDE
    
    # 返回检索器
    return bm25_retriever


# ====================== 第四部分：在线问答核心模块 ======================
# 说明：负责问题检索、答案生成等在线推理流程

@st.cache_resource
def init_llm():
    """
    初始化大语言模型（全局缓存）
    
    作用：加载本地运行的LLM，用于生成答案和问题改写
    
    返回：
        Ollama: Ollama LLM对象
    
    工作原理：
        1. 导入Ollama库
        2. 创建Ollama实例，配置模型参数
        3. 返回LLM对象
    """
    from langchain_community.llms import Ollama
    
    # 创建Ollama实例
    # model: 使用的模型名称
    # temperature: 温度参数（控制随机性）
    # device: 运行设备
    return Ollama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


@st.cache_resource
def init_reranker():
    """
    初始化重排序模型（全局缓存）
    
    作用：加载交叉编码器模型，对检索结果进行精细排序
    
    返回：
        CrossEncoderReranker: 重排序检索器对象
    
    工作原理：
        1. 加载交叉编码器模型
        2. 创建重排序器包装器
        3. 配置返回的文档数量
    
    应用场景：
        - 提升检索精度
        - 从大量候选文档中筛选最相关的
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    
    # 创建交叉编码器模型
    # model_name: 模型名称
    # model_kwargs: 模型参数（设备等）
    model = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL,
        model_kwargs={"device": DEVICE}
    )
    
    # 创建重排序器
    # model: 交叉编码器模型
    # top_n: 最终保留的文档数量
    return CrossEncoderReranker(model=model, top_n=TOP_K_FINAL)


def build_final_retriever(_vector_db, _bm25_retriever, _reranker):
    """
    构建最终检索器
    
    作用：整合向量检索、BM25检索、重排序等功能，形成完整的检索流程
    
    参数：
        _vector_db: 向量数据库对象
        _bm25_retriever: BM25检索器对象（可能为None）
        _reranker: 重排序模型对象（可能为None）
    
    返回：
        Retriever: 最终的检索器对象
    
    工作流程：
        1. 创建向量检索器
        2. 如果启用混合检索，创建混合检索器
        3. 如果启用重排序，创建重排序检索器
        4. 返回最终检索器
    
    检索链路：
        用户问题 → 向量检索 → (混合检索) → (重排序) → 最终结果
    """
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    
    # ========== 步骤1：创建基础向量检索器 ==========
    
    # as_retriever() - 将向量库转换为检索器
    # search_kwargs: 检索参数
    #   - k: 返回的文档数量
    vector_retriever = _vector_db.as_retriever(search_kwargs={"k": TOP_K_WIDE})
    
    # ========== 步骤2：混合检索（向量 + BM25） ==========
    
    # 检查是否启用混合检索且BM25检索器可用
    if USE_MIXED_RETRIEVAL and _bm25_retriever is not None:
        # 创建集成检索器（混合检索）
        # EnsembleRetriever - 组合多个检索器，加权融合结果
        base_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, _bm25_retriever],  # 检索器列表
            weights=[VECTOR_WEIGHT, BM25_WEIGHT]             # 对应的权重
        )
    else:
        # 不使用混合检索，只用向量检索
        base_retriever = vector_retriever
    
    # ========== 步骤3：重排序 ==========
    
    # 检查是否启用重排序且重排序模型可用
    if USE_RERANKER and _reranker is not None:
        # 创建上下文压缩检索器（包含重排序功能）
        # ContextualCompressionRetriever - 包装基础检索器，添加压缩/重排序
        # base_compressor: 压缩器（这里用作重排序）
        # base_retriever: 基础检索器
        final_retriever = ContextualCompressionRetriever(
            base_compressor=_reranker,
            base_retriever=base_retriever
        )
    else:
        # 不使用重排序
        final_retriever = base_retriever
    
    # 返回最终检索器
    return final_retriever


# ---------------------- 提示词模板 ----------------------
# 说明：控制大模型生成行为的指令模板

# 单轮问答提示词
SINGLE_TURN_PROMPT = """你是一个专业的文档问答助手，必须严格遵守以下规则：

1. 仅基于【检索到的文档内容】回答用户的问题，绝对不允许编造知识库中没有的信息。
2. 如果【检索到的文档内容】中没有相关信息，请直接说"抱歉，当前知识库中没有找到与您问题相关的内容，请您更换问题或补充相关文档到知识库。"，不要编造答案。
3. 回答要清晰、准确、有条理，分点说明复杂内容。
4. 每一句话的结尾必须标注【来源：文件名 第X页】，必须严格使用检索内容里的文件名和页码。

【检索到的文档内容】：
{context}

【用户的问题】：
{question}

【你的回答】：
"""

# 多轮对话提示词
MULTI_TURN_PROMPT = """你是一个专业的文档问答助手，必须严格遵守以下规则：

1. 仅基于【对话历史】和【检索到的文档内容】回答用户的问题，绝对不允许编造知识库中没有的信息。
2. 如果【检索到的文档内容】中没有相关信息，请直接说"抱歉，当前知识库中没有找到与您问题相关的内容，请您更换问题或补充相关文档到知识库。"，不要编造答案。
3. 回答要贴合对话上下文，连贯自然，清晰准确，分点说明复杂内容。
4. 每一句话的结尾必须标注【来源：文件名 第X页】，必须严格使用检索内容里的文件名和页码。

【对话历史】：
{chat_history}

【检索到的文档内容】：
{context}

【用户的当前问题】：
{question}

【你的回答】：
"""

# 问题改写提示词
QUERY_REWRITE_PROMPT = """请结合以下【对话历史】，把用户的【当前问题】改写成一个完整、无指代、适合文档检索的独立问题。

要求：
1. 补全问题中的指代（比如"它""这个""那个"），明确问题的核心主体。
2. 保留原问题的核心意图，不添加额外内容。
3. 只输出改写后的问题，不要输出其他任何内容。

【对话历史】：
{chat_history}

【用户的当前问题】：
{question}

【改写后的问题】：
"""


def format_chat_history(history):
    """
    格式化对话历史
    
    作用：将对话历史列表转换为字符串，方便插入到提示词中
    
    参数：
        history (list): 对话历史列表，每个元素是 {"role": "user/assistant", "content": "..."}
    
    返回：
        str: 格式化后的对话历史字符串
    
    输出示例：
        user：你好
        assistant：您好！有什么可以帮您的？
        user：什么是RAG？
    """
    # 初始化格式化列表
    formatted = []
    
    # 遍历对话历史
    for msg in history:
        # 格式化每条消息
        formatted.append(f"{msg['role']}：{msg['content']}")
    
    # 用换行符连接所有消息
    # "\n".join() - 将列表元素用换行符连接成字符串
    return "\n".join(formatted)


def rag_answer(question, chat_history, _llm, _final_retriever):
    """
    RAG核心问答函数
    
    作用：整合在线推理全流程，从问题到答案
    
    参数：
        question (str): 用户当前问题
        chat_history (list): 对话历史列表
        _llm: 大语言模型对象
        _final_retriever: 最终检索器对象
    
    返回:
        dict: 包含答案和元数据的字典
            - answer: 生成的答案
            - source_docs: 检索到的文档列表
            - rewritten_question: 改写后的问题（多轮模式下）
    
    工作流程：
        1. 如果启用多轮对话且有历史记录，进行问题改写
        2. 使用检索器检索相关文档
        3. 格式化检索到的文档上下文
        4. 根据是否有历史记录选择提示词模板
        5. 调用LLM生成答案
        6. 返回结果
    """
    # ========== 步骤1：多轮对话问题改写 ==========
    
    # 初始化改写后的问题变量
    rewritten_question = None
    
    # 检查是否启用多轮对话且有对话历史
    if USE_MULTI_TURN and len(chat_history) > 0:
        # 格式化对话历史
        formatted_history = format_chat_history(chat_history)
        
        # 生成问题改写提示词
        # format() - 字符串格式化，用实际值替换占位符
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=formatted_history,
            question=question
        )
        
        # 调用LLM进行问题改写
        # invoke() - 调用LLM生成文本
        # strip() - 去除首尾空白字符
        rewritten_question = _llm.invoke(rewrite_prompt).strip()
        
        # 使用改写后的问题进行检索
        search_query = rewritten_question
    else:
        # 单轮对话，直接使用原问题
        search_query = question
    
    # ========== 步骤2：检索相关文档 ==========
    
    # 使用检索器检索相关文档
    # invoke() - 执行检索，返回Document对象列表
    docs = _final_retriever.invoke(search_query)
    
    # ========== 步骤3：格式化检索到的上下文 ==========
    
    # 将检索到的文档格式化为字符串
    # f-string - 格式化字符串
    # 遍历每个文档，提取元数据和内容
    context = "\n\n".join([
        f"【来源：{doc.metadata.get('file_name', '未知')} 第{doc.metadata.get('page', '未知')}页】\n{doc.page_content}"
        for doc in docs
    ])
    
    # ========== 步骤4：生成回答 ==========
    
    # 检查是否使用多轮对话模式
    if USE_MULTI_TURN and len(chat_history) > 0:
        # 多轮对话，使用多轮提示词
        prompt = MULTI_TURN_PROMPT.format(
            chat_history=format_chat_history(chat_history),
            context=context,
            question=question
        )
    else:
        # 单轮对话，使用单轮提示词
        prompt = SINGLE_TURN_PROMPT.format(
            context=context,
            question=question
        )
    
    # 调用LLM生成答案
    answer = _llm.invoke(prompt).strip()
    
    # ========== 步骤5：返回结果 ==========
    
    # 返回包含答案和元数据的字典
    return {
        "answer": answer,              # 生成的答案
        "source_docs": docs,           # 检索到的文档
        "rewritten_question": rewritten_question  # 改写后的问题
    }


# ====================== 第五部分：Web交互界面 ======================
# 说明：使用Streamlit构建用户界面，实现可视化操作

def main():
    """
    主函数，构建Web应用
    
    作用：初始化所有组件，创建用户界面，处理用户交互
    
    工作流程：
        1. 配置页面
        2. 初始化系统组件
        3. 创建侧边栏（系统状态、知识库管理）
        4. 创建主界面（对话区域）
        5. 处理用户输入和系统响应
    """
    # ========== 页面配置 ==========
    
    # 设置页面配置
    # page_title: 页面标题
    # page_icon: 页面图标
    # layout: 页面布局（wide-宽屏）
    st.set_page_config(
        page_title="完整RAG问答系统",
        page_icon="📚",
        layout="wide"
    )
    
    # 设置页面标题
    st.title("📚 完整RAG文档问答系统")
    
    # 设置页面副标题
    st.caption("基于LangChain + Chroma + Ollama 全本地部署，整合离线知识库与在线问答全链路")
    
    # ========== 侧边栏：系统状态与知识库管理 ==========
    
    # 创建侧边栏
    with st.sidebar:
        # 侧边栏标题
        st.header("⚙️ 系统配置与状态")
        
        # --- 1. 系统状态检查 ---
        
        # 创建可折叠区域
        with st.expander("系统状态检查", expanded=True):
            # 检查Ollama服务状态
            ollama_status, ollama_msg = check_ollama_running()
            
            # 根据检查结果显示不同颜色和图标
            if ollama_status:
                # 成功：显示绿色成功消息
                st.success(ollama_msg)
            else:
                # 失败：显示红色错误消息
                st.error(ollama_msg)
        
        # --- 2. 初始化核心组件 ---
        
        # 显示加载提示
        with st.spinner("正在初始化系统组件..."):
            # 初始化嵌入模型（全局缓存）
            embeddings = init_embeddings()
            
            # 初始化大模型（全局缓存）
            llm = init_llm()
            
            # 初始化重排序模型（如果启用）
            reranker = init_reranker() if USE_RERANKER else None
            
            # 初始化向量库
            vector_db, all_docs = init_vector_db(embeddings)
            
            # 初始化BM25检索器
            bm25_retriever = init_bm25_retriever(all_docs)
            
            # 构建最终检索器
            final_retriever = build_final_retriever(vector_db, bm25_retriever, reranker)
        
        # --- 3. 知识库管理 ---
        
        # 添加分隔线
        st.divider()
        
        # 侧边栏标题
        st.header("📂 知识库管理")
        
        # 显示知识库信息
        st.info(f"知识库文件夹：{KNOWLEDGE_BASE_DIR}")
        st.info(f"当前知识库文档数量：{len(list(set([meta.get('file_name', '') for meta in all_docs['metadatas']])))} 个")
        st.info(f"当前知识库分块数量：{len(all_docs['documents'])} 个")
        
        # 创建更新按钮
        # button() - 创建按钮
        # type: 按钮类型（primary-主要按钮）
        # use_container_width: 占满容器宽度
        if st.button("🔄 更新知识库", type="primary", use_container_width=True):
            # 显示加载提示
            with st.spinner("正在更新知识库..."):
                # 更新知识库
                chunks, msg = build_or_update_knowledge_base(embeddings, vector_db, all_docs)
                
                # 检查是否有新增分块
                if chunks:
                    # 显示成功消息
                    st.success(msg)
                    
                    # 清除缓存，重新加载组件
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    # 显示警告消息
                    st.warning(msg)
        
        # --- 4. 对话管理 ---
        
        # 添加分隔线
        st.divider()
        
        # 侧边栏标题
        st.header("💬 对话管理")
        
        # 创建清空对话按钮
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            # 清空对话历史
            st.session_state.messages = []
            # 重新运行页面
            st.rerun()
        
        # --- 5. 功能开关说明 ---
        
        # 添加分隔线
        st.divider()
        
        # 侧边栏标题
        st.header("✅ 已启用功能")
        
        # 显示各功能开关状态
        if USE_MIXED_RETRIEVAL:
            st.write("☑️ 混合检索（向量+BM25关键词）")
        if USE_RERANKER:
            st.write("☑️ 检索重排序")
        if USE_MULTI_TURN:
            st.write("☑️ 多轮对话+问题改写")
        if USE_CONTEXT_COMPRESSION:
            st.write("☑️ 上下文压缩")
    
    # ========== 主界面：对话区域 ==========
    
    # 初始化对话历史
    # st.session_state - Streamlit的会话状态，用于在页面刷新时保持数据
    if "messages" not in st.session_state:
        # 如果不存在，初始化为空列表
        st.session_state.messages = []
    
    # 显示历史对话
    # 遍历对话历史，显示每条消息
    for msg in st.session_state.messages:
        # 创建聊天消息框
        with st.chat_message(msg["role"]):
            # 显示消息内容
            st.markdown(msg["content"])
            
            # 如果是助手回复，显示源文档
            if msg["role"] == "assistant" and "source_docs" in msg:
                # 创建可折叠区域
                with st.expander("📄 查看检索到的源文档片段"):
                    # 遍历源文档
                    for i, doc in enumerate(msg["source_docs"]):
                        # 显示文档标题
                        st.markdown(f"**【源文档 {i+1}】来源：{doc.metadata.get('file_name', '未知')} 第{doc.metadata.get('page', '未知')}页**")
                        # 显示文档内容（代码格式）
                        st.code(doc.page_content, language="text")
    
    # 用户输入问题
    # chat_input() - 创建聊天输入框
    user_question = st.chat_input("请输入你的问题...")
    
    # 如果用户输入了问题
    if user_question:
        # 显示用户问题
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # 将用户消息添加到对话历史
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        # 调用RAG核心问答
        with st.chat_message("assistant"):
            # 显示加载提示
            with st.spinner("正在检索文档并生成回答..."):
                # 调用问答函数
                result = rag_answer(
                    question=user_question,
                    chat_history=st.session_state.messages[:-1],  # 传入除当前消息外的历史
                    _llm=llm,
                    _final_retriever=final_retriever
                )
            
            # 显示回答
            st.markdown(result["answer"])
            
            # 如果有多轮对话改写，显示改写后的问题
            if result["rewritten_question"]:
                st.caption(f"🔍 检索用改写问题：{result['rewritten_question']}")
            
            # 显示源文档
            with st.expander("📄 查看检索到的源文档片段"):
                for i, doc in enumerate(result["source_docs"]):
                    st.markdown(f"**【源文档 {i+1}】来源：{doc.metadata.get('file_name', '未知')} 第{doc.metadata.get('page', '未知')}页**")
                    st.code(doc.page_content, language="text")
        
        # 将助手回复添加到对话历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "source_docs": result["source_docs"],
            "rewritten_question": result["rewritten_question"]
        })


# ====================== 程序入口 ======================
# 说明：Python程序的入口点，启动Web应用

# __name__ 是Python内置变量
# 当直接运行该文件时，__name__ 的值为 "__main__"
# 当被其他文件导入时，__name__ 的值为模块名
if __name__ == "__main__":
    # 调用主函数，启动应用
    main()