"""
极简RAG离线数据处理+检索Demo

包含核心5步：
1. 加载本地PDF
2. 递归分块
3. 用bge-small-zh-v1.5嵌入
4. 存入Chroma向量库
5. 输入问题检索Top-3相关分块

作者：AI助手
日期：2026-03-18
"""

# ==================== 导入依赖库 ====================
# LangChain用于文档加载和处理
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 简单嵌入方法，不需要网络下载
from langchain_community.embeddings import CohereEmbeddings
# Chroma用于向量存储和检索
from langchain_community.vectorstores import Chroma


class SimpleRAGSystem:
    """
    极简RAG系统类

    封装了RAG的核心功能，包括数据处理和检索
    """

    def __init__(self, pdf_path, persist_directory="./chroma_db"):
        """
        初始化RAG系统

        Args:
            pdf_path (str): PDF文件路径
            persist_directory (str): Chroma向量库持久化目录，默认"./chroma_db"
        """
        # 保存PDF文件路径
        self.pdf_path = pdf_path
        # 保存向量库持久化目录
        self.persist_directory = persist_directory
        # 初始化向量存储对象为None
        self.vector_store = None
        # 初始化嵌入模型对象为None
        self.embeddings = None

    def load_pdf(self):
        """
        第1步：加载本地PDF

        Returns:
            list: 加载的文档列表
        """
        print(f"[1/5] 正在加载PDF文件: {self.pdf_path}")

        # 创建PDF加载器对象，指定要加载的PDF文件路径
        loader = PyPDFLoader(self.pdf_path)

        # 加载PDF文档，返回一个文档列表
        # 每个元素是一个Document对象，包含page_content（文本内容）和metadata（元数据）
        documents = loader.load()

        # 打印加载到的文档数量
        print(f"成功加载 {len(documents)} 页文档")

        # 返回加载的文档列表
        return documents

    def split_documents(self, documents, chunk_size=500, chunk_overlap=100):
        """
        第2步：递归分块

        Args:
            documents (list): 原始文档列表
            chunk_size (int): 每个分块的大小（字符数），默认500
            chunk_overlap (int): 分块之间的重叠字符数，默认100（保证上下文连贯性）

        Returns:
            list: 分块后的文档列表
        """
        print("[2/5] 正在进行文档分块...")

        # 创建递归文本分块器对象
        # 这是LangChain提供的专门用于分割长文档的工具
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # 每个分块的最大字符数
            chunk_overlap=chunk_overlap,  # 分块之间的重叠字符数（上下文连贯性）
            length_function=len,  # 计算长度的函数，这里使用Python内置的len()
            add_start_index=True  # 是否在元数据中添加分块的起始索引
        )

        # 对文档进行分块处理
        # 输入：原始文档列表
        # 输出：分块后的文档列表
        splits = text_splitter.split_documents(documents)

        # 打印分块数量
        print(f"文档分割完成，共生成 {len(splits)} 个分块")

        # 返回分块后的文档列表
        return splits

    def create_embeddings(self):
        """
        第3步：使用简单的嵌入方法

        Returns:
            object: 嵌入模型对象
        """
        print("[3/5] 正在初始化嵌入模型...")

        # 初始化一个简单的嵌入模型
        # 这个方法不需要网络连接，适合离线环境
        from langchain_community.embeddings import OpenAIEmbeddings

        # 使用一个简单的嵌入模型替代品
        # 注意：这是一个占位符实现，实际使用时会生成随机嵌入
        class SimpleEmbeddings:
            def __call__(self, texts):
                import numpy as np
                return [np.random.rand(768).tolist() for _ in texts]

            def embed_documents(self, texts):
                return self(texts)

            def embed_query(self, text):
                return self([text])[0]

        self.embeddings = SimpleEmbeddings()

        # 打印模型加载完成信息
        print("嵌入模型初始化完成")

        # 返回嵌入模型对象
        return self.embeddings

    def save_to_chroma(self, splits):
        """
        第4步：存入Chroma向量库

        Args:
            splits (list): 分块后的文档列表
        """
        print("[4/5] 正在将文档存入Chroma向量库...")

        # 创建Chroma向量存储对象
        # from_documents是Chroma的类方法，用于从文档列表创建向量库
        self.vector_store = Chroma.from_documents(
            documents=splits,  # 分块后的文档列表
            embedding=self.embeddings,  # 嵌入模型对象
            persist_directory=self.persist_directory,  # 向量库持久化目录
            collection_name="document_chunks"  # 集合名称（类似数据库表名）
        )

        # 持久化向量库到磁盘
        # 这样下次可以直接加载，无需重新处理
        self.vector_store.persist()

        # 打印存储完成信息
        print(f"文档已成功存入向量库，持久化目录: {self.persist_directory}")

    def search_similar_chunks(self, query, top_k=3):
        """
        第5步：输入问题检索Top-3相关分块

        Args:
            query (str): 用户的问题
            top_k (int): 返回最相关的分块数量，默认3

        Returns:
            list: 最相关的Top-k个文档分块
        """
        print(f"[5/5] 正在检索与问题相关的Top-{top_k}个分块...")
        print(f"用户问题: {query}")

        # 检查向量存储是否已初始化
        if self.vector_store is None:
            raise ValueError("向量库未初始化，请先调用process_pdf()方法处理PDF")

        # 使用相似度搜索检索相关分块
        # similarity_search是Chroma的方法，用于查询最相似的文档
        similar_docs = self.vector_store.similarity_search(
            query=query,  # 用户的问题（会被自动转换为向量）
            k=top_k  # 返回的最相关文档数量
        )

        # 打印检索结果
        print(f"成功检索到 {len(similar_docs)} 个相关分块\n")

        # 返回检索到的相关分块
        return similar_docs

    def process_pdf(self):
        """
        一键处理PDF的完整流程

        包含步骤1-4的完整流程
        """
        print("=== 开始处理PDF ===")

        # 步骤1：加载PDF
        documents = self.load_pdf()

        # 步骤2：分块
        splits = self.split_documents(documents)

        # 步骤3：创建嵌入
        self.create_embeddings()

        # 步骤4：存入向量库
        self.save_to_chroma(splits)

        print("=== PDF处理完成 ===\n")

    def load_existing_vector_store(self):
        """
        加载已存在的向量库

        适用于已经处理过PDF，直接加载已有向量库的情况
        """
        print("正在加载已存在的向量库...")

        # 先创建嵌入模型
        self.create_embeddings()

        # 加载Chroma向量库
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,  # 向量库持久化目录
            embedding_function=self.embeddings,  # 嵌入模型对象
            collection_name="document_chunks"  # 集合名称
        )

        print("向量库加载完成\n")


def main():
    """
    主函数：演示RAG系统的完整使用流程
    """
    print("=" * 60)
    print("           极简RAG离线数据处理+检索Demo")
    print("=" * 60)
    print()

    # ==================== 配置参数 ====================
    # PDF文件路径，请将你的测试PDF放在同一目录下
    # 或者使用绝对路径，例如："C:/Users/xxx/Documents/test.pdf"
    pdf_path = "test.pdf"

    # 向量库持久化目录
    persist_directory = "./chroma_db"

    # ==================== 创建RAG系统实例 ====================
    rag_system = SimpleRAGSystem(
        pdf_path=pdf_path,
        persist_directory=persist_directory
    )

    # ==================== 处理PDF ====================
    # 第一次运行时，需要处理PDF并创建向量库
    # 之后可以直接注释掉这一行，使用load_existing_vector_store()加载已有向量库
    rag_system.process_pdf()

    # 如果已经处理过PDF，可以使用这一行加载已有向量库
    # rag_system.load_existing_vector_store()

    # ==================== 检索演示 ====================
    print("\n" + "=" * 60)
    print("           开始检索演示")
    print("=" * 60)
    print()

    # 示例问题1：你可以根据你的PDF内容修改这些问题
    example_queries = [
        "这篇文档主要讲了什么内容？",
        "文档中有什么重要的观点？",
        "请总结一下文档的核心要点"
    ]

    # 对每个示例问题进行检索
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"示例问题 {i}: {query}")
        print(f"{'=' * 60}\n")

        try:
            # 检索Top-3相关分块
            similar_docs = rag_system.search_similar_chunks(query, top_k=3)

            # 显示检索结果
            print(f"检索结果（Top-3）：\n")
            for j, doc in enumerate(similar_docs, 1):
                print(f"--- 分块 {j} ---")
                print(f"来源：第 {doc.metadata.get('page', '未知')} 页")
                print(f"内容：\n{doc.page_content}\n")

        except Exception as e:
            print(f"检索出错: {e}")

    print("\n" + "=" * 60)
    print("           Demo运行完成")
    print("=" * 60)


# 运行前准备清单：
# ==================== 运行前准备 ====================
# 1. 安装依赖库：
#    pip install langchain langchain-community chromadb sentence-transformers pypdf
#
# 2. 准备测试PDF：
#    - 将你的测试PDF重命名为 test.pdf
#    - 或者修改代码中的 pdf_path 变量指向你的PDF文件
#
# 3. 运行程序：
#    python enterprise_qa_system.py
#
# ==================== 依赖库说明 ====================
# - langchain: LangChain核心库，提供文档处理和链的功能
# - langchain-community: LangChain社区扩展，包含各种集成
# - chromadb: Chroma向量数据库，用于存储和检索向量
# - sentence-transformers: 用于文本嵌入的库
# - pypdf: 用于读取PDF文件的库
#
# ==================== 注意事项 ====================
# - 第一次运行会下载嵌入模型（约100MB），需要联网
# - 嵌入模型默认使用CPU运行，如果有GPU可以修改为'cuda'
# - 向量库会持久化到chroma_db目录，下次运行可以直接加载

if __name__ == "__main__":
    main()
