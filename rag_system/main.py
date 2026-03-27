"""
RAG完整系统主程序
作用：Web用户界面，提供交互式问答体验
"""

import streamlit as st
from config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    DEVICE,
    USE_RERANKER,
    USE_MIXED_RETRIEVAL,
    USE_MULTI_TURN,
    USE_CONTEXT_COMPRESSION
)
from utils import check_ollama_running
from knowledge import KnowledgeBase
from retrieval import build_retriever, create_reranker
from generation import create_qa_chain


def init_system_components():
    """
    初始化系统核心组件
    
    返回：
        dict: 包含所有核心组件的字典
    """
    # 初始化嵌入模型
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )
    
    # 初始化大模型
    from langchain_community.llms import Ollama
    llm = Ollama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )
    
    # 初始化知识库
    knowledge_base = KnowledgeBase(embeddings)
    vector_db, all_docs = knowledge_base.load_vector_db()
    
    # 初始化重排序器（如果启用）
    reranker = None
    if USE_RERANKER:
        reranker = create_reranker(model_name=RERANKER_MODEL)
    
    # 构建最终检索器
    bm25_retriever = knowledge_base.get_bm25_retriever()
    final_retriever = build_retriever(vector_db, bm25_retriever, reranker)
    
    # 创建问答链
    qa_chain = create_qa_chain(llm, final_retriever)
    
    return {
        "knowledge_base": knowledge_base,
        "llm": llm,
        "vector_db": vector_db,
        "retriever": final_retriever,
        "qa_chain": qa_chain
    }


def render_sidebar(components):
    """
    渲染侧边栏
    
    参数：
        components (dict): 系统组件字典
    """
    with st.sidebar:
        st.header("⚙️ 系统配置与状态")
        
        # 系统状态检查
        with st.expander("系统状态检查", expanded=True):
            ollama_status, ollama_msg = check_ollama_running()
            if ollama_status:
                st.success(ollama_msg)
            else:
                st.error(ollama_msg)
        
        st.divider()
        
        # 知识库管理
        st.header("📂 知识库管理")
        knowledge_base = components["knowledge_base"]
        
        # 显示知识库信息
        stats = knowledge_base.get_stats()
        from config import KNOWLEDGE_BASE_DIR
        st.info(f"知识库文件夹：{KNOWLEDGE_BASE_DIR}")
        st.info(f"当前知识库文档数量：{stats['document_count']} 个")
        st.info(f"当前知识库分块数量：{stats['chunk_count']} 个")
        
        # 更新知识库按钮
        if st.button("🔄 更新知识库", type="primary", use_container_width=True):
            with st.spinner("正在更新知识库..."):
                chunks, msg = knowledge_base.update_knowledge_base()
                if chunks:
                    st.success(msg)
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.warning(msg)
        
        st.divider()
        
        # 对话管理
        st.header("💬 对话管理")
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # 功能开关说明
        st.header("✅ 已启用功能")
        if USE_MIXED_RETRIEVAL:
            st.write("☑️ 混合检索（向量+BM25关键词）")
        if USE_RERANKER:
            st.write("☑️ 检索重排序")
        if USE_MULTI_TURN:
            st.write("☑️ 多轮对话+问题改写")
        if USE_CONTEXT_COMPRESSION:
            st.write("☑️ 上下文压缩")


def render_chat_interface(components):
    """
    渲染聊天界面
    
    参数：
        components (dict): 系统组件字典
    """
    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 显示历史对话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # 如果是助手回复，显示源文档
            if msg["role"] == "assistant" and "source_docs" in msg:
                with st.expander("📄 查看检索到的源文档片段"):
                    for i, doc in enumerate(msg["source_docs"]):
                        st.markdown(
                            f"**【源文档 {i+1}】来源：{doc.metadata.get('file_name', '未知')} "
                            f"第{doc.metadata.get('page', '未知')}页**"
                        )
                        st.code(doc.page_content, language="text")
    
    # 用户输入
    user_question = st.chat_input("请输入你的问题...")
    
    if user_question:
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # 添加到对话历史
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        # 生成回答
        qa_chain = components["qa_chain"]
        with st.chat_message("assistant"):
            with st.spinner("正在检索文档并生成回答..."):
                result = qa_chain.answer(
                    question=user_question,
                    chat_history=st.session_state.messages[:-1]
                )
            
            # 显示回答
            st.markdown(result["answer"])
            
            # 显示改写后的问题
            if result["rewritten_question"]:
                st.caption(f"🔍 检索用改写问题：{result['rewritten_question']}")
            
            # 显示源文档
            with st.expander("📄 查看检索到的源文档片段"):
                for i, doc in enumerate(result["source_docs"]):
                    st.markdown(
                        f"**【源文档 {i+1}】来源：{doc.metadata.get('file_name', '未知')} "
                        f"第{doc.metadata.get('page', '未知')}页**"
                    )
                    st.code(doc.page_content, language="text")
        
        # 添加助手回复到对话历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "source_docs": result["source_docs"],
            "rewritten_question": result["rewritten_question"]
        })


def main():
    """
    主函数
    """
    # 页面配置
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT
    )
    
    # 页面标题
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("基于LangChain + Chroma + Ollama 全本地部署，整合离线知识库与在线问答全链路")
    
    # 初始化系统组件
    with st.spinner("正在初始化系统组件..."):
        components = init_system_components()
    
    # 渲染侧边栏
    render_sidebar(components)
    
    # 渲染聊天界面
    render_chat_interface(components)


if __name__ == "__main__":
    main()
