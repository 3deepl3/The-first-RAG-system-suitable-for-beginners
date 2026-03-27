"""
问答链模块
作用：整合在线推理全流程，从问题到答案
"""

from config import USE_MULTI_TURN
from .prompts import (
    SINGLE_TURN_PROMPT,
    MULTI_TURN_PROMPT,
    QUERY_REWRITE_PROMPT,
    format_chat_history,
    format_source_docs
)


class QAChain:
    """
    问答链类
    负责从问题到答案的完整流程
    """
    
    def __init__(self, llm, retriever):
        """
        初始化问答链
        
        参数：
            llm: 大语言模型对象
            retriever: 检索器对象
        """
        self.llm = llm
        self.retriever = retriever
    
    def answer(self, question, chat_history=None):
        """
        回答用户问题
        
        参数：
            question (str): 用户当前问题
            chat_history (list): 对话历史列表
        
        返回：
            dict: 包含答案和元数据的字典
                - answer: 生成的答案
                - source_docs: 检索到的文档列表
                - rewritten_question: 改写后的问题（多轮模式下）
        """
        if chat_history is None:
            chat_history = []
        
        # 步骤1：问题改写（多轮对话）
        rewritten_question = self._rewrite_question(question, chat_history)
        
        # 步骤2：检索相关文档
        docs = self.retriever.invoke(rewritten_question)
        
        # 步骤3：格式化上下文
        context = format_source_docs(docs)
        
        # 步骤4：生成回答
        prompt = self._build_prompt(question, chat_history, context)
        answer = self.llm.invoke(prompt).strip()
        
        # 步骤5：返回结果
        return {
            "answer": answer,
            "source_docs": docs,
            "rewritten_question": rewritten_question if rewritten_question != question else None
        }
    
    def _rewrite_question(self, question, chat_history):
        """
        改写问题（多轮对话模式）
        
        参数：
            question (str): 原始问题
            chat_history (list): 对话历史
        
        返回：
            str: 改写后的问题
        """
        # 如果不启用多轮对话或没有历史记录，直接返回原问题
        if not USE_MULTI_TURN or len(chat_history) == 0:
            return question
        
        # 格式化对话历史
        formatted_history = format_chat_history(chat_history)
        
        # 生成问题改写提示词
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=formatted_history,
            question=question
        )
        
        # 调用LLM进行问题改写
        rewritten = self.llm.invoke(rewrite_prompt).strip()
        
        return rewritten
    
    def _build_prompt(self, question, chat_history, context):
        """
        构建提示词
        
        参数：
            question (str): 用户问题
            chat_history (list): 对话历史
            context (str): 检索到的上下文
        
        返回：
            str: 完整的提示词
        """
        # 根据是否有历史记录选择提示词模板
        if USE_MULTI_TURN and len(chat_history) > 0:
            return MULTI_TURN_PROMPT.format(
                chat_history=format_chat_history(chat_history),
                context=context,
                question=question
            )
        else:
            return SINGLE_TURN_PROMPT.format(
                context=context,
                question=question
            )


# 便捷函数
def create_qa_chain(llm, retriever):
    """
    便捷函数：创建问答链
    
    参数：
        llm: 大语言模型对象
        retriever: 检索器对象
    
    返回：
        QAChain: 问答链对象
    """
    return QAChain(llm, retriever)
