"""
系统检查工具模块
作用：提供系统健康检查功能
"""

from langchain_community.llms import Ollama
from config import LLM_MODEL


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
