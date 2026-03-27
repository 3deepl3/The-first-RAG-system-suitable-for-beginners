"""
生成模块
包含提示词模板、问答链等功能
"""

from .prompts import (
    SINGLE_TURN_PROMPT,
    MULTI_TURN_PROMPT,
    QUERY_REWRITE_PROMPT
)
from .qa_chain import QAChain

__all__ = [
    'SINGLE_TURN_PROMPT',
    'MULTI_TURN_PROMPT',
    'QUERY_REWRITE_PROMPT',
    'QAChain'
]
