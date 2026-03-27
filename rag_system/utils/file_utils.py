"""
文件处理工具模块
作用：提供文件操作相关的通用函数
"""

import hashlib
import os


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
        1. 创建MD5哈希对象
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


def get_file_extension(file_path):
    """
    获取文件扩展名（小写）
    
    参数：
        file_path (str): 文件路径
    
    返回：
        str: 文件扩展名（如.pdf, .docx）
    """
    # os.path.splitext() - 分离文件名和扩展名，返回 (name, ext)
    return os.path.splitext(file_path)[1].lower()


def get_file_name(file_path):
    """
    获取文件名（不含路径）
    
    参数：
        file_path (str): 文件路径
    
    返回：
        str: 文件名
    """
    # os.path.basename() - 提取文件名部分
    return os.path.basename(file_path)


def is_supported_format(file_path, supported_formats):
    """
    检查文件格式是否支持
    
    参数：
        file_path (str): 文件路径
        supported_formats (list): 支持的格式列表
    
    返回：
        bool: 是否支持
    """
    extension = get_file_extension(file_path)
    return extension in supported_formats


def ensure_directory_exists(directory_path):
    """
    确保目录存在，不存在则创建
    
    参数：
        directory_path (str): 目录路径
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
