📚 完整模块化RAG问答系统

基于 LangChain + Chroma + Ollama 的全本地化 RAG（检索增强生成）问答系统，支持混合检索、重排序、多轮对话等高级功能。

✨ 特性

🏠 全本地部署：无需联网，保护数据隐私
🔍 混合检索：向量检索 + BM25 关键词检索
🎯 智能重排序：使用 BGE-Reranker 提升检索精度
💬 多轮对话：支持上下文理解和问题改写
📄 多格式支持：支持 PDF、Word、Markdown、TXT 等文档
🎨 Web界面：基于 Streamlit 的交互式界面
🔧 高度可配置：所有参数可通过 .env 文件配置

🏗️ 系统架构

rag_system/
├── config.py           # 配置文件(优先读取 .env)
├── main.py             # Streamlit 主程序
├── 1.knowledge/          # 知识库模块
│   ├── document_loader.py   # 文档加载
│   ├── chunker.py          # 文档分块
│   └── knowledge_base.py   # 知识库管理
├── 2.retrieval/          # 检索模块
│   ├── retriever.py        # 检索器(混合检索)
│   └── reranker.py         # 重排序器
├── 3.generation/         # 生成模块
│   ├── prompts.py          # 提示词模板
│   └── qa_chain.py         # 问答链
└── 4.utils/             # 工具模块
    ├── file_utils.py       # 文件工具
    └── system_utils.py     # 系统工具


🚀 快速开始

环境要求

Python 3.8+
Ollama（本地大模型运行环境）

安装步骤

1. 克隆仓库

bash
git clone https://github.com/your-username/rag-system.git
cd rag-system


2. 创建虚拟环境（推荐）

bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate


3. 安装依赖

bash
pip install -r requirements.txt


4. 安装 Ollama

访问 Ollama 官网 下载安装，或使用命令：

bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# 下载运行模型
ollama pull qwen2.5:3b


5. 配置环境变量

bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，修改以下关键配置：
# - KNOWLEDGE_BASE_DIR: 您的知识库文件夹路径
# - LLM_MODEL: 您使用的 Ollama 模型名称
# - DEVICE: cpu 或 cuda（如果有 GPU）


6. 准备知识库

将您的文档（PDF、Word、MD、TXT）放入 data/knowledge_base/ 文件夹（或您在 .env 中配置的路径）。

7. 运行系统

bash
streamlit run rag_system/main.py


浏览器会自动打开 http://localhost:8501

📖 配置说明

核心参数

参数	说明	推荐值
EMBEDDING_MODEL	嵌入模型	BAAI/bge-small-zh-v1.5
RERANKER_MODEL	重排序模型	BAAI/bge-reranker-base
LLM_MODEL	大语言模型	qwen2.5:3b
KNOWLEDGE_BASE_DIR	知识库路径	./data/knowledge_base
CHUNK_SIZE	分块大小	500
TOP_K_FINAL	最终召回数量	3

高级功能开关

USE_MIXED_RETRIEVAL: 启用混合检索（向量+BM25）
USE_RERANKER: 启用重排序
USE_MULTI_TURN: 启用多轮对话
USE_CONTEXT_COMPRESSION: 启用上下文压缩

🎯 使用示例

单轮问答

直接在聊天界面输入问题，系统会自动检索相关文档并生成回答。

多轮对话

系统支持上下文理解，可以基于前文进行追问。

示例：

plaintext
用户: 什么是 RAG？
系统: RAG 是检索增强生成技术...

用户: 它有哪些应用场景？
系统: 基于上文提到的 RAG 技术，它的主要应用场景包括...


查看源文档

每次回答都会显示检索到的源文档片段，方便追溯信息来源。

🔧 故障排查

问题：Ollama 连接失败

解决方案：

确保 Ollama 已启动：ollama serve
检查模型是否已下载：ollama list
运行测试：curl http://localhost:11434/api/tags

问题：知识库更新失败

解决方案：

检查 KNOWLEDGE_BASE_DIR 路径是否正确
确认文档格式是否支持（.pdf, .docx, .md, .txt）
查看错误日志获取详细信息

问题：检索结果不准确

解决方案：

调整 CHUNK_SIZE 和 CHUNK_OVERLAP 参数
尝试启用重排序：USE_RERANKER=True
调整混合检索权重：VECTOR_WEIGHT 和 BM25_WEIGHT

📊 技术栈

框架: LangChain
向量数据库: ChromaDB
嵌入模型: BGE (Beijing Academy of Artificial Intelligence)
重排序模型: BGE-Reranker
大语言模型: Ollama (支持 Qwen、Llama 等)
Web框架: Streamlit

🤝 贡献

欢迎提交 Issue 和 Pull Request！

📄 许可证

MIT License

🙏 致谢

LangChain
ChromaDB
Ollama
BGE

📞 联系方式

如有问题，欢迎通过以下方式联系：


邮箱: 1811325263@qq.com

⭐ 如果这个项目对您有帮助，请给个 Star！
