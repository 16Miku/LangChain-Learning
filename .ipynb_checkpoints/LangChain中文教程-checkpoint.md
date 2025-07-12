# LangChain 中文教程：从原理到实战 (v2.0 - Gemini 版)

## 引言：什么是 LangChain？

LangChain 是一个开源的、功能强大的框架，旨在简化和增强基于大型语言模型 (LLM) 的**上下文感知推理 (Context-aware Reasoning)** 应用程序的开发。它不仅仅是一个模型包装器，更是一个从开发、生产到部署的全生命周期应用框架。

随着框架的成熟，LangChain 已经演变成一个更加模块化的生态系统，其核心功能被拆分到不同的包中（如 `langchain-core`, `langchain-community`, `langchain-google-genai`），使得开发者可以只引入需要的部分，保持项目的轻量化。

**为什么使用 LangChain？**

1.  **组件化与标准化**: 提供标准化的接口和可组合的构建块（组件），让复杂的应用构建变得清晰。
2.  **LangChain 表达式语言 (LCEL)**: 以声明式的方式构建和组合链，原生支持流式传输、并行执行和异步处理，是现代 LangChain 开发的核心。
3.  **数据感知 (Data-aware)**: 能够轻松连接到各种外部数据源（PDF、数据库、API），构建强大的**检索增强生成 (RAG)** 应用，让 LLM 基于私有数据或特定领域知识进行回答。
4.  **代理能力 (Agentic)**: 允许 LLM 使用工具（如搜索引擎、计算器、代码执行器），使其能够与环境交互并自主决策以完成复杂任务。
5.  **生态系统支持**: 拥有 **LangSmith** 用于调试、监控和评估，以及 **LangServe** 用于快速将应用部署为 API，构成了完整的开发运维闭环。

---

## LangChain 表达式语言 (LCEL)

LCEL 是 LangChain 的灵魂，它提供了一种声明式的方法来构建生产级的应用程序。通过类似管道符 `|` 的操作，可以将不同的组件无缝地“链接”在一起。所有使用 LCEL 构建的链都自动拥有同步、异步、批处理和流式处理的能力。

一个最基础的 LCEL 链如下所示：

```python
# 安装必要的库
# pip install langchain langchain-google-genai

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 设置您的 Google API 密钥
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# 1. 定义提示模板
prompt = ChatPromptTemplate.from_template(
    "你是一位世界级的历史学家。请回答以下问题：\n问题: {question}"
)

# 2. 初始化模型
# 使用 Gemini 1.5 Flash 模型，它速度快且成本效益高
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0)

# 3. 定义输出解析器
output_parser = StrOutputParser()

# 4. 使用 LCEL | 符号构建链
chain = prompt | model | output_parser

# 5. 调用链
question = "亚历山大大帝的老师是谁？"
response = chain.invoke({"question": question})

print(response)
# 输出: 亚历山大大帝的老师是古希腊著名哲学家亚里士多德。
```

这个例子中，`prompt` 的输出被“管道”给了 `model`，`model` 的输出又被“管道”给了 `output_parser`，形成了一个清晰、可读的数据流。

---

## LangChain 的核心组件

### 1. 模型 (Models)

LangChain 将模型抽象为标准接口，主要分为两类：

*   **聊天模型 (Chat Models)**: 这是目前主流的模型接口，它接收一个聊天消息列表 (`list[BaseMessage]`) 作为输入，返回一个 `BaseMessage`。Gemini 就是一个典型的聊天模型。
*   **文本嵌入模型 (Text Embedding Models)**: 将文本转换为向量（浮点数列表），用于语义表示和相似度计算，是 RAG 的基础。

**示例：使用 Gemini 模型**

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# 聊天模型
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")
messages = [
    SystemMessage(content="你是一个专业的营养师，请用中文回答。"),
    HumanMessage(content="一个成年男性每天需要摄入多少克蛋白质？")
]
response = chat.invoke(messages)
print(response.content)

# 文本嵌入模型
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embedding = embeddings_model.embed_query("你好，LangChain！")
print(embedding[:5]) # 打印向量的前5个维度
```

### 2. 提示 (Prompts)

提示模板负责动态地构建模型输入。`ChatPromptTemplate` 是最常用的模板，它可以组合不同角色的消息（`System`, `Human`, `AI`）。

```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的翻译官，请将以下英文翻译成中文。"),
    ("human", "{text}")
])

prompt_value = prompt_template.invoke({"text": "Hello, world!"})
print(prompt_value)
```

### 3. 输出解析器 (Output Parsers)

输出解析器负责将 LLM 返回的原始输出（通常是字符串或 `AIMessage`）转换为更易于使用的格式，例如结构化的 JSON 对象或 Python 数据类。

*   `StrOutputParser`: 最简单的解析器，将输出直接转为字符串。
*   `JsonOutputParser`: 将模型的 JSON 字符串输出解析为 Python 字典。

### 4. 检索 (Retrieval) - 构建 RAG 应用

检索是 LangChain 实现“数据感知”能力的核心，是构建 RAG (Retrieval-Augmented Generation) 应用的关键。它让 LLM 能够访问和利用外部私有数据。

**RAG 的工作流程**: 

1.  **加载 (Load)**: 使用 **文档加载器 (Document Loaders)** 从各种来源（PDF, TXT, 网页, 数据库）加载数据为 `Document` 对象。
2.  **分割 (Split)**: 使用 **文本分割器 (Text Splitters)** 将长文档切分成更小的、语义相关的块 (chunks)，以适应模型上下文窗口。
3.  **存储 (Store)**: 使用 **文本嵌入模型** 将每个文本块转换为向量，并与原文一起存入 **向量存储 (Vector Stores)** 中（如 FAISS, Chroma, Pinecone）。
4.  **检索 (Retrieve)**: 当用户提问时，**检索器 (Retriever)** 接收用户的问题，将其向量化，然后在向量存储中进行相似度搜索，找出与问题最相关的文档块。
5.  **生成 (Generate)**: 将用户的问题和检索到的相关文档块一起放入提示中，交给 LLM，LLM 会基于这些上下文信息生成最终答案。

### 5. 代理 (Agents)

代理让 LLM 具备了决策和执行能力。代理内部的 LLM 作为一个“推理引擎”，可以使用一系列 **工具 (Tools)** 来与外部世界交互。

*   **工具 (Tools)**: 工具是代理可以执行的具体动作，例如：
    *   `DuckDuckGoSearchRun`: 进行网络搜索。
    *   `Calculator`: 进行数学计算。
    *   自定义函数或 API。
*   **ReAct 框架**: 代理通常遵循一个 “Reason and Act” (ReAct) 的模式，循环地进行 **思考 (Thought)** -> **行动 (Action)** -> **观察 (Observation)**，直到完成任务。

---

## 综合实例：使用 LCEL 构建 Gemini RAG 问答机器人

这个例子将展示如何使用最新的 LangChain 组件和 LCEL，构建一个能够回答 PDF 文档内容的 RAG 应用。

**目标**: 让用户可以就一个 PDF 文档进行提问，并得到基于文档内容的回答。

**前提**: 请先创建一个名为 `example.pdf` 的文件，或替换成你自己的 PDF 文件路径。

```python
# 安装必要的库
# pip install langchain langchain-google-genai faiss-cpu pypdf

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 设置您的 Google API 密钥
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# 1. 初始化模型和嵌入
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. 加载和分割文档
loader = PyPDFLoader("LangChain中文教程.md") # 替换成你的 PDF 文件
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = text_splitter.split_documents(docs)

# 3. 创建向量存储和检索器
vector_store = FAISS.from_documents(splitted_docs, embeddings)
retriever = vector_store.as_retriever()

# 4. 创建提示模板
prompt = ChatPromptTemplate.from_template('''
你是一个用于回答问题的AI助手。
请根据下面提供的上下文来回答用户的问题。

<context>
{context}
</context>

问题: {input}''')

# 5. 创建并组合链 (使用 LCEL)
# 创建一个将文档塞入提示的链
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建一个在检索之前传递问题的链
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 6. 调用链进行问答
question = "LangChain的核心组件有哪些？"
response = retrieval_chain.invoke({"input": question})

print("回答:", response["answer"])

# 我们可以检查一下检索到的上下文
print("\n--- 检索到的上下文 ---")
for i, doc in enumerate(response["context"]):
    print(f"\n## 文档 {i+1} ##")
    print(doc.page_content)
```

---

## LangChain 生态系统

*   **LangSmith**: 一个强大的调试、测试、评估和监控平台。它能让你可视化地追踪链和代理的每一步执行过程（输入、输出、调用的工具等），是排查问题和优化性能的必备工具。
*   **LangServe**: 一个用于将 LangChain 链和代理一键部署为生产级 REST API 的库。它自动处理并发、流式传输，并提供 API 文档页面。
*   **LangGraph**: LangChain 生态的最新成员，用于构建有状态的、多步骤的代理和应用。它将复杂的任务流表示为图（Graph），其中节点是计算单元，边是它们之间的连接，非常适合创建循环和需要更精细控制流程的代理。

## 总结与后续学习

LangChain 已经从一个简单的“链”式框架，发展成为一个成熟的、端到端的 LLM 应用开发平台。其核心优势在于**模块化的组件**和强大的 **LCEL**，使得开发者可以快速、灵活地构建从简单的 RAG 应用到复杂的多代理系统。

要深入学习，强烈建议访问 [LangChain 官方 Python 文档](https://python.langchain.com/)，那里有最全面、最及时的信息、教程和 API 参考。