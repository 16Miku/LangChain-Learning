



# RAG-App-FastAPI


好的，作为一名高级软件工程师，我将指导您如何将之前构建的RAG应用程序（包括基于链和基于代理的版本）与FastAPI结合，将其部署为Web服务。这将使您的RAG应用能够通过HTTP请求进行访问，并支持对话记忆。

### 为什么选择 FastAPI？

*   **高性能**: FastAPI 基于 Starlette (用于Web部分) 和 Pydantic (用于数据验证和序列化)，提供了极高的性能，与 NodeJS 和 Go 相当。
*   **易于使用**: 现代且直观的API设计，利用 Python 类型提示，代码编写效率高。
*   **自动文档**: 自动生成交互式API文档 (Swagger UI 和 ReDoc)，方便测试和使用。
*   **异步支持**: 原生支持 `async/await`，非常适合处理I/O密集型任务，如LLM调用和数据库操作。

### 项目结构

我们将创建一个简单的项目结构：

```
rag_app_fastapi/
├── main.py             # FastAPI 应用程序的入口文件
├── .gitignore          # Git 忽略文件配置
└── chroma_langchain_db/ # Chroma 向量数据库的持久化目录 (会被 Git 忽略)
```

### 准备工作

1.  **确保所有依赖已安装**:
    请确保您已经安装了之前教程中提到的所有LangChain、LangGraph、Google GenAI相关的依赖。
    ```powershell
    pip install --quiet --upgrade langchain langchain-core langgraph langchain-text-splitters langchain-community langchain-chroma langchain-google-genai bs4 typing-extensions
    pip install -qU "langchain[google-genai]"
    pip install -qU langchain-google-genai
    pip install -qU langchain-chroma
    ```

2.  **安装 FastAPI 和 Uvicorn**:
    Uvicorn 是一个 ASGI (Asynchronous Server Gateway Interface) 服务器，用于运行 FastAPI 应用程序。
    ```powershell
    pip install fastapi uvicorn
    ```
    如果您希望获得更多可选的依赖项（如 `python-dotenv` 用于加载环境变量），可以使用 `uvicorn[standard]`：
    ```powershell
    pip install "uvicorn[standard]"
    ```

3.  **设置环境变量**:
    确保您的 `GOOGLE_API_KEY` 已经设置为环境变量。如果您想使用 LangSmith 跟踪，也请设置 `LANGSMITH_TRACING` 和 `LANGSMITH_API_KEY`。
    在 PowerShell 中，您可以这样做：
    ```powershell
    $env:GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    $env:LANGSMITH_TRACING="true"
    $env:LANGSMITH_API_KEY="YOUR_LANGSMITH_API_KEY"
    ```
    请将 `YOUR_GOOGLE_GEMINI_API_KEY` 和 `YOUR_LANGSMITH_API_KEY` 替换为您自己的实际密钥。

### `main.py` 文件内容

现在，我们将把RAG应用程序的初始化逻辑和FastAPI的接口定义结合起来。

```python
# main.py

import os
import getpass # 用于在开发时通过交互式输入获取API Key，生产环境应避免
import bs4
from typing import List, Dict, Any, Union

from fastapi import FastAPI, HTTPException # 导入 FastAPI 核心类和 HTTPException
from pydantic import BaseModel # 导入 Pydantic 用于数据模型定义

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver # 导入内存检查点器，用于会话记忆

# --- 全局变量声明 ---
# 这些变量将在应用启动时初始化，并在整个应用生命周期中共享。
llm: Any = None # 语言模型
embeddings: Any = None # 嵌入模型
vector_store: Any = None # 向量存储
rag_chain_graph: Any = None # 基于链的 RAG 图
rag_agent_executor: Any = None # 基于代理的 RAG 执行器
# MemorySaver 用于在不同请求之间持久化对话状态。
# 在生产环境中，您可能会使用 SQLiteSaver 或其他数据库支持的检查点器。
memory_saver: MemorySaver = MemorySaver()

# --- FastAPI 应用实例初始化 ---
app = FastAPI(
    title="RAG 应用与 FastAPI 结合", # API 标题
    description="一个使用 LangChain 和 LangGraph 进行对话式 RAG 的 Web 服务。", # API 描述
    version="1.0.0", # API 版本
)

# --- Pydantic 模型定义 ---
# 定义 HTTP 请求体的数据结构。
class ChatRequest(BaseModel):
    question: str # 用户提出的问题
    session_id: str = "default_session" # 会话ID，用于区分不同用户的对话历史，默认值为 "default_session"

# 定义 HTTP 响应体的数据结构。
class ChatResponse(BaseModel):
    answer: str # LLM 生成的答案
    session_id: str # 返回会话ID

# --- RAG 组件初始化 (FastAPI 启动事件) ---
# `@app.on_event("startup")` 装饰器确保 `startup_event` 函数在 FastAPI 应用程序启动时只运行一次。
# 这是加载和初始化所有耗时 RAG 组件的理想位置，避免每个请求都重新加载。
@app.on_event("startup")
async def startup_event():
    """
    FastAPI 启动时初始化所有 RAG 组件。
    这确保了 LLM、嵌入模型、向量存储和 RAG 图在服务器开始处理请求之前只加载一次。
    """
    print("--- 正在初始化 RAG 组件... ---")
    global llm, embeddings, vector_store, rag_chain_graph, rag_agent_executor

    # 1. API 密钥检查 (生产环境应使用更安全的配置管理)
    # 检查 GOOGLE_API_KEY 是否已设置。如果未设置，抛出错误。
    if not os.environ.get("GOOGLE_API_KEY"):
        # 在生产环境中，避免使用 getpass。请通过环境变量或配置服务安全地提供密钥。
        # 对于本地测试，您可以取消注释以下行，但请确保不要将密钥提交到版本控制。
        # os.environ["GOOGLE_API_KEY"] = getpass.getpass("请输入 Google Gemini 的 API 密钥: ")
        raise ValueError("GOOGLE_API_KEY 环境变量未设置。请设置您的 Google Gemini API 密钥。")

    # 检查 LangSmith 跟踪配置 (可选)
    if os.environ.get("LANGSMITH_TRACING") == "true" and not os.environ.get("LANGSMITH_API_KEY"):
        print("LangSmith 跟踪已启用，但 LANGSMITH_API_KEY 未设置。跟踪可能无法正常工作。")

    # 2. 初始化聊天模型 (LLM)
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    print("聊天模型 (LLM) 已初始化。")

    # 3. 初始化嵌入模型 (Embeddings Model)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print("嵌入模型 (Embeddings Model) 已初始化。")

    # 4. 初始化向量存储 (Chroma)
    # collection_name: 向量存储的集合名称。
    # embedding_function: 用于生成嵌入的函数。
    # persist_directory: 数据持久化到本地的目录，如果目录不存在会自动创建。
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    print("向量存储 (Chroma) 已初始化。")

    # 5. 索引阶段: 加载并分块文档
    print("正在加载和分块文档...")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    # 将文档添加到向量存储。`_` 表示我们不关心返回的文档ID。
    _ = vector_store.add_documents(documents=all_splits)
    print(f"文档已加载并索引: {len(all_splits)} 个文本块。")

    # 6. 定义检索工具
    # `@tool` 装饰器将普通 Python 函数转换为可被 LLM 调用的工具。
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """
        检索工具：根据查询从向量存储中检索相关信息。
        此工具被设计为由LLM调用，用于执行信息检索。

        Args:
            query (str): 用于搜索向量存储的查询字符串。

        Returns:
            tuple[str, List[Document]]:
                - str: 序列化后的文档内容，作为 ToolMessage 的主要内容。
                - List[Document]: 原始的检索到的文档对象列表，作为 ToolMessage 的附加工件。
        """
        print(f"--- 工具调用: retrieve，查询: '{query}' ---")
        # 从向量存储中执行相似性搜索，k=2 表示检索最相似的2个文档。
        retrieved_docs = vector_store.similarity_search(query, k=2)
        # 将检索到的文档内容格式化为字符串，每个文档包含其元数据和内容。
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        print(f"--- 工具 'retrieve' 执行完成，检索到 {len(retrieved_docs)} 篇文档 ---")
        return serialized, retrieved_docs

    # --- 7. 设置基于链的 RAG 应用程序 ---
    print("正在设置基于链的 RAG 应用程序...")
    # MessagesState 是 LangGraph 提供的一个特殊状态类型，它将所有消息存储在一个列表中。
    chain_graph_builder = StateGraph(MessagesState)

    # 节点 1: query_or_respond - 生成工具调用或直接响应
    def chain_query_or_respond(state: MessagesState):
        """
        LLM 根据对话历史和当前用户输入决定是调用检索工具还是直接回复。
        """
        print("--- 正在执行链节点: query_or_respond ---")
        llm_with_tools = llm.bind_tools([retrieve]) # 将 retrieve 工具绑定到 LLM
        response = llm_with_tools.invoke(state["messages"]) # LLM 根据消息决定行动
        return {"messages": [response]} # 将 LLM 响应追加到状态消息列表

    # 节点 2: tools - 执行检索工具
    chain_tools_node = ToolNode([retrieve]) # ToolNode 会自动执行 LLM 生成的工具调用

    # 节点 3: generate - 使用检索到的内容生成最终答案
    def chain_generate(state: MessagesState):
        """
        在工具执行完毕后，整合检索到的信息和对话历史来生成用户最终看到的答案。
        """
        print("--- 正在执行链节点: generate ---")
        recent_tool_messages = []
        # 从消息列表末尾向前遍历，查找类型为 "tool" 的消息，以获取当前回合的检索结果。
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1] # 反转列表，按时间顺序排列

        # 从工具消息中提取检索到的文档内容
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        
        # 构建系统消息，包含通用指令和检索到的上下文
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"Retrieved Context:\n{docs_content}"
        )
        # 过滤出用于生成最终答案的对话消息，排除 LLM 内部的工具调用思考过程。
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages # 构建最终提示
        response = llm.invoke(prompt) # 调用 LLM 生成最终响应
        print(f"--- 链节点 'generate' 完成，生成答案: {response.content[:50]}... ---")
        return {"messages": [response]} # 将生成的响应追加到状态消息列表

    # 添加节点到图构建器
    chain_graph_builder.add_node("query_or_respond", chain_query_or_respond)
    chain_graph_builder.add_node("tools", chain_tools_node)
    chain_graph_builder.add_node("generate", chain_generate)

    # 设置图的入口点
    chain_graph_builder.set_entry_point("query_or_respond")
    # 添加条件边：根据 query_or_respond 的输出决定下一步走向
    # tools_condition 检查 LLM 响应是否包含工具调用。
    # 如果有工具调用，转向 "tools" 节点；否则，图执行结束 (END)。
    chain_graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    # 添加常规边
    chain_graph_builder.add_edge("tools", "generate") # 工具执行后，总是转向生成答案
    chain_graph_builder.add_edge("generate", END) # 生成答案后，图执行结束

    # 编译图，并指定 checkpointer 以支持会话记忆
    rag_chain_graph = chain_graph_builder.compile(checkpointer=memory_saver)
    print("基于链的 RAG 应用程序已编译。")

    # --- 8. 设置基于代理的 RAG 应用程序 ---
    print("正在设置基于代理的 RAG 应用程序...")
    # create_react_agent 是 LangGraph 预构建的 ReAct 代理构造函数。
    # 它会自动处理 LLM 的思考和行动循环，包括多次工具调用。
    rag_agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory_saver)
    print("基于代理的 RAG 应用程序已编译。")

    print("--- RAG 组件初始化完成！ ---")

# --- FastAPI 接口定义 ---

# 定义根路径接口，用于健康检查或欢迎信息
@app.get("/")
async def read_root():
    """提供欢迎信息和可用接口指引。"""
    return {"message": "欢迎使用 RAG FastAPI 应用程序！请使用 /chat/chain 或 /chat/agent 接口。"}

# 基于链的 RAG 对话接口
@app.post("/chat/chain", response_model=ChatResponse)
async def chat_with_chain(request: ChatRequest):
    """
    使用基于链的方法进行对话式 RAG。
    通过 `session_id` 维护聊天历史。
    """
    if rag_chain_graph is None:
        # 如果 RAG 链未初始化，返回 503 Service Unavailable 错误。
        raise HTTPException(status_code=503, detail="RAG 链尚未初始化，请稍后再试。")

    print(f"\n--- 收到基于链的聊天请求 (会话ID: {request.session_id}): {request.question} ---")
    try:
        # 准备图的输入消息：将用户问题封装为 HumanMessage。
        input_messages = [HumanMessage(content=request.question)]
        
        # 配置图的运行参数，通过 `thread_id` 启用会话记忆。
        config = {"configurable": {"thread_id": request.session_id}}
        
        # 异步调用 RAG 链图。`ainvoke` 是异步版本。
        final_state = await rag_chain_graph.ainvoke(
            {"messages": input_messages}, # 传递初始消息状态
            config=config                 # 传递配置，包括会话ID
        )
        
        # 从最终状态中提取答案。
        answer = "未能生成直接答案。" # 默认答案
        if final_state and final_state["messages"]:
            last_message = final_state["messages"][-1] # 获取最后一条消息
            # 如果最后一条消息是 AIMessage 且不包含工具调用，则认为是最终答案。
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                answer = last_message.content
            else:
                # 这种情况通常不应该发生在链中，因为链会以 generate 节点结束。
                # 但作为通用处理，如果出现，记录警告。
                print(f"警告: 链的最后一条消息类型为 {type(last_message)} 或包含工具调用。内容: {last_message.content[:100]}...")
                answer = "生成答案失败，请检查日志或重试。" # 更友好的错误提示

        # 返回 ChatResponse 对象。
        return ChatResponse(answer=answer, session_id=request.session_id)
    except Exception as e:
        # 捕获并处理异常，返回 500 Internal Server Error。
        print(f"基于链的聊天发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

# 基于代理的 RAG 对话接口
@app.post("/chat/agent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    使用基于代理 (ReAct) 的方法进行对话式 RAG。
    通过 `session_id` 维护聊天历史。
    """
    if rag_agent_executor is None:
        # 如果 RAG 代理未初始化，返回 503 Service Unavailable 错误。
        raise HTTPException(status_code=503, detail="RAG 代理尚未初始化，请稍后再试。")

    print(f"\n--- 收到基于代理的聊天请求 (会话ID: {request.session_id}): {request.question} ---")
    try:
        # 准备代理的输入消息。
        input_messages = [HumanMessage(content=request.question)]

        # 配置代理的运行参数，通过 `thread_id` 启用会话记忆。
        config = {"configurable": {"thread_id": request.session_id}}

        # 异步调用 RAG 代理执行器。
        # 代理执行器会在完成所有思考和行动步骤后返回最终状态。
        final_state = await rag_agent_executor.ainvoke(
            {"messages": input_messages},
            config=config
        )

        # 从最终状态中提取答案。
        answer = "未能生成直接答案。" # 默认答案
        if final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            # 代理的最后一条消息可能是直接答案，也可能是工具调用（如果它认为还需要更多信息）。
            # 这里我们尝试获取其内容作为答案。
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                 answer = last_message.content
            else:
                # 如果代理以工具调用或其他非直接回复结束，我们仍然可以尝试返回其内容。
                # 在更复杂的代理设计中，您可能需要更精细的逻辑来处理这种情况。
                print(f"警告: 代理的最后一条消息类型为 {type(last_message)} 或包含工具调用。内容: {last_message.content[:100]}...")
                answer = last_message.content # 尝试返回其内容，即使是工具调用信息
        else:
            answer = "代理未生成任何响应。"

        return ChatResponse(answer=answer, session_id=request.session_id)
    except Exception as e:
        # 捕获并处理异常。
        print(f"基于代理的聊天发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

```

### 代码解读与实现细节

1.  **全局变量**:
    *   `llm`, `embeddings`, `vector_store`, `rag_chain_graph`, `rag_agent_executor`, `memory_saver` 都被声明为全局变量。这是因为这些组件的初始化成本较高，应该在应用程序启动时进行一次，并在所有请求中共享。
    *   `memory_saver = MemorySaver()`：创建了一个内存中的检查点器。在生产环境中，您应该替换为支持持久化的检查点器，例如 `SQLiteSaver`，以便在服务器重启后仍能保留对话历史。

2.  **FastAPI 应用程序实例**:
    *   `app = FastAPI(...)` 创建了 FastAPI 应用程序实例，并添加了标题、描述和版本，这些信息将显示在自动生成的API文档中。

3.  **`Pydantic` 模型**:
    *   `ChatRequest(BaseModel)`：定义了客户端发送给API的请求体结构。`question` 是用户的问题，`session_id` 是一个可选的字符串，用于标识不同的对话会话，默认值为 "default_session"。这个 `session_id` 对于维护对话记忆至关重要。
    *   `ChatResponse(BaseModel)`：定义了API返回给客户端的响应体结构，包含LLM生成的 `answer` 和 `session_id`。

4.  **`@app.on_event("startup")`**:
    *   这是一个FastAPI的生命周期事件装饰器。它修饰的 `startup_event` 异步函数会在FastAPI应用程序启动时自动运行一次。
    *   **重要性**: 所有RAG组件（LLM、嵌入模型、向量存储、文档加载与索引、LangGraph图的编译）都在这里完成。这样，当HTTP请求到达时，这些组件已经准备就绪，无需重复初始化，大大提高了响应速度。
    *   **API Key 检查**: 增加了对 `GOOGLE_API_KEY` 环境变量的检查，如果未设置则会抛出 `ValueError`，防止应用在没有密钥的情况下运行。

5.  **RAG 组件初始化**:
    *   这部分代码与之前教程中的初始化和索引步骤相同，确保了LLM、嵌入模型、向量存储和数据索引的正确设置。
    *   `@tool(response_format="content_and_artifact")` 装饰器将 `retrieve` 函数转换为FastAPI可用的工具。

6.  **基于链的 RAG 应用程序设置**:
    *   `chain_graph_builder = StateGraph(MessagesState)`：使用 `MessagesState` 来自动管理对话消息列表。
    *   `chain_query_or_respond`, `chain_tools_node`, `chain_generate`：这三个函数/节点构成了基于链的RAG逻辑，与教程第二部分中的非代理版本相同。
    *   `rag_chain_graph = chain_graph_builder.compile(checkpointer=memory_saver)`：编译图时传入 `memory_saver`，使得这个链支持会话记忆。

7.  **基于代理的 RAG 应用程序设置**:
    *   `rag_agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory_saver)`：使用LangGraph预构建的 `create_react_agent` 函数，一行代码即可创建一个ReAct代理，同样传入 `memory_saver` 来支持会话记忆。

8.  **FastAPI 接口 (`@app.post`)**:
    *   `@app.post("/chat/chain", response_model=ChatResponse)` 和 `@app.post("/chat/agent", response_model=ChatResponse)` 定义了两个 POST 接口，分别用于基于链和基于代理的RAG对话。
    *   `request: ChatRequest`：FastAPI 会自动解析传入的 JSON 请求体，并将其验证为 `ChatRequest` Pydantic 模型的实例。
    *   `config = {"configurable": {"thread_id": request.session_id}}`：这是关键的记忆集成点。`thread_id` 参数告诉LangGraph的 `checkpointer` 为哪个会话加载和保存状态。每次使用相同的 `session_id` 发送请求，都将继续同一个对话。
    *   `await rag_chain_graph.ainvoke(...)` 和 `await rag_agent_executor.ainvoke(...)`：使用异步 `ainvoke` 方法来运行LangGraph图或代理，这与FastAPI的异步特性相匹配，提高了并发处理能力。
    *   **答案提取**: 从 `final_state["messages"]` 中提取最后一条 `AIMessage` 的 `content` 作为最终答案。

### 运行 FastAPI 应用程序

1.  **保存文件**:
    将上述代码保存为 `rag_app_fastapi/main.py`。

2.  **打开终端/PowerShell**:
    导航到 `rag_app_fastapi` 目录。

3.  **启动 Uvicorn 服务器**:
    ```powershell
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    *   `main:app`: 指示 Uvicorn 运行 `main.py` 文件中的 `app` 实例。
    *   `--host 0.0.0.0`: 允许从任何IP地址访问服务器（在生产环境中可能需要更严格的配置）。
    *   `--port 8000`: 服务器将在 8000 端口监听请求。
    *   `--reload`: 在代码文件更改时自动重新加载服务器（开发时非常有用）。

    当您运行此命令时，您会看到控制台输出显示RAG组件的初始化过程。这可能需要一些时间来下载模型和索引文档。

### 测试 FastAPI 应用程序

服务器成功启动后，您可以通过以下方式测试您的API：

1.  **访问交互式 API 文档**:
    在浏览器中打开 `http://127.0.0.1:8000/docs` (Swagger UI) 或 `http://127.0.0.1:8000/redoc` (ReDoc)。
    您将看到 `/`、`/chat/chain` 和 `/chat/agent` 接口的详细说明。

2.  **使用 Swagger UI 进行测试**:

    *   **测试 `/chat/chain` (链式 RAG)**:
        1.  点击 `/chat/chain` 接口，然后点击 `Try it out` 按钮。
        2.  在 `Request body` 中，输入第一个问题和会话ID：
            ```json
            {
              "question": "What is Task Decomposition?",
              "session_id": "my_user_session_1"
            }
            ```
        3.  点击 `Execute`。您会收到一个答案。
        4.  再次点击 `Try it out`，输入第二个问题，**使用相同的 `session_id`**:
            ```json
            {
              "question": "Can you look up some common ways of doing it?",
              "session_id": "my_user_session_1"
            }
            ```
        5.  点击 `Execute`。您会发现LLM能够根据之前的对话上下文（即“Task Decomposition”）生成更相关的查询和答案。

    *   **测试 `/chat/agent` (代理式 RAG)**:
        1.  点击 `/chat/agent` 接口，然后点击 `Try it out` 按钮。
        2.  在 `Request body` 中，输入一个复杂的问题和会话ID：
            ```json
            {
              "question": "What is the standard method for Task Decomposition? Once you get the answer, look up common extensions of that method.",
              "session_id": "my_user_session_2"
            }
            ```
        3.  点击 `Execute`。您会观察到代理执行多步检索（在控制台日志中可见），然后给出一个综合性的答案。
        4.  尝试使用不同的 `session_id` (`my_user_session_3`) 开始一个新的对话，以验证会话隔离。

通过上述步骤，您已经成功地将您的RAG应用程序与FastAPI结合，并部署为一个支持对话记忆的Web服务！这为将您的LLM应用集成到更大的系统中奠定了基础。








