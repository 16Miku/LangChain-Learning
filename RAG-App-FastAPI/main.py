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

