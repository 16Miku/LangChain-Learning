import os
import asyncio
import aiosqlite
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


import logging
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
# 细微优化
# 之前日志中有个小 Warning：
# WARNING:root:Failed to validate notification: 11 validation errors...
# 这是 MCP 协议的底层日志，不影响业务，但看着心烦。可以通过调整 logging 级别来屏蔽：



# Import custom tools
from tools.search_tools import generate_search_queries, execute_searches_and_get_urls
from tools.rag_tools import ingest_knowledge, query_knowledge_base
from tools.structure_tools import format_paper_analysis, format_linkedin_profile

load_dotenv()

# Global variables
_agent_executor = None
_mcp_client = None
_mcp_tools = []
_sqlite_conn = None

# --- Persistence Config ---
# On Vercel, only /tmp is writable
DB_PATH = "/tmp/data/state.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SYSTEM_PROMPT = """
你是一个全能的 AI 研究助理 (Stream-Agent v6.0)。
你可以处理多种任务，包括分析学术论文、查询个人资料、执行复杂的网络搜索，以及深入学习和查询特定的网页/文件知识库。

**你的能力 (工具箱):**
1.  **RAG 知识库工具 (统一入口)**:
    *   `ingest_knowledge(source, type)`: 学习新知识。`source`可以是URL或上传的文件名。
    *   `query_knowledge_base(query, source_filter)`: 查询知识库。可以指定 `source_filter` 来只查特定文档。
2.  **搜索与分析工具**:
    *   `generate_search_queries`: 分析用户意图并生成搜索策略。
    *   `execute_searches_and_get_urls`: 执行搜索。
    *   以及来自 MCP (如 BrightData, PaperSearch) 的其他强大工具（如果已配置）。
3.  **结构化报告工具**:
    *   `format_paper_analysis`: 生成论文分析报告。
    *   `format_linkedin_profile`: 生成领英个人主页报告。

**你的行动指南 (ReAct 思考模式):**
1.  **分析与规划**: 仔细阅读用户的请求。
    *   用户上传了文件? -> 自动调用 `ingest_knowledge(filename, 'file')`。
    *   用户发了链接? -> 自动调用 `ingest_knowledge(url, 'url')`。
    *   用户问关于刚才文件的问题? -> `query_knowledge_base(query, filename)`。
    *   用户需要做研究? -> `generate_search_queries` -> `execute_searches`。
2.  **信息收集**: 灵活组合使用你的工具。
3.  **生成回答**: 综合所有信息给出最终答案。

**注意事项**:
*   如果用户提到“刚上传的文件”，请检查上下文中的文件名。
*   对于 RAG 任务，优先尝试精确过滤查询 (`source_filter`)，如果无结果再尝试全局查询。
"""

async def initialize_agent(api_keys: Dict[str, str] = None):
    """
    Initialize the LangGraph agent with MCP tools, custom tools, and SQLite persistence.
    """
    global _agent_executor, _mcp_client, _mcp_tools, _sqlite_conn

    print("🚀 [Agent Service] Initializing Agent with Persistence...")
    
    # 1. Configure MCP Client (Same as before)
    mcp_servers = {}
    bd_key = api_keys.get("BRIGHT_DATA_API_KEY") if api_keys else os.environ.get("BRIGHT_DATA_API_KEY")
    if bd_key:
        mcp_servers["bright_data"] = {
            "url": f"https://mcp.brightdata.com/mcp?token={bd_key}&pro=1",
            "transport": "streamable_http",
        }
    ps_key = api_keys.get("PAPER_SEARCH_API_KEY") if api_keys else os.environ.get("PAPER_SEARCH_API_KEY")
    if ps_key:
        mcp_servers["paper_search"] = {
            "url": f"https://server.smithery.ai/@adamamer20/paper-search-mcp-openai/mcp?api_key={ps_key}",
            "transport": "streamable_http",
        }

    custom_tools = [
        generate_search_queries, 
        execute_searches_and_get_urls,
        ingest_knowledge, 
        query_knowledge_base,
        format_paper_analysis,
        format_linkedin_profile
    ]

    if mcp_servers:
        try:
            _mcp_client = MultiServerMCPClient(mcp_servers)
            try:
                _mcp_tools = await _mcp_client.get_tools()
                print(f"✅ [Agent Service] Loaded {len(_mcp_tools)} MCP tools.")
            except Exception as e:
                print(f"⚠️ [Agent Service] Failed to load MCP tools: {e}")
                _mcp_tools = []
        except Exception as e:
            print(f"⚠️ [Agent Service] Failed to connect to MCP servers: {e}")
            _mcp_tools = []
    else:
        _mcp_tools = []

    all_tools = _mcp_tools + custom_tools

    # 2. Configure LLM
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY is missing!")
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0
    )

    # 3. Create LangGraph Agent with AsyncSqliteSaver
    if _sqlite_conn is None:
        _sqlite_conn = await aiosqlite.connect(DB_PATH)
        
    checkpointer = AsyncSqliteSaver(_sqlite_conn)
    
    _agent_executor = create_react_agent(
        model=llm,
        tools=all_tools,
        checkpointer=checkpointer
    )
    
    print("✅ [Agent Service] Persistent Agent initialized successfully.")
    return _agent_executor

async def get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        await initialize_agent()
    return _agent_executor

async def chat_with_agent(message: str, thread_id: str, api_keys: Dict[str, str] = None):
    """
    Main entry point for chatting (Synchronous return for now, will be upgraded to stream).
    """
    if api_keys:
        await initialize_agent(api_keys)
    
    agent = await get_agent_executor()
    config = {"configurable": {"thread_id": thread_id}}
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message)
    ]
    
    final_state = await agent.ainvoke(
        {"messages": messages},
        config=config
    )
    
    return final_state["messages"][-1].content

async def chat_with_agent_stream(message: str, thread_id: str, api_keys: Dict[str, str] = None):
    """
    Generator function for streaming agent responses and thoughts.
    """
    if api_keys:
        await initialize_agent(api_keys)
    
    agent = await get_agent_executor()
    config = {"configurable": {"thread_id": thread_id}}
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message)
    ]

    async for event in agent.astream_events({"messages": messages}, config=config, version="v1"):
        kind = event["event"]
        
        # Yield different event types for the frontend to consume
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield f"data: {content}\n\n"
        
        elif kind == "on_tool_start":
            tool_name = event["name"]
            yield f"event: tool_start\ndata: {tool_name}\n\n"
            
        elif kind == "on_tool_end":
            tool_name = event["name"]
            output = str(event["data"].get("output"))
            # Truncate long outputs for display
            safe_output = (output[:200] + '...') if len(output) > 200 else output
            # JSON encoded to avoid newline issues in SSE
            import json
