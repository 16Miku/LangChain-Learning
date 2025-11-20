import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import custom tools
from tools.search_tools import generate_search_queries, execute_searches_and_get_urls
from tools.rag_tools import ingest_knowledge, query_knowledge
from tools.structure_tools import format_paper_analysis, format_linkedin_profile

load_dotenv()

# Global variables to hold the initialized agent and client
_agent_executor = None
_mcp_client = None
_mcp_tools = []

SYSTEM_PROMPT = """
你是一个全能的 AI 研究助理 (My-Chat-LangChain v5.0)。
你可以处理多种任务，包括分析学术论文、查询个人资料、执行复杂的网络搜索，以及深入学习和查询特定的网页知识库。

**你的能力 (工具箱):**
1.  **RAG 知识库工具**:
    *   `ingest_knowledge(url)`: 当用户要求学习某个新网页，或基于某个网页回答问题时，**必须**先调用此工具。
    *   `query_knowledge(query, url)`: 当需要从已学习的网页中检索详细信息时使用。
2.  **搜索与分析工具**:
    *   `generate_search_queries`: 生成专业的搜索策略。
    *   `execute_searches_and_get_urls`: 执行搜索并获取 URL。
    *   以及来自 MCP (如 BrightData, PaperSearch) 的其他强大工具（如果已配置）。
3.  **结构化报告工具**:
    *   `format_paper_analysis`: 当用户明确要求对论文进行分析报告时，在收集完信息后调用此工具输出结果。
    *   `format_linkedin_profile`: 当用户明确要求提取领英个人主页信息时，在收集完信息后调用此工具输出结果。

**你的行动指南 (ReAct 思考模式):**
1.  **分析与规划**: 仔细阅读用户的请求。
    *   如果是关于特定网页的问答 -> 1. `ingest_knowledge` -> 2. `query_knowledge`。
    *   如果是生成报告 -> 收集信息 -> 调用 `format_paper_analysis` 或 `format_linkedin_profile`。
2.  **信息收集**: 灵活组合使用你的工具。
3.  **生成回答**: 综合所有信息给出最终答案。如果用户只是闲聊，直接回答即可。

**注意事项**:
*   在回答之前，仔细检查是否有可用的结构化工具适合当前任务。
*   对于 RAG 任务，确保 URL 准确无误。
"""

async def initialize_agent(api_keys: Dict[str, str] = None):
    """
    Initialize the LangGraph agent with MCP tools and custom tools.
    This can be re-called if API keys are updated dynamically.
    """
    global _agent_executor, _mcp_client, _mcp_tools

    print("🚀 [Agent Service] Initializing Agent...")
    
    # 1. Configure MCP Client
    mcp_servers = {}
    
    # BrightData
    bd_key = api_keys.get("BRIGHT_DATA_API_KEY") if api_keys else os.environ.get("BRIGHT_DATA_API_KEY")
    if bd_key:
        mcp_servers["bright_data"] = {
            "url": f"https://mcp.brightdata.com/mcp?token={bd_key}&pro=1",
            "transport": "streamable_http", # Reverted to streamable_http as per demo code
        }
    
    # Paper Search
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
        query_knowledge,
        format_paper_analysis,
        format_linkedin_profile
    ]

    # Try to connect to MCP servers if configured
    if mcp_servers:
        try:
            # Using async context manager usually, but MultiServerMCPClient might need specific handling
            # For simplicity in this demo, we assume direct initialization if supported or handle clean up
            # Note: langchain-mcp-adapters usage pattern:
            _mcp_client = MultiServerMCPClient(mcp_servers)
            # await _mcp_client.__aenter__() # Manually enter context if needed, or use context manager wrapper
            # For now let's try to get tools directly. If it fails, we fallback.
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
        print("ℹ️ [Agent Service] No MCP keys provided, skipping MCP initialization.")
        _mcp_tools = []

    all_tools = _mcp_tools + custom_tools

    # 2. Configure LLM
    # Use Gemini 2.5 Flash or Pro as the brain
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY is missing!")
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # 根据用户指示使用 gemini-2.5-flash
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0
    )

    # 3. Create LangGraph Agent
    checkpointer = InMemorySaver()
    
    # 为了最大程度兼容不同版本的 LangGraph，我们不再向 create_react_agent 传递 system prompt
    # 而是选择在调用 agent 时手动添加 SystemMessage。
    # 这样可以避免 state_modifier / messages_modifier 参数名不匹配的问题。
    _agent_executor = create_react_agent(
        model=llm,
        tools=all_tools,
        checkpointer=checkpointer
    )
    
    print("✅ [Agent Service] Agent initialized successfully.")
    return _agent_executor

async def get_agent_executor():
    """Get the singleton agent executor, initializing it if necessary."""
    global _agent_executor
    if _agent_executor is None:
        await initialize_agent()
    return _agent_executor

async def chat_with_agent(message: str, thread_id: str, api_keys: Dict[str, str] = None):
    """
    Main entry point for chatting with the agent.
    """
    # Re-initialize if new keys are provided (simple logic for now)
    if api_keys:
        await initialize_agent(api_keys)
    
    agent = await get_agent_executor()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Invoke the graph
    # 手动注入 System Prompt 以确保兼容性
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message)
    ]
    
    final_state = await agent.ainvoke(
        {"messages": messages},
        config=config
    )
    
    # Extract the last message
    last_message = final_state["messages"][-1]
    return last_message.content

async def cleanup_mcp():
    """Cleanup MCP client resources."""
    global _mcp_client
    if _mcp_client:
        # await _mcp_client.__aexit__(None, None, None) # If using context manager manual control
        pass
