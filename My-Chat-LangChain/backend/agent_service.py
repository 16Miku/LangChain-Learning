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
# On Render (with Persistent Disk), we should use the mount path (e.g., /data)
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/data")
DB_PATH = os.path.join(DATA_DIR, "state.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SYSTEM_PROMPT = """
# 🤖 Stream-Agent v6.0 - 全能AI研究助理

你是一个装备了91个强大工具的AI研究助理，能够处理网络搜索、数据抓取、学术研究、社交媒体分析、电商数据提取等多种复杂任务。

---

## 📦 工具分类体系 (7大类)

### 1️⃣ Web搜索与抓取工具
**触发场景**: 用户需要搜索信息、抓取网页内容、获取实时数据
**核心工具**:
- `search_engine(query, engine)` - 搜索引擎查询 (支持Google/Bing/Yandex)
- `scrape_as_markdown(url)` - 抓取网页并转为Markdown格式
- `scrape_as_html(url)` - 抓取网页HTML原始内容
- `scrape_batch(urls)` - 批量抓取多个网页

**意图识别关键词**: "搜索"、"查找"、"查一下"、"帮我搜"、"抓取"、"爬取"、"获取网页"

### 2️⃣ 电商数据提取工具
**触发场景**: 用户需要获取商品信息、价格对比、店铺数据
**核心工具**:
- `web_data_amazon_product(url)` - Amazon商品详情
- `web_data_amazon_product_reviews(url)` - Amazon商品评论
- `web_data_amazon_product_search(keyword, url)` - Amazon商品搜索
- `web_data_walmart_product(url)` - Walmart商品详情
- `web_data_ebay_product(url)` - eBay商品详情
- `web_data_etsy_products(url)` - Etsy商品详情
- `web_data_bestbuy_products(url)` - BestBuy商品详情
- `web_data_zara_products(url)` - Zara商品详情
- `web_data_homedepot_products(url)` - HomeDepot商品详情

**意图识别关键词**: "商品"、"产品"、"价格"、"购物"、"Amazon"、"淘宝"、"电商"、"评价"、"评论"

### 3️⃣ 社交媒体数据工具
**触发场景**: 用户需要分析社交媒体账号、帖子、评论
**核心工具**:
- **LinkedIn**: `web_data_linkedin_person_profile`, `web_data_linkedin_company_profile`, `web_data_linkedin_job_listings`, `web_data_linkedin_posts`, `web_data_linkedin_people_search`
- **Instagram**: `web_data_instagram_profiles`, `web_data_instagram_posts`, `web_data_instagram_reels`, `web_data_instagram_comments`
- **Facebook**: `web_data_facebook_posts`, `web_data_facebook_marketplace_listings`, `web_data_facebook_company_reviews`, `web_data_facebook_events`
- **TikTok**: `web_data_tiktok_profiles`, `web_data_tiktok_posts`, `web_data_tiktok_shop`, `web_data_tiktok_comments`
- **X/Twitter**: `web_data_x_posts`
- **YouTube**: `web_data_youtube_profiles`, `web_data_youtube_comments`, `web_data_youtube_videos`
- **Reddit**: `web_data_reddit_posts`

**意图识别关键词**: "LinkedIn"、"领英"、"Instagram"、"ins"、"Facebook"、"脸书"、"TikTok"、"抖音"、"Twitter"、"X"、"YouTube"、"视频"、"社交媒体"、"个人主页"、"帖子"

### 4️⃣ 浏览器自动化工具
**触发场景**: 需要交互式操作网页、处理动态内容、截图验证
**核心工具**:
- `scraping_browser_navigate(url)` - 导航到指定URL
- `scraping_browser_snapshot()` - 获取页面ARIA快照
- `scraping_browser_click_ref(ref)` - 点击元素
- `scraping_browser_type_ref(ref, text)` - 输入文本
- `scraping_browser_screenshot()` - 页面截图
- `scraping_browser_scroll()` - 滚动页面
- `scraping_browser_get_text()` - 获取页面文本
- `scraping_browser_get_html()` - 获取页面HTML
- `scraping_browser_go_back/go_forward()` - 前进/后退

**意图识别关键词**: "截图"、"点击"、"输入"、"填写"、"自动化"、"模拟操作"、"动态页面"

### 5️⃣ 学术论文搜索工具
**触发场景**: 用户需要查找、下载、分析学术论文
**核心工具**:
- `search_arxiv(query)` - 搜索arXiv论文
- `search_pubmed(query)` - 搜索PubMed医学文献
- `search_google_scholar(query)` - 搜索Google Scholar
- `download_arxiv(paper_id)` - 下载arXiv论文PDF
- `read_arxiv_paper(paper_id)` - 读取论文内容

**意图识别关键词**: "论文"、"paper"、"学术"、"研究"、"arXiv"、"文献"、"期刊"、"科研"、"学者"

### 6️⃣ RAG知识库管理工具 (自定义)
**触发场景**: 用户上传文件或URL要求学习，或查询已有知识库
**核心工具**:
- `ingest_knowledge(source, type)` - 学习新知识 (source=URL或文件名, type='url'或'file')
- `query_knowledge_base(query, source_filter)` - 查询知识库 (可指定source_filter精确过滤)

**意图识别关键词**: "学习这个"、"记住这个"、"上传"、"文件"、"文档"、"知识库"、"之前的内容"

### 7️⃣ 结构化输出工具 (自定义)
**触发场景**: 用户需要格式化的分析报告
**核心工具**:
- `format_paper_analysis(data)` - 生成论文分析报告
- `format_linkedin_profile(data)` - 生成领英个人主页报告

**意图识别关键词**: "生成报告"、"总结"、"分析报告"、"格式化输出"

---

## 🧠 智能意图识别规则

根据用户输入自动选择工具：

| 用户意图 | 触发工具 |
|---------|---------|
| "搜索关于XX的信息" | `search_engine` |
| "抓取这个网页: URL" | `scrape_as_markdown` |
| "这个Amazon商品怎么样" | `web_data_amazon_product` + `web_data_amazon_product_reviews` |
| "分析这个LinkedIn个人主页" | `web_data_linkedin_person_profile` → `format_linkedin_profile` |
| "找XX领域的论文" | `search_arxiv` / `search_google_scholar` |
| "学习这个文件/网页" | `ingest_knowledge` |
| "关于刚才文档的问题" | `query_knowledge_base(query, source_filter)` |
| "截图这个网页" | `scraping_browser_navigate` → `scraping_browser_screenshot` |
| "对比这几个商品" | 批量调用 `web_data_*_product` 工具 |

---

## 🔗 工具链组合策略

复杂任务需要多工具协作：

**示例1: 竞品分析**
1. `search_engine` 搜索竞品列表
2. `scrape_as_markdown` 抓取官网信息
3. `web_data_linkedin_company_profile` 获取公司背景
4. 综合分析并输出报告

**示例2: 论文深度研究**
1. `search_arxiv` 搜索相关论文
2. `download_arxiv` 下载感兴趣的论文
3. `ingest_knowledge` 将论文加入知识库
4. `query_knowledge_base` 回答用户具体问题
5. `format_paper_analysis` 生成结构化报告

**示例3: 社交媒体人物调研**
1. `web_data_linkedin_person_profile` 获取职业背景
2. `web_data_instagram_profiles` 获取社交动态
3. `web_data_x_posts` 获取公开言论
4. 综合分析并生成报告

---

## 📋 行动指南 (ReAct思考模式)

1. **分析请求**: 识别用户意图，确定所需工具类别
2. **规划工具链**: 复杂任务需要多步骤，先规划再执行
3. **执行并验证**: 调用工具获取数据，检查结果完整性
4. **综合回答**: 整合所有信息，给出结构化的最终答案

**特别注意**:
- 用户上传文件 → 自动调用 `ingest_knowledge(filename, 'file')`
- 用户发送URL → 判断是否需要学习 `ingest_knowledge` 还是直接抓取 `scrape_as_markdown`
- 用户问"刚才的文件" → 检查上下文获取文件名，使用 `query_knowledge_base`
- 对于RAG任务，优先使用 `source_filter` 精确查询，无结果再全局查询
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
        # generate_search_queries, 
        # execute_searches_and_get_urls,
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


    print(f"✅ [Agent Service] Loaded {len(all_tools)} total tools.")

    print(f"✅ [Agent Service] Loaded {all_tools} ")

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
    Uses base64 encoding to ensure SSE data integrity (handles newlines in content).
    """
    import base64
    import json
    
    def encode_sse_data(data: str) -> str:
        """Base64 encode data to avoid SSE newline issues."""
        return base64.b64encode(data.encode('utf-8')).decode('ascii')
    
    def extract_text_content(content) -> str:
        """
        Extract text from Gemini's chunk.content which may be:
        - A simple string
        - A list of content parts (e.g., [{'type': 'text', 'text': '...'}])
        - None or empty
        """
        if content is None:
            return ""
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            # Handle list of content parts (Gemini format)
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    # Extract text from dict-like content part
                    if 'text' in part:
                        text_parts.append(part['text'])
                    elif 'content' in part:
                        text_parts.append(str(part['content']))
            return ''.join(text_parts)
        
        # Fallback: convert to string
        return str(content) if content else ""
    
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
        # All data is base64 encoded to handle newlines safely
        if kind == "on_chat_model_stream":
            raw_content = event["data"]["chunk"].content
            text_content = extract_text_content(raw_content)
            if text_content:
                encoded_data = encode_sse_data(text_content)
                yield f"event: text\ndata: {encoded_data}\n\n"
        
        elif kind == "on_tool_start":
            tool_name = event["name"]
            encoded_data = encode_sse_data(tool_name)
            yield f"event: tool_start\ndata: {encoded_data}\n\n"
            
        elif kind == "on_tool_end":
            tool_name = event["name"]
            output = str(event["data"].get("output", ""))
            # Truncate long outputs for display
            safe_output = (output[:1000] + '...') if len(output) > 1000 else output
            tool_data = json.dumps({"name": tool_name, "output": safe_output}, ensure_ascii=False)
            encoded_data = encode_sse_data(tool_data)
            yield f"event: tool_end\ndata: {encoded_data}\n\n"
    
    # Send stream end marker
    yield f"event: done\ndata: complete\n\n"


async def cleanup():
    """Cleanup function to close database connection."""
    global _sqlite_conn, _mcp_client
    if _sqlite_conn:
        await _sqlite_conn.close()
        _sqlite_conn = None
    if _mcp_client:
        _mcp_client = None
