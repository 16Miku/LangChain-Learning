import json
import asyncio
import os
import http.client
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

@tool
async def generate_search_queries(user_requirement: str):
    """
    根据用户的具体需求 (user_requirement)，智能生成最优的 Google 搜索策略。
    返回包含针对不同侧重点（如通用搜索、学术搜索、深度挖掘）的搜索指令。
    """
    print(f"\n🧠 [Profiler] 正在为需求 '{user_requirement}' 生成搜索策略...")

    prompt = f"""
    你是一位世界顶级的搜索情报专家。你的任务是根据用户的需求，生成一组极其精准、专业的 Google 搜索指令 (Search Queries)。
    你需要分析用户的意图：
    1. 如果是**事实查询**（如“奥运会金牌榜”），生成直接的关键词。
    2. 如果是**深度研究**（如“AI Agent 架构设计”），使用 `site:`, `filetype:pdf`, `OR`, `AND` 等高级语法。
    3. 如果是**寻找特定资源**（如“Python 教程”），定向搜索 GitHub, Medium 等平台。
    4. 如果是**学术/技术研究**，包含 Google Scholar 风格的查询。

    # 用户需求:
    "{user_requirement}"

    # 输出格式 (必须严格遵守，直接输出JSON):
    {{
      "google_search": [
        "指令1 (侧重广度/通用)",
        "指令2 (侧重特定网站/资源, 如 site:github.com)",
        "指令3 (侧重文件/报告, 如 filetype:pdf)"
      ],
      "google_scholar": [
        "指令1 (侧重学术论文/作者)",
        "指令2 (侧重技术概念)"
      ]
    }}
    
    注意：
    - 必须返回 JSON 格式。
    - 如果用户需求明显不需要学术搜索（如“今天天气”），google_scholar 列表可以为空。
    """

    def _sync_call():
        try:
            if "GOOGLE_API_KEY" not in os.environ:
                 return {"error": "GOOGLE_API_KEY missing"}
                 
            # 使用 LangChain 的 ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
            
            # 使用 bind 强制 JSON 模式 (Gemini 支持)
            json_llm = llm.bind(response_mime_type="application/json")
            response = json_llm.invoke(prompt)
            
            return json.loads(response.content)
        except Exception as e:
            print(f"Gemini Generate Error: {e}")
            return None

    try:
        result = await asyncio.to_thread(_sync_call)
        if isinstance(result, dict) and "google_search" in result:
            print("✅ [Profiler] 搜索策略生成成功且格式正确！")
            return result
        else:
            print(f"🟡 [Profiler] LLM返回了非预期的格式: {result}")
            # Fallback structure
            return {"google_search": [user_requirement], "google_scholar": []}
    except Exception as e:
        print(f"❌ [Profiler] 调用LLM或解析其响应时发生错误: {e}")
        return {"google_search": [user_requirement], "google_scholar": []}

@tool
async def execute_searches_and_get_urls(search_queries_dict: dict, serper_api_key: str = None):
    """根据search_queries_dict，调用SerperAPI进行批量google search，获取大量网页url。
    如果未传入 serper_api_key，将尝试从环境变量 SERPER_API_KEY 读取。
    """
    
    if not serper_api_key:
        serper_api_key = os.environ.get("SERPER_API_KEY")
    
    if not serper_api_key:
        return "Error: Serper API Key is missing. Please provide it in the arguments or set SERPER_API_KEY environment variable."

    all_urls = set()
    print("\n🔍 [Scout] 开始执行多平台搜索...")

    for platform, queries in search_queries_dict.items():
        for query in queries:
            if not query: continue
            print(f"  -> 正在搜索 '{query}'")
            try:
                conn = http.client.HTTPSConnection("google.serper.dev")
                payload_obj = {"q": query, "num": 10} # 减少单次请求数量以加快速度
                if platform == "google_scholar":
                    payload_obj["engine"] = "google_scholar"
                else:
                    payload_obj["engine"] = "google"

                payload = json.dumps(payload_obj)
                headers = {
                  'X-API-KEY': serper_api_key,
                  'Content-Type': 'application/json'
                }

                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                data = res.read()
                results = json.loads(data.decode("utf-8"))
                conn.close()

                search_results = []
                if "organic" in results: 
                    search_results.extend(results["organic"])
                if "scholar" in results: 
                    search_results.extend(results["scholar"])
                if "organic_results" in results: 
                    search_results.extend(results["organic_results"])

                for result in search_results:
                    link = result.get("link")
                    # 过滤掉 Google 自身的链接
                    if link and not any(domain in link for domain in ["google.com/search", "support.google.com"]):
                      all_urls.add(link)
            except Exception as e:
                print(f"  -> ❌ 执行搜索 '{query}' 时发生错误: {e}")
    
    print(f"✅ [Scout] 搜索完成！共找到 {len(all_urls)} 个不重复的URL。")
