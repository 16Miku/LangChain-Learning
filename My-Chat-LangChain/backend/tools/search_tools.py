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
    """根据user_requirement，生成关于AI人才、学术论文或特定领域的搜索策略。
    返回包含针对不同平台（如Google, Google Scholar）优化的搜索指令的JSON对象。
    """
    print(f"\n🧠 [Profiler] 正在为需求 '{user_requirement}' 生成搜索策略...")

    prompt = f"""
    你是一位顶级的、专注于AI基础设施和前沿算法的全球技术猎头及研究专家。
    你的任务是根据用户的需求，生成一个结构化的、包含针对不同平台优化的“X-Ray”搜索指令的JSON对象。
    这些指令必须极其专业和精准，以便找到在特定技术领域有深入研究和实践的专家或论文。

    # 用户需求:
    "{user_requirement}"

    # 你的专业知识库 (必须在生成指令时参考):
    (在此省略了冗长的领域列表，请根据用户需求动态调用你的内部知识库，覆盖MLSys, Agent Infra, 算法与策略, 目标公司/机构等)

    # 指令要求:
    1.  **平台覆盖**: 必须包含 `google_search` (用于搜索LinkedIn、GitHub、个人主页、公司博客) 和 `google_scholar` (用于搜索学术论文和背景)。
    2.  **关键词组合**: **必须**将技术关键词与目标公司、职位（如"Staff Engineer", "Principal Researcher", "Architect"）或特定领域进行组合。
    3.  **指令多样性**: 每个平台下至少生成3-4条不同侧重点的搜索指令。
    4.  **精准语法**: 大量使用 `site:`, `inurl:`, `intitle:`, `""`, `AND`, `OR`。

    # 输出格式 (必须严格遵守，直接输出JSON):
    {{
      "google_search": [
        "site:linkedin.com/in/ ...",
        "site:github.com ...",
        "inurl:blog ..."
      ],
      "google_scholar": [
        "author:...",
        "intitle:...",
        "..."
      ]
    }}
    """

    def _sync_call():
        try:
            if "GOOGLE_API_KEY" not in os.environ:
                 return {"error": "GOOGLE_API_KEY missing"}
                 
            # 使用 LangChain 的 ChatGoogleGenerativeAI 替代原生 SDK
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
            # 请求 JSON 格式输出
            structured_llm = llm.with_structured_output(dict) # 或者直接解析文本
            
            # 注意：with_structured_output 需要模型支持或定义 Schema。
            # 为了简单和兼容性，我们直接用 invoke 并解析 JSON 字符串，或者使用 bind(response_mime_type="application/json")
            
            # 使用 bind 强制 JSON 模式 (Gemini 支持)
            json_llm = llm.bind(response_mime_type="application/json")
            response = json_llm.invoke(prompt)
            
            return json.loads(response.content)
        except Exception as e:
            print(f"Gemini Generate Error: {e}")
            return None

    try:
        result = await asyncio.to_thread(_sync_call)
        if isinstance(result, dict) and "google_search" in result and "google_scholar" in result:
            print("✅ [Profiler] 搜索策略生成成功且格式正确！")
            return result
        else:
            print(f"🟡 [Profiler] LLM返回了非预期的格式: {result}")
            return None
    except Exception as e:
        print(f"❌ [Profiler] 调用LLM或解析其响应时发生错误: {e}")
        return None

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
            print(f"  -> 正在搜索 '{query}'")
            try:
                conn = http.client.HTTPSConnection("google.serper.dev")
                payload_obj = {"q": query, "num": 20}
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
                if "organic" in results: # SerperAPI 的普通搜索结果键
                    search_results.extend(results["organic"])
                if "scholar" in results: # SerperAPI 的学术搜索结果键
                    search_results.extend(results["scholar"])
                if "organic_results" in results: # 兼容 SerpApi 的 organic_results
                    search_results.extend(results["organic_results"])

                for result in search_results:
                    link = result.get("link")
                    # 过滤掉 Google 自身的链接
                    if link and not any(domain in link for domain in ["google.com/search", "support.google.com"]):
                      all_urls.add(link)
            except Exception as e:
                print(f"  -> ❌ 执行搜索 '{query}' 时发生错误: {e}")
    
    print(f"✅ [Scout] 搜索完成！共找到 {len(all_urls)} 个不重复的URL。")
    return list(all_urls)
