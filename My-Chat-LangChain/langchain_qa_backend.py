# langchain_qa_backend.py
import os
import streamlit as st
# 导入 SitemapLoader
from langchain_community.document_loaders import SitemapLoader
# 导入用于解析URL的库
from urllib.parse import urlparse
# 导入 LangChain 核心消息类型，用于处理聊天历史
from langchain_core.messages import HumanMessage, AIMessage

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

# 尝试导入 tqdm，如果未安装则给出提示
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    st.warning("为了更好地显示文档加载进度，请安装 `tqdm`：`pip install tqdm`")


# 使用 Streamlit secrets 获取 API 密钥
# 推荐做法：在 .streamlit/secrets.toml 文件中添加 GOOGLE_API_KEY = "你的API密钥"
# 或者设置系统环境变量 GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def load_and_process_documents(url: str):
    """
    从指定URL（通过其站点地图）加载文档，进行分块，创建嵌入并构建向量存储。
    此函数会缓存结果，避免重复加载和处理。
    """
    st.write(f"正在从 {url} 对应的知识库加载并处理文档，请稍候...")
    try:
        # 1. 构造站点地图URL
        parsed_url = urlparse(url)
        base_domain_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = f"{base_domain_url}/sitemap.xml"

        st.info(f"尝试从站点地图URL: {sitemap_url} 加载文档。")

        # 1.1. 加载文档
        # 使用 SitemapLoader
        loader = SitemapLoader(
            sitemap_url,
            filter_urls=[url],
            # ****** 关键修改部分：移除自定义的 requests_per_second 和 requests_kwargs ******
            # 允许 SitemapLoader 使用其内部默认的并发和请求参数，这可能更适合目标网站
            # 如果之前是为了解决 TypeError 而添加的 verify=False，
            # 那么现在请确保您的 Python 环境能够正确验证 SSL 证书，
            # 或者接受 InsecureRequestWarning（在生产环境不推荐）。
            # 如果继续遇到 SSL 验证问题，可以考虑在 requests_kwargs 中仅添加 "verify": False
            # 但首先尝试不添加任何 requests_kwargs
            # requests_per_second=0.1, # 移除此行
            # requests_kwargs={...}, # 移除此字典
            # ****** 调整结束 ******
            continue_on_failure=True, # 保持遇到错误时继续加载其他URL
            # 添加 tqdm 进度条支持
            show_progress=True if tqdm else False
        )
        
        documents = loader.load() # 直接调用 load()

        if not documents:
            st.warning(f"无法从站点地图 {sitemap_url} 或通过过滤条件 '{url}' 加载任何文档。")
            st.info("这可能是由于站点地图不存在、内容为空，或者过滤条件过于严格。将尝试使用 RecursiveUrlLoader 仅加载当前URL页面作为备用方案。")
            
            from langchain_community.document_loaders import RecursiveUrlLoader
            st.write(f"正在尝试使用 RecursiveUrlLoader 递归加载 {url} 页面...")
            # 对于 RecursiveUrlLoader 备用方案，参考官方项目设置较长超时
            loader_fallback = RecursiveUrlLoader(
                url,
                max_depth=2 # 加载2层页面
                
                
                # ****** 备用方案调整结束 ******
            )
            documents = loader_fallback.load()
            if not documents:
                 st.error(f"备用方案也未能从 {url} 加载任何文档。请检查URL是否有效且可访问。")
                 return None

        st.info(f"成功从知识库加载了 {len(documents)} 篇文档。")
        # 移除了 failed_urls_count 的自定义逻辑，因为 SitemapLoader 内部会处理并记录错误
        # if failed_urls_count > 0:
        #     st.warning(f"注意：有 {failed_urls_count} 个URL未能成功抓取。这可能由于网站反爬或链接失效。")

        # 2. 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(documents)
        st.info(f"文档分割成 {len(all_splits)} 个块。")

        # 3. 初始化嵌入模型
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.info("嵌入模型初始化完成。")

        # 4. 创建内存向量存储并添加文档
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_splits)
        st.info("向量存储构建完成。")

        # 5. 创建检索器
        retriever = vector_store.as_retriever(
            search_type="mmr",  # 最大边际相关性检索
            search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        )
        st.success("文档加载、处理和检索器准备完成！")
        return retriever
    except Exception as e:
        st.error(f"加载或处理文档时发生错误: {e}")
        st.exception(e) # 打印详细错误信息，方便调试
        return None

# 通过在参数名 retriever 前加上 _，将其改为 _retriever，你是在告诉 Streamlit 的缓存机制：“嘿，这个参数 _retriever 是不可哈希的，请不要尝试对其进行哈希计算，直接跳过它作为缓存键的一部分。” Streamlit 仍然会缓存函数的结果，但它会忽略带有下划线的参数在哈希计算中的作用。
@st.cache_resource
def get_retrieval_chain(_retriever):
    """
    根据检索器获取LangChain的检索问答链。
    此函数会缓存问答链。
    """
    if _retriever is None:
        return None

    # 1. 初始化聊天模型
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5) # 使用gemini-2.0-flash，并设置温度

    # 2. 拉取预设的检索问答Prompt
    # 这个prompt通常接受 'input' 和 'chat_history' 变量
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # 3. 创建文档组合链
    combine_docs_chain = create_stuff_documents_chain(
        model, retrieval_qa_chat_prompt
    )

    # 4. 创建检索问答链
    # 检索链现在可以接收 'chat_history'
    retrieval_chain = create_retrieval_chain(_retriever, combine_docs_chain)
    return retrieval_chain

def answer_question(retrieval_chain, query: str):
    """
    使用检索问答链回答问题。
    """
    if retrieval_chain is None:
        return "知识库未准备好，请检查URL或API密钥。"
    try:
        # 从 Streamlit session_state 中获取聊天历史
        # 将 Streamlit 的消息格式转换为 LangChain chain 期望的格式 (HumanMessage, AIMessage)
        chat_history = []
        # 遍历除当前查询以外的所有历史消息
        # st.session_state.messages 包含当前用户问题，但我们只将之前的历史传递给 LLM
        for msg in st.session_state.messages[:-1]: # 排除最新的用户问题
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))

        # 调用检索链，并传入当前问题和格式化后的聊天历史
        response = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
        return response["answer"]
    except Exception as e:
        return f"回答问题时发生错误: {e}"

