# backend/langchain_qa_backend.py

import os
import asyncio
import logging
from urllib.parse import urlparse

# 导入 LangChain 核心组件
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ****** 关键修改 1: 导入新的 HuggingFaceEmbeddings ******
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI # 我们仍然使用 Google 的 LLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage

# 导入 dotenv，用于从 .env 文件加载环境变量
from dotenv import load_dotenv

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载 .env 文件中的环境变量
load_dotenv()

# 确保 API 密钥已设置 (这对于 ChatGoogleGenerativeAI 仍然是必需的)
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")


async def load_and_process_documents(url: str):
    """
    (本地嵌入模型版本) 从URL加载文档，使用本地模型进行嵌入，速度更快且无速率限制。
    """
    logging.info(f"开始从 URL 加载和处理文档: {url}")
    try:
        # 1. 文档加载 (这部分代码保持不变)
        parsed_url = urlparse(url)
        base_domain_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = f"{base_domain_url}/sitemap.xml"
        
        logging.info(f"尝试从站点地图加载: {sitemap_url}")
        loader = SitemapLoader(
            sitemap_url,
            filter_urls=[url],
            continue_on_failure=True,
            show_progress=True
        )
        documents = await asyncio.to_thread(loader.load)

        if not documents:
            logging.warning(f"无法从站点地图加载文档。回退到 RecursiveUrlLoader。")
            loader_fallback = RecursiveUrlLoader(url, max_depth=1)
            documents = await asyncio.to_thread(loader_fallback.load)
            if not documents:
                logging.error(f"备用方案也未能从 {url} 加载任何文档。")
                return None
        
        logging.info(f"成功加载 {len(documents)} 篇文档。")

        # 2. 文本分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(documents)
        logging.info(f"文档被分割成 {len(all_splits)} 个块。")

        # --- 核心修改 2: 初始化本地嵌入模型 ---
        logging.info("初始化本地嵌入模型 (HuggingFace)...")
        # 我们选用 'all-MiniLM-L6-v2'，这是一个性能优秀且轻量级的模型，非常适合入门。
        # 第一次运行时，LangChain 会自动从 Hugging Face Hub 下载模型文件并缓存到本地。
        # 这可能需要几分钟时间，但之后每次运行都会直接从本地加载，速度飞快。
        model_name = "all-MiniLM-L6-v2"
        # 你可以指定模型在 CPU ('cpu') 或 GPU ('cuda') 上运行。对于这个模型，CPU 已经足够快。
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logging.info(f"本地嵌入模型 '{model_name}' 加载完成。")
        # --- 修改结束 ---

        # --- 核心修改 3: 回归简洁的向量存储创建 ---
        # 因为本地模型没有速率限制，我们不再需要复杂的分批和延迟代码。
        # 直接使用 Chroma.from_documents 一步到位，它会利用你的 CPU 核心全速运行。
        logging.info("开始创建向量存储 (这将利用CPU全速进行，可能需要几分钟)...")
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory="./chroma_langchain_db"
        )
        logging.info("向量存储构建完成。")
        # --- 修改结束 ---

        # 5. 创建检索器
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10},
        )
        logging.info("检索器创建成功！")
        return retriever

    except Exception as e:
        logging.error(f"加载或处理文档时发生严重错误: {e}", exc_info=True)
        return None


def get_retrieval_chain(retriever):
    """
    根据给定的检索器，创建并返回一个完整的RAG问答链。
    """
    if retriever is None:
        return None
    
    # LLM 部分我们仍然使用 Google Gemini API
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) 
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        model, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    logging.info("RAG 问答链创建成功。")
    return retrieval_chain