# backend/langchain_qa_backend.py

import os
import asyncio
import logging
from urllib.parse import urlparse
import hashlib # 导入 hashlib 用于生成 MD5 哈希

# 导入 LangChain 核心组件
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader
from langchain_community.document_loaders import PyPDFLoader # 新增 PyPDFLoader

# 新增导入
from langchain_community.document_transformers import BeautifulSoupTransformer


from langchain.text_splitter import RecursiveCharacterTextSplitter
# ****** 关键修改 1: 导入新的 HuggingFaceEmbeddings ******
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI # 我们仍然使用 Google 的 LLM

# --- 核心修改 1: 使用最新的、最正确的导入路径 ---
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

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



# --- 新增辅助函数：为 URL 生成唯一的目录名 ---
def get_persist_directory_for_url(url: str) -> str:
    """根据 URL 生成一个唯一的、安全的文件夹名"""
    # 使用 MD5 哈希算法，确保任何 URL 都能转换成一个固定长度的字符串
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    # 返回一个基于哈希值的路径
    return f"./chroma_db_{url_hash}"






# --- 新增辅助函数：为文件生成唯一的目录名 ---
def get_persist_directory_for_file(filename: str, file_content: bytes) -> str:
    """
    根据文件名和文件内容的哈希生成唯一的、安全的文件夹名。
    这样即使用户上传同名但内容不同的文件，也能被区分。
    """
    # 计算文件内容的 MD5 哈希值
    file_hash = hashlib.md5(file_content).hexdigest()
    # 获取文件名（不含扩展名），并确保其对于路径是安全的
    basename = os.path.splitext(filename)[0].replace(" ", "_")
    # 结合文件名和内容哈希，创建唯一目录名
    return f"./chroma_db_{basename}_{file_hash}"




# --- 核心重构 1: URL 处理函数，专门负责从零构建向量数据库 ---
async def create_vector_store_from_url(url: str, persist_directory: str):
    """
    从 URL 抓取、处理文档，并创建一个新的 Chroma 向量数据库并持久化。
    """
    logging.info(f"知识库 '{persist_directory}' 不存在，开始从零创建...")
    # 1. 文档加载
    # ... (这部分逻辑从原函数移动过来，保持不变) ...
    parsed_url = urlparse(url)
    base_domain_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    sitemap_url = f"{base_domain_url}/sitemap.xml"
    loader = SitemapLoader(sitemap_url, filter_urls=[url], continue_on_failure=True, show_progress=True)
    documents = await asyncio.to_thread(loader.load)
    if not documents:
        loader_fallback = RecursiveUrlLoader(url, max_depth=1)
        documents = await asyncio.to_thread(loader_fallback.load)
        if not documents:
            logging.error(f"无法从 {url} 加载任何文档。")
            return None
    logging.info(f"成功加载 {len(documents)} 篇文档。")

    # 1.5. HTML 清洗
    bs_transformer = BeautifulSoupTransformer()
    cleaned_documents = bs_transformer.transform_documents(documents, unwanted_tags=["script", "style"])

    # 2. 文本分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(cleaned_documents)

    # 3. 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # 4. 创建并持久化向量存储
    logging.info(f"开始为新知识库创建向量存储于 '{persist_directory}'...")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logging.info("新向量存储创建并持久化完成。")
    return vector_store





# --- 核心重构 2: 新增的文件处理函数 ---
async def create_vector_store_from_file(filepath: str, persist_directory: str):
    """
    从本地文件路径加载文档，并创建一个新的 Chroma 向量数据库。
    """
    logging.info(f"知识库 '{persist_directory}' 不存在，开始从文件 {filepath} 创建...")
    try:
        # 1. 文档加载
        # 根据文件扩展名选择合适的加载器
        if filepath.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        # 未来可以在这里添加对 .txt, .docx, .md 等文件的支持
        # elif filepath.lower().endswith(".txt"):
        #     loader = TextLoader(filepath)
        else:
            logging.error(f"不支持的文件类型: {filepath}")
            return None
        
        # PyPDFLoader 的 load 是同步阻塞的，所以也用 to_thread
        documents = await asyncio.to_thread(loader.load)
        if not documents:
            logging.error(f"无法从 {filepath} 加载任何文档。")
            return None
        logging.info(f"成功从文件加载 {len(documents)} 页/篇文档。")

        # 2. 文本分割 (PDF 通常不需要复杂的HTML清洗)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(documents)
        logging.info(f"文档被分割成 {len(all_splits)} 个块。")

        # 3. 初始化嵌入模型 (与 URL 版本完全相同)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

        # 4. 创建并持久化向量存储 (与 URL 版本完全相同)
        logging.info(f"开始为新知识库创建向量存储于 '{persist_directory}'...")
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logging.info("新向量存储创建并持久化完成。")
        return vector_store
    except Exception as e:
        logging.error(f"从文件创建向量存储时发生错误: {e}", exc_info=True)
        return None






# --- 核心重构 2: 创建一个函数，负责加载现有的数据库 ---
def load_vector_store(persist_directory: str):
    """
    从指定的磁盘目录加载一个已存在的 Chroma 向量数据库。
    """
    logging.info(f"开始从 '{persist_directory}' 加载现有知识库...")
    # 嵌入模型必须和创建时使用的模型完全一样
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    # 直接使用 Chroma 的构造函数加载
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    logging.info("现有知识库加载完成。")
    return vector_store






def get_retrieval_chain(base_retriever):
    """
    (函数被重构) 根据基础检索器，创建一个包含本地开源 Rerank 模型的高级 RAG 链。
    """
    if base_retriever is None: return None

    
    
    # --- 核心修改 2: 初始化本地 FlashrankRerank ---
    logging.info("初始化本地 FlashrankRerank 模型...")
    # FlashrankRerank 会自动从 Hugging Face 下载并缓存重排序模型
    # 第一次运行时会需要一些时间下载
    reranker = FlashrankRerank( top_n=20 )
    logging.info("本地 Rerank 模型加载完成。")

    # --- 核心修改 3: 创建上下文压缩检索器 (逻辑不变) ---
    # 这里的逻辑和使用 Cohere 时完全一样，我们只是把“复赛评委”换成了本地模型
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=base_retriever
    )
    logging.info("上下文压缩检索器 (带本地重排功能) 创建成功。")

    # --- 后续构建 RAG 链的步骤完全不变 ---
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) 
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    combine_docs_chain = create_stuff_documents_chain(
        model, retrieval_qa_chat_prompt
    )
    
    retrieval_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)
    
    logging.info("带本地 Rerank 功能的高级 RAG 问答链创建成功。")
    return retrieval_chain