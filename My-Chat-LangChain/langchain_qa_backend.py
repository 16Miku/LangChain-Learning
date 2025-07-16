# langchain_qa_backend.py
import os
import streamlit as st
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

# 使用 Streamlit secrets 获取 API 密钥
# 推荐做法：在 .streamlit/secrets.toml 文件中添加 GOOGLE_API_KEY = "你的API密钥"
# 或者设置系统环境变量 GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def load_and_process_documents(url: str):
    """
    从指定URL加载文档，进行分块，创建嵌入并构建向量存储。
    此函数会缓存结果，避免重复加载和处理。
    """
    st.write(f"正在从 {url} 加载并处理文档，请稍候...")
    try:
        # 1. 加载文档
        # max_depth=1 表示只加载当前URL页面，不深入抓取子链接
        loader = RecursiveUrlLoader(url, max_depth=1)
        documents = loader.load()

        if not documents:
            st.error(f"无法从 {url} 加载任何文档。请检查URL是否有效且可访问。")
            return None, None

        # 2. 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(documents)

        # 3. 初始化嵌入模型
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # 4. 创建内存向量存储并添加文档
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_splits)

        # 5. 创建检索器
        retriever = vector_store.as_retriever(
            search_type="mmr",  # 最大边际相关性检索
            search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        )
        st.success("文档加载和处理完成！")
        return retriever
    except Exception as e:
        st.error(f"加载或处理文档时发生错误: {e}")
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
    # 使用 ChatGoogleGenerativeAI 代替 init_chat_model
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5) # 使用gemini-2.0-flash，并设置温度

    # 2. 拉取预设的检索问答Prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # 3. 创建文档组合链
    combine_docs_chain = create_stuff_documents_chain(
        model, retrieval_qa_chat_prompt
    )

    # 4. 创建检索问答链
    retrieval_chain = create_retrieval_chain(_retriever, combine_docs_chain)
    return retrieval_chain

def answer_question(retrieval_chain, query: str):
    """
    使用检索问答链回答问题。
    """
    if retrieval_chain is None:
        return "知识库未准备好，请检查URL或API密钥。"
    try:
        response = retrieval_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"回答问题时发生错误: {e}"

