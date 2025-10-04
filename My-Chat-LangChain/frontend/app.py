# frontend/app.py

import streamlit as st
import requests
import json
import os

# --- 1. API 配置 (保持不变) ---
BACKEND_URL_ENDPOINT = "http://127.0.0.1:8000/chat_url"
BACKEND_FILE_ENDPOINT = "http://127.0.0.1:8000/chat_file"

# --- 2. 页面配置 & 样式加载 (保持不变) ---
st.set_page_config(
    page_title="Chat LangChain | Enterprise Edition",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

# --- 3. API 调用函数 (保持不变) ---
def get_backend_response_from_url(url: str, query: str, chat_history: list):
    # ... (函数内容不变)
    try:
        payload = {"url": url, "query": query, "chat_history": chat_history}
        proxies = {"http": None, "https": None}
        response = requests.post(BACKEND_URL_ENDPOINT, json=payload, timeout=180, proxies=proxies)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"请求后端服务时出错 (URL): {e}", "source_documents": []}

def get_backend_response_from_file(query: str, chat_history: list, uploaded_file):
    # ... (函数内容不变)
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {'query': query, 'chat_history_str': json.dumps(chat_history)}
        proxies = {"http": None, "https": None}
        response = requests.post(BACKEND_FILE_ENDPOINT, files=files, data=data, timeout=300, proxies=proxies)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"请求后端服务时出错 (File): {e}", "source_documents": []}

# --- 4. 侧边栏内容 (保持不变) ---
with st.sidebar:
    # ... (内容不变)
    st.markdown("## 🔗 Chat LangChain v4.0", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**v4.0 新增功能:**\n- **文档知识库:** 新增了通过上传 PDF 文件进行问答的功能。\n\n**工作模式:**\n1.  **网页知识库:** 在 Tab 中输入 URL 进行在线内容问答。\n2.  **文档知识库:** 在 Tab 中上传 PDF 文件进行本地文档问答。\n")
    st.markdown("---")
    st.markdown("**核心技术:**\n- 前端: Streamlit\n- 后端: FastAPI\n- RAG: LangChain, ChromaDB, SentenceTransformers, Flashrank\n")

# --- 5. 主内容区域 ---
st.title("My Chat LangChain 🤖 (Enterprise Edition)")

tab_url, tab_file = st.tabs(["🔗 网页知识库", "📄 文档知识库"])

# --- Tab 1: 网页知识库 (逻辑微调) ---
with tab_url:
    st.header("与在线网页内容对话")

    if "url_messages" not in st.session_state:
        st.session_state.url_messages = []
    if "current_url" not in st.session_state:
        st.session_state.current_url = "https://python.langchain.com/docs/modules/agents/"

    col1, col2 = st.columns([3, 1])
    with col1:
        new_url = st.text_input("知识库 URL:", st.session_state.current_url, key="url_input")
    with col2:
        st.selectbox("模型:", ["Gemini 2.5 Flash (Backend)"], disabled=True, key="url_model_select")

    if st.session_state.current_url != new_url:
        st.session_state.current_url = new_url
        st.session_state.url_messages = []
        st.info(f"网页知识库已切换到: {new_url}。")
        st.rerun()

    # 渲染历史消息 (逻辑不变)
    for message in st.session_state.url_messages:
        # ... (渲染逻辑不变)
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📖 查看答案来源"):
                    for i, source in enumerate(message["sources"]):
                        source_url = source.get("metadata", {}).get("source", "未知来源")
                        st.markdown(f"**来源 {i+1}:** [{source_url}]({source_url})")
                        st.markdown(f"> {source['page_content']}")
                        if i < len(message["sources"]) - 1: st.markdown("---")

    # --- 核心修改：将输入框移到 Tab 逻辑的末尾 ---
    if prompt := st.chat_input("就当前网页提问..."):
        st.session_state.url_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("正在基于网页内容思考..."):
                response_data = get_backend_response_from_url(
                    url=st.session_state.current_url,
                    query=prompt,
                    chat_history=st.session_state.url_messages[:-1]
                )
                answer = response_data.get("answer", "抱歉，出错了。")
                sources = response_data.get("source_documents", [])
                st.markdown(answer)
                if sources:
                    with st.expander("📖 查看答案来源"):
                        for i, source in enumerate(sources):
                            source_url = source.get("metadata", {}).get("source", "未知来源")
                            st.markdown(f"**来源 {i+1}:** [{source_url}]({source_url})")
                            st.markdown(f"> {source['page_content']}")
                            if i < len(sources) - 1: st.markdown("---")
                
                st.session_state.url_messages.append({"role": "assistant", "content": answer, "sources": sources})
                # 添加 rerun 确保来源展开器状态正确更新
                st.rerun()

# --- Tab 2: 文档知识库 (核心重构) ---
with tab_file:
    st.header("与您上传的 PDF 文档对话")

    if "file_messages" not in st.session_state:
        st.session_state.file_messages = []
    if "current_file_id" not in st.session_state:
        st.session_state.current_file_id = None

    uploaded_file = st.file_uploader(
        "请在此处上传您的 PDF 文件", 
        type=['pdf'],
        help="上传后，您可以就该文档的内容进行提问。"
    )

    # --- 核心修改：使用 uploaded_file.file_id 替换 .id ---
    if uploaded_file and (st.session_state.current_file_id != uploaded_file.file_id):
        st.session_state.current_file_id = uploaded_file.file_id
        st.session_state.file_messages = []
        st.info(f"文档知识库已切换到: {uploaded_file.name}。")

    # 渲染历史消息 (逻辑不变)
    for message in st.session_state.file_messages:
        # ... (渲染逻辑不变)
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📖 查看答案来源"):
                    for i, source in enumerate(message["sources"]):
                        page_num = source.get("metadata", {}).get("page", -1)
                        st.markdown(f"**来源 {i+1}:** 第 {page_num + 1} 页")
                        st.markdown(f"> {source['page_content']}")
                        if i < len(message["sources"]) - 1: st.markdown("---")

    # --- 核心修改：将输入框移到 Tab 逻辑的末尾，并用 disabled 参数控制 ---
    # 如果没有上传文件，输入框会显示但不可用
    if prompt := st.chat_input(
        f"就 {uploaded_file.name} 提问..." if uploaded_file else "请先上传一个 PDF 文件", 
        disabled=not uploaded_file
    ):
        st.session_state.file_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("正在基于文档内容思考..."):
                response_data = get_backend_response_from_file(
                    query=prompt,
                    chat_history=st.session_state.file_messages[:-1],
                    uploaded_file=uploaded_file
                )
                answer = response_data.get("answer", "抱歉，出错了。")
                sources = response_data.get("source_documents", [])
                st.markdown(answer)
                if sources:
                    with st.expander("📖 查看答案来源"):
                        for i, source in enumerate(sources):
                            page_num = source.get("metadata", {}).get("page", -1)
                            st.markdown(f"**来源 {i+1}:** 第 {page_num + 1} 页")
                            st.markdown(f"> {source['page_content']}")
                            if i < len(sources) - 1: st.markdown("---")
                
                st.session_state.file_messages.append({"role": "assistant", "content": answer, "sources": sources})
                # 添加 rerun 确保来源展开器状态正确更新
                st.rerun()