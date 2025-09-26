# frontend/app.py

import streamlit as st
import requests
import json

# --- 1. API 配置 ---
BACKEND_API_URL = "http://127.0.0.1:8000/chat"

# --- 2. 页面配置 ---
st.set_page_config(
    page_title="Chat LangChain | Enterprise Edition",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. 加载外部 CSS 文件 ---
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

# --- 4. 封装后端 API 调用逻辑 ---
def get_backend_response(url: str, query: str, chat_history: list):
    """
    修改函数名以反映它现在返回整个响应体，而不仅仅是答案。
    """
    try:
        payload = {
            "url": url,
            "query": query,
            "chat_history": chat_history
        }
        proxies = {"http": None, "https": None}
        
        response = requests.post(
            BACKEND_API_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=180,
            proxies=proxies
        )
        response.raise_for_status()
        # --- 核心修改：返回整个 JSON 响应体 ---
        return response.json()

    except requests.exceptions.Timeout:
        return {"answer": "请求超时。后端可能正在处理大型知识库，请稍后再试。", "source_documents": []}
    except requests.exceptions.RequestException as e:
        return {"answer": f"请求后端服务时出错: {e}", "source_documents": []}
    except Exception as e:
        return {"answer": f"发生未知错误: {e}", "source_documents": []}

# --- 5. 侧边栏内容 ---
with st.sidebar:
    st.markdown("## 🔗 Chat LangChain", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "这是一个基于 LangChain 和 Streamlit 构建的企业级 RAG 应用。\n\n"
        "**v2.1 新增功能:**\n"
        "- **答案溯源:** 每个回答下方都会附带其参考的原文片段，增强可信度。\n\n"
        "**工作流程:**\n"
        "1.  **输入 URL:** 指定知识库来源。\n"
        "2.  **后台处理:** 后端服务构建向量索引。\n"
        "3.  **开始提问:** 与专属知识库对话。\n\n"
        "**技术栈:**\n"
        "- **前端:** Streamlit\n"
        "- **后端:** FastAPI\n"
        "- **核心:** LangChain, SentenceTransformers, ChromaDB\n"
    )

# --- 6. 主内容区域 ---
st.title("My Chat LangChain 🤖 (Enterprise Edition)")

# ... (URL输入框、模型选择框、session_state 初始化等代码保持不变) ...
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_url" not in st.session_state:
    st.session_state.current_url = "https://python.langchain.com/docs/"
col1, col2 = st.columns([3, 1])
with col1:
    new_url = st.text_input("知识库 URL:", st.session_state.current_url)
with col2:
    st.selectbox("模型:", ["Gemini 1.5 Flash (Backend)"], disabled=True)
if st.session_state.current_url != new_url:
    st.session_state.current_url = new_url
    st.session_state.messages = []
    st.info(f"知识库已切换到: {new_url}。聊天历史已清空。")
    st.rerun()

# --- 欢迎语和示例问题 ---
if not st.session_state.messages:
    # ... (这部分代码保持不变) ...
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("你好！我是你的专属知识库问答助手。")
    st.subheader("可以试试这些示例问题：")
    cols = st.columns(2)
    example_questions = ["LangChain 是什么？", "什么是 LCEL？", "如何使用检索器？", "这个知识库的主要内容？"]
    for i, question in enumerate(example_questions):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state.prompt_from_button = question
            st.rerun()

# --- 显示聊天历史 (核心修改) ---
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        # --- 核心修改：如果消息是AI的，并且包含源文档，就显示它们 ---
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📖 查看答案来源"):
                for i, source in enumerate(message["sources"]):
                    # 尝试从元数据中获取源 URL
                    source_url = source.get("metadata", {}).get("source", "未知来源")
                    st.markdown(f"**来源 {i+1}:** [{source_url}]({source_url})")
                    # 使用 st.code 或 st.markdown(f"```{...}") 来展示文本片段，保持格式
                    st.markdown(f"> {source['page_content']}")
                    if i < len(message["sources"]) - 1:
                        st.markdown("---")

# --- 统一处理用户输入 (核心修改) ---
def handle_user_query(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        spinner_text = "正在思考中..."
        if st.session_state.current_url not in st.session_state:
             spinner_text = "首次加载知识库，可能需要几分钟..."
        
        with st.spinner(spinner_text):
            # 调用封装好的函数获取完整的响应
            response_data = get_backend_response(
                url=st.session_state.current_url,
                query=prompt,
                chat_history=[msg for msg in st.session_state.messages if msg["role"] != "user"]
            )
            answer = response_data.get("answer", "抱歉，出错了。")
            sources = response_data.get("source_documents", [])
            
            st.markdown(answer)
            
            # --- 核心修改：在AI回答后，立即显示来源 ---
            if sources:
                with st.expander("📖 查看答案来源"):
                    for i, source in enumerate(sources):
                        source_url = source.get("metadata", {}).get("source", "未知来源")
                        st.markdown(f"**来源 {i+1}:** [{source_url}]({source_url})")
                        st.markdown(f"> {source['page_content']}")
                        if i < len(sources) - 1:
                            st.markdown("---")
            
            # --- 核心修改：将源文档也存入 session_state ---
            # 这样当页面刷新时，历史消息的来源也能被重新渲染出来
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources # 将源文档列表一起存储
            })

# ... (处理按钮和输入框的逻辑保持不变) ...
if prompt_from_button := st.session_state.get("prompt_from_button"):
    del st.session_state.prompt_from_button
    handle_user_query(prompt_from_button)
elif prompt_from_input := st.chat_input("输入你的问题..."):
    handle_user_query(prompt_from_input)