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
    layout="wide", # 将布局从 "centered" 改为 "wide" 以获得更多空间
    initial_sidebar_state="expanded" # 默认展开侧边栏
)

# --- 3. 加载外部 CSS 文件 (新方式) ---
def load_css(file_path):
    """一个用于加载外部CSS文件的函数"""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 在这里调用函数，加载我们的样式文件
load_css("style.css")



# --- 4. 封装后端 API 调用逻辑 (代码重构) ---
def get_backend_answer(url: str, query: str, chat_history: list):
    """
    一个专门用于调用后端 API 的函数，封装了所有网络请求的细节。
    这使得主逻辑更清晰，并且便于未来维护（例如添加重试机制）。
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
            timeout=180,  # 延长超时时间以应对超大知识库的首次加载
            proxies=proxies
        )
        response.raise_for_status()
        return response.json().get("answer", "抱歉，未能从后端获取有效回答。")

    except requests.exceptions.Timeout:
        return "请求超时。后端可能正在处理大型知识库，请稍后再试。"
    except requests.exceptions.RequestException as e:
        return f"请求后端服务时出错: {e}"
    except Exception as e:
        return f"发生未知错误: {e}"

# --- 5. 侧边栏内容 (UI 优化) ---
with st.sidebar:
    st.markdown("## 🔗 Chat LangChain", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "这是一个基于 LangChain 和 Streamlit 构建的企业级 RAG (检索增强生成) 应用。\n\n"
        "**工作流程:**\n"
        "1.  **输入 URL:** 在下方输入框中指定一个网页作为知识库来源。\n"
        "2.  **后台处理:** 后端服务会加载、处理该网页内容并构建向量索引。\n"
        "3.  **开始提问:** 你可以就该网页的内容进行提问。\n\n"
        "**技术栈:**\n"
        "- **前端:** Streamlit\n"
        "- **后端:** FastAPI\n"
        "- **核心:** LangChain, SentenceTransformers, ChromaDB\n"
    )
    st.markdown("---")


# --- 6. 主内容区域 ---
st.title("My Chat LangChain 🤖 (Enterprise Edition)")

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_url" not in st.session_state:
    st.session_state.current_url = "https://python.langchain.com/docs/"

# URL输入框和模型选择框放在一行
col1, col2 = st.columns([3, 1])
with col1:
    new_url = st.text_input("知识库 URL:", st.session_state.current_url)
with col2:
    st.selectbox("模型:", ["Gemini 2.5 Flash (Backend)"], disabled=True)

# 当URL改变时，清空聊天历史
if st.session_state.current_url != new_url:
    st.session_state.current_url = new_url
    st.session_state.messages = []
    st.info(f"知识库已切换到: {new_url}。聊天历史已清空。")
    st.rerun()

# --- 欢迎语和示例问题 (UX 优化) ---
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "你好！我是你的专属知识库问答助手。\n\n"
            "请在上方输入框中提供一个网页 URL 作为知识库，然后我们就可以开始对话了！"
        )
    
    st.markdown("---")
    st.subheader("或者，可以试试这些示例问题：")
    # 使用列布局来排列示例问题按钮
    cols = st.columns(2)
    example_questions = [
        "LangChain 是什么？",
        "什么是 LCEL？",
        "如何使用 LangChain 的检索器？",
        "这个知识库的主要内容是什么？"
    ]
    # 将按钮均匀分配到列中
    for i, question in enumerate(example_questions):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state.prompt_from_button = question
            st.rerun()


# 显示聊天历史
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- 统一处理用户输入 (代码重构 & UX 优化) ---
def handle_user_query(prompt: str):
    """统一处理来自输入框或按钮的查询"""
    # 1. 将用户输入添加到聊天历史并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 2. 显示加载动画，并调用后端
    with st.chat_message("assistant", avatar="🤖"):
        # 优化加载提示，告知用户新 URL 可能需要时间
        spinner_text = "正在思考中..."
        if st.session_state.current_url not in st.session_state:
             spinner_text = "首次加载知识库，可能需要几分钟，请耐心等待..."
        
        with st.spinner(spinner_text):
            # 调用封装好的函数获取答案
            answer = get_backend_answer(
                url=st.session_state.current_url,
                query=prompt,
                chat_history=st.session_state.messages[:-1]
            )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# 检查是否有来自按钮的查询
if prompt_from_button := st.session_state.get("prompt_from_button"):
    # 清除状态，避免重复触发
    del st.session_state.prompt_from_button
    handle_user_query(prompt_from_button)

# 检查是否有来自聊天输入框的查询
elif prompt_from_input := st.chat_input("输入你的问题..."):
    handle_user_query(prompt_from_input)