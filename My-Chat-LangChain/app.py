# app.py
import streamlit as st
import os
from langchain_qa_backend import load_and_process_documents, get_retrieval_chain, answer_question

# --- 自定义 CSS 样式 ---
# 仿照图片中的暗色主题、圆角按钮和输入框
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="st-emotion"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0; /* 浅灰色文字 */
    }

    /* 整体背景色 */
    .stApp {
        background-color: #202123; /* 深灰色背景 */
    }

    /* 侧边栏背景色 */
    .stSidebar {
        background-color: #2c2c2e; /* 稍浅的深灰色 */
        border-right: 1px solid #3c3c3e; /* 侧边栏分隔线 */
    }

    /* 聊天历史标题 */
    .stSidebar h2 {
        color: #e0e0e0;
        padding-top: 20px;
        padding-left: 20px;
        font-weight: 500;
    }

    /* 聊天历史编辑图标 */
    .stSidebar [data-testid="stSidebar"] > div:first-child > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-right: 20px;
    }
    .stSidebar [data-testid="stSidebar"] > div:first-child > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:last-child {
        font-size: 1.2rem;
        cursor: pointer;
    }

    /* 主标题 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .st-emotion-cache-1avcm0c { /* 调整主内容区域的内边距 */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }


    h1 {
        color: #e0e0e0;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    /* 模型选择下拉框 */
    .stSelectbox > div > div {
        background-color: #3c3c3e;
        border-radius: 8px;
        border: none;
        color: #e0e0e0;
    }
    .stSelectbox > div > div:hover {
        border-color: #555;
    }
    .stSelectbox .st-bd { /* 下拉菜单选项 */
        background-color: #3c3c3e;
        color: #e0e0e0;
    }
    .stSelectbox .st-cd { /* 下拉菜单选项hover */
        background-color: #4a4a4c;
    }

    /* 预设问题按钮 */
    .stButton > button {
        background-color: #3c3c3e;
        color: #e0e0e0;
        border-radius: 12px; /* 更大的圆角 */
        border: 1px solid #4a4a4c;
        padding: 10px 20px;
        font-size: 1rem;
        margin: 5px;
        transition: background-color 0.2s ease, border-color 0.2s ease;
        box-shadow: none; /* 移除默认阴影 */
    }
    .stButton > button:hover {
        background-color: #4a4a4c;
        border-color: #5a5a5c;
        color: #ffffff;
    }

    /* 聊天消息容器 */
    .chat-message {
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 10px;
        display: flex;
        align-items: flex-start;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-message.user {
        background-color: #3c3c3e; /* 用户消息背景 */
        align-self: flex-end;
        justify-content: flex-end;
        margin-left: auto;
    }
    .chat-message.ai {
        background-color: #2c2c2e; /* AI消息背景 */
        align-self: flex-start;
        justify-content: flex-start;
        margin-right: auto;
    }
    .chat-message .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: bold;
        color: #fff;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .chat-message.user .avatar {
        background-color: #4a90e2; /* 用户头像颜色 */
        margin-left: 10px;
        margin-right: 0;
    }
    .chat-message.ai .avatar {
        background-color: #8e44ad; /* AI头像颜色 */
    }
    .chat-message .content {
        flex-grow: 1;
        word-break: break-word; /* 单词换行 */
    }

    /* 输入框容器 */
    .stForm {
        background-color: #2c2c2e; /* 输入框区域背景 */
        padding: 15px 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* 文本输入框 */
    .stTextInput > div > div > input {
        background-color: #3c3c3e;
        color: #e0e0e0;
        border-radius: 10px;
        border: 1px solid #4a4a4c;
        padding: 12px 15px;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 0.1rem #4a90e2;
    }

    /* 发送按钮 */
    .stForm button[type="submit"] {
        background-color: #4a90e2; /* 蓝色发送按钮 */
        color: white;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        transition: background-color 0.2s ease;
    }
    .stForm button[type="submit"]:hover {
        background-color: #357ABD; /* 鼠标悬停时的颜色 */
    }

    /* 文本框占位符颜色 */
    .stTextInput > div > div > input::placeholder {
        color: #999;
    }

    /* 滚动条样式 (Webkit browsers) */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #2c2c2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #777;
    }

    /* 调整默认的 Streamlit 容器宽度 */
    .css-1dp5atx.e1fqkh3o1 {
        max-width: 1000px; /* 设置最大宽度 */
        padding-left: 20px;
        padding-right: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- 页面配置 ---
st.set_page_config(
    page_title="Chat LangChain",
    page_icon="🔗",
    layout="centered" # 保持内容居中，但通过CSS调整宽度
)

# --- 侧边栏：聊天历史 ---
with st.sidebar:
    st.markdown("<h2>Chat History <span style='float:right;'>✏️</span></h2>", unsafe_allow_html=True)
    st.markdown("---") # 分隔线

    # 这里可以添加实际的聊天历史管理功能，目前仅为占位符
    # 例如：显示过去的会话列表
    st.write("（暂无历史记录）")

# --- 主内容区域 ---
st.title("Chat LangChain 🔗")

# 模型选择（仿照图片，当前只用一个模型）
selected_model = st.selectbox(
    " ", # 空标签，让下拉框更紧凑
    ["Gemini 2.0 Flash"], # 实际使用的是langchain_qa_backend中定义的模型
    label_visibility="collapsed" # 隐藏标签
)

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "current_url" not in st.session_state:
    st.session_state.current_url = "https://python.langchain.com/docs/" # 默认URL

# URL输入框，用于指定知识库来源
new_url = st.text_input(
    "输入知识库URL (例如: https://python.langchain.com/docs/)",
    st.session_state.current_url,
    placeholder="输入一个有效的URL，例如：https://python.langchain.com/docs/"
)

if st.session_state.current_url != new_url:
    st.session_state.current_url = new_url
    # 当URL改变时，清空缓存并重新加载
    st.session_state.retriever = None
    st.session_state.retrieval_chain = None
    st.session_state.messages = [] # 清空聊天历史

# 初始化知识库和问答链
if st.session_state.retriever is None:
    if st.session_state.current_url:
        st.session_state.retriever = load_and_process_documents(st.session_state.current_url)
    else:
        st.warning("请在上方输入一个有效的知识库URL。")

if st.session_state.retrieval_chain is None and st.session_state.retriever:
    st.session_state.retrieval_chain = get_retrieval_chain(st.session_state.retriever)

# 预设问题按钮
st.markdown("<div style='display: flex; flex-wrap: wrap; justify-content: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
suggested_questions = [
    "如何使用RecursiveUrlLoader加载网页内容？",
    "如何定义我的LangGraph图的状态模式？",
    "如何使用Ollama在本地运行模型？",
    "解释RAG技术以及LangGraph如何实现它们。",
]
for q in suggested_questions:
    if st.button(q):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("正在思考..."):
            response = answer_question(st.session_state.retrieval_chain, q)
            st.session_state.messages.append({"role": "ai", "content": response})
        st.rerun() # 重新运行以显示新消息
st.markdown("</div>", unsafe_allow_html=True)


# 显示聊天历史
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user">
                    <div class="content">{message["content"]}</div>
                    <div class="avatar">U</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message ai">
                    <div class="avatar">AI</div>
                    <div class="content">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)

# 用户输入框和发送按钮
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_input = st.text_input(
            "How can I...",
            placeholder="输入你的问题...",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.form_submit_button("▶️") # 使用一个简单的箭头作为发送按钮

    if send_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("正在思考..."):
            response = answer_question(st.session_state.retrieval_chain, user_input)
            st.session_state.messages.append({"role": "ai", "content": response})
        st.rerun() # 重新运行以显示新消息

