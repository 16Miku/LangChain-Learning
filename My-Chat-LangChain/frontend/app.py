import streamlit as st
import requests
import json
import os

# --- 1. API Config ---
BACKEND_URL_ENDPOINT = "http://127.0.0.1:8000/chat_url"
BACKEND_FILE_ENDPOINT = "http://127.0.0.1:8000/chat_file"
BACKEND_AGENT_ENDPOINT = "http://127.0.0.1:8000/chat_agent"

# --- 2. Page Config & Style ---
st.set_page_config(
    page_title="Chat LangChain | Agentic Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

# --- 3. API Helper Functions ---
def get_backend_response_from_url(url: str, query: str, chat_history: list):
    try:
        payload = {"url": url, "query": query, "chat_history": chat_history}
        proxies = {"http": None, "https": None}
        response = requests.post(BACKEND_URL_ENDPOINT, json=payload, timeout=180, proxies=proxies)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Backend Error (URL): {e}", "source_documents": []}

def get_backend_response_from_file(query: str, chat_history: list, uploaded_file):
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {'query': query, 'chat_history_str': json.dumps(chat_history)}
        proxies = {"http": None, "https": None}
        response = requests.post(BACKEND_FILE_ENDPOINT, files=files, data=data, timeout=300, proxies=proxies)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Backend Error (File): {e}", "source_documents": []}

def get_agent_response(message: str, thread_id: str, api_keys: dict):
    try:
        payload = {
            "message": message,
            "thread_id": thread_id,
            "api_keys": api_keys
        }
        proxies = {"http": None, "https": None}
        # Use stream=False for now, will just await final response
        response = requests.post(BACKEND_AGENT_ENDPOINT, json=payload, timeout=300, proxies=proxies)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"response": f"Agent Error: {e}"}

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("## 🤖 Agentic RAG Platform v5.0", unsafe_allow_html=True)
    st.markdown("---")
    
    # API Key Configuration
    with st.expander("⚙️ Agent Configuration", expanded=True):
        serper_key = st.text_input("Serper API Key (Search)", type="password", value=os.environ.get("SERPER_API_KEY", ""))
        brightdata_key = st.text_input("BrightData API Key", type="password", value=os.environ.get("BRIGHT_DATA_API_KEY", ""))
        papersearch_key = st.text_input("Paper Search API Key", type="password", value=os.environ.get("PAPER_SEARCH_API_KEY", ""))
        
        # Store keys in session for current run
        st.session_state.api_keys = {
            "SERPER_API_KEY": serper_key,
            "BRIGHT_DATA_API_KEY": brightdata_key,
            "PAPER_SEARCH_API_KEY": papersearch_key
        }

    st.markdown("---")
    st.markdown("**Modes:**\n1. **🤖 Universal Agent:** Autonomous research, search, and RAG orchestration.\n2. **🔗 Web RAG:** Focused Q&A on specific URLs.\n3. **📄 Doc RAG:** Local PDF analysis.")

# --- 5. Main Content ---
st.title("My Chat LangChain 🤖")

# Create tabs
tab_agent, tab_url, tab_file = st.tabs(["🤖 Universal Agent", "🔗 Web RAG", "📄 Doc RAG"])

# --- Tab 1: Universal Agent ---
with tab_agent:
    st.header("Autonomous Research Assistant")
    
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
        # Welcome message
        st.session_state.agent_messages.append({
            "role": "assistant", 
            "content": "Hello! I am your universal AI assistant. I can search the web, analyze papers, find people on LinkedIn, and learn from webpages you provide. How can I help you today?"
        })

    # Display history
    for msg in st.session_state.agent_messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            # Check if content is JSON (structured output)
            try:
                content_json = json.loads(msg["content"])
                if isinstance(content_json, dict) and "type" in content_json:
                    # Handle structured rendering
                    data = content_json.get("data", {})
                    if content_json["type"] == "paper_analysis":
                        st.markdown(f"### 📄 Paper Analysis: {data.get('title')}")
                        st.caption(f"**Authors:** {', '.join(data.get('authors', []))}")
                        st.info(f"**Field:** {data.get('research_field')}")
                        st.markdown(f"**Summary:**\n{data.get('summary')}")
                        st.success(f"**Contact:** {data.get('author_contact')}")
                    elif content_json["type"] == "linkedin_profile":
                        st.markdown(f"### 👔 LinkedIn Profile: {data.get('full_name')}")
                        st.caption(f"{data.get('headline')} | {data.get('location')}")
                        st.markdown(f"**Summary:**\n{data.get('summary')}")
                        st.markdown("**Experience:**")
                        for exp in data.get('experience', []):
                            st.text(f"• {exp}")
                        st.success(f"**Contact:** {data.get('contact')}")
                    else:
                         st.markdown(msg["content"])
                else:
                    st.markdown(msg["content"])
            except json.JSONDecodeError:
                # Not JSON, render as markdown
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask me anything (e.g., 'Analyze this paper...', 'Find web info about...')"):
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Agent is thinking & using tools..."):
                # Prepare keys, filtering out empty ones
                active_keys = {k: v for k, v in st.session_state.api_keys.items() if v}
                
                response_data = get_agent_response(
                    message=prompt,
                    thread_id="demo_thread_1", # Fixed thread for demo
                    api_keys=active_keys
                )
                response_content = response_data.get("response", "Error processing request.")
                
                # Render response (Markdown or Structured)
                try:
                    content_json = json.loads(response_content)
                    if isinstance(content_json, dict) and "type" in content_json:
                        # Handle structured rendering (duplicated logic for immediate display)
                        data = content_json.get("data", {})
                        if content_json["type"] == "paper_analysis":
                            st.markdown(f"### 📄 Paper Analysis: {data.get('title')}")
                            st.caption(f"**Authors:** {', '.join(data.get('authors', []))}")
                            st.info(f"**Field:** {data.get('research_field')}")
                            st.markdown(f"**Summary:**\n{data.get('summary')}")
                            st.success(f"**Contact:** {data.get('author_contact')}")
                        elif content_json["type"] == "linkedin_profile":
                            st.markdown(f"### 👔 LinkedIn Profile: {data.get('full_name')}")
                            st.caption(f"{data.get('headline')} | {data.get('location')}")
                            st.markdown(f"**Summary:**\n{data.get('summary')}")
                            st.markdown("**Experience:**")
                            for exp in data.get('experience', []):
                                st.text(f"• {exp}")
                            st.success(f"**Contact:** {data.get('contact')}")
                        else:
                             st.markdown(response_content)
                    else:
                        st.markdown(response_content)
                except json.JSONDecodeError:
                    st.markdown(response_content)

                st.session_state.agent_messages.append({"role": "assistant", "content": response_content})


# --- Tab 2: Web RAG (Legacy) ---
with tab_url:
    st.header("Legacy Web RAG")
    # (Keep existing logic minimal or exactly as before)
    if "url_messages" not in st.session_state:
        st.session_state.url_messages = []
    if "current_url" not in st.session_state:
        st.session_state.current_url = "https://python.langchain.com/docs/modules/agents/"

    col1, col2 = st.columns([3, 1])
    with col1:
        new_url = st.text_input("URL:", st.session_state.current_url, key="url_input")
    with col2:
        st.selectbox("Model:", ["Gemini 2.5 Flash"], disabled=True, key="url_model_select")

    if st.session_state.current_url != new_url:
        st.session_state.current_url = new_url
        st.session_state.url_messages = []
        st.rerun()

    for message in st.session_state.url_messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                 with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**{i+1}.** {source['page_content'][:200]}...")

    if prompt := st.chat_input("Ask about the URL...", key="url_chat"):
        st.session_state.url_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = get_backend_response_from_url(st.session_state.current_url, prompt, st.session_state.url_messages[:-1])
                answer = response.get("answer", "Error")
                sources = response.get("source_documents", [])
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                         for i, source in enumerate(sources):
                            st.markdown(f"**{i+1}.** {source['page_content'][:200]}...")
                st.session_state.url_messages.append({"role": "assistant", "content": answer, "sources": sources})

# --- Tab 3: Doc RAG (Legacy) ---
with tab_file:
    st.header("Legacy Doc RAG")
    # (Keep existing logic minimal)
    if "file_messages" not in st.session_state:
        st.session_state.file_messages = []
    
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        if "current_file_id" not in st.session_state or st.session_state.current_file_id != uploaded_file.file_id:
            st.session_state.current_file_id = uploaded_file.file_id
            st.session_state.file_messages = []
            st.info("File loaded.")

        for message in st.session_state.file_messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Sources"):
                         for i, source in enumerate(message["sources"]):
                            st.markdown(f"**{i+1}.** {source['page_content'][:200]}...")

        if prompt := st.chat_input("Ask about the file...", key="file_chat"):
            st.session_state.file_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = get_backend_response_from_file(prompt, st.session_state.file_messages[:-1], uploaded_file)
                    answer = response.get("answer", "Error")
                    sources = response.get("source_documents", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                             for i, source in enumerate(sources):
                                st.markdown(f"**{i+1}.** {source['page_content'][:200]}...")
                    st.session_state.file_messages.append({"role": "assistant", "content": answer, "sources": sources})
