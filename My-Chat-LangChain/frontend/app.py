import streamlit as st
import requests
import json
import os
import time
import uuid

# --- 1. API Config ---
# Read from Environment Variable for Cloud Deployment
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
STREAM_ENDPOINT = f"{BACKEND_URL}/chat/stream"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload_file"

# --- 2. Page Config & Style ---
st.set_page_config(
    page_title="Stream-Agent | AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# --- 3. Session State Management ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am your unified Stream-Agent. I can search the web, read your uploaded PDFs, and analyze papers. Try uploading a file or asking a question!"
    })

def reset_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I started a new session for you."
    })
    # Clear file uploader if possible, though Streamlit file_uploader state is tricky to clear programmatically without a key hack.
    # For now we just reset chat history.

# --- 4. Sidebar & Configuration ---
with st.sidebar:
    st.title("🤖 Stream-Agent v6.0")
    
    if st.button("🔄 New Chat", type="primary"):
        reset_chat()
        st.rerun()
        
    st.markdown("---")
    
    # API Key Config
    with st.expander("⚙️ API Keys", expanded=True):
        serper_key = st.text_input("Serper API Key", type="password", value=os.environ.get("SERPER_API_KEY", ""))
        brightdata_key = st.text_input("BrightData API Key", type="password", value=os.environ.get("BRIGHT_DATA_API_KEY", ""))
        papersearch_key = st.text_input("Paper Search API Key", type="password", value=os.environ.get("PAPER_SEARCH_API_KEY", ""))
        
        st.session_state.api_keys = {
            "SERPER_API_KEY": serper_key,
            "BRIGHT_DATA_API_KEY": brightdata_key,
            "PAPER_SEARCH_API_KEY": papersearch_key
        }
    
    st.markdown("---")
    st.markdown("### 📂 File Upload")
    uploaded_file = st.file_uploader("Upload PDF for RAG", type=['pdf'], key="file_uploader")
    
    if uploaded_file:
        # Handle file upload automatically
        # Using a simple key based check to avoid re-uploading on every rerun
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            with st.spinner("Uploading and Ingesting file..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    # Only bypass proxies if running locally (localhost/127.0.0.1)
                    proxies = {"http": None, "https": None} if "127.0.0.1" in BACKEND_URL or "localhost" in BACKEND_URL else None
                    
                    response = requests.post(UPLOAD_ENDPOINT, files=files, proxies=proxies)
                    
                    if response.status_code == 200:
                        result = response.json()
                        msg = result.get("message", f"Uploaded: {uploaded_file.name}")
                        st.success(msg)
                        st.session_state.last_uploaded_file = uploaded_file.name
                    else:
                        st.error("Upload failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.caption(f"Session ID: `{st.session_state.thread_id}`")

# --- 5. Main Chat Interface ---
st.subheader("💬 Universal AI Assistant")

# Display Message History
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        # Try to render JSON content (Paper Analysis / Profile)
        content = msg["content"]
        try:
            # Heuristic check if it looks like JSON
            if content.strip().startswith("{") and "type" in content:
                data_obj = json.loads(content)
                if data_obj.get("type") == "paper_analysis":
                    d = data_obj.get("data", {})
                    st.markdown(f"### 📄 {d.get('title')}")
                    st.caption(f"**Authors:** {', '.join(d.get('authors', []))}")
                    st.info(d.get('summary'))
                elif data_obj.get("type") == "linkedin_profile":
                    d = data_obj.get("data", {})
                    st.markdown(f"### 👔 {d.get('full_name')}")
                    st.caption(f"{d.get('headline')} | {d.get('location')}")
                    st.info(d.get('summary'))
                else:
                    st.markdown(content)
            else:
                st.markdown(content)
        except:
            st.markdown(content)

# --- 6. Streaming Logic ---
def stream_generator(prompt):
    """
    Generator that yields chunks from the backend SSE stream.
    Handles both text tokens and tool events.
    """
    active_keys = {k: v for k, v in st.session_state.api_keys.items() if v}
    payload = {
        "message": prompt,
        "thread_id": st.session_state.thread_id,
        "api_keys": active_keys
    }
    
    try:
        # Only bypass proxies if running locally (localhost/127.0.0.1)
        proxies = {"http": None, "https": None} if "127.0.0.1" in BACKEND_URL or "localhost" in BACKEND_URL else None
        
        with requests.post(STREAM_ENDPOINT, json=payload, stream=True, proxies=proxies) as response:
            response.raise_for_status()
            
            # Streamlit's status container for tool outputs
            status_container = st.status("Thinking...", expanded=True)
            
            # Standard requests.iter_lines() splits by \n.
            # SSE sends "event: ...\ndata: ...\n\n"
            # We need a slightly smarter parser or assumption.
            
            event_type = None
            
            for line in response.iter_lines():
                if not line:
                    # End of event block (empty line)
                    event_type = None
                    continue
                
                decoded_line = line.decode('utf-8')
                
                if decoded_line.startswith("event: "):
                    event_type = decoded_line[7:].strip()
                
                elif decoded_line.startswith("data: "):
                    data = decoded_line[6:]
                    
                    if event_type == "tool_start":
                        status_container.write(f"🛠️ Calling Tool: **{data}**...")
                    
                    elif event_type == "tool_end":
                        try:
                            tool_data = json.loads(data)
                            status_container.markdown(f"✅ **{tool_data['name']}** finished.")
                            with status_container.expander(f"Result ({tool_data['name']})"):
                                st.code(tool_data['output'])
                        except:
                            status_container.write("Tool finished.")
                            
                    else:
                        # Default text content (no event type or event='on_chat_model_stream')
                        # Yield text for st.write_stream
                        yield data

            status_container.update(label="Finished thinking!", state="complete", expanded=False)
            
    except Exception as e:
        yield f"Error: {str(e)}"

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Display Assistant Response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Use st.write_stream to consume our generator
        full_response = st.write_stream(stream_generator(prompt))
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
