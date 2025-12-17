import streamlit as st
import requests
import json
import os
import time
import uuid
import base64
import re

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
    st.title("🤖 Stream-Agent v8.0")

    if st.button("🔄 New Chat", type="primary"):
        reset_chat()
        st.rerun()

    st.markdown("---")

    # LLM Provider Selection
    with st.expander("🧠 LLM Configuration", expanded=True):
        llm_provider = st.radio(
            "Select LLM Provider",
            options=["google", "openai_compatible"],
            format_func=lambda x: "Google Gemini (Official)" if x == "google" else "OpenAI Compatible (Proxy)",
            index=0,
            help="Choose between Google's official API or a third-party OpenAI-compatible proxy"
        )

        if llm_provider == "google":
            st.caption("Using Google Gemini API directly")
            google_model = st.text_input(
                "Model Name",
                value="gemini-2.0-flash-lite",
                help="e.g., gemini-2.0-flash-lite, gemini-1.5-flash, gemini-2.5-flash"
            )
            # Store LLM config
            st.session_state.llm_config = {
                "LLM_PROVIDER": "google",
                "GOOGLE_MODEL": google_model
            }
        else:
            st.caption("Using OpenAI-compatible API endpoint")
            openai_base_url = st.text_input(
                "Base URL",
                value="https://api.openai.com/v1",
                help="e.g., https://api.openrouter.ai/api/v1"
            )
            openai_api_key = st.text_input(
                "API Key",
                type="password",
                help="Your proxy platform's API key"
            )
            openai_model = st.text_input(
                "Model Name",
                value="gpt-4o-mini",
                help="e.g., gpt-4o-mini, google/gemini-2.0-flash"
            )
            # Store LLM config
            st.session_state.llm_config = {
                "LLM_PROVIDER": "openai_compatible",
                "OPENAI_BASE_URL": openai_base_url,
                "OPENAI_API_KEY": openai_api_key,
                "OPENAI_MODEL": openai_model
            }

    st.markdown("---")

    # API Key Config
    with st.expander("⚙️ Tool API Keys", expanded=False):
        serper_key = st.text_input("Serper API Key", type="password", value=os.environ.get("SERPER_API_KEY", ""))
        brightdata_key = st.text_input("BrightData API Key", type="password", value=os.environ.get("BRIGHT_DATA_API_KEY", ""))
        papersearch_key = st.text_input("Paper Search API Key", type="password", value=os.environ.get("PAPER_SEARCH_API_KEY", ""))
        e2b_key = st.text_input("E2B API Key", type="password", value=os.environ.get("E2B_API_KEY", ""), help="For code execution sandbox")

        # Store tool API keys (will be merged with LLM config when sending)
        st.session_state.tool_api_keys = {
            "SERPER_API_KEY": serper_key,
            "BRIGHT_DATA_API_KEY": brightdata_key,
            "PAPER_SEARCH_API_KEY": papersearch_key,
            "E2B_API_KEY": e2b_key
        }

    st.markdown("---")
    st.markdown("### 📂 File Upload")
    st.caption("Supports: PDF, CSV, Excel, JSON, TXT, Python")
    uploaded_file = st.file_uploader(
        "Upload file for analysis",
        type=['pdf', 'csv', 'xlsx', 'xls', 'json', 'txt', 'py'],
        key="file_uploader"
    )
    
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

# Helper function to render content with embedded images
def render_content_with_images(content):
    """
    Render content that may contain embedded base64 images.
    Images are marked as [IMAGE_BASE64:xxx]
    """
    # Check for embedded images
    image_pattern = r'\[IMAGE_BASE64:([A-Za-z0-9+/=]+)\]'
    matches = list(re.finditer(image_pattern, content))

    if not matches:
        # No images, just render markdown
        st.markdown(content)
        return

    # Split content and render parts with images
    last_end = 0
    for match in matches:
        # Render text before the image
        text_before = content[last_end:match.start()]
        if text_before.strip():
            st.markdown(text_before)

        # Render the image
        try:
            image_b64 = match.group(1)
            image_bytes = base64.b64decode(image_b64)
            # Use try/except for Streamlit version compatibility
            try:
                st.image(image_bytes, caption="📊 Generated Chart", use_container_width=True)
            except TypeError:
                st.image(image_bytes, caption="📊 Generated Chart", use_column_width=True)
        except Exception as e:
            st.warning(f"Failed to render image: {e}")

        last_end = match.end()

    # Render remaining text after last image
    remaining_text = content[last_end:]
    if remaining_text.strip():
        st.markdown(remaining_text)

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
                    render_content_with_images(content)
            else:
                render_content_with_images(content)
        except:
            render_content_with_images(content)

# --- 6. Streaming Logic ---
def stream_generator(prompt):
    """
    Generator that yields chunks from the backend SSE stream.
    Handles both text tokens and tool events.
    All data from backend is base64 encoded to handle newlines safely.
    """

    def decode_sse_data(encoded_data: str) -> str:
        """Decode base64 encoded SSE data."""
        try:
            return base64.b64decode(encoded_data.encode('ascii')).decode('utf-8')
        except Exception:
            # Fallback: return as-is if decoding fails
            return encoded_data

    def render_tool_output(output_str, container):
        """Render tool output, handling embedded images."""
        image_pattern = r'\[IMAGE_BASE64:([A-Za-z0-9+/=]+)\]'
        matches = list(re.finditer(image_pattern, output_str))

        if matches:
            # Has images - render them directly in container
            for match in matches:
                try:
                    image_b64 = match.group(1)
                    image_bytes = base64.b64decode(image_b64)
                    # Use try/except for Streamlit version compatibility
                    try:
                        container.image(image_bytes, caption="📊 Generated Chart", use_container_width=True)
                    except TypeError:
                        # Fallback for older Streamlit versions
                        container.image(image_bytes, caption="📊 Generated Chart", use_column_width=True)
                except Exception as e:
                    container.warning(f"Failed to render chart: {e}")

            # Show text output (without image markers) - clean up markdown formatting
            clean_output = re.sub(image_pattern, '', output_str).strip()
            # Remove raw markdown and extra formatting for cleaner display
            clean_output = clean_output.replace('**结果**:', '').replace('**图表已生成并显示在上方**', '')
            clean_output = re.sub(r'```\n?', '', clean_output).strip()
            clean_output = re.sub(r'📊|🖼️', '', clean_output).strip()
            if clean_output and len(clean_output) > 10:
                container.caption(clean_output[:500])
        else:
            # No images, just show text (truncated)
            if output_str:
                container.text(output_str[:1500])

    # Merge LLM config and tool API keys
    llm_config = st.session_state.get("llm_config", {"LLM_PROVIDER": "google"})
    tool_keys = st.session_state.get("tool_api_keys", {})

    # Combine all configs, filtering out empty values
    all_api_keys = {**llm_config, **tool_keys}
    active_keys = {k: v for k, v in all_api_keys.items() if v}

    payload = {
        "message": prompt,
        "thread_id": st.session_state.thread_id,
        "api_keys": active_keys
    }

    try:
        # Only bypass proxies if running locally (localhost/127.0.0.1)
        proxies = {"http": None, "https": None} if "127.0.0.1" in BACKEND_URL or "localhost" in BACKEND_URL else None

        # Headers to ensure proper SSE streaming
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }

        with requests.post(STREAM_ENDPOINT, json=payload, stream=True, proxies=proxies, headers=headers, timeout=300) as response:
            response.raise_for_status()

            # Streamlit's status container for tool outputs
            status_container = st.status("Thinking...", expanded=True)

            # SSE Parser: event type is stored and applied to next data line
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
                    raw_data = decoded_line[6:]

                    if event_type == "text":
                        # Text content - decode base64 and yield for streaming display
                        text_content = decode_sse_data(raw_data)

                        # Filter out Base64 image data from text stream
                        # Images should only be rendered in tool_end results
                        if "[IMAGE_BASE64:" in text_content or "data:image" in text_content.lower():
                            # Check if this looks like base64 data (long alphanumeric string)
                            if len(text_content) > 100 and re.match(r'^[A-Za-z0-9+/=\s]+$', text_content.replace('\n', '')):
                                continue  # Skip pure base64 chunks
                            # Filter out image markers from mixed content
                            text_content = re.sub(r'\[IMAGE_BASE64:[A-Za-z0-9+/=]+\]', '[图表已在上方工具结果中显示]', text_content)
                            text_content = re.sub(r'!\[.*?\]\(data:image/[^)]+\)', '[图表已生成]', text_content)

                        if text_content.strip():
                            yield text_content

                    elif event_type == "tool_start":
                        # Tool start event - decode and display
                        tool_name = decode_sse_data(raw_data)
                        status_container.write(f"🛠️ Calling Tool: **{tool_name}**...")

                    elif event_type == "tool_end":
                        # Tool end event - decode JSON and display result
                        try:
                            decoded_json = decode_sse_data(raw_data)
                            tool_data = json.loads(decoded_json)
                            tool_name = tool_data.get('name', 'unknown')
                            tool_output = tool_data.get('output', '')

                            status_container.markdown(f"✅ **{tool_name}** finished.")

                            # Render output directly in status_container (no nested expanders)
                            if tool_name in ['create_visualization', 'generate_chart_from_data', 'execute_python_code']:
                                # Check if output contains image
                                if "[IMAGE_BASE64:" in tool_output:
                                    render_tool_output(tool_output, status_container)
                                elif tool_output:
                                    status_container.text(tool_output[:1500])
                            elif tool_output:
                                status_container.text(tool_output[:1000])
                        except Exception as e:
                            status_container.write(f"Tool finished. (Error: {str(e)[:80]})")

                    elif event_type == "done":
                        # Stream finished signal, don't yield anything
                        pass

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
