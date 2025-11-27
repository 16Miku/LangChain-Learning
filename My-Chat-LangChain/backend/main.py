# Patch sqlite3 with pysqlite3-binary to satisfy chroma requirements on Vercel
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import shutil
import json

# Import new agent service
from agent_service import chat_with_agent, chat_with_agent_stream
# Import RAG backend functions
from langchain_qa_backend import ingest_file

app = FastAPI(
    title="Stream-Agent Backend API",
    description="An API for the Unified Agent application powered by LangChain and Google Gemini.",
    version="6.0.0",
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Configuration ---
TEMP_UPLOAD_DIR = "/tmp/temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Data Models ---

class AgentChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"
    api_keys: Optional[Dict[str, str]] = None

class AgentChatResponse(BaseModel):
    response: str

class FileUploadResponse(BaseModel):
    filename: str
    message: str

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Stream-Agent Backend API v6.0!"}

# --- Agent Chat Endpoint (Streaming) ---
@app.post("/chat/stream", tags=["Agent Chat"])
async def chat_stream_endpoint(request: AgentChatRequest):
    """
    Stream endpoint for chatting with the autonomous agent.
    Returns SSE (Server-Sent Events) with tokens and tool events.
    """
    return StreamingResponse(
        chat_with_agent_stream(
            message=request.message, 
            thread_id=request.thread_id, 
            api_keys=request.api_keys
        ),
        media_type="text/event-stream"
    )

# --- Agent Chat Endpoint (Sync - Legacy/Backup) ---
@app.post("/chat_agent", response_model=AgentChatResponse, tags=["Agent Chat"])
async def chat_agent_endpoint(request: AgentChatRequest):
    """
    Endpoint for chatting with the autonomous agent (Synchronous).
    Now supports unified RAG through tools.
    """
    try:
        response_content = await chat_with_agent(
            message=request.message, 
            thread_id=request.thread_id,
            api_keys=request.api_keys
        )
        return AgentChatResponse(response=response_content)
    except Exception as e:
        print(f"Error in Agent Chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- File Upload Endpoint ---
@app.post("/upload_file", response_model=FileUploadResponse, tags=["File Management"])
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Upload a file to the backend temporary storage and automatically ingest it into the knowledge base.
    The Agent can then query this knowledge.
    """
    try:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"✅ File uploaded: {file.filename}")
        
        # Automatically ingest the file
        print(f"📚 Auto-ingesting file: {file.filename} ...")
        success = await ingest_file(file_path, file.filename)
        
        if success:
            return FileUploadResponse(
                filename=file.filename,
                message=f"File {file.filename} uploaded and learned successfully! You can now ask questions about it."
            )
        else:
             return FileUploadResponse(
                filename=file.filename,
                message=f"File {file.filename} uploaded, but ingestion failed. Agent might not know about it."
            )
            
    except Exception as e:
        print(f"❌ File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
