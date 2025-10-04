# backend/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any # 引入 Dict 和 Any 用于更灵活的类型定义
import numpy as np # 导入 numpy 库，以便我们能识别它的类型
import os # 导入 os 库来检查文件夹是否存在
import json
import tempfile
import hashlib



# 导入我们重构后的后端逻辑模块
from langchain_qa_backend import (
    create_vector_store_from_url,
    create_vector_store_from_file,
    load_vector_store, 
    get_retrieval_chain, 
    get_persist_directory_for_url,
    get_persist_directory_for_file
)
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. 初始化 FastAPI 应用 ---
app = FastAPI(
    title="Enterprise RAG Backend API",
    description="An API for the RAG application powered by LangChain and Google Gemini.",
    version="1.0.0",
)

# --- 2. 简单的内存缓存 ---
rag_chain_cache = {}

# --- 3. 定义 API 数据模型 (核心修改) ---

class ChatHistoryItem(BaseModel):
    """定义聊天历史中单条消息的结构"""
    role: str
    content: str

class ChatRequest(BaseModel):
    """定义 /chat 接口的请求体结构"""
    url: str
    query: str
    chat_history: List[ChatHistoryItem]

# --- 新增模型：定义单个源文档的数据结构 ---
class SourceDocument(BaseModel):
    """定义返回给前端的单个源文档的结构"""
    page_content: str = Field(..., description="源文档的文本内容片段")
    metadata: Dict[str, Any] = Field({}, description="源文档的元数据，通常包含来源URL等")

class ChatResponse(BaseModel):
    """
    定义 /chat 接口的响应体结构
    --- 核心修改：新增 source_documents 字段 ---
    """
    answer: str = Field(..., description="由RAG系统生成的回答")
    source_documents: List[SourceDocument] = Field([], description="答案所依据的源文档列表")


# --- 4. 新增：一个用于清理 NumPy 类型的辅助函数 ---
def clean_metadata(metadata: dict) -> dict:
    """
    递归地遍历元数据字典，将所有 numpy.float32 类型转换为标准的 float 类型。
    """
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, np.float32):
            cleaned[key] = float(value)
        elif isinstance(value, dict):
            cleaned[key] = clean_metadata(value)
        else:
            cleaned[key] = value
    return cleaned


# --- 4. API 端点 ---
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the RAG Backend API v4.0!"}

# --- URL 问答端点 (逻辑重构) ---
@app.post("/chat_url", response_model=ChatResponse, tags=["RAG Chat"])
async def chat_url_endpoint(request: ChatRequest):
    url = request.url
    query = request.query
    
    if url in rag_chain_cache:
        retrieval_chain = rag_chain_cache[url]
        print(f"从内存缓存中获取 RAG 链 (URL): {url}")
    else:
        persist_directory = get_persist_directory_for_url(url)
        
        if os.path.exists(persist_directory):
            print(f"从磁盘加载知识库 (URL): {persist_directory}")
            vector_store = load_vector_store(persist_directory)
        else:
            print(f"创建新知识库 (URL): {url}")
            vector_store = await create_vector_store_from_url(url, persist_directory)
        
        if not vector_store:
            raise HTTPException(status_code=500, detail="Failed to process URL.")
        
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        retrieval_chain = get_retrieval_chain(base_retriever)
        if not retrieval_chain:
            raise HTTPException(status_code=500, detail="Failed to create RAG chain.")
        rag_chain_cache[url] = retrieval_chain
        print(f"RAG 链已为 URL {url} 创建并缓存。")

    # --- 后续调用逻辑 (与文件端点复用) ---
    return await invoke_rag_chain(retrieval_chain, query, request.chat_history)

# --- 新增：文件问答端点 ---
@app.post("/chat_file", response_model=ChatResponse, tags=["RAG Chat"])
async def chat_file_endpoint(
    query: str = Form(...),
    chat_history_str: str = Form("[]"),
    file: UploadFile = File(...)
):
    # 1. 安全地处理上传的文件
    # 使用 with 语句确保临时目录在操作完成后被自动清理
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, file.filename)
        
        # 读取文件内容用于计算哈希和写入临时文件
        file_content = await file.read()
        with open(temp_filepath, "wb") as f:
            f.write(file_content)
        
        # 2. 持久化与加载逻辑
        persist_directory = get_persist_directory_for_file(file.filename, file_content)
        
        # 使用持久化目录作为内存缓存的 key，因为它是唯一的
        if persist_directory in rag_chain_cache:
            retrieval_chain = rag_chain_cache[persist_directory]
            print(f"从内存缓存中获取 RAG 链 (File): {file.filename}")
        else:
            if os.path.exists(persist_directory):
                print(f"从磁盘加载知识库 (File): {persist_directory}")
                vector_store = load_vector_store(persist_directory)
            else:
                print(f"创建新知识库 (File): {file.filename}")
                vector_store = await create_vector_store_from_file(temp_filepath, persist_directory)

            if not vector_store:
                raise HTTPException(status_code=500, detail="Failed to process File.")
            
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
            retrieval_chain = get_retrieval_chain(base_retriever)
            if not retrieval_chain:
                raise HTTPException(status_code=500, detail="Failed to create RAG chain.")
            rag_chain_cache[persist_directory] = retrieval_chain
            print(f"RAG 链已为文件 {file.filename} 创建并缓存。")

    # 3. 解析聊天历史并调用链
    chat_history = json.loads(chat_history_str)
    return await invoke_rag_chain(retrieval_chain, query, chat_history)

# --- 修改：复用的 RAG 调用函数 ---
async def invoke_rag_chain(chain, query: str, history: List[Any]): # 将类型提示改为更通用的 List[Any]
    """
    一个可复用的函数，用于格式化历史记录、调用 RAG 链并处理响应。
    现在它可以同时接受字典列表和 Pydantic 对象列表。
    """
    # 格式化聊天历史
    formatted_chat_history = []
    for item in history:
        # --- 核心修改：使用 hasattr 和 getattr 来安全地访问属性 ---
        # 这种方式对字典 (用 .get()) 和对象 (用 .) 都有效
        if isinstance(item, dict):
            # 如果是字典，使用 .get()
            role = item.get("role")
            content = item.get("content")
        else:
            # 如果是 Pydantic 对象，使用 .role 和 .content
            role = item.role
            content = item.content

        if role == "user":
            formatted_chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_chat_history.append(AIMessage(content=content))
    
    try:
        # 调用链 (后续逻辑不变)
        response = chain.invoke({
            "input": query,
            "chat_history": formatted_chat_history
        })
        
        # 清洗并格式化源文档
        source_documents = response.get("context", [])
        formatted_sources = [
            SourceDocument(page_content=doc.page_content, metadata=clean_metadata(doc.metadata))
            for doc in source_documents
        ]
        return ChatResponse(answer=response["answer"], source_documents=formatted_sources)
    except Exception as e:
        print(f"调用 RAG 链时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))