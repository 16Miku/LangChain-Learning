# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any # 引入 Dict 和 Any 用于更灵活的类型定义
import numpy as np # 导入 numpy 库，以便我们能识别它的类型
import os # 导入 os 库来检查文件夹是否存在


# 导入我们重构后的后端逻辑模块
from langchain_qa_backend import (
    create_vector_store, 
    load_vector_store, 
    get_retrieval_chain, 
    get_persist_directory_for_url
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


# --- 5. 定义 API 端点 (Endpoint) ---
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the RAG Backend API!"}

@app.post("/chat", response_model=ChatResponse, tags=["RAG Chat"])
async def chat_endpoint(request: ChatRequest):
    url = request.url
    query = request.query
    
    if url in rag_chain_cache:
        retrieval_chain = rag_chain_cache[url]
        print(f"从缓存中获取 RAG 链: {url}")
    else:
        # --- 核心重构：实现持久化加载逻辑 ---
        # 1. 根据 URL 生成唯一的持久化目录路径
        persist_directory = get_persist_directory_for_url(url)
        
        # 2. 检查这个目录是否已经存在
        if os.path.exists(persist_directory):
            # 如果存在，直接从磁盘加载
            print(f"从磁盘持久化目录加载知识库: {persist_directory}")
            vector_store = load_vector_store(persist_directory)
        else:
            # 如果不存在，才执行完整的创建流程
            print(f"磁盘上无此知识库，开始为 URL 创建新的知识库: {url}")
            vector_store = await create_vector_store(url, persist_directory)
        
        if not vector_store:
            raise HTTPException(status_code=500, detail="Failed to load or create vector store.")
        
        # 3. 用获取到的 vector_store 创建检索器和 RAG 链
        # 我们让基础检索器召回更多的文档，为重排器提供充足的候选材料
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 200})
        retrieval_chain = get_retrieval_chain(base_retriever)
        
        if not retrieval_chain:
            raise HTTPException(status_code=500, detail="Failed to create RAG chain.")
        
        # 4. 将最终的链存入内存缓存
        rag_chain_cache[url] = retrieval_chain
        print(f"新的 RAG 链已创建并缓存到内存: {url}")

    formatted_chat_history = []
    for item in request.chat_history:
        if item.role.lower() == "user":
            formatted_chat_history.append(HumanMessage(content=item.content))
        elif item.role.lower() == "ai":
            formatted_chat_history.append(AIMessage(content=item.content))

    try:
        # 调用RAG链
        response = retrieval_chain.invoke({
            "input": query,
            "chat_history": formatted_chat_history
        })
        
        # --- 核心修改：从 RAG 链的响应中提取源文档 ---
        # create_retrieval_chain 返回的响应字典中，'context' 键对应的值就是检索到的文档列表
        source_documents = response.get("context", [])
        
        # --- 核心修改：在构造响应前，清洗源文档的元数据 ---
        formatted_sources = []
        for doc in source_documents:
            # 对每个文档的 metadata 调用我们的清洗函数
            cleaned_meta = clean_metadata(doc.metadata)
            formatted_sources.append(
                SourceDocument(page_content=doc.page_content, metadata=cleaned_meta)
            )
        
        # 构造并返回包含答案和源文档的完整响应
        return ChatResponse(answer=response["answer"], source_documents=formatted_sources)

    except Exception as e:
        print(f"调用 RAG 链时出错: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the answer: {str(e)}")