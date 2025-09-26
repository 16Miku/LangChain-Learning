# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any # 引入 Dict 和 Any 用于更灵活的类型定义

# 导入我们重构后的后端逻辑模块
from langchain_qa_backend import load_and_process_documents, get_retrieval_chain
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


# --- 4. 定义 API 端点 (Endpoint) ---
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
        print(f"缓存未命中。为 URL 创建新的 RAG 链: {url}")
        retriever = await load_and_process_documents(url)
        if not retriever:
            raise HTTPException(status_code=500, detail="Failed to load or process documents from the given URL.")
        retrieval_chain = get_retrieval_chain(retriever)
        if not retrieval_chain:
            raise HTTPException(status_code=500, detail="Failed to create the RAG retrieval chain.")
        rag_chain_cache[url] = retrieval_chain
        print(f"新的 RAG 链已创建并缓存: {url}")

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
        
        # 将 LangChain 的 Document 对象转换为我们定义的 Pydantic SourceDocument 模型列表
        # 这是一个标准的适配器模式，确保 API 的输出格式稳定可控
        formatted_sources = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata) 
            for doc in source_documents
        ]
        
        # 构造并返回包含答案和源文档的完整响应
        return ChatResponse(answer=response["answer"], source_documents=formatted_sources)

    except Exception as e:
        print(f"调用 RAG 链时出错: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the answer: {str(e)}")