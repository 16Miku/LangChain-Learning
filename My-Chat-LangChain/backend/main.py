# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

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

# --- 3. 定义 API 数据模型 ---
class ChatHistoryItem(BaseModel):
    role: str = Field(..., description="消息发送者的角色 ('user' 或 'ai')")
    content: str = Field(..., description="消息的具体内容")

class ChatRequest(BaseModel):
    url: str = Field(..., description="作为知识库来源的网页URL", example="https://python.langchain.com/docs/")
    query: str = Field(..., description="用户提出的问题", example="How to use RecursiveUrlLoader?")
    chat_history: List[ChatHistoryItem] = Field([], description="之前的聊天历史记录")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="由RAG系统生成的回答")

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
        
        # ****** 关键修改: 使用 await 来调用异步函数 load_and_process_documents ******
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
        # RAG 链的 invoke 方法本身是同步的，但其内部的 LLM 调用可能是异步的。
        # LangChain 会自动处理。如果未来使用 ainvoke，这里也需要 await。
        response = retrieval_chain.invoke({
            "input": query,
            "chat_history": formatted_chat_history
        })
        
        return ChatResponse(answer=response["answer"])

    except Exception as e:
        print(f"调用 RAG 链时出错: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the answer: {str(e)}")