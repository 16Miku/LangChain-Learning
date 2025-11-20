import os
import asyncio
from langchain_core.tools import tool
from langchain_qa_backend import (
    create_vector_store_from_url,
    load_vector_store,
    get_retrieval_chain,
    get_persist_directory_for_url
)
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any
import numpy as np

# RAG Chain Cache (Shared with main app if needed, or separate)
# Note: Ideally this should be a shared cache, but for tool simplicity we might maintain a local one 
# or import the one from main if structure allows. For now, let's keep a local cache for the tool.
tool_rag_chain_cache = {}

def clean_metadata(metadata: dict) -> dict:
    """Recursively convert numpy types to python types for JSON serialization"""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, np.float32):
            cleaned[key] = float(value)
        elif isinstance(value, dict):
            cleaned[key] = clean_metadata(value)
        else:
            cleaned[key] = value
    return cleaned

@tool
async def ingest_knowledge(url: str):
    """
    将指定的网页URL内容摄取并处理为知识库。
    当用户要求学习某个新网页或基于某个网页进行问答时，首先调用此工具。
    """
    print(f"\n📚 [Knowledge] 正在摄取知识库: {url} ...")
    
    # Check if chain already exists in cache
    if url in tool_rag_chain_cache:
        print(f"  -> 知识库已在缓存中: {url}")
        return f"知识库已准备就绪 (Cached): {url}"

    # Check persistence
    persist_directory = get_persist_directory_for_url(url)
    if os.path.exists(persist_directory):
        print(f"  -> 从磁盘加载知识库: {persist_directory}")
        vector_store = load_vector_store(persist_directory)
    else:
        print(f"  -> 创建新知识库: {url}")
        vector_store = await create_vector_store_from_url(url, persist_directory)
    
    if not vector_store:
        return f"❌ 错误: 无法处理 URL {url}"

    # Create Chain
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    retrieval_chain = get_retrieval_chain(base_retriever)
    
    if not retrieval_chain:
        return f"❌ 错误: 无法为 {url} 创建 RAG 链"
        
    tool_rag_chain_cache[url] = retrieval_chain
    print(f"✅ [Knowledge] 知识库摄取完成: {url}")
    return f"成功学习了网页内容: {url}"

@tool
async def query_knowledge(query: str, url: str):
    """
    基于已摄取的网页知识库回答问题。
    必须先调用 `ingest_knowledge` 确保该 URL 已被处理。
    """
    print(f"\n🤔 [RAG] 正在查询知识库 ({url}): {query} ...")
    
    if url not in tool_rag_chain_cache:
        # Try to auto-ingest if not found (optional, but robust)
        print(f"  -> 警告: URL {url} 未在缓存中，尝试自动摄取...")
        await ingest_knowledge(url)
        if url not in tool_rag_chain_cache:
            return f"❌ 错误: 知识库未找到且无法自动加载: {url}"

    chain = tool_rag_chain_cache[url]
    
    try:
        # Minimal history for single-turn tool usage, or pass full history if available in context
        response = await chain.ainvoke({
            "input": query,
            "chat_history": [] # Tool call usually handles single specific query
        })
        
        answer = response["answer"]
        source_documents = response.get("context", [])
        
        # Format sources for the Agent
        sources_text = ""
        for i, doc in enumerate(source_documents[:3]): # Limit to top 3 sources
            cleaned_meta = clean_metadata(doc.metadata)
            source_url = cleaned_meta.get("source", "Unknown")
            sources_text += f"\n- Source {i+1} ({source_url}): {doc.page_content[:100]}..."

        final_output = f"{answer}\n\n参考来源:{sources_text}"
        print(f"✅ [RAG] 查询完成。")
        return final_output

    except Exception as e:
        print(f"❌ [RAG] 查询出错: {e}")
        return f"查询知识库时发生错误: {e}"
