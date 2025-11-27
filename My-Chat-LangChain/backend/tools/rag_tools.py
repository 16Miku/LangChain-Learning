import os
import asyncio
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langchain_qa_backend import (
    ingest_url,
    ingest_file,
    get_retrieval_chain
)
import numpy as np

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
async def ingest_knowledge(source: str, type: str):
    """
    统一的知识摄取工具。将网页URL或本地文件内容摄取到统一的向量知识库中。
    
    Args:
        source (str): 资源路径。如果是URL则为http链接，如果是文件则为文件名（后端已保存到临时区）。
        type (str): 资源类型。可选值: 'url', 'file'。
    """
    print(f"\n📚 [Knowledge] 正在摄取知识库: {source} (Type: {type})...")
    
    success = False
    if type == 'url':
        success = await ingest_url(source)
    elif type == 'file':
        # For file ingestion, the backend API should have already saved the file 
        # and passed the path. However, 'source' here is likely just the filename 
        # if called by the LLM based on user context.
        # We assume the file is in a temporary holding area known to the backend.
        # For simplicity in this tool, we might need the full path.
        # Let's assume 'source' passed by Agent is the filename user sees.
        # We need to look it up in a temp dir.
        # On Vercel, only /tmp is writable
        temp_dir = "/tmp/temp_uploads" 
        filepath = os.path.join(temp_dir, source)
        
        if os.path.exists(filepath):
            success = await ingest_file(filepath, source)
        else:
            return f"❌ 错误: 找不到文件 {source}。请确认文件已上传。"
    else:
        return f"❌ 错误: 不支持的类型 {type}"
    
    if success:
        print(f"✅ [Knowledge] 知识库摄取完成: {source}")
        return f"成功学习了 {type} 内容: {source}"
    else:
        return f"❌ 错误: 无法处理 {source}"

@tool
async def query_knowledge_base(query: str, source_filter: Optional[str] = None):
    """
    统一的知识库查询工具。
    
    Args:
        query (str): 用户的具体问题。
        source_filter (Optional[str]): 可选。如果指定，仅从该特定的 URL 或文件名中检索答案。
                                     如果不指定，则从整个知识库中检索。
    """
    filter_msg = f" (Filter: {source_filter})" if source_filter else " (Global Search)"
    print(f"\n🤔 [RAG] 正在查询知识库: {query}{filter_msg} ...")
    
    try:
        # Create chain on the fly (lightweight)
        chain = get_retrieval_chain(source_filter)
        
        response = await chain.ainvoke({
            "input": query,
            "chat_history": [] 
        })
        
        answer = response["answer"]
        source_documents = response.get("context", [])
        
        # Format sources
        sources_text = ""
        for i, doc in enumerate(source_documents[:3]):
            cleaned_meta = clean_metadata(doc.metadata)
            src = cleaned_meta.get("source", "Unknown")
            sources_text += f"\n- [{i+1}] {src}: {doc.page_content[:100]}..."

        final_output = f"{answer}\n\n参考来源:{sources_text}"
        print(f"✅ [RAG] 查询完成。")
        return final_output

    except Exception as e:
        print(f"❌ [RAG] 查询出错: {e}")
        return f"查询知识库时发生错误: {e}"
