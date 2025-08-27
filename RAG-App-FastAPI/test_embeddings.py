# test_embeddings.py

import os
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# 确保 GOOGLE_API_KEY 环境变量已设置
if not os.environ.get("GOOGLE_API_KEY"):
    print("GOOGLE_API_KEY 环境变量未设置。尝试通过输入获取...")
    try:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("请输入 Google Gemini 的 API 密钥: ")
    except Exception as e:
        print(f"获取 API 密钥失败: {e}")
        exit(1)

if not os.environ.get("GOOGLE_API_KEY"):
    print("错误: GOOGLE_API_KEY 仍未设置。请确保已设置或正确输入。")
    exit(1)
else:
    print("GOOGLE_API_KEY 已设置。")

try:
    # 初始化嵌入模型
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print("嵌入模型已初始化。")

    # 准备一些测试文档
    test_docs = [
        Document(page_content="Hello world, this is a test sentence for embedding."),
        Document(page_content="Another example sentence to check the embedding functionality."),
        Document(page_content="Short text.")
    ]
    
    print(f"准备嵌入 {len(test_docs)} 个测试文档。")
    
    # 尝试生成嵌入
    # embed_documents 期望一个 Document 对象列表或字符串列表
    test_embeddings = embeddings_model.embed_documents([doc.page_content for doc in test_docs])
    
    if test_embeddings:
        print(f"成功生成 {len(test_embeddings)} 个嵌入。")
        print(f"第一个嵌入向量的维度: {len(test_embeddings[0])}")
        print("嵌入模型工作正常。")
    else:
        print("警告: 嵌入模型返回了空的嵌入列表。")
        print("这可能表示 API 密钥有问题，或网络连接失败。")

except Exception as e:
    print(f"嵌入模型测试失败: {e}")
    print("请检查您的 GOOGLE_API_KEY 是否有效，以及网络连接是否正常。")

