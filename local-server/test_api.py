# test_api.py
from langgraph_sdk import get_client
import asyncio

async def main():
    # 获取 LangGraph 客户端实例，连接到本地运行的服务器
    client = get_client(url="http://localhost:2024")

    # 使用 stream 方法向助手发送消息并流式接收响应
    # 第一个参数 None 表示这是一个无线程运行 (threadless run)
    # 第二个参数 "agent" 是助手的名称，通常在 langgraph.json 中定义
    # input 字典包含了发送给助手的消息内容
    async for chunk in client.runs.stream(
        None,  # 无线程运行
        "agent", # 助手的名称，在 langgraph.json 中定义。
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
    ):
        # 打印接收到的事件类型
        print(f"Receiving new event of type: {chunk.event}...")
        # 打印事件数据
        print(chunk.data)
        print("\n\n")

# 运行主异步函数
asyncio.run(main())
