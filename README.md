

# LangChain

## Build a Chatbot

代码
[Chatbot](Chatbot.ipynb)

链接
https://python.langchain.com/docs/tutorials/chatbot/



## Simple-LLM-Application

代码
[texSimple-LLM-Applicationt](Simple-LLM-Application.ipynb)

链接
https://python.langchain.com/docs/tutorials/llm_chain/



## SimpleAgent

代码
[SimpleAgent](SimpleAgent.ipynb)

链接
https://python.langchain.com/docs/tutorials/agents/


## Build a semantic search engine

[Build a semantic search engine代码](Build-a-semantic-search-engine.ipynb)

https://python.langchain.com/docs/tutorials/retrievers/


## RecursiveUrlLoader



[RecursiveUrlLoader](RecursiveUrlLoader.ipynb)

https://python.langchain.com/docs/integrations/document_loaders/recursive_url/



## SitemapLoader

代码
[SitemapLoader](SitemapLoader.ipynb)


链接
https://python.langchain.com/docs/integrations/document_loaders/sitemap/



## create_retrieval_chain


[create_retrieval_chain](create_retrieval_chain.ipynb)

## create_retrieval_chain-4399


[create_retrieval_chain-4399](create_retrieval_chain-4399.ipynb)




## My-Chat-LangChain子项目


[My-Chat-LangChain子项目目录](My-Chat-LangChain)

[My-Chat-LangChain/README.md](My-Chat-LangChain/README.md)


[CSDN文章：从0到1，构建你的专属AI知识库：My-Chat-LangChain项目深度解析](https://blog.csdn.net/m0_73479109/article/details/152751205?spm=1001.2014.3001.5501)



`My-Chat-LangChain`是一个设计简洁、功能强大的问答平台，它提供了两种构建知识库的核心模式：

1.  **网页知识库 (Webpage Knowledge Base):** 你只需输入任意一个网站的URL，系统便会自动抓取、解析该网站的内容，并在几分钟内构建一个可供对话的知识库。你可以用它来学习在线教程、分析新闻文章，或者快速理解任何网页的核心信息。

2.  **文档知识库 (Document Knowledge Base):** 你可以直接从本地上传PDF文件。系统会智能地解析文档内容，并为你创建一个完全私密的、基于该文档的问答机器人。这对于学习研究报告、阅读法律文件或理解产品手册等场景非常有用。

为了实现优雅、高效的人机交互，整个应用在设计上遵循了几个关键原则：

*   **清晰的功能分区：** 前端界面采用`Streamlit Tabs`（选项卡）设计，将“网页”和“文档”两大功能清晰地隔离开，用户可以自由切换，操作流程一目了然。
*   **前后端分离架构：** 采用现代Web开发模式，前端（Streamlit）负责用户交互和展示，后端（FastAPI）负责繁重的AI计算和数据处理。这种模式让项目结构更清晰，也更容易维护和扩展。
*   **智能缓存机制：** 为了提升效率和节省资源，后端设计了一套智能持久化策略。无论是URL还是上传的文件，只要内容不变，系统处理过一次后就会将知识库保存在本地。下次再处理相同内容时，系统会直接加载缓存，实现秒级响应，极大提升了用户体验。






## RAG-App-Part1


中文文档
[RAG-App-Part1文档](Note/RAG-App-Part1.md)

代码
[RAG-App-Part1代码](RAG-App-Part1.ipynb)

链接
https://python.langchain.com/docs/tutorials/rag/

## RAG-App-Part2


中文文档
[RAG-App-Part2文档](Note/RAG-App-Part2.md)

代码
[RAG-App-Part2代码](RAG-App-Part2.ipynb)

链接
https://python.langchain.com/docs/tutorials/qa_chat_history/



## RAG-App-FastAPI

代码文件夹

[RAG-App-FastAPI](RAG-App-FastAPI)

文档

[RAG-App-FastAPI\RAG-App-FastAPI.md](RAG-App-FastAPI/RAG-App-FastAPI.md)

将 RAG 应用程序（包括基于链和基于代理的版本）与FastAPI结合，将其部署为Web服务。使得 RAG 应用能够通过 HTTP 请求进行访问，并支持对话记忆。





# LangGraph

## Build a basic chatbot

代码
[1-build-basic-chatbot代码](BasicChatbot.ipynb)

链接
https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/

## ChatbotWithTavilySearch

代码
[ChatbotWithTavilySearch](ChatbotWithTavilySearch.ipynb)

链接
https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/






## ChatbotWithMemory

代码
[ChatbotWithMemory](ChatbotWithMemory.ipynb)

链接
https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/






## Add human-in-the-loop controls

代码
[ChatbotWithHITLControls](ChatbotWithHITLControls.ipynb)

链接
https://langchain-ai.github.io/langgraph/tutorials/get-started/4-human-in-the-loop/




## ChatbotWithCustomizeState

代码
[ChatbotWithCustomizeState](ChatbotWithCustomizeState.ipynb)

链接
https://langchain-ai.github.io/langgraph/tutorials/get-started/5-customize-state/





## ChatbotWithTimeTravel


代码
[ChatbotWithTimeTravel](ChatbotWithTimeTravel.ipynb)

链接
https://langchain-ai.github.io/langgraph/tutorials/get-started/6-time-travel/


## Run-a-LangGraph-Local-Server

代码文件夹
[local-server](local-server)


实践文档
[Run-a-LangGraph-Local-Server](Run-a-LangGraph-Local-Server.md)


链接
https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/



## LangGraph+BrightData+PaperSearch的研究助理




[Agent-Demo](My-Chat-LangChain/Agent-Demo.md)



[稀土掘金文章：LangGraph+BrightData+PaperSearch的研究助理](https://juejin.cn/post/7572714389942337588)





