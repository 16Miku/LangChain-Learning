---
typora-copy-images-to: media
---





@高级软件工程师 @📁 scrape_as_markdown 

https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/

阅读该网页教程，指导我完成该教程内容





好的，我将帮助你阅读 `https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/` 这个网页教程，并一步步指导你完成教程内容。

首先，我将抓取该网页的内容。

好的，我将指导您完成 LangGraph 本地服务器教程。请按照以下步骤操作：

## LangGraph 本地服务器教程

本教程将向您展示如何在本地运行 LangGraph 应用程序。

### 先决条件

在开始之前，请确保您具备以下条件：

* **Python >= 3.11**：这是运行 LangGraph Python 服务器所必需的。

  ```powershell
  (Gemini) PS A:\> conda install python=3.12
  (Gemini) PS A:\> python --version
  Python 3.12.11
  ```

  

* **LangSmith API Key**：您可以访问 [LangSmith](https://smith.langchain.com/settings) 免费注册并获取 API 密钥。



### 1. 安装 LangGraph CLI

首先，您需要安装 LangGraph 命令行界面（CLI）。打开您的 PowerShell，并执行以下命令：

```powershell
pip install --upgrade "langgraph-cli[inmem]"
```

**注释：**
*   `pip install`：这是 Python 的包管理器命令，用于安装软件包。
*   `--upgrade`：确保将 `langgraph-cli` 升级到最新版本（如果已安装）。
*   `"langgraph-cli[inmem]"`：安装 `langgraph-cli` 包，并包含 `inmem` 扩展，用于在内存中运行服务器。

```powershell
(Gemini) PS A:\> pip install --upgrade "langgraph-cli[inmem]"
...
...
...
Installing collected packages: forbiddenfruit, truststore, structlog, python-dotenv, pyjwt, jsonschema-rs, cloudpickle, click, blockbuster, watchfiles, uvicorn, starlette, cryptography, sse-starlette, langgraph-cli, langgraph-runtime-inmem, langgraph-api
Successfully installed blockbuster-1.5.25 click-8.2.1 cloudpickle-3.1.1 cryptography-44.0.3 forbiddenfruit-0.1.4 jsonschema-rs-0.29.1 langgraph-api-0.2.86 langgraph-cli-0.3.4 langgraph-runtime-inmem-0.3.4 pyjwt-2.10.1 python-dotenv-1.1.1 sse-starlette-2.1.3 starlette-0.47.1 structlog-25.4.0 truststore-0.10.1 uvicorn-0.35.0 watchfiles-1.1.0
```



### 2. 创建一个 LangGraph 应用

接下来，我们将创建一个新的 LangGraph 应用程序。该教程推荐使用 `new-langgraph-project-python` 模板，它演示了一个可以扩展的单节点应用程序。

在 PowerShell 中执行以下命令，将 `path/to/your/app` 替换为您希望创建应用程序的实际路径（例如 `my_langgraph_app`）：

```powershell
langgraph new my_langgraph_app --template new-langgraph-project-python
```

**注释：**
*   `langgraph new`：这是 LangGraph CLI 的命令，用于创建一个新的 LangGraph 项目。
*   `my_langgraph_app`：您可以将此替换为您想要的项目文件夹名称。
*   `--template new-langgraph-project-python`：指定使用 Python 版本的 LangGraph 项目模板。

```powershell
(Gemini) PS A:\> langgraph new A:\study\Python\jupyter\LangChain\local-server --template new-langgraph-project-python
📥 Attempting to download repository as a ZIP archive...
URL: https://github.com/langchain-ai/new-langgraph-project/archive/refs/heads/main.zip
✅ Downloaded and extracted repository to A:\study\Python\jupyter\LangChain\local-server
🎉 New project created at A:\study\Python\jupyter\LangChain\local-server
```



### 3. 安装依赖项

进入您新创建的 LangGraph 应用程序的根目录，并安装其依赖项。这将以“可编辑”模式安装，以便服务器可以使用您本地的更改。

```powershell
cd my_langgraph_app
pip install -e .
```

**注释：**
*   `cd my_langgraph_app`：进入您在上一步中创建的应用程序目录。请确保将 `my_langgraph_app` 替换为您的实际目录名。
*   `pip install -e .`：在当前目录（`.`）中以可编辑模式（`-e`）安装项目依赖。这意味着 Python 将链接到您的项目源代码，而不是复制它，这对于开发非常方便。

```powershell
(Gemini) PS A:\> cd A:\study\Python\jupyter\LangChain\local-server
(Gemini) PS A:\study\Python\jupyter\LangChain\local-server> pip install -e .
Obtaining file:///A:/study/Python/jupyter/LangChain/local-server
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
...
...
...
Successfully built agent
Installing collected packages: agent
Successfully installed agent-0.0.1

```



### 4. 创建一个 `.env` 文件

在您的 LangGraph 应用程序的根目录中，您会找到一个名为 `.env.example` 的文件。您需要创建一个名为 `.env` 的新文件，并将 `.env.example` 的内容复制到其中，然后填入您的 LangSmith API 密钥。

1.  在 `my_langgraph_app` 目录下，右键点击或使用文本编辑器创建一个新文件，并将其命名为 `.env`。
2.  打开 `.env.example` 文件，复制其内容。
3.  将内容粘贴到新创建的 `.env` 文件中。
4.  找到 `LANGSMITH_API_KEY=` 这一行，将您的 LangSmith API 密钥粘贴到等号后面。

您的 `.env` 文件内容应类似于：

```
LANGSMITH_API_KEY=lsv2... # 替换为您的实际密钥
```

**注释：**
*   `.env` 文件用于存储环境变量，特别是敏感信息（如 API 密钥），这样可以避免将它们硬编码到代码中，提高安全性。
*   `LANGSMITH_API_KEY`：这是 LangSmith 平台用于认证您的请求的密钥。



```
# To separate your traces from other application
LANGSMITH_PROJECT=new-agent

# Add API keys for connecting to LLM providers, data sources, and other integrations here
LANGSMITH_API_KEY=lsv2_........


```



### 5. 启动 LangGraph 服务器 🚀

现在，您可以启动 LangGraph API 服务器了。在您的应用程序根目录（`my_langgraph_app`）中，执行以下命令：

```powershell
langgraph dev
```

**注释：**

*   `langgraph dev`：这个命令会启动 LangGraph 服务器，默认以内存模式运行，适用于开发和测试。

成功启动后，您将看到类似以下的输出：

```
> Ready!
>
> - API: http://localhost:2024
>
> - Docs: http://localhost:2024/docs
>
> - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

请记下这些 URL，特别是 `LangGraph Studio Web UI` 的 URL，它将用于下一步。



```powershell
(Gemini) PS A:\study\Python\jupyter\LangChain\local-server> langgraph dev
INFO:langgraph_api.cli:

        Welcome to

╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

This in-memory server is designed for development and testing.
For production use, please use LangGraph Platform.


2025-07-15T15:32:18.859680Z [info     ] Using langgraph_runtime_inmem  [langgraph_runtime] api_variant=local_dev langgraph_api_version=0.2.86 thread_name=MainThread
2025-07-15T15:32:18.898402Z [info     ] Using auth of type=noop        [langgraph_api.auth.middleware] api_variant=local_dev langgraph_api_version=0.2.86 thread_name=MainThread
```



### 6. 在 LangGraph Studio 中测试您的应用程序

LangGraph Studio 是一个专门的 UI，您可以连接到 LangGraph API 服务器，以可视化、交互和调试您的本地应用程序。

复制上一步输出中 `LangGraph Studio Web UI` 后面的 URL（例如 `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`），并在您的浏览器中打开它。

您将能够在该界面中与您的本地运行的 LangGraph 应用程序进行交互和调试。



![image-20250716000824480](A:%5Cstudy%5CPython%5Cjupyter%5CLangChain%5Cmedia%5Cimage-20250716000824480.png)

### 7. 测试 API

您可以使用 Python SDK 来测试 LangGraph 服务器的 API。

1.  **安装 LangGraph Python SDK**：
    在保持 LangGraph 服务器运行的同时，打开一个新的 PowerShell 窗口，并安装 LangGraph SDK：

    ```powershell
    pip install langgraph-sdk
    ```

2.  **发送消息到助手（无线程运行）**：
    在同一个新的 PowerShell 窗口中，创建一个 Python 文件（例如 `test_api.py`），并将以下异步代码粘贴进去：

    ```python
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
    ```

    保存文件后，在 PowerShell 中运行该 Python 脚本：

    ```powershell
    python test_api.py
    ```

    您将看到从 LangGraph 服务器流式传输回来的响应数据。结果如下所示。
    
    

```powershell
(Gemini) PS A:\> cd A:\study\Python\jupyter\LangChain\local-server
(Gemini) PS A:\study\Python\jupyter\LangChain\local-server> pip install langgraph-sdk
Requirement already satisfied: langgraph-sdk in a:\anaconda\envs\gemini\lib\site-packages (0.1.72)
Requirement already satisfied: httpx>=0.25.2 in a:\anaconda\envs\gemini\lib\site-packages (from langgraph-sdk) (0.28.1)
Requirement already satisfied: orjson>=3.10.1 in a:\anaconda\envs\gemini\lib\site-packages (from langgraph-sdk) (3.10.18)
Requirement already satisfied: anyio in a:\anaconda\envs\gemini\lib\site-packages (from httpx>=0.25.2->langgraph-sdk) (4.9.0)
Requirement already satisfied: certifi in a:\anaconda\envs\gemini\lib\site-packages (from httpx>=0.25.2->langgraph-sdk) (2025.1.31)
Requirement already satisfied: httpcore==1.* in a:\anaconda\envs\gemini\lib\site-packages (from httpx>=0.25.2->langgraph-sdk) (1.0.7)
Requirement already satisfied: idna in a:\anaconda\envs\gemini\lib\site-packages (from httpx>=0.25.2->langgraph-sdk) (3.10)
Requirement already satisfied: h11<0.15,>=0.13 in a:\anaconda\envs\gemini\lib\site-packages (from httpcore==1.*->httpx>=0.25.2->langgraph-sdk) (0.14.0)
Requirement already satisfied: sniffio>=1.1 in a:\anaconda\envs\gemini\lib\site-packages (from anyio->httpx>=0.25.2->langgraph-sdk) (1.3.1)
Requirement already satisfied: typing_extensions>=4.5 in a:\anaconda\envs\gemini\lib\site-packages (from anyio->httpx>=0.25.2->langgraph-sdk) (4.12.2)
(Gemini) PS A:\study\Python\jupyter\LangChain\local-server> python test_api.py
Receiving new event of type: metadata...
{'run_id': '1f061923-bc44-68a0-a838-7edcb98bcd56', 'attempt': 1}



Receiving new event of type: values...
{'changeme': 'output from call_model. Configured with None'}



(Gemini) PS A:\study\Python\jupyter\LangChain\local-server>
```

