# My-Chat-LangChain 分离部署指南 (Split Deployment)

本指南详细说明如何将 **My-Chat-LangChain (v6.0)** 项目进行分离部署：后端托管于 **Vercel**，前端托管于 **Streamlit Cloud**。

## 🏗️ 架构概览

由于 Vercel Serverless Functions 的无状态特性和只读文件系统限制，我们对代码进行了适配：
1.  **Backend (FastAPI)**: 部署在 Vercel。
    *   所有的文件写入操作（如 SQLite 数据库、上传的临时文件）都已指向 `/tmp` 目录（这是 Vercel 唯一可写的临时目录）。
    *   通过 HTTP API 提供服务。
2.  **Frontend (Streamlit)**: 部署在 Streamlit Cloud。
    *   通过 `BACKEND_URL` 环境变量连接到 Vercel 后端。

---

## ✅ 准备工作

1.  **GitHub 账号**: 拥有此项目的代码仓库。
2.  **Vercel 账号**: 用于部署后端 ([vercel.com](https://vercel.com))。
3.  **Streamlit Cloud 账号**: 用于部署前端 ([share.streamlit.io](https://share.streamlit.io))。
4.  **API Keys**:
    *   `GOOGLE_API_KEY` (必填，用于 Gemini 模型)
    *   `BRIGHT_DATA_API_KEY` (可选，用于网络搜索)
    *   `PAPER_SEARCH_API_KEY` (可选，用于论文检索)

---

## 🚀 第一步：部署后端 (Vercel)

### 1. 检查项目文件
确保你的 GitHub 仓库根目录下包含以下关键文件：
*   `requirements.txt`: 包含 `fastapi`, `uvicorn`, `langchain` 等依赖。
*   `vercel.json`: 配置重写规则，将请求转发给 `backend/main.py`。
    *   *内容确认*:
        ```json
        {
          "rewrites": [{ "source": "/(.*)", "destination": "/backend/main.py" }]
        }
        ```

### 2. 导入项目到 Vercel
1.  登录 Vercel 控制台。
2.  点击 **"Add New..."** -> **"Project"**。
3.  选择包含 `My-Chat-LangChain` 代码的 GitHub 仓库并点击 **"Import"**。

### 3. 配置部署设置 (Configure Project)
*   **Root Directory (根目录)**:
    *   如果你的仓库直接就是项目文件（即 `requirements.txt` 在仓库根目录），保持默认 `./`。
    *   如果项目在子文件夹（例如仓库叫 `LangChain-Learning`，代码在 `My-Chat-LangChain` 下），请点击 **Edit** 并选择 `My-Chat-LangChain` 目录。
*   **Framework Preset**: 选择 **FastAPI** 或保持默认 (Vercel 通常会自动检测 Python 环境)。
*   **Build & Output Settings**: 保持默认。

### 4. 设置环境变量 (Environment Variables)
在 **"Environment Variables"** 部分，添加以下变量：

| Key | Value (示例) | 说明 |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | `AIzaSy...` | **必填**。Gemini 模型密钥。 |
| `BRIGHT_DATA_API_KEY` | `...` | 可选。用于高级搜索 MCP 工具。 |
| `PAPER_SEARCH_API_KEY` | `...` | 可选。用于论文搜索 MCP 工具。 |

> ⚠️ **注意**: 不需要设置 `BACKEND_URL`，因为这是后端自己。

### 5. 点击 Deploy
*   点击 **"Deploy"** 按钮。
*   等待构建完成（Install Dependencies 可能需要几分钟）。
*   部署成功后，你会获得一个 **Project Domains**（例如 `https://my-chat-langchain-alpha.vercel.app`）。
*   **测试**: 访问 `https://<你的域名>/docs`。如果看到 Swagger UI 界面，说明后端部署成功！
*   👉 **复制这个域名**，下一步前端部署需要用到。

---

## 🎨 第二步：部署前端 (Streamlit Cloud)

### 1. 导入项目
1.  登录 [Streamlit Cloud](https://share.streamlit.io/)。
2.  点击 **"New app"**。
3.  **Repository**: 选择同一个 GitHub 仓库。
4.  **Branch**: `main` (或你开发的分支)。

### 2. 配置主文件路径
*   **Main file path**: 输入 `frontend/app.py`。
    *   *注意*: 这里的路径是相对于你第一步选择的 Repository 根目录的。如果之前在 Vercel 设置了子目录作为 Root，这里要在仓库层级找到该文件。

### 3. 配置 Secrets (环境变量)
点击 **"Advanced settings..."**，在 **Secrets** 输入框中添加以下 TOML 格式配置：

```toml
# Vercel 后端地址 (必填，末尾不要带斜杠 /)
BACKEND_URL = "https://my-chat-langchain-alpha.vercel.app"

# 可选：如果想预置 Key 方便前端直接使用，可以在这里添加
# BRIGHT_DATA_API_KEY = "xxx"
# PAPER_SEARCH_API_KEY = "xxx"
```

### 4. 点击 Deploy
*   点击 **"Deploy!"** 按钮。
*   等待应用启动。Streamlit 会自动安装 `requirements.txt` 中的依赖。

---

## 📝 限制与注意事项

1.  **Vercel 函数超时**:
    *   Vercel Hobby (免费版) 的 Serverless Function 执行超时时间通常为 **10秒** (有时可配置到 60秒)。
    *   **影响**: 如果上传特别大的 PDF 文件进行 RAG 索引，或者执行非常复杂的 Agent 搜索任务，可能会因为超时导致前端报错 (504 Gateway Timeout)。
    *   **建议**: 仅用于演示或处理轻量级任务（小文件、简单搜索）。

2.  **临时存储 (/tmp)**:
    *   由于使用了 Serverless 架构，`/tmp` 目录下的文件（上传的文件、SQLite 数据库）是**临时的**。
    *   一段时间不活动后，实例会被销毁，**聊天记录和知识库索引将会丢失**。这是 Serverless 的特性。如果需要持久化存储，建议对接云数据库 (如 MongoDB Atlas, Supabase)。

3.  **依赖包体积**:
    *   Vercel 对 Serverless Function 的包体积有限制 (250MB)。如果添加了过多大型 Python 库（如 pytorch, heavy transformers），可能会导致部署失败。目前的 `requirements.txt` 包含了 `sentence-transformers` 和 `chromadb`，体积较大，勉强可能在边缘。如果部署失败，请尝试精简依赖。
