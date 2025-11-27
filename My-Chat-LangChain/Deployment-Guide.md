# Vercel 部署指南 (中文版)

为了解决 Vercel 部署时出现的 "Error: No fastapi entrypoint found" 错误，我们对项目结构进行了调整以符合 Vercel 的 Serverless Function 规范。

## 1. 项目变更说明

我们添加了以下文件来适配 Vercel 的部署环境：

- **`api/index.py`**: 这是 Vercel 识别的 Python 入口文件。它负责将请求转发给 `backend/main.py` 中的 FastAPI 应用。
- **`vercel.json`**: Vercel 的配置文件，定义了路由重写规则，确保所有 API 请求都指向 `api/index.py`。
- **`backend/__init__.py`**: 使 backend 目录成为一个 Python 包，以便在 `api/index.py` 中可以正确导入。

## 2. 部署步骤

### 第一步：推送到 GitHub
确保你已经将最新的代码（包括上述新文件）推送到你的 GitHub 仓库。

### 第二步：在 Vercel 上创建项目
1. 登录 [Vercel Dashboard](https://vercel.com/dashboard)。
2. 点击 "Add New..." -> "Project"。
3. 导入你的 `LangChain-Learning` 仓库。

### 第三步：配置项目
在 "Configure Project" 页面：

1. **Framework Preset**: 选择 "Other" (或者如果 Vercel 自动检测到 Python 也可以，但通常默认即可)。
2. **Root Directory**:
   - 点击 "Edit"。
   - 选择 `My-Chat-LangChain` 目录作为项目的根目录。
   - **重要**: 必须选择 `My-Chat-LangChain`，因为 `vercel.json` 和 `api/` 目录都在这里。

3. **Environment Variables (环境变量)**:
   展开 "Environment Variables" 部分，添加你在 `.env` 文件中使用的所有变量，例如：
   - `GOOGLE_API_KEY`
   - `HUGGINGFACEHUB_API_TOKEN`
   - 以及其他后端需要的 API Key。

### 第四步：部署
点击 "Deploy" 按钮。Vercel 将开始构建和部署你的应用。

## 3. 验证部署
部署完成后，Vercel 会提供一个访问域（例如 `https://your-project.vercel.app`）。
你可以尝试访问 API 文档路径来验证后端是否正常运行：
- `https://your-project.vercel.app/docs`

## 常见问题
- **依赖安装慢**: Vercel 的构建时间有限制。如果依赖项非常多（特别是像 torch 这种大型库），可能会导致超时。目前的 `requirements.txt` 看起来是标准的，应该没问题。
- **ChromaDB**: ChromaDB 是一个基于文件的向量数据库。在 Vercel 的 Serverless 环境中，文件系统是临时的（ephemeral）。这意味着**重新部署后，数据库中的数据会丢失**。如果你需要持久化存储，建议使用 ChromaDB 的 Client-Server 模式（连接到远程服务器）或其他云端向量数据库（如 Pinecone）。

## 前端部署 (可选)
该项目包含一个基于 Streamlit 的前端。Streamlit 应用通常部署在 [Streamlit Cloud](https://streamlit.io/cloud) 上，而不是 Vercel。
如果你想部署前端：
1. 在 Streamlit Cloud 上创建一个新应用。
2. 仓库连接到同一个 GitHub 仓库。
3. "Main file path" 设置为 `My-Chat-LangChain/frontend/app.py`。
4. 记得在 Streamlit Cloud 的设置中配置 `BACKEND_URL` 环境变量，指向你在 Vercel 上部署的后端地址。
