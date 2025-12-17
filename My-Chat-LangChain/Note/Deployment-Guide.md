# My-Chat-LangChain Render 部署指南

本指南详细说明如何将 `My-Chat-LangChain` 项目部署到 [Render](https://render.com) 云平台。文档涵盖了从环境准备、配置详解、详细部署步骤到故障排查的全过程。

## 📖 项目概述

`My-Chat-LangChain` 是一个基于 LangChain 和 FastAPI 构建的 AI 聊天应用，前端使用 Streamlit (或 HTML/JS)。本项目已针对 Render 容器化部署进行了优化，包含自动化配置文件 `render.yaml`。

### 关键配置文说明
- **`render.yaml`**: Render 的基础设施即代码 (IaC) 配置文件 (Blueprint)，定义了服务类型、环境变构建命令等。
- **`Dockerfile`**: 定义了应用的运行环境，基于 Python 3.11-slim，安装了必要的系统依赖（如 `curl`, `gcc`）和 Python 库。
- **`requirements.txt`**: 列出了所有 Python 依赖包。**注意：** 其中的 `requests` 库版本已特别处理以解决冲突。

---

## 🚀 部署方案选择

Render 提供多种实例类型，针对本项目主要有两种部署策略：

| 特性 | **方案 A: 快速体验 (推荐)** | **方案 B: 生产环境** |
| :--- | :--- | :--- |
| **Render 计划** | **Free Tier (免费)** | **Starter ($7/mo)** + Disk ($0.25/GB) |
| **数据持久性** | ❌ **无** (无状态) | ✅ **有** (持久化) |
| **应用场景** | 演示、测试、临时使用 | 长期运行、需要保存聊天记录/向量库 |
| **限制** | 实例会在闲置 15 分钟后休眠<br>重启后所有上传文件丢失 | 持续运行<br>数据重启不丢失 |

---

## 🛠️ 部署步骤详解

本指南主要介绍基于 **Render Blueprint** 的自动部署流程，这也是最简单的方式。

### 第一步：准备工作
1. **GitHub 仓库**: 确保本项目代码已完整推送到你的 GitHub 账户。
2. **注册 Render**: 访问 [dashboard.render.com](https://dashboard.render.com/) 并注册/登录。
3. **API Keys**: 准备好项目运行所需的 API Keys (如 OpenAI, Google Search 等)。









### 第二步：创建 Blueprint 实例
1. 在 Render Dashboard 点击右上角的 **"New +"** 按钮。
2. 选择 **"Blueprint"**。
3. 在列表中找到并点击 **"Connect"** 连接你的 `My-Chat-LangChain` 仓库。
   - 如果未看到仓库，点击 "Configure account" 授权 Render 访问该仓库。
4. Render 会自动读取仓库根目录下的 `render.yaml` 文件。
5. **服务名称**: 默认为 `my-chat-langchain`，你可以根据需要修改。
6. **Apply**: 点击页面底部的 **"Apply"** 按钮开始部署。



![alt text](media/Snipaste_2025-12-02_11-35-45.png)








### 第三步：配置环境变量
在部署过程中（或者部署完成后），你需要检查并填写以下环境变量。可以在 Render 控制台的 **"Environment"** 标签页中进行管理。

| 变量名 | 必填 | 说明 | 示例值 |
| :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | ✅ | 用于驱动 LLM 核心功能 | `sk-proj-...` |
| `GOOGLE_API_KEY` | ❌ | 用于 Google 搜索工具 | `AIzaSy...` |
| `BRIGHT_DATA_API_KEY`| ❌ | 用于高级网页抓取 (可选) | `...` |
| `PAPER_SEARCH_API_KEY`| ❌ | 用于论文搜索工具 (可选) | `...` |
| `DATA_DIR` | ✅ | 数据存储目录 (默认已配置) | `/var/lib/data` |




![alt text](media/Snipaste_2025-12-02_11-42-07.png)






### 第四步：等待构建与验证
Render 将自动开始构建 Docker 镜像。
1. 点击 **"Logs"** 标签页查看实时日志。
2. **构建阶段**: 你会看到 `pip install` 安装依赖的过程。
3. **启动阶段**: 看到以下日志即表示启动成功：
   ```log
   INFO:     Started server process [1]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:10000 (Press CTRL+C to quit)
   ```
4. **访问应用**: 点击页面左上角的 URL (例如 `https://my-chat-langchain.onrender.com`)，应用界面应正常加载。


![alt text](media/Snipaste_2025-12-02_14-07-13.png)


![alt text](media/Snipaste_2025-12-02_12-20-46.png)



![alt text](media/Snipaste_2025-12-02_14-00-39.png)


![alt text](media/Snipaste_2025-12-02_14-00-45.png)

![alt text](media/Snipaste_2025-12-02_14-03-57.png)





---

## 🔧 故障排查与维护 (Troubleshooting)

### 1. 依赖冲突解决 (`requests` 库)
**症状**: 部署失败，日志显示 `Conflict causing the package install to fail`，主要涉及 `langchain-community` 和 `requests`。

**原因**: `langchain-community >= 0.3.29` 强制要求 `requests >= 2.32.5`，而旧的 `requirements.txt` 可能锁定了较低版本 (如 `2.32.4`)。

**解决方案 (已应用)**:
确保 `My-Chat-LangChain/requirements.txt` 中 `requests` 的版本定义如下：
```text
requests>=2.32.5
```
本项目代码库**已经包含此修复**。如果未来遇到类似问题，请检查 `langchain` 相关包的依赖要求。



### 2. 内存不足 (OOM Killed)
**症状**: 服务启动中途突然停止，日志显示 `Exited with status 137` 或 `OOM Killed`。

**原因**: 免费实例只有 512MB 内存。加载大型嵌入模型 (Embedding Models) 或处理大量数据时可能超限。

**解决方案**:
- 升级到 **Starter** 计划 (增加内存)。
- 优化代码，减少启动时加载的大型对象。

---

## 💾 进阶：启用数据持久化

如果你决定从免费版升级到生产版，请按以下步骤启用磁盘挂载，防止数据丢失。

1. **升级 Plan**: 在 Settings -> Instance Type 中选择 Starter ($7/mo)。
2. **添加 Disk**: 在 Disks 菜单中点击 "Add Disk"。
   - **Name**: `chat-data`
   - **Mount Path**: `/var/lib/data` (必须与环境变量 `DATA_DIR` 一致)
   - **Size**: 1 GB (或更多)
3. **保存**: Render 会自动重新部署并挂载磁盘。

---

## 📝 部署核对清单

- [x] 代码推送到 GitHub
- [x] `requirements.txt` 中 `requests` 版本已修复 (>=2.32.5)
- [x] Render Blueprint 已连接仓库
- [x] 环境变量 (`OPENAI_API_KEY` 等) 已在 Render 后台填入
- [x] 部署日志显示 "Application startup complete"
- [x] 浏览器访问 URL 成功加载聊天界面
