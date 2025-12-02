# My-Chat-LangChain Render 部署指南

本指南详细说明如何将 My-Chat-LangChain 部署到 Render 平台。项目支持两种部署模式：**无状态模式 (免费层)** 和 **持久化模式 (付费层)**。

## 部署模式概览

| 特性 | 方案 2: 无状态模式 (默认) | 方案 1: 持久化模式 (升级) |
| :--- | :--- | :--- |
| **适用场景** | 测试、演示、免费体验 | 生产环境、需要保存数据 |
| **Render 方案** | Free Tier (免费) | Starter ($7/mo) + Disk ($0.25/GB/mo) |
| **数据持久性** | **无** (重启后数据丢失) | **有** (数据保存在持久化磁盘) |
| **配置复杂度** | 低 | 中 (需配置磁盘) |

---

## 快速开始：部署方案 2 (免费/无状态)

此方案完全兼容 Render 的免费实例类型，适合快速体验和演示。

### 1. 准备工作
- 确保代码已推送到 GitHub。
- 准备好必要的 API Keys (`GOOGLE_API_KEY`, `OPENAI_API_KEY` 等)。

### 2. 部署步骤 (使用 Blueprint)
本项目包含 `render.yaml`，推荐使用 Render Blueprint 自动部署。

1. 登录 [Render Dashboard](https://dashboard.render.com/)。
2. 点击 **"New +"** 按钮，选择 **"Blueprint"**。
3. 连接包含本项目的 GitHub 仓库。
4. Render 会自动检测 `render.yaml` 配置文件。
5. 点击 **"Apply"** 开始创建服务。

### 3. 环境变量配置
在创建过程中或创建后，请在 Render 控制台的 **"Environment"** 标签页中检查并填入以下变量：

| Key | 说明 |
| :--- | :--- |
| `GOOGLE_API_KEY` | Google API Key (用于搜索功能) |
| `BRIGHT_DATA_API_KEY` | Bright Data API Key (可选) |
| `PAPER_SEARCH_API_KEY` | 论文搜索 API Key (可选) |
| `OPENAI_API_KEY` | OpenAI API Key (用于 LLM) |
| `DATA_DIR` | 默认设置为 `/var/lib/data` (无需修改) |

### 4. 验证部署
部署完成后，访问 Render 提供的 URL。应用应正常启动。
**注意：** 在此模式下上传的文件、创建的向量数据库 (ChromaDB) 或生成的日志，**在应用重启或重新部署后将会丢失**。

---

## 升级指南：切换到方案 1 (付费/持久化)

如果你需要保存 ChromaDB 向量库、上传的文件或对话历史，请按以下步骤升级到持久化模式。

### 1. 升级实例类型
Render Free Tier 不支持挂载磁盘 (Disk)。
1. 在 Render Dashboard 进入你的服务页面。
2. 点击 **"Settings"**。
3. 在 **"Instance Type"** 部分，选择 **"Starter"** ($7/mo) 或更高配置。
4. 点击 **"Save Changes"**。

### 2. 添加持久化磁盘 (Persistent Disk)
1. 在服务页面的侧边栏选择 **"Disks"**。
2. 点击 **"Add Disk"**。
3. 配置磁盘参数：
   - **Name**: `chat-data` (建议命名)
   - **Mount Path**: `/var/lib/data` (**必须完全匹配此路径**)
   - **Size**: 根据需要选择 (例如 1GB)
4. 点击 **"Create Disk"**。

Render 将会自动重新部署服务并挂载磁盘。此时，写入 `/var/lib/data` 的所有数据都将被持久化保存。

### 3. (可选) 通过 `render.yaml` 更新
如果你更喜欢通过代码管理配置 (IaC)，可以修改项目根目录下的 `render.yaml`，将配置更新为：

```yaml
services:
  - type: web
    name: my-chat-langchain
    runtime: docker
    plan: starter  # 将 free 改为 starter 或其他付费计划
    # ... (其他配置保持不变) ...
    disk:
      name: chat-data
      mountPath: /var/lib/data
      sizeGB: 1
```

提交并推送代码后，Render Blueprint 将自动应用变更（可能需要在 Dashboard 确认支付信息）。

---

## 常见问题 (FAQ)

### Q: 为什么应用重启后我的知识库没了？
**A:** 如果你使用的是**方案 2 (免费层)**，这是预期行为。Docker 容器的文件系统是临时的，重启后会重置。如需保存数据，请参考"升级指南"切换到方案 1 并挂载磁盘。

### Q: 部署失败，提示 SQLite 版本过低？
**A:** LangChain/ChromaDB 对 SQLite 版本有要求。本项目已在 `backend/main.py` 中集成了 `pysqlite3-binary` 补丁，确保在 Linux 环境下覆盖系统默认的 SQLite。**请确保不要删除 `main.py` 顶部的相关补丁代码。**

### Q: 为什么我在本地运行正常，Render 上报错？
**A:** 请检查 Render 的 **Logs**。常见原因包括：
1. **环境变量缺失**：确保所有 API Key 都已正确设置。
2. **内存不足**：Free Tier 限制 512MB RAM。如果遇到 `OOM Killed` 或 `Out Of Memory` 错误，可能需要升级实例类型。

### Q: 部署需要多长时间？
**A:** 首次构建 Docker 镜像可能需要几分钟。后续部署如果利用了缓存会更快。
