# My-Chat-LangChain Render 部署指南

> **版本**: V8.0
> **更新日期**: 2025-12-17
> **新增**: E2B 代码执行沙箱配置、OpenAI 兼容模式

本指南详细说明如何将 `My-Chat-LangChain` 项目部署到 [Render](https://render.com) 云平台。文档涵盖了从环境准备、配置详解、详细部署步骤到故障排查的全过程。

---

## 📖 项目概述

`My-Chat-LangChain` (Stream-Agent V8.0) 是一个基于 LangChain 和 FastAPI 构建的 AI 聊天应用，前端使用 Streamlit。

### V8.0 新增功能
- **E2B 云沙箱**: 安全执行 Python 代码、数据分析、生成图表
- **OpenAI 兼容模式**: 支持第三方 LLM 中转平台
- **图表渲染**: 前端支持 matplotlib 图表显示

### 关键配置文件说明
| 文件 | 说明 |
|------|------|
| `render.yaml` | Render Blueprint 配置 (IaC) |
| `Dockerfile` | Docker 镜像定义 |
| `requirements.txt` | Python 依赖 (含 E2B) |
| `start.sh` | 启动脚本 |

---

## 🚀 部署方案选择

| 特性 | **方案 A: 免费体验** | **方案 B: 生产环境** |
|:---|:---|:---|
| **Render 计划** | Free Tier | Starter ($7/mo) + Disk |
| **数据持久性** | ❌ 无 (无状态) | ✅ 有 (持久化) |
| **E2B 功能** | ✅ 可用 | ✅ 可用 |
| **应用场景** | 演示、测试 | 长期运行 |
| **限制** | 闲置 15 分钟休眠 | 持续运行 |

---

## 🛠️ 部署步骤详解

### 第一步：准备工作

1. **GitHub 仓库**: 确保代码已推送到 GitHub
2. **注册 Render**: 访问 [dashboard.render.com](https://dashboard.render.com/)
3. **获取 API Keys**:

| API Key | 获取地址 | 必须 |
|---------|---------|------|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | ✅ 是* |
| `E2B_API_KEY` | https://e2b.dev/dashboard | ✅ 是 |
| `BRIGHT_DATA_API_KEY` | https://brightdata.com | 可选 |
| `PAPER_SEARCH_API_KEY` | Smithery.ai | 可选 |

*如使用 OpenAI 兼容模式，可不设置 GOOGLE_API_KEY

### 第二步：创建 Blueprint 实例

1. 在 Render Dashboard 点击 **"New +"** → **"Blueprint"**
2. 连接你的 `My-Chat-LangChain` 仓库
3. Render 自动读取 `render.yaml` 配置
4. 点击 **"Apply"** 开始部署

![Blueprint 配置](media/Snipaste_2025-12-02_11-35-45.png)

### 第三步：配置环境变量 (V8.0 更新)

在 Render Dashboard → **"Environment"** 中配置：

#### 必须配置

| 变量名 | 说明 | 示例值 |
|:---|:---|:---|
| `GOOGLE_API_KEY` | Google AI API Key | `AIzaSy...` |
| `E2B_API_KEY` | E2B 沙箱 API Key (V8.0 新增) | `e2b_...` |

#### 可选配置 - LLM 模式切换

**方式 1: Google Gemini (默认)**
```
GOOGLE_API_KEY=your_key
GOOGLE_MODEL=gemini-2.0-flash-lite
```

**方式 2: OpenAI 兼容模式 (第三方中转)**
```
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://api.openrouter.ai/api/v1
OPENAI_API_KEY=your_proxy_key
OPENAI_MODEL=google/gemini-2.0-flash-exp:free
```

#### 可选配置 - MCP 工具

| 变量名 | 说明 |
|:---|:---|
| `BRIGHT_DATA_API_KEY` | 网页抓取工具 |
| `PAPER_SEARCH_API_KEY` | 论文搜索工具 |

#### 系统配置 (已默认设置)

| 变量名 | 默认值 | 说明 |
|:---|:---|:---|
| `DATA_DIR` | `/var/lib/data` | 数据存储目录 |
| `PYTHONUNBUFFERED` | `1` | 禁用 Python 输出缓冲 |
| `PYTHONPATH` | `/app/backend` | Python 模块路径 |

![环境变量配置](media/Snipaste_2025-12-02_11-42-07.png)

### 第四步：等待构建与验证

1. 查看 **"Logs"** 标签页
2. **构建阶段**: `pip install` 安装依赖 (包括 E2B)
3. **启动阶段**: 看到以下日志即表示成功：
   ```log
   INFO:     Started server process [1]
   INFO:     Uvicorn running on http://0.0.0.0:10000
   ```
4. **访问应用**: 点击 URL (如 `https://my-chat-langchain.onrender.com`)


![alt text](media/Snipaste_2025-12-02_14-07-13.png)


![alt text](media/Snipaste_2025-12-02_12-20-46.png)



![alt text](media/Snipaste_2025-12-02_14-00-39.png)


![alt text](media/Snipaste_2025-12-02_14-00-45.png)



![部署成功](media/Snipaste_2025-12-02_14-03-57.png)

### 第五步：验证 E2B 功能

部署成功后，测试代码执行功能：

1. 打开应用
2. 在侧边栏确认 E2B API Key 已配置
3. 发送测试消息：
   ```
   帮我画一个正弦波图表
   ```
4. 应看到图表正常显示

---

## 📋 环境变量完整清单 (V8.0)

```yaml
# render.yaml 中的完整配置
envVars:
  # LLM 配置 (Google Gemini)
  - key: GOOGLE_API_KEY
    sync: false
  - key: GOOGLE_MODEL
    value: gemini-2.0-flash-lite

  # E2B 代码执行 (V8.0 必须)
  - key: E2B_API_KEY
    sync: false

  # MCP 工具 (可选)
  - key: BRIGHT_DATA_API_KEY
    sync: false
  - key: PAPER_SEARCH_API_KEY
    sync: false

  # 系统配置
  - key: DATA_DIR
    value: /var/lib/data
  - key: PYTHONUNBUFFERED
    value: "1"
  - key: PYTHONPATH
    value: /app/backend
```

---

## 🔧 故障排查与维护

### 1. E2B 相关错误 (V8.0 新增)

**症状**: `E2B_API_KEY 环境变量未设置`

**解决方案**:
1. 访问 https://e2b.dev/dashboard 获取 API Key
2. 在 Render Environment 中添加 `E2B_API_KEY`
3. 重新部署

---

**症状**: `The sandbox was not found (502)`

**原因**: E2B 沙箱超时被销毁

**解决方案**: V8.0 已自动处理，沙箱会自动重建

---

**症状**: 图表不显示

**排查步骤**:
1. 确认代码以 `plt.show()` 结尾
2. 检查 Render 日志中是否有 `[IMAGE_BASE64:` 输出
3. 检查浏览器控制台错误

### 2. 依赖冲突 (`requests` 库)

**症状**: 部署失败，`Conflict causing the package install to fail`

**解决方案** (已在 V8.0 修复):
```text
# requirements.txt
requests>=2.32.5
```

### 3. 内存不足 (OOM Killed)

**症状**: 服务启动中途突然停止，日志显示 `Exited with status 137` 或 `OOM Killed`。

**原因**: 免费实例只有 512MB 内存。加载大型嵌入模型 (Embedding Models) 或处理大量数据时可能超限。


**解决方案**:
- 升级到 **Starter** 计划 (增加内存)
- 或减少同时加载的大型模型

### 4. LLM 连接失败

**症状**: 前端卡在 "Thinking..."

**排查步骤**:
1. 检查 `GOOGLE_API_KEY` 是否正确
2. 如使用代理，检查网络节点是否支持 Gemini API
3. 尝试切换到 OpenAI 兼容模式

---

## 💾 进阶：启用数据持久化

如需长期运行，建议升级到付费计划并启用磁盘：

1. **升级 Plan**: 在 Settings -> Instance Type 中选择 Starter ($7/mo)。
2. **添加 Disk**: 在 Disks 菜单中点击 "Add Disk"。
   - **Name**: `chat-data`
   - **Mount Path**: `/var/lib/data` (必须与环境变量 `DATA_DIR` 一致)
   - **Size**: 1 GB (或更多)
3. **保存**: Render 会自动重新部署并挂载磁盘。

---

## 📝 V8.0 部署核对清单

- [ ] 代码推送到 GitHub (包含 E2B 工具)
- [ ] `requirements.txt` 包含 `e2b>=1.0.0` 和 `e2b-code-interpreter>=1.0.0`
- [ ] `requirements.txt` 包含 `langchain-openai>=0.3.0`
- [ ] Render Blueprint 已连接仓库
- [ ] `GOOGLE_API_KEY` 已配置
- [ ] `E2B_API_KEY` 已配置 (**V8.0 必须**)
- [ ] 部署日志显示 "Application startup complete"
- [ ] 浏览器访问 URL 成功
- [ ] 测试 "画一个正弦波" 图表显示正常

---

## 📚 相关文档

- [Note-V8.md](Note-V8.md) - V8.0 详细开发说明
- [Plan-V8.md](Plan-V8.md) - V8.0 开发计划
- [E2B 官方文档](https://e2b.dev/docs)
- [Render 官方文档](https://render.com/docs)

---

> **文档版本**: V8.0
> **最后更新**: 2025-12-17
