# Plan-V8: E2B 代码执行沙箱集成方案

> **版本**: V8.0
> **日期**: 2025-12-17
> **目标**: 为 My-Chat-LangChain 集成 E2B 云沙箱，实现安全的代码执行能力
> **状态**: ✅ 开发完成 (所有核心功能已测试通过)

---

## 📋 开发进度总览

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| 阶段 1: 基础集成 | ✅ 已完成 | 100% |
| 阶段 2: 数据分析能力 | ✅ 已完成 | 100% |
| 阶段 3: 前端增强 | ✅ 已完成 | 100% |
| 阶段 4: 测试与优化 | ✅ 已完成 | 100% |

---

## 一、背景与动机

### 1.1 当前项目能力

My-Chat-LangChain 是一个全功能实时流式 Agentic RAG 平台，目前拥有：

- **90+ 工具**: Web 搜索、电商数据、社交媒体、浏览器自动化、学术论文、RAG 知识库
- **LangGraph Agent**: 基于 ReAct 模式的智能代理
- **实时流式响应**: SSE 推送 AI 回复和工具调用状态
- **会话持久化**: SQLite 存储对话历史

### 1.2 能力缺口

当前项目**缺少代码执行能力**：

| 用户需求 | 当前状态 | 期望状态 |
|---------|---------|---------|
| "帮我分析这个 CSV 数据" | ❌ 无法执行 | ✅ 执行 pandas 代码分析 |
| "画一个销售趋势图" | ❌ 无法生成 | ✅ 生成 matplotlib 图表 |
| "验证这段代码是否正确" | ❌ 只能静态分析 | ✅ 实际运行验证 |
| "计算这个数学公式" | ❌ LLM 计算易出错 | ✅ Python 精确计算 |

### 1.3 为什么选择 E2B

| 方案 | 安全性 | 易用性 | 成本 | 适合场景 |
|------|--------|--------|------|---------|
| 本地执行 | ❌ 危险 | ✅ 简单 | 免费 | 仅开发测试 |
| Docker 容器 | ⚠️ 需配置 | ⚠️ 复杂 | 服务器成本 | 自托管场景 |
| **E2B 云沙箱** | ✅ 隔离 | ✅ SDK 简单 | 按需付费 | **生产环境首选** |

**E2B 核心优势**:
- 每次执行在独立 VM 中，完全隔离
- Python/JS SDK，与 LangChain 无缝集成
- 支持文件系统操作，可处理用户上传文件
- 预装 Python 环境，可按需安装包

---

## 二、技术方案

### 2.1 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      My-Chat-LangChain                          │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)                                           │
│  ├── 聊天界面                                                    │
│  ├── 文件上传 (支持 .py, .csv, .xlsx, .json)                     │
│  └── 代码执行结果展示 (文本 + 图表)                                │
├─────────────────────────────────────────────────────────────────┤
│  Backend (FastAPI)                                              │
│  ├── /chat/stream (现有)                                         │
│  ├── /upload_file (现有)                                         │
│  └── API 端点保持不变，工具层自动处理                               │
├─────────────────────────────────────────────────────────────────┤
│  Agent Service (LangGraph)                                      │
│  ├── 现有 90+ 工具                                               │
│  └── 新增 E2B 工具集 ────────────────────────┐                   │
│       ├── execute_python_code               │                   │
│       ├── execute_shell_command             │                   │
│       ├── install_python_package            │                   │
│       ├── upload_data_to_sandbox            │                   │
│       ├── download_file_from_sandbox        │                   │
│       ├── create_visualization              │                   │
│       └── analyze_csv_data                  │                   │
├─────────────────────────────────────────────┼───────────────────┤
│                                             │                   │
│                                             ▼                   │
│                                    ┌────────────────┐           │
│                                    │   E2B Cloud    │           │
│                                    │   Sandbox      │           │
│                                    │  ┌──────────┐  │           │
│                                    │  │ Python   │  │           │
│                                    │  │ Runtime  │  │           │
│                                    │  └──────────┘  │           │
│                                    │  ┌──────────┐  │           │
│                                    │  │ Files    │  │           │
│                                    │  └──────────┘  │           │
│                                    └────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 新增工具清单 ✅ 已实现

| 工具名称 | 功能描述 | 触发场景 | 状态 |
|---------|---------|---------|------|
| `execute_python_code` | 执行 Python 代码 | 数据分析、计算、验证代码 | ✅ 已实现 |
| `execute_shell_command` | 执行 Shell 命令 | 查看文件、系统信息 | ✅ 已实现 |
| `install_python_package` | 安装 Python 包 | 需要额外依赖时 | ✅ 已实现 |
| `upload_data_to_sandbox` | 上传文件到沙箱 | 分析用户上传的数据文件 | ✅ 已实现 |
| `download_file_from_sandbox` | 从沙箱下载文件 | 获取生成的结果文件 | ✅ 已实现 |
| `create_visualization` | 生成可视化图表 | 数据可视化需求 | ✅ 已实现 |
| `analyze_csv_data` | 快速分析 CSV | 数据探索 | ✅ 已实现 |
| `generate_chart_from_data` | 快速生成图表 | 简单图表需求 | ✅ 已实现 (新增) |

### 2.3 数据流设计

```
用户上传 CSV 文件
        │
        ▼
┌───────────────────┐
│  /upload_file API │ ──► 保存到 /tmp/temp_uploads/
└───────────────────┘
        │
        ▼
用户: "分析这个数据的趋势"
        │
        ▼
┌───────────────────┐
│  Agent 意图识别   │ ──► 识别为数据分析任务
└───────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  工具调用链:                                       │
│  1. upload_data_to_sandbox("data.csv")            │
│  2. analyze_csv_data("/tmp/data/data.csv", ...)   │
│  3. execute_python_code(trend_analysis_code)      │
│  4. create_visualization(...)                     │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│  返回分析结果     │ ──► 文本 + 图表 Base64
└───────────────────┘
```

---

## 三、实现细节

### 3.1 依赖安装 ✅ 已完成

**文件**: `backend/requirements.txt`

```txt
# 新增 E2B 依赖 ✅
e2b>=1.0.0
e2b-code-interpreter>=1.0.0
```

> ⚠️ **注意**: 原计划使用 `e2b==1.2.5`，但该版本不存在，已改为 `>=1.0.0`

### 3.2 E2B 工具模块 ✅ 已完成

**新建文件**: `backend/tools/e2b_tools.py`

> ⚠️ **实际实现与计划的差异**:
> - 使用 `AsyncSandbox` 替代同步 `Sandbox`（支持异步操作）
> - 使用 `sandbox.run_code()` 替代 `sandbox.commands.run()`（E2B Code Interpreter API）
> - 添加了 `_get_lock()` 懒加载机制解决事件循环问题
> - E2B v1 API 变更：`execution.logs.stdout` 返回字符串列表而非对象列表

```python
import os
import base64
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from e2b import Sandbox

# ============================================================
# E2B Sandbox 管理
# ============================================================

_sandbox: Optional[Sandbox] = None

def get_sandbox() -> Sandbox:
    """
    获取或创建 E2B Sandbox 单例。
    使用单例模式避免频繁创建/销毁沙箱，节省成本和时间。
    """
    global _sandbox
    if _sandbox is None or not _sandbox.is_running():
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            raise ValueError("E2B_API_KEY 环境变量未设置")

        _sandbox = Sandbox(
            api_key=api_key,
            timeout=300,  # 5分钟超时
            metadata={"project": "my-chat-langchain"}
        )

        # 预装常用数据分析库
        print("📦 [E2B] 正在初始化沙箱环境...")
        _sandbox.commands.run(
            "pip install pandas numpy matplotlib seaborn plotly openpyxl xlrd scipy scikit-learn",
            timeout=120
        )
        print("✅ [E2B] 沙箱环境初始化完成")

    return _sandbox

async def close_sandbox():
    """关闭沙箱（应用关闭时调用）"""
    global _sandbox
    if _sandbox and _sandbox.is_running():
        _sandbox.close()
        _sandbox = None
        print("🔒 [E2B] 沙箱已关闭")

# ============================================================
# 核心工具定义
# ============================================================

@tool
def execute_python_code(code: str) -> str:
    """
    在安全的云沙箱中执行 Python 代码。

    适用场景:
    - 数据分析和处理 (pandas, numpy)
    - 数学计算和统计分析
    - 生成可视化图表 (matplotlib, seaborn, plotly)
    - 验证代码逻辑
    - 文件处理和转换

    Args:
        code (str): 要执行的 Python 代码。支持多行代码。

    Returns:
        str: 执行结果，包括 stdout、stderr 和执行状态

    注意:
    - 代码在隔离的云环境中运行，不会影响主系统
    - 如需生成图表，请将图片保存到 /tmp 目录
    - 如需读取用户上传的文件，文件位于 /tmp/data/ 目录
    - 单次执行超时时间为 60 秒
    - 预装库: pandas, numpy, matplotlib, seaborn, plotly, scipy, scikit-learn
    """
    try:
        sandbox = get_sandbox()

        # 将代码写入临时文件并执行
        sandbox.files.write("/tmp/script.py", code)
        result = sandbox.commands.run("python /tmp/script.py", timeout=60)

        output_parts = []

        if result.stdout:
            output_parts.append(f"📤 **输出**:\n```\n{result.stdout}\n```")

        if result.stderr:
            # 过滤掉常见的无害警告
            stderr_lines = [
                line for line in result.stderr.split('\n')
                if not any(ignore in line for ignore in [
                    'FutureWarning', 'DeprecationWarning', 'UserWarning'
                ])
            ]
            if stderr_lines:
                filtered_stderr = '\n'.join(stderr_lines)
                output_parts.append(f"⚠️ **警告/错误**:\n```\n{filtered_stderr}\n```")

        if result.exit_code == 0:
            output_parts.append("✅ 代码执行成功")
        else:
            output_parts.append(f"❌ 退出码: {result.exit_code}")

        return "\n\n".join(output_parts) if output_parts else "代码执行完成，无输出"

    except Exception as e:
        return f"❌ 执行错误: {str(e)}"


@tool
def execute_shell_command(command: str) -> str:
    """
    在云沙箱中执行 Shell 命令。

    适用场景:
    - 查看文件列表 (ls, find)
    - 检查系统信息 (uname, df, free)
    - 简单的文件操作 (cat, head, tail, wc)
    - 查看已安装的包 (pip list)

    Args:
        command (str): 要执行的 Shell 命令

    Returns:
        str: 命令执行结果

    限制:
    - 超时时间 30 秒
    - 禁止执行危险命令 (rm -rf /, etc.)
    """
    # 安全检查：禁止危险命令
    dangerous_patterns = ['rm -rf /', 'mkfs', 'dd if=', ':(){', 'fork bomb']
    for pattern in dangerous_patterns:
        if pattern in command.lower():
            return f"❌ 安全限制: 禁止执行危险命令"

    try:
        sandbox = get_sandbox()
        result = sandbox.commands.run(command, timeout=30)

        output = ""
        if result.stdout:
            output += f"📤 **输出**:\n```\n{result.stdout}\n```"
        if result.stderr:
            output += f"\n⚠️ **错误**:\n```\n{result.stderr}\n```"

        return output if output else "命令执行完成，无输出"

    except Exception as e:
        return f"❌ 执行错误: {str(e)}"


@tool
def install_python_package(package_name: str) -> str:
    """
    在沙箱中安装 Python 包。

    Args:
        package_name (str): 要安装的包名，支持版本指定，如 "requests" 或 "pandas==2.0.0"

    Returns:
        str: 安装结果

    注意:
    - 安装可能需要一些时间，超时设置为 120 秒
    - 常用数据分析包已预装，无需重复安装
    """
    # 预装包列表
    preinstalled = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
                    'openpyxl', 'xlrd', 'scipy', 'scikit-learn']

    base_package = package_name.split('==')[0].split('>=')[0].split('<=')[0]
    if base_package.lower() in preinstalled:
        return f"ℹ️ {base_package} 已预装，无需重复安装"

    try:
        sandbox = get_sandbox()
        result = sandbox.commands.run(f"pip install {package_name}", timeout=120)

        if result.exit_code == 0:
            return f"✅ 成功安装 {package_name}"
        else:
            return f"❌ 安装失败:\n```\n{result.stderr}\n```"

    except Exception as e:
        return f"❌ 安装错误: {str(e)}"


@tool
def upload_data_to_sandbox(filename: str) -> str:
    """
    将用户上传的文件传输到沙箱环境以供分析。

    Args:
        filename (str): 本地临时目录中的文件名（用户上传时的原始文件名）

    Returns:
        str: 上传结果和沙箱中的文件路径

    说明:
    - 文件将被上传到沙箱的 /tmp/data/ 目录
    - 上传后可使用 execute_python_code 读取和分析文件
    """
    try:
        # 读取本地临时文件
        temp_dir = "/tmp/temp_uploads"
        local_path = os.path.join(temp_dir, filename)

        if not os.path.exists(local_path):
            return f"❌ 找不到文件: {filename}。请确认文件已上传。"

        sandbox = get_sandbox()
        sandbox_path = f"/tmp/data/{filename}"

        # 确保目录存在
        sandbox.commands.run("mkdir -p /tmp/data")

        # 读取并上传文件（自动处理二进制/文本）
        with open(local_path, "rb") as f:
            content = f.read()

        sandbox.files.write(sandbox_path, content)

        # 获取文件信息
        file_size = len(content)
        size_str = f"{file_size / 1024:.1f} KB" if file_size > 1024 else f"{file_size} bytes"

        return f"""✅ 文件上传成功

📁 **文件信息**:
- 文件名: {filename}
- 大小: {size_str}
- 沙箱路径: `{sandbox_path}`

💡 **使用提示**:
```python
import pandas as pd
df = pd.read_csv("{sandbox_path}")  # 或其他适合的读取方法
print(df.head())
```"""

    except Exception as e:
        return f"❌ 上传错误: {str(e)}"


@tool
def download_file_from_sandbox(sandbox_path: str) -> Dict[str, Any]:
    """
    从沙箱下载文件。

    Args:
        sandbox_path (str): 沙箱中的文件路径，如 "/tmp/result.csv"

    Returns:
        Dict: 包含文件内容（Base64编码）和元信息
    """
    try:
        sandbox = get_sandbox()
        content = sandbox.files.read(sandbox_path)

        if isinstance(content, bytes):
            content_b64 = base64.b64encode(content).decode("utf-8")
            return {
                "success": True,
                "filename": os.path.basename(sandbox_path),
                "content_base64": content_b64,
                "size_bytes": len(content)
            }
        else:
            return {
                "success": True,
                "filename": os.path.basename(sandbox_path),
                "content_text": content,
                "size_bytes": len(content.encode('utf-8'))
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def create_visualization(
    data_description: str,
    chart_type: str,
    code: str
) -> str:
    """
    生成数据可视化图表。

    Args:
        data_description (str): 数据和图表的简要描述
        chart_type (str): 图表类型，如 "bar", "line", "scatter", "pie", "heatmap", "histogram"
        code (str): 生成图表的完整 Python 代码

    Returns:
        str: 执行结果，包含图表的 Base64 编码（如果成功生成）

    代码要求:
    - 必须将图表保存到 /tmp/chart.png
    - 示例: plt.savefig('/tmp/chart.png', dpi=150, bbox_inches='tight')
    - 建议设置中文字体: plt.rcParams['font.sans-serif'] = ['SimHei']
    """
    try:
        sandbox = get_sandbox()

        # 确保代码保存图片
        if "savefig" not in code:
            code += "\nimport matplotlib.pyplot as plt\nplt.savefig('/tmp/chart.png', dpi=150, bbox_inches='tight')"

        sandbox.files.write("/tmp/viz_script.py", code)
        result = sandbox.commands.run("python /tmp/viz_script.py", timeout=60)

        output_parts = []

        if result.stdout:
            output_parts.append(f"📤 **输出**:\n```\n{result.stdout}\n```")

        if result.stderr:
            stderr_clean = '\n'.join([
                line for line in result.stderr.split('\n')
                if 'Warning' not in line
            ])
            if stderr_clean.strip():
                output_parts.append(f"⚠️ **警告**:\n```\n{stderr_clean}\n```")

        # 尝试读取生成的图片
        try:
            image_content = sandbox.files.read("/tmp/chart.png")
            if isinstance(image_content, bytes):
                image_b64 = base64.b64encode(image_content).decode("utf-8")
                output_parts.append(f"✅ **图表生成成功**")
                output_parts.append(f"📊 图表类型: {chart_type}")
                output_parts.append(f"📝 描述: {data_description}")
                output_parts.append(f"\n[IMAGE_BASE64:{image_b64}]")
        except Exception as img_err:
            output_parts.append(f"⚠️ 图表文件读取失败: {img_err}")

        return "\n\n".join(output_parts)

    except Exception as e:
        return f"❌ 可视化生成错误: {str(e)}"


@tool
def analyze_csv_data(filename: str, analysis_request: str = "基础分析") -> str:
    """
    快速分析 CSV 数据文件，返回数据概览和基础统计信息。

    Args:
        filename (str): 沙箱中的 CSV 文件路径（如 /tmp/data/sales.csv）
                       或仅文件名（将自动添加 /tmp/data/ 前缀）
        analysis_request (str): 分析需求描述，如 "找出销售趋势" 或 "统计各类别分布"

    Returns:
        str: 数据分析结果，包括数据概览、统计摘要、缺失值分析等
    """
    # 自动补全路径
    if not filename.startswith('/'):
        filename = f"/tmp/data/{filename}"

    analysis_code = f'''
import pandas as pd
import numpy as np

# 读取数据
try:
    df = pd.read_csv("{filename}")
except Exception as e:
    print(f"❌ 读取文件失败: {{e}}")
    exit(1)

print("=" * 50)
print("📊 数据概览")
print("=" * 50)
print(f"📐 数据维度: {{df.shape[0]}} 行 × {{df.shape[1]}} 列")
print(f"📋 列名: {{list(df.columns)}}")

print("\\n" + "=" * 50)
print("🔤 数据类型")
print("=" * 50)
print(df.dtypes.to_string())

print("\\n" + "=" * 50)
print("👀 数据预览 (前5行)")
print("=" * 50)
print(df.head().to_string())

print("\\n" + "=" * 50)
print("📈 数值列统计摘要")
print("=" * 50)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe().to_string())
else:
    print("没有数值列")

print("\\n" + "=" * 50)
print("❓ 缺失值分析")
print("=" * 50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({{"缺失数量": missing, "缺失比例(%)": missing_pct}})
print(missing_df[missing_df["缺失数量"] > 0].to_string() if missing.sum() > 0 else "没有缺失值 ✅")

print("\\n" + "=" * 50)
print("🏷️ 分类列统计")
print("=" * 50)
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols[:3]:  # 只显示前3个分类列
    print(f"\\n【{{col}}】唯一值数量: {{df[col].nunique()}}")
    print(df[col].value_counts().head(5).to_string())

print("\\n" + "=" * 50)
print(f"💡 分析需求: {analysis_request}")
print("=" * 50)
print("数据已加载完成，可以进行进一步分析。")
'''

    return execute_python_code.invoke({"code": analysis_code})
```

### 3.3 Agent Service 集成 ✅ 已完成

**修改文件**: `backend/agent_service.py`

```python
# ============================================================
# 在文件顶部添加导入
# ============================================================
from tools.e2b_tools import (
    execute_python_code,
    execute_shell_command,
    install_python_package,
    upload_data_to_sandbox,
    download_file_from_sandbox,
    create_visualization,
    analyze_csv_data,
    close_sandbox
)

# ============================================================
# 更新 custom_tools 列表
# ============================================================
custom_tools = [
    # 现有工具
    ingest_knowledge,
    query_knowledge_base,
    format_paper_analysis,
    format_linkedin_profile,

    # E2B 代码执行工具 (新增)
    execute_python_code,
    execute_shell_command,
    install_python_package,
    upload_data_to_sandbox,
    download_file_from_sandbox,
    create_visualization,
    analyze_csv_data,
]

# ============================================================
# 更新 cleanup() 函数
# ============================================================
async def cleanup():
    """Cleanup function to close database connection and sandbox."""
    global _sqlite_conn, _mcp_client
    if _sqlite_conn:
        await _sqlite_conn.close()
        _sqlite_conn = None
    if _mcp_client:
        _mcp_client = None
    # 清理 E2B 沙箱
    await close_sandbox()
```

### 3.4 System Prompt 更新 ✅ 已完成

在 `SYSTEM_PROMPT` 中添加第 8 类工具（版本更新为 v7.0）：

```python
### 8️⃣ 代码执行工具 (E2B 云沙箱)
**触发场景**: 用户需要执行代码、数据分析、生成图表、验证算法
**核心工具**:
- `execute_python_code(code)` - 执行 Python 代码（支持 pandas, numpy, matplotlib 等）
- `execute_shell_command(command)` - 执行 Shell 命令
- `install_python_package(package)` - 安装额外的 Python 包
- `upload_data_to_sandbox(filename)` - 上传数据文件到沙箱
- `download_file_from_sandbox(path)` - 从沙箱下载文件
- `create_visualization(desc, type, code)` - 生成可视化图表
- `analyze_csv_data(filename, request)` - 快速分析 CSV 数据

**意图识别关键词**: "执行代码"、"运行"、"计算"、"分析数据"、"画图"、"可视化"、"统计"、"验证"

**工具链示例: 数据分析任务**
1. 用户上传 sales.csv 文件
2. `upload_data_to_sandbox("sales.csv")` 传输文件到沙箱
3. `analyze_csv_data("/tmp/data/sales.csv", "分析销售趋势")` 获取数据概览
4. `execute_python_code(detailed_analysis)` 执行深度分析代码
5. `create_visualization("月度销售趋势", "line", plot_code)` 生成趋势图

**工具链示例: 代码验证任务**
1. 用户: "写一个快速排序并验证"
2. 生成快速排序代码
3. `execute_python_code(quicksort_with_tests)` 运行并验证
4. 返回执行结果和测试输出
```

### 3.5 环境变量配置 ✅ 已完成

**更新文件**: `backend/.env`

```bash
# 现有配置
GOOGLE_API_KEY=your_google_api_key
BRIGHT_DATA_API_KEY=your_bright_data_key
PAPER_SEARCH_API_KEY=your_paper_search_key

# 新增 E2B 配置
E2B_API_KEY=your_e2b_api_key
```

**Render 部署**: 在 Render Dashboard 添加环境变量 `E2B_API_KEY`

---

## 四、前端增强 ✅ 已完成

### 4.1 图表展示支持 ✅ 已完成

**修改文件**: `frontend/app.py`

已添加 `render_content_with_images()` 和 `render_tool_output()` 函数处理 Base64 图片渲染。

**关键修复**:
- 修复 Streamlit expander 嵌套错误，改为直接在 `status_container` 中渲染
- 添加 Streamlit 版本兼容处理 (`use_container_width` vs `use_column_width`)
- 过滤流式文本中的 Base64 数据，避免重复显示

```python
def render_tool_output(output_str, container):
    """Render tool output, handling embedded images."""
    image_pattern = r'\[IMAGE_BASE64:([A-Za-z0-9+/=]+)\]'
    matches = list(re.finditer(image_pattern, output_str))

    if matches:
        for match in matches:
            try:
                image_b64 = match.group(1)
                image_bytes = base64.b64decode(image_b64)
                try:
                    container.image(image_bytes, caption="📊 Generated Chart", use_container_width=True)
                except TypeError:
                    container.image(image_bytes, caption="📊 Generated Chart", use_column_width=True)
            except Exception as e:
                container.warning(f"Failed to render chart: {e}")
```

### 4.2 支持更多文件类型上传 ✅ 已完成

**修改文件**: `frontend/app.py`

```python
# 扩展支持的文件类型 ✅
uploaded_file = st.file_uploader(
    "Upload file for analysis",
    type=['pdf', 'csv', 'xlsx', 'xls', 'json', 'txt', 'py'],
    key="file_uploader"
)
```

### 4.3 已优化项 ✅

- [x] 图表直接显示在工具结果区域（修复嵌套 expander 问题）
- [x] 添加 Streamlit 版本兼容处理
- [x] 过滤 LLM 回复中的 Base64 数据

---

## 五、使用场景示例 ✅ 已验证

> 以下场景已在 2025-12-17 进行端到端测试验证

### 场景 1: 销售数据分析

**用户输入**:
> 我上传了 sales_2024.csv，帮我分析一下哪个产品销量最好，并画一个月度销售趋势图

**Agent 执行流程**:

```
1. upload_data_to_sandbox("sales_2024.csv")
   → ✅ 文件上传到 /tmp/data/sales_2024.csv

2. analyze_csv_data("/tmp/data/sales_2024.csv", "产品销量分析")
   → 返回数据概览、各产品销量统计

3. execute_python_code("""
   import pandas as pd
   df = pd.read_csv('/tmp/data/sales_2024.csv')
   top_product = df.groupby('product')['sales'].sum().idxmax()
   print(f"销量最好的产品: {top_product}")
   """)
   → 输出最佳产品

4. create_visualization(
   "月度销售趋势",
   "line",
   """
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.read_csv('/tmp/data/sales_2024.csv')
   monthly = df.groupby('month')['sales'].sum()
   plt.figure(figsize=(10, 6))
   plt.plot(monthly.index, monthly.values, marker='o')
   plt.title('2024年月度销售趋势')
   plt.xlabel('月份')
   plt.ylabel('销售额')
   plt.savefig('/tmp/chart.png', dpi=150, bbox_inches='tight')
   """
   )
   → 返回趋势图
```

### 场景 2: 算法验证

**用户输入**:
> 帮我实现一个二分查找算法，并用几个测试用例验证

**Agent 执行流程**:

```
1. execute_python_code("""
   def binary_search(arr, target):
       left, right = 0, len(arr) - 1
       while left <= right:
           mid = (left + right) // 2
           if arr[mid] == target:
               return mid
           elif arr[mid] < target:
               left = mid + 1
           else:
               right = mid - 1
       return -1

   # 测试用例
   test_cases = [
       ([1, 3, 5, 7, 9], 5, 2),
       ([1, 3, 5, 7, 9], 1, 0),
       ([1, 3, 5, 7, 9], 9, 4),
       ([1, 3, 5, 7, 9], 4, -1),
       ([], 1, -1),
   ]

   print("二分查找测试结果:")
   for arr, target, expected in test_cases:
       result = binary_search(arr, target)
       status = "✅" if result == expected else "❌"
       print(f"{status} binary_search({arr}, {target}) = {result}, 期望: {expected}")
   """)

   → 输出所有测试结果
```

### 场景 3: 学术论文 + 数据分析

**用户输入**:
> 搜索关于 "transformer attention mechanism" 的最新论文，如果有实验数据就帮我可视化

**Agent 执行流程**:

```
1. search_arxiv("transformer attention mechanism 2024")
   → 返回论文列表

2. download_arxiv("2401.xxxxx")
   → 下载论文 PDF

3. ingest_knowledge("2401.xxxxx.pdf", "file")
   → 加入知识库

4. query_knowledge_base("实验结果和数据", "2401.xxxxx.pdf")
   → 提取实验数据

5. execute_python_code(data_extraction_code)
   → 解析数据

6. create_visualization("注意力机制性能对比", "bar", comparison_code)
   → 生成对比图
```

---

## 六、成本与性能考量

### 6.1 E2B 定价 (参考)

| 计划 | 沙箱时长 | 超时上限 | 价格 |
|------|---------|---------|------|
| Hobby | 有限 | 1小时 | 免费 |
| Pro | 无限 | 24小时 | $20/月起 |
| 按用量 | - | - | $0.10/沙箱小时 |

### 6.2 优化策略

| 优化项 | 实现方式 | 效果 |
|--------|---------|------|
| 沙箱复用 | 单例模式 | 减少创建开销 |
| 预装依赖 | 初始化时安装常用包 | 减少运行时等待 |
| 超时控制 | 各操作设置合理超时 | 防止资源浪费 |
| 按需启动 | 只在需要代码执行时创建 | 降低空闲成本 |

### 6.3 性能指标

| 操作 | 预期耗时 |
|------|---------|
| 首次创建沙箱 | 3-5秒 |
| 预装依赖 | 30-60秒（仅首次） |
| 执行简单代码 | 1-2秒 |
| 执行数据分析 | 2-10秒 |
| 生成图表 | 3-8秒 |

---

## 七、安全考虑

### 7.1 E2B 内置安全

- ✅ 每个沙箱运行在独立 VM 中
- ✅ 沙箱间完全隔离，无法相互访问
- ✅ 自动超时销毁，防止资源泄露
- ✅ 无法访问宿主机文件系统

### 7.2 应用层安全

| 风险 | 缓解措施 |
|------|---------|
| 恶意代码执行 | E2B 沙箱隔离 |
| 资源耗尽 | 超时限制 (60秒) |
| 危险命令 | Shell 命令白名单检查 |
| 数据泄露 | 沙箱超时后自动清理 |

---

## 八、实施计划

### 阶段 1: 基础集成 ✅ 已完成

- [x] 安装 E2B SDK (`e2b>=1.0.0`, `e2b-code-interpreter>=1.0.0`)
- [x] 创建 `backend/tools/e2b_tools.py`
- [x] 实现 `execute_python_code` 和 `execute_shell_command`
- [x] 集成到 Agent Service
- [x] 测试基本代码执行 ✅ 2025-12-17 验证通过

### 阶段 2: 数据分析能力 ✅ 已完成

- [x] 实现 `upload_data_to_sandbox`
- [x] 实现 `analyze_csv_data`
- [x] 实现 `create_visualization` (后移除，改用 `execute_python_code` 统一处理)
- [x] 实现 `generate_chart_from_data` (后移除，改用 `execute_python_code` 统一处理)
- [x] 数据分析完整流程测试通过（上传CSV → 分析 → 可视化）

### 阶段 3: 前端增强 ✅ 已完成

- [x] 添加图表渲染支持 (`render_content_with_images`)
- [x] 扩展文件上传类型 (PDF, CSV, Excel, JSON, TXT, Python)
- [x] 添加 E2B API Key 输入框
- [x] 修复 Streamlit expander 嵌套错误
- [x] 添加 Streamlit 版本兼容处理
- [x] 过滤 LLM 回复中的 Base64 图片数据

### 阶段 4: 测试与优化 ✅ 已完成

**端到端测试用例**:
- [x] 简单计算: "帮我计算 1+1 并验证" ✅ 通过
- [x] 数学图表: "画一个正弦波图表" ✅ 通过
- [x] 算法验证: "写一个快速排序并测试" ✅ 通过
- [x] 数据分析: 上传 CSV → "分析这个数据并画趋势图" ✅ 通过

**性能优化**:
- [x] 沙箱超时自动重建机制（10分钟超时 + ping 检测）
- [x] 工具集简化（移除 `create_visualization` 和 `generate_chart_from_data`，统一使用 `execute_python_code`）
- [x] System Prompt 优化（强制先读取列名再画图，减少试错次数）

**优化效果**:
| 测试场景 | 优化前工具调用 | 优化后工具调用 |
|---------|--------------|--------------|
| CSV 分析+画图 | 6-8 次 | 3 次 |
| 数学图表 | 3-4 次 | 1-2 次 |

---

## 九、风险与回退方案

| 风险 | 可能性 | 影响 | 回退方案 |
|------|--------|------|---------|
| E2B 服务不可用 | 低 | 高 | 工具返回友好错误提示 |
| 超出免费额度 | 中 | 中 | 升级计划或限制使用频率 |
| 代码执行超时 | 中 | 低 | 增加超时时间或拆分任务 |
| 依赖安装失败 | 低 | 低 | 预装更多常用包 |

---

## 十、总结

### 预期收益

| 维度 | 提升 |
|------|------|
| 功能完整度 | 从 90+ 工具扩展到 99+ 工具 ✅ |
| 用户体验 | 支持代码执行、数据分析、图表生成 ✅ |
| 应用场景 | 新增数据科学、算法验证等场景 ✅ |
| 竞争力 | 对标 ChatGPT Code Interpreter ✅ |

### 开发资源

| 资源 | 预估 | 实际 |
|------|------|------|
| 开发时间 | 2-3 天 | 1 天 (核心功能) |
| 测试时间 | 0.5 天 | 🔴 待进行 |
| 代码量 | ~400 行 | ~500 行 (`e2b_tools.py`) |
| 依赖增加 | 2 个包 | ✅ 2 个包 |

---

## 🚨 已解决的技术问题

| 问题 | 原因 | 解决方案 | 日期 |
|------|------|----------|------|
| 前端卡在 "Thinking..." | Clash 代理节点选择香港，Gemini API 不可用 | 切换到新加坡节点 | 2025-12-17 |
| `'str' object has no attribute 'line'` | E2B SDK v1 API 变更 | 修改 `execution.logs.stdout` 处理逻辑 | 2025-12-17 |
| `asyncio.Lock()` 事件循环问题 | 模块加载时创建 Lock | 改为懒加载 `_get_lock()` | 2025-12-17 |
| 图表不显示 | Base64 图片数据被截断（1000字符限制） | 修改截断逻辑，保留完整图片数据 | 2025-12-17 |
| Streamlit expander 嵌套错误 | st.status 内嵌套 st.expander | 直接在 status_container 中渲染 | 2025-12-17 |
| use_container_width 不兼容 | Streamlit 版本差异 | 添加 try/except 回退到 use_column_width | 2025-12-17 |
| E2B 沙箱超时 502 错误 | 沙箱 5 分钟超时后失效 | 增加超时到 10 分钟，添加 ping 检测自动重建 | 2025-12-17 |
| Windows 文件路径错误 | `/tmp/temp_uploads` 不存在 | 根据 platform.system() 选择路径 | 2025-12-17 |
| LLM 输出 Base64 数据 | Agent 在回复中复述图片数据 | 前端过滤 + Prompt 明确禁止 | 2025-12-17 |
| 工具调用混乱（6-8次） | 工具集过多，列名猜测 | 简化工具集，强制先读取列名 | 2025-12-17 |

---

## 📌 下一步行动 (可选优化)

### 🟢 低优先级

1. **更新 README 文档**: 添加 E2B 功能说明和使用示例
2. **部署配置**: 在 Render 添加 E2B_API_KEY 环境变量
3. **添加更多图表类型支持**: 饼图、热力图、3D 图表等

---

## 十一、开发日志

### 2025-12-17 开发总结

**主要成就**:
1. ✅ E2B 云沙箱集成完成
2. ✅ 代码执行、数据分析、图表生成功能全部可用
3. ✅ 前端图表渲染正常
4. ✅ 工具调用效率优化（从 6-8 次减少到 3 次）

**最终工具集**:
- `execute_python_code` - 核心工具，处理所有代码执行和图表生成
- `execute_shell_command` - Shell 命令执行
- `install_python_package` - 安装 Python 包
- `upload_data_to_sandbox` - 上传文件到沙箱
- `download_file_from_sandbox` - 从沙箱下载文件
- `analyze_csv_data` - 快速 CSV 数据分析

**移除的工具**:
- `create_visualization` - 功能合并到 `execute_python_code`
- `generate_chart_from_data` - 功能合并到 `execute_python_code`

**关键配置修改**:
- `agent_service.py`: 简化工具集，优化 System Prompt
- `e2b_tools.py`: 沙箱超时 10 分钟，自动重建机制
- `main.py`: Windows 路径兼容
- `app.py`: 图表渲染，Streamlit 版本兼容

---

> **审阅日期**: 2025-12-17
> **状态**: ✅ 开发完成
