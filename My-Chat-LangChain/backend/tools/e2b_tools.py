"""
E2B Code Interpreter Tools for My-Chat-LangChain

This module provides secure code execution capabilities using E2B's cloud sandbox.
All code runs in an isolated environment, ensuring safety and security.
"""

import os
import base64
import asyncio
from typing import Optional, Dict, Any
from langchain_core.tools import tool

# E2B Code Interpreter imports
from e2b_code_interpreter import AsyncSandbox

# ============================================================
# E2B Sandbox 管理
# ============================================================

_sandbox: Optional[AsyncSandbox] = None
_sandbox_lock: Optional[asyncio.Lock] = None  # Lazy initialization


def _get_lock() -> asyncio.Lock:
    """Lazily create the lock in the current event loop."""
    global _sandbox_lock
    if _sandbox_lock is None:
        _sandbox_lock = asyncio.Lock()
    return _sandbox_lock


async def _create_new_sandbox() -> AsyncSandbox:
    """Create a new E2B sandbox instance."""
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise ValueError("E2B_API_KEY 环境变量未设置。请在 .env 文件中配置 E2B_API_KEY。")

    print("📦 [E2B] 正在创建云沙箱...")
    sandbox = await AsyncSandbox.create(
        api_key=api_key,
        timeout=600,  # 10分钟超时 (increased from 5 min)
    )

    # 预装常用数据分析库
    print("📦 [E2B] 正在安装常用数据分析库...")
    await sandbox.run_code(
        """
import subprocess
subprocess.run(['pip', 'install', '-q', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'openpyxl', 'xlrd', 'scipy'],
               capture_output=True)
print("✅ 常用库安装完成")
""",
        timeout=120
    )
    print("✅ [E2B] 沙箱环境初始化完成")
    return sandbox


async def get_sandbox() -> AsyncSandbox:
    """
    获取或创建 E2B Sandbox 单例。
    使用单例模式避免频繁创建/销毁沙箱，节省成本和时间。
    如果沙箱超时失效，会自动重新创建。
    """
    global _sandbox

    async with _get_lock():
        # 如果沙箱不存在，创建新的
        if _sandbox is None:
            _sandbox = await _create_new_sandbox()
            return _sandbox

        # 检查沙箱是否仍然有效（通过尝试执行简单命令）
        try:
            await _sandbox.run_code("print('ping')", timeout=5)
            return _sandbox
        except Exception as e:
            print(f"⚠️ [E2B] 沙箱已失效 ({str(e)[:50]}...)，正在重新创建...")
            _sandbox = None
            _sandbox = await _create_new_sandbox()
            return _sandbox


async def close_sandbox():
    """关闭沙箱（应用关闭时调用）"""
    global _sandbox, _sandbox_lock
    if _sandbox_lock is None:
        return  # Lock never created, sandbox never used
    async with _sandbox_lock:
        if _sandbox is not None:
            try:
                await _sandbox.kill()
                print("🔒 [E2B] 沙箱已关闭")
            except Exception as e:
                print(f"⚠️ [E2B] 关闭沙箱时出错: {e}")
            finally:
                _sandbox = None


# ============================================================
# 核心工具定义
# ============================================================

@tool
async def execute_python_code(code: str) -> str:
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
    - 如需生成图表，请将图片保存到文件并使用 plt.savefig()
    - 如需读取用户上传的文件，文件位于 /home/user/data/ 目录
    - 单次执行超时时间为 60 秒
    - 预装库: pandas, numpy, matplotlib, seaborn, plotly, scipy
    """
    try:
        sandbox = await get_sandbox()

        print(f"🔄 [E2B] 正在执行代码...")
        execution = await sandbox.run_code(code, timeout=60)

        output_parts = []

        # 收集标准输出 - E2B v1 返回字符串列表
        if execution.logs and execution.logs.stdout:
            # stdout 是字符串列表
            stdout_content = '\n'.join(execution.logs.stdout) if isinstance(execution.logs.stdout, list) else str(execution.logs.stdout)
            if stdout_content.strip():
                output_parts.append(f"📤 **输出**:\n```\n{stdout_content}\n```")

        # 收集标准错误（过滤常见无害警告）
        if execution.logs and execution.logs.stderr:
            stderr_lines = []
            stderr_list = execution.logs.stderr if isinstance(execution.logs.stderr, list) else [str(execution.logs.stderr)]
            for line in stderr_list:
                # 过滤常见的无害警告
                if not any(ignore in str(line) for ignore in [
                    'FutureWarning', 'DeprecationWarning', 'UserWarning',
                    'Warning:', 'warnings.warn'
                ]):
                    stderr_lines.append(str(line))
            if stderr_lines:
                stderr_content = '\n'.join(stderr_lines)
                output_parts.append(f"⚠️ **警告/错误**:\n```\n{stderr_content}\n```")

        # 处理执行结果
        if execution.results:
            for result in execution.results:
                # 处理文本结果
                if hasattr(result, 'text') and result.text:
                    output_parts.append(f"📊 **结果**:\n```\n{result.text}\n```")
                # 处理图片结果
                if hasattr(result, 'png') and result.png:
                    # Note: The [IMAGE_BASE64:...] marker will be rendered as an image by the frontend
                    # Do NOT repeat this data in your response - just tell the user the chart was generated
                    output_parts.append(f"🖼️ **图表已生成并显示在上方**\n[IMAGE_BASE64:{result.png}]")

        # 处理执行错误
        if execution.error:
            error_name = getattr(execution.error, 'name', 'Error')
            error_value = getattr(execution.error, 'value', str(execution.error))
            error_traceback = getattr(execution.error, 'traceback', '')
            output_parts.append(f"❌ **执行错误**:\n```\n{error_name}: {error_value}\n{error_traceback}\n```")
        elif not output_parts:
            output_parts.append("✅ 代码执行成功")

        return "\n\n".join(output_parts) if output_parts else "代码执行完成，无输出"

    except Exception as e:
        return f"❌ 执行错误: {str(e)}"


@tool
async def execute_shell_command(command: str) -> str:
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
    - 禁止执行危险命令
    """
    # 安全检查：禁止危险命令
    dangerous_patterns = ['rm -rf /', 'mkfs', 'dd if=', ':(){', 'fork bomb', '> /dev/sda']
    for pattern in dangerous_patterns:
        if pattern in command.lower():
            return f"❌ 安全限制: 禁止执行危险命令"

    try:
        sandbox = await get_sandbox()

        # 使用 Python 的 subprocess 来执行 shell 命令
        shell_code = f'''
import subprocess
result = subprocess.run({repr(command)}, shell=True, capture_output=True, text=True, timeout=30)
if result.stdout:
    print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("EXIT_CODE:", result.returncode)
'''

        execution = await sandbox.run_code(shell_code, timeout=35)

        output_parts = []

        if execution.logs and execution.logs.stdout:
            stdout_content = '\n'.join(execution.logs.stdout) if isinstance(execution.logs.stdout, list) else str(execution.logs.stdout)
            output_parts.append(f"📤 **输出**:\n```\n{stdout_content}\n```")

        if execution.error:
            error_value = getattr(execution.error, 'value', str(execution.error))
            output_parts.append(f"❌ **错误**:\n```\n{error_value}\n```")

        return "\n\n".join(output_parts) if output_parts else "命令执行完成，无输出"

    except Exception as e:
        return f"❌ 执行错误: {str(e)}"


@tool
async def install_python_package(package_name: str) -> str:
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
                    'openpyxl', 'xlrd', 'scipy']

    base_package = package_name.split('==')[0].split('>=')[0].split('<=')[0].lower()
    if base_package in preinstalled:
        return f"ℹ️ {base_package} 已预装，无需重复安装"

    try:
        sandbox = await get_sandbox()

        install_code = f'''
import subprocess
result = subprocess.run(['pip', 'install', '-q', {repr(package_name)}], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ 成功安装 {repr(package_name)}")
else:
    print(f"❌ 安装失败: {{result.stderr}}")
'''

        execution = await sandbox.run_code(install_code, timeout=120)

        if execution.logs and execution.logs.stdout:
            stdout_content = '\n'.join(execution.logs.stdout) if isinstance(execution.logs.stdout, list) else str(execution.logs.stdout)
            return stdout_content

        if execution.error:
            error_value = getattr(execution.error, 'value', str(execution.error))
            return f"❌ 安装错误: {error_value}"

        return f"✅ 成功安装 {package_name}"

    except Exception as e:
        return f"❌ 安装错误: {str(e)}"


@tool
async def upload_data_to_sandbox(filename: str) -> str:
    """
    将用户上传的文件传输到沙箱环境以供分析。

    Args:
        filename (str): 本地临时目录中的文件名（用户上传时的原始文件名）

    Returns:
        str: 上传结果和沙箱中的文件路径

    说明:
    - 文件将被上传到沙箱的 /home/user/data/ 目录
    - 上传后可使用 execute_python_code 读取和分析文件
    """
    import platform

    try:
        # Determine temp upload directory based on platform
        if platform.system() == "Windows":
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_uploads")
        else:
            temp_dir = "/tmp/temp_uploads"

        local_path = os.path.join(temp_dir, filename)

        if not os.path.exists(local_path):
            # List available files to help user
            available_files = []
            if os.path.exists(temp_dir):
                available_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]

            if available_files:
                return f"❌ 找不到文件: {filename}。\n可用文件: {', '.join(available_files[:5])}"
            else:
                return f"❌ 找不到文件: {filename}。临时上传目录为空，请先上传文件。"

        sandbox = await get_sandbox()
        sandbox_path = f"/home/user/data/{filename}"

        # 确保目录存在
        await sandbox.run_code("import os; os.makedirs('/home/user/data', exist_ok=True)")

        # 读取文件内容
        with open(local_path, "rb") as f:
            content = f.read()

        # 上传文件到沙箱
        await sandbox.files.write(sandbox_path, content)

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
df = pd.read_csv("{sandbox_path}")  # CSV 文件
# 或
df = pd.read_excel("{sandbox_path}")  # Excel 文件
print(df.head())
```"""

    except Exception as e:
        return f"❌ 上传错误: {str(e)}"


@tool
async def download_file_from_sandbox(sandbox_path: str) -> str:
    """
    从沙箱下载文件内容。

    Args:
        sandbox_path (str): 沙箱中的文件路径，如 "/home/user/result.csv"

    Returns:
        str: 文件内容或错误信息
    """
    try:
        sandbox = await get_sandbox()
        content = await sandbox.files.read(sandbox_path)

        if isinstance(content, bytes):
            # 尝试解码为文本
            try:
                text_content = content.decode('utf-8')
                return f"📄 **文件内容** (`{sandbox_path}`):\n```\n{text_content[:5000]}\n```" + (
                    "\n\n... (内容已截断)" if len(text_content) > 5000 else ""
                )
            except UnicodeDecodeError:
                # 二进制文件，返回 base64
                content_b64 = base64.b64encode(content).decode("utf-8")
                return f"📦 **二进制文件** (`{sandbox_path}`)\n大小: {len(content)} bytes\nBase64: [FILE_BASE64:{content_b64[:100]}...]"
        else:
            return f"📄 **文件内容** (`{sandbox_path}`):\n```\n{str(content)[:5000]}\n```"

    except Exception as e:
        return f"❌ 下载错误: {str(e)}"


@tool
async def create_visualization(
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
    - 代码应该使用 matplotlib 或其他可视化库
    - E2B 会自动捕获生成的图表
    - 可以使用 plt.show() 或直接返回图表对象
    """
    try:
        sandbox = await get_sandbox()

        # 添加图表显示的支持代码
        viz_code = f'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体支持（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 用户代码
{code}

# 确保显示图表
plt.tight_layout()
plt.show()
'''

        execution = await sandbox.run_code(viz_code, timeout=60)

        output_parts = []
        output_parts.append(f"📊 **图表类型**: {chart_type}")
        output_parts.append(f"📝 **描述**: {data_description}")

        # 收集输出
        if execution.logs and execution.logs.stdout:
            stdout_content = '\n'.join(execution.logs.stdout) if isinstance(execution.logs.stdout, list) else str(execution.logs.stdout)
            if stdout_content.strip():
                output_parts.append(f"📤 **输出**:\n```\n{stdout_content}\n```")

        # 处理图片结果
        image_found = False
        if execution.results:
            for result in execution.results:
                if hasattr(result, 'png') and result.png:
                    output_parts.append(f"✅ **图表生成成功** (图片已显示在前端)")
                    # Note: Frontend will render this as an image - do not repeat in LLM response
                    output_parts.append(f"[IMAGE_BASE64:{result.png}]")
                    image_found = True
                    break

        if not image_found:
            if execution.error:
                error_value = getattr(execution.error, 'value', str(execution.error))
                output_parts.append(f"❌ **错误**:\n```\n{error_value}\n```")
            else:
                output_parts.append("⚠️ 未能捕获图表，请检查代码是否正确调用了 plt.show()")

        return "\n\n".join(output_parts)

    except Exception as e:
        return f"❌ 可视化生成错误: {str(e)}"


@tool
async def analyze_csv_data(filename: str, analysis_request: str = "基础分析") -> str:
    """
    快速分析 CSV 数据文件，返回数据概览和基础统计信息。

    Args:
        filename (str): 文件路径，可以是：
                       - 沙箱完整路径（如 /home/user/data/sales.csv）
                       - 仅文件名（将自动添加 /home/user/data/ 前缀）
        analysis_request (str): 分析需求描述，如 "找出销售趋势" 或 "统计各类别分布"

    Returns:
        str: 数据分析结果，包括数据概览、统计摘要、缺失值分析等
    """
    # 自动补全路径
    if not filename.startswith('/'):
        filename = f"/home/user/data/{filename}"

    analysis_code = f'''
import pandas as pd
import numpy as np

# 读取数据
try:
    df = pd.read_csv("{filename}")
except Exception as e:
    print(f"❌ 读取文件失败: {{e}}")
    raise

print("=" * 60)
print("📊 数据概览")
print("=" * 60)
print(f"📐 数据维度: {{df.shape[0]}} 行 × {{df.shape[1]}} 列")
print(f"📋 列名: {{list(df.columns)}}")

print("\\n" + "=" * 60)
print("🔤 数据类型")
print("=" * 60)
print(df.dtypes.to_string())

print("\\n" + "=" * 60)
print("👀 数据预览 (前5行)")
print("=" * 60)
print(df.head().to_string())

print("\\n" + "=" * 60)
print("📈 数值列统计摘要")
print("=" * 60)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe().round(2).to_string())
else:
    print("没有数值列")

print("\\n" + "=" * 60)
print("❓ 缺失值分析")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({{"缺失数量": missing, "缺失比例(%)": missing_pct}})
missing_info = missing_df[missing_df["缺失数量"] > 0]
if len(missing_info) > 0:
    print(missing_info.to_string())
else:
    print("没有缺失值 ✅")

print("\\n" + "=" * 60)
print("🏷️ 分类列统计 (前3列)")
print("=" * 60)
cat_cols = df.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    for col in list(cat_cols)[:3]:
        print(f"\\n【{{col}}】唯一值数量: {{df[col].nunique()}}")
        print(df[col].value_counts().head(5).to_string())
else:
    print("没有分类列")

print("\\n" + "=" * 60)
print(f"💡 分析需求: {analysis_request}")
print("=" * 60)
print("数据已加载完成，可以进行进一步分析。")
'''

    return await execute_python_code.ainvoke({"code": analysis_code})


# ============================================================
# 额外工具：数据分析辅助
# ============================================================

@tool
async def generate_chart_from_data(
    filename: str,
    x_column: str,
    y_column: str,
    chart_type: str = "line",
    title: str = "数据图表"
) -> str:
    """
    根据数据文件快速生成图表。

    Args:
        filename (str): 数据文件路径（沙箱路径或文件名）
        x_column (str): X 轴列名
        y_column (str): Y 轴列名
        chart_type (str): 图表类型 - "line"(折线图), "bar"(柱状图), "scatter"(散点图), "pie"(饼图)
        title (str): 图表标题

    Returns:
        str: 图表生成结果，包含 Base64 编码的图片
    """
    # 自动补全路径
    if not filename.startswith('/'):
        filename = f"/home/user/data/{filename}"

    chart_code = f'''
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("{filename}")

# 创建图表
plt.figure(figsize=(10, 6))

chart_type = "{chart_type}"
x_col = "{x_column}"
y_col = "{y_column}"

if chart_type == "line":
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=6)
elif chart_type == "bar":
    plt.bar(df[x_col], df[y_col], color='steelblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
elif chart_type == "scatter":
    plt.scatter(df[x_col], df[y_col], alpha=0.6, edgecolors='black')
elif chart_type == "pie":
    plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%', startangle=90)
    plt.axis('equal')

plt.title("{title}", fontsize=14, fontweight='bold')
if chart_type != "pie":
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"✅ 图表生成成功: {title}")
'''

    return await create_visualization.ainvoke({
        "data_description": f"{title} - {x_column} vs {y_column}",
        "chart_type": chart_type,
        "code": chart_code
    })
