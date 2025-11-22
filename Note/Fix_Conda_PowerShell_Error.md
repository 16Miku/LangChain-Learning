# 修复 PowerShell 7 中 Conda 激活失败报错 `invalid choice: ''` 的全过程记录

## 1. 问题描述

在 Windows 11 的 PowerShell 7 环境下，尝试使用 `conda activate <env_name>` 激活环境时，出现以下错误：

```powershell
(base) PS A:\path\to\project> conda activate Gemini
usage: conda-script.py [-h] [-v] [--no-plugins] [-V] COMMAND ...
conda-script.py: error: argument COMMAND: invalid choice: '' (choose from 'activate', 'clean', 'commands', 'compare', 'config', 'create', 'deactivate', 'env', 'export', 'info', 'init', 'install', 'list', 'notices', 'package', 'build', 'content-trust', 'convert', 'debug', 'develop', 'doctor', 'index', 'inspect', 'metapackage', 'render', 'repoquery', 'skeleton', 'repo', 'token', 'server', 'pack', 'remove', 'uninstall', 'rename', 'run', 'search', 'update', 'upgrade')
Invoke-Expression: Cannot bind argument to parameter 'Command' because it is an empty string.
```

同时，PowerShell 可能会提示：
```
Invoke-Expression: Cannot bind argument to parameter 'Command' because it is an empty string.
```

这意味着 Conda 接收到了一个空的命令参数，导致解析失败。

## 2. 问题排查

### 2.1 检查 Shell 集成
首先尝试重新初始化 Conda 的 Shell 支持：
```powershell
conda init powershell
conda init pwsh
```
并确保 PowerShell 执行策略允许运行脚本：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
但这并没有解决问题。

### 2.2 调试 Conda Hook
通过查看 Conda 在 PowerShell 中的加载脚本（通常位于 `Anaconda/shell/condabin/Conda.psm1`），发现它定义了一个 `Invoke-Conda` 函数来拦截 `conda` 命令。

Conda 的 PowerShell 钩子（Hook）会尝试调用 `conda.exe` 并传递一些内部环境变量，特别是 `$Env:_CE_M` 和 `$Env:_CE_CONDA`。

我们通过以下命令手动测试了参数传递：
```powershell
& "A:\Anaconda\Scripts\conda.exe" shell.powershell hook > debug_output.txt
```
输出显示这些环境变量在当前 Shell 中是 `$null`（空的）。

### 2.3 根源分析
在 `A:\Anaconda\shell\condabin\Conda.psm1` 文件中，原始代码是这样写的：

```powershell
& $Env:CONDA_EXE $Env:_CE_M $Env:_CE_CONDA $Command @OtherArgs;
```

当 `$Env:_CE_M` 和 `$Env:_CE_CONDA` 为空时，PowerShell 可能会将它们作为空字符串 `""` 传递给 `conda.exe`。
相当于执行了：
```bash
conda.exe "" "" activate Gemini
```
Conda 解析器看到第一个参数是 `""`（空字符串），但这不在它的合法命令列表（activate, install, list 等）中，因此报错 `invalid choice: ''`。

## 3. 解决方案

修改 `Anaconda/shell/condabin/Conda.psm1` 文件，优化参数传递逻辑。我们不再直接传递变量，而是先判断变量是否存在，只有非空时才加入参数列表。

### 修改前（示例片段）：
```powershell
function Invoke-Conda() {
    # ...
    if ($Args.Count -eq 0) {
        & $Env:CONDA_EXE $Env:_CE_M $Env:_CE_CONDA;
    }
    else {
        # ...
        switch ($Command) {
            "activate" {
                Enter-CondaEnvironment @OtherArgs;
            }
            "deactivate" {
                Exit-CondaEnvironment;
            }
            default {
                & $Env:CONDA_EXE $Env:_CE_M $Env:_CE_CONDA $Command @OtherArgs;
            }
        }
    }
}
```

### 修改后：
我们需要构建一个动态的参数数组 `$condaArgs`。

```powershell
function Invoke-Conda() {
    # 构建基础参数列表
    $condaArgs = @()
    if ($Env:_CE_M) { $condaArgs += $Env:_CE_M }
    if ($Env:_CE_CONDA) { $condaArgs += $Env:_CE_CONDA }

    if ($Args.Count -eq 0) {
        & $Env:CONDA_EXE $condaArgs;
    }
    else {
        $Command = $Args[0];
        if ($Args.Count -ge 2) {
            $OtherArgs = $Args[1..($Args.Count - 1)];
        } else {
            $OtherArgs = @();
        }
        switch ($Command) {
            "activate" {
                Enter-CondaEnvironment @OtherArgs;
            }
            "deactivate" {
                Exit-CondaEnvironment;
            }
            default {
                # 将命令和后续参数加入列表
                $condaArgs += $Command
                $condaArgs += $OtherArgs
                & $Env:CONDA_EXE $condaArgs;
            }
        }
    }
}
```

**注意：** 需要对文件中的 `Get-CondaEnvironment`、`Enter-CondaEnvironment`、`Exit-CondaEnvironment` 和 `Invoke-Conda` 四个函数都做类似的修改，将直接插值的变量改为动态数组构建。

## 4. 结果验证

保存文件后，**重启 PowerShell 终端**。再次运行：
```powershell
conda activate Gemini
```
环境成功激活，不再报错。

---
*记录时间：2025年11月23日*
