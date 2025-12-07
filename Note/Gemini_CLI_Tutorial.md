# Gemini CLI 综合使用教程 (v2.0)

欢迎使用 Gemini CLI！这是一个强大的人工智能命令行助手，旨在帮助您安全、高效地完成各种软件工程任务。本教程将为您提供全面、详细的指导，帮助您从入门到精通。

---

## 目录

1.  [简介](#1-简介)
    *   [什么是 Gemini CLI？](#什么是-gemini-cli)
    *   [核心设计理念](#核心设计理念)
2.  [核心概念](#2-核心概念)
    *   [交互式对话](#交互式对话)
    *   [工作区与上下文](#工作区与上下文)
    *   [安全确认机制](#安全确认机制)
3.  [工具命令详解](#3-工具命令详解)
    *   [3.1 文件与目录操作](#31-文件与目录操作)
        *   [`list_directory`](#list_directory)
        *   [`glob`](#glob)
        *   [`read_file`](#read_file)
        *   [`write_file`](#write_file)
        *   [`replace`](#replace)
    *   [3.2 搜索与分析](#32-搜索与分析)
        *   [`search_file_content`](#search_file_content)
        *   [`codebase_investigator`](#codebase_investigator)
    *   [3.3 命令执行](#33-命令执行)
        *   [`run_shell_command`](#run_shell_command)
    *   [3.4 信息获取](#34-信息获取)
        *   [`google_web_search`](#google_web_search)
        *   [`web_fetch`](#web_fetch)
    *   [3.5 任务与记忆](#35-任务与记忆)
        *   [`write_todos`](#write_todos)
        *   [`save_memory`](#save_memory)
4.  [最佳实践与技巧](#4-最佳实践与技巧)
5.  [实战示例：修复一个 Bug](#5-实战示例修复一个-bug)

---

## 1. 简介

### 什么是 Gemini CLI？

Gemini CLI 是一个交互式的命令行代理，专门用于处理软件工程任务。它将大型语言模型 (LLM) 的强大能力与一套用于与您的文件系统、代码库和系统环境交互的工具相结合。您可以像与一位初级软件工程师结对编程一样，通过自然语言下达指令，让它帮您完成编码、测试、重构、文档编写等工作。

### 核心设计理念

-   **安全高效:** 所有危险或修改性的操作（如执行命令、修改文件）都会在执行前向您请求确认。
-   **遵循惯例:** Gemini CLI 会努力分析并遵循您项目现有的代码风格、框架和约定。
-   **测试驱动:** 在添加新功能或修复 Bug 时，它会主动编写或运行测试来验证其工作的正确性。
-   **清晰明确:** 避免做出超出请求范围的重大操作，对于模糊指令会主动寻求您的澄清。

---

## 2. 核心概念

### 交互式对话

您可以像聊天一样与 Gemini CLI 互动。它会记住对话的上下文，并根据您的指令执行操作。

### 工作区与上下文

Gemini CLI 在您启动它的项目目录（工作区）中运行。它对当前目录结构、操作系统等环境信息有感知。这些信息会在每次交互开始时显示，以确保您和它对当前状态有一致的理解。

### 安全确认机制

-   **命令解释:** 在执行任何可能修改文件系统或系统状态的 `run_shell_command` 命令之前，Gemini CLI **必须** 解释该命令的用途和潜在影响。
-   **用户确认:** 您将有机会在工具调用实际执行前批准或取消它。这是保障系统安全的核心机制。

---

## 3. 工具命令详解

Gemini CLI 的核心能力来源于其内置的工具集。以下是所有可用工具的详细说明。

### 3.1 文件与目录操作

#### `list_directory`
-   **功能:** 列出指定目录下的文件和子目录。
-   **使用场景:** 当您想快速了解一个文件夹的内部结构时。
-   **示例:**
    > "看看 `My-Chat-LangChain/backend/tools` 目录下有什么"
    ```tool_code
    list_directory(dir_path='My-Chat-LangChain/backend/tools')
    ```

#### `glob`
-   **功能:** 使用 glob 模式（例如 `*.py`, `**/*.md`）高效地查找文件。`**`代表递归匹配所有子目录。
-   **使用场景:** 当您需要根据文件名或路径模式查找一组文件时，这比 `list_directory` 更强大。
-   **示例:**
    > "帮我找到项目里所有的 `requirements.txt` 文件"
    ```tool_code
    glob(pattern='**/requirements.txt')
    ```
    > "列出 `tests` 目录下所有的 Python 测试文件"
    ```tool_code
    glob(pattern='tests/**/test_*.py')
    ```

#### `read_file`
-   **功能:** 读取并返回指定文件的内容。对于大文件，内容会被截断，并提示如何读取更多。
-   **使用场景:** 查看代码、配置文件、文档等任何文本文件的内容。
-   **示例:**
    > "请阅读 `README.md` 的内容"
    ```tool_code
    read_file(file_path='README.md')
    ```

#### `write_file`
-   **功能:** 将指定内容写入文件。如果文件已存在，它将被**完全覆盖**；如果文件不存在，则会创建新文件。
-   **使用场景:** 创建新代码文件、配置文件，或者用全新内容替换旧文件。
-   **示例:**
    > "创建一个名为 `config.json` 的新文件，内容为 `{\"theme\": \"dark\"}`"
    ```tool_code
    write_file(file_path='config.json', content='{"theme": "dark"}')
    ```

#### `replace`
-   **功能:** 在文件中进行精确的文本替换。这是最强大但也需要最谨慎使用的工具之一。
-   **核心要求:**
    1.  **`old_string` 必须精确匹配:** 它必须是包含上下文（通常是前后几行代码）、缩进、换行和空格的**字面量**。
    2.  **`new_string`** 是您想替换成的确切内容。
    3.  **`instruction`** 需要清晰地描述修改意图。
-   **使用场景:** 修改代码、修正拼写错误、更新配置文件中的值。
-   **正确流程:**
    1.  **先 `read_file`:** 读取文件以获取要修改部分的确切上下文。
    2.  **再 `replace`:** 基于读取到的内容，给出包含足够上下文的修改指令。
-   **示例:**
    > (在阅读了 `app.py` 后) "在 `app.py` 中，我想把 `st.title(\"Old Title\")` 改为 `st.title(\"New Shiny Title\")`"
    ```tool_code
    replace(
      file_path='app.py',
      instruction="Update the application title in the Streamlit app.",
      old_string='''
    # ... some line before
    st.title("Old Title")
    # ... some line after
    ''',
      new_string='''
    # ... some line before
    st.title("New Shiny Title")
    # ... some line after
    '''
    )
    ```

### 3.2 搜索与分析

#### `search_file_content`
-   **功能:** 在整个项目（或指定目录/文件）中进行快速、区分大小写的文本/正则表达式搜索。
-   **使用场景:** 定位函数定义、查找特定变量的使用位置、搜索错误信息等。
-   **示例:**
    > "在整个项目中搜索 'get_agent_executor' 这个函数是在哪里定义的？"
    ```tool_code
    search_file_content(pattern='def get_agent_executor')
    ```
    > "查找所有使用了 `os.getenv` 的地方"
    ```tool_code
    search_file_content(pattern='os.getenv', case_sensitive=True)
    ```

#### `codebase_investigator`
-   **功能:** 对代码库进行深度分析，以理解架构、依赖关系和关键逻辑。它返回一份结构化的报告，包含关键文件、符号和架构洞见。
-   **使用场景:** 
    *   刚接触一个新项目，需要快速理解其工作原理。
    *   需要在一个复杂的系统中进行重构或添加功能。
    *   排查一个涉及多个模块、原因不明的 Bug。
-   **示例:**
    > "我需要理解这个 `My-Chat-LangChain` 应用的认证流程是如何工作的。请帮我深入分析一下。"
    ```tool_code
    codebase_investigator(objective="Understand the authentication flow in the 'My-Chat-LangChain' application.")
    ```

### 3.3 命令执行

#### `run_shell_command`
-   **功能:** 执行任意 shell 命令。这是与您的系统环境（如包管理器、构建工具、版本控制）交互的主要方式。
-   **安全提示:** 在执行任何可能产生副作用（如安装、删除、修改）的命令前，Gemini CLI 都会向您解释命令的意图并请求批准。
-   **示例:**
    > "检查一下 Python 版本"
    ```tool_code
    run_shell_command(command='python --version')
    ```
    > "在 `My-Chat-LangChain` 目录下运行 `pip install -r requirements.txt` 来安装依赖"
    ```tool_code
    run_shell_command(command='pip install -r requirements.txt', dir_path='My-Chat-LangChain')
    ```
    > "运行 `git status`"
    ```tool_code
    run_shell_command(command='git status')
    ```

### 3.4 信息获取

#### `google_web_search`
-   **功能:** 使用 Google 搜索获取外部信息。
-   **使用场景:** 查找第三方库的文档、搜索解决方案来修复错误、了解最新的技术趋势。
-   **示例:**
    > "搜索一下 'fastapi middleware cors' 的用法"
    ```tool_code
    google_web_search(query='fastapi middleware cors usage')
    ```

#### `web_fetch`
-   **功能:** 直接从一个或多个 URL 提取内容，并让模型处理这些内容（如总结、提取信息）。
-   **使用场景:** 
    *   总结一篇网络文章。
    *   从 API 端点获取数据并进行分析。
    *   比较多个在线文档页面的信息。
-   **示例:**
    > "请帮我总结这篇 LangChain 的博文: https://blog.langchain.dev/new-in-langchain/"
    ```tool_code
    web_fetch(prompt='Please summarize the content of the URL https://blog.langchain.dev/new-in-langchain/')
    ```

### 3.5 任务与记忆

#### `write_todos`
-   **功能:** 创建和管理一个任务清单，以跟踪复杂任务的进度。
-   **使用场景:** 当一个请求需要多个步骤才能完成时（例如“为项目添加 Docker 支持”），此工具用于规划和展示进度。
-   **示例:**
    > "我们来给这个项目添加 Docker 支持。请制定一个计划。"
    (Gemini CLI 会调用 `write_todos` 创建一个包含多个步骤的计划)
    ```tool_code
    write_todos(todos=[
      {'description': 'Create a Dockerfile', 'status': 'pending'},
      {'description': 'Create a .dockerignore file', 'status': 'pending'},
      {'description': 'Build the Docker image and test it', 'status': 'pending'},
      {'description': 'Add documentation for the Docker setup', 'status': 'pending'}
    ])
    ```
    在执行过程中，它会不断更新这个列表的状态（`pending`, `in_progress`, `completed`）。

#### `save_memory`
-   **功能:** 让 Gemini CLI 长期记住一个特定的事实或偏好。
-   **使用场景:** 保存您的个人偏好、项目特定的命令或配置，以便在未来的对话中直接使用。
-   **注意:** 不应用于保存机密信息。
-   **示例:**
    > "我习惯的 Python 格式化工具是 `black`。请记住。"
    ```tool_code
    save_memory(fact="The user's preferred Python formatter is 'black'.")
    ```
    > "请记住，这个项目的后端服务在 8080 端口启动。"
    ```tool_code
    save_memory(fact="The project's backend service starts on port 8080.")
    ```

---

## 4. 最佳实践与技巧

-   **指令要明确具体:** 避免模糊的指令。提供文件名、函数名等具体信息。
-   **小步快跑:** 将大任务分解成小步骤。这使得每一步都更容易验证，也更容易从错误中恢复。
-   **先侦察，后行动:** 在修改代码前，先使用 `read_file` 和 `search_file_content` 充分理解上下文。
-   **信任但要验证:** 在 Gemini CLI 完成修改后，要求它运行测试或再次读取文件内容，以确认修改符合预期。
-   **善用 `/help`:** 如果不确定某个功能，可以输入 `/help` 查看可用的工具和信息。

---

## 5. 实战示例：修复一个 Bug

**场景:** 假设我们的应用在处理没有用户名的用户时会崩溃。

**您的指令流可能如下:**

1.  **描述问题:**
    > "应用在处理匿名用户时会崩溃。似乎是在 `render_user_profile` 函数里，当 `user.name` 不存在时引发了错误。"
2.  **定位代码 (使用 `search_file_content`):**
    > "请帮我找到 `render_user_profile` 函数的定义。"
3.  **分析代码 (使用 `read_file`):**
    > "好的，现在把这个函数所在文件的内容展示给我。"
4.  **下达修复指令 (触发 `replace`):**
    > "我看到问题了。请在这个函数开头添加一个检查，如果 `user` 对象或者 `user.name` 为空，就显示 'Guest'，而不是尝试访问 `user.name`。"
5.  **验证修复 (再次 `read_file`):**
    > "修改完成了吗？请再次显示文件内容让我确认一下。"
6.  **运行测试 (使用 `run_shell_command`):**
    > "很好。现在请运行相关的单元测试来确保没有引入新的问题。"

---

希望这份详尽的教程能帮助您充分利用 Gemini CLI 的强大功能。祝您使用愉快！