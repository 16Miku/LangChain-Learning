# GitHub Markdown 视频嵌入问题解决方案

## 问题背景

在为 `My-Chat-LangChain` 项目编写文档 (`Note-V7.md`) 时，需要在 GitHub 仓库的 Markdown 文件中展示演示视频。尝试了以下方法均未成功：

```markdown
<!-- 方法1: 直接粘贴视频 URL -->
https://raw.githubusercontent.com/16Miku/LangChain-Learning/master/My-Chat-LangChain/media/search_asuka.mp4

<!-- 方法2: 使用 HTML video 标签 -->
<video controls src="media/search_asuka.mp4" title="Title"></video>

<!-- 方法3: 使用图片链接到 raw URL -->
[![Watch](封面图.png)](https://raw.githubusercontent.com/.../search_asuka.mp4)
```

---

## 原因分析

GitHub 的 Markdown 渲染器出于安全考虑，**不支持以下方式嵌入视频**：

1.  **`<video>` HTML 标签**: 会被 GitHub 的 HTML 过滤器移除。
2.  **直接链接到 `.mp4` 文件**: GitHub 不会将其转换为可播放的播放器，只会显示为普通链接。
3.  **`raw.githubusercontent.com` 链接**: 该域名用于提供原始文件下载，浏览器会尝试下载文件而非播放。

**GitHub 官方支持的视频嵌入方式：**
*   仅限在 **Issue**, **Pull Request** 或 **Discussion** 的评论区，通过拖拽上传视频文件。上传后，GitHub 会托管该视频并生成可播放链接。
*   在 **README.md** 或其他普通 Markdown 文件中，**不支持**视频内嵌播放。

---

## 解决方案

### 方案一：将视频转换为 GIF 动图 (推荐)

GIF 格式可以在 GitHub Markdown 中直接使用图片语法嵌入并自动播放。

**步骤:**

1.  **安装 `moviepy` 库** (Python):
    ```bash
    pip install moviepy
    ```
    `moviepy` 会自动下载并集成 `imageio-ffmpeg`，无需手动安装 FFmpeg。

2.  **执行转换命令**:
    ```python
    # 使用 Python 单行命令执行
    python -c "
    from moviepy.video.io.VideoFileClip import VideoFileClip
    clip = VideoFileClip('path/to/your_video.mp4')
    clip.resized(width=640).write_gif('path/to/output.gif', fps=10)
    clip.close()
    print('GIF created successfully!')
    "
    ```

    **参数说明:**
    *   `resized(width=640)`: 缩放宽度至 640 像素，高度按比例自动调整。可根据需要调整，或使用 `resized(height=...)` 按高度缩放。
    *   `fps=10`: 设置 GIF 帧率为 10fps。降低帧率可显著减小文件体积。

3.  **在 Markdown 中嵌入 GIF**:
    ```markdown
    ![演示动图描述](media/output.gif)
    ```

**优点:**
*   GIF 可在 GitHub 页面上自动循环播放。
*   兼容性极佳，所有浏览器和平台都支持。

**缺点:**
*   GIF 文件体积通常远大于 MP4。
*   GIF 不支持音频。
*   对于较长的视频，GIF 体积可能过大，不适合直接嵌入。

---

### 方案二：上传到视频托管平台

将视频上传到 YouTube、Bilibili 等平台，然后在 Markdown 中使用"图片链接"的方式展示封面，点击后跳转到视频页面。

**Markdown 格式:**
```markdown
[![视频封面描述](封面图URL)](视频页面URL)
```

**示例:**
```markdown
[![点击观看演示](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
```

**优点:**
*   不受文件大小限制。
*   支持音频和更长的视频内容。

**缺点:**
*   用户需要跳转到外部网站观看。
*   需要额外管理视频托管账号。

---

## 本次开发总结

| 步骤 | 描述 |
| :--- | :--- |
| 1 | 分析 GitHub Markdown 不支持 `<video>` 标签的原因 |
| 2 | 确定使用 GIF 替代方案 |
| 3 | 安装 `moviepy` Python 库 |
| 4 | 将 `search_asuka.mp4` 转换为 `search_asuka.gif` |
| 5 | 将 `search_teresa.mp4` 转换为 `search_teresa.gif` |
| 6 | 更新 `My-Chat-LangChain/Note-V7.md`，用 `![](gif)` 语法替代 `<video>` 标签 |
| 7 | 提交更改到 GitHub |

**转换命令示例:**
```bash
# 转换 search_asuka.mp4
python -c "from moviepy.video.io.VideoFileClip import VideoFileClip; clip = VideoFileClip('My-Chat-LangChain/media/search_asuka.mp4'); clip.resized(width=640).write_gif('My-Chat-LangChain/media/search_asuka.gif', fps=10); clip.close(); print('search_asuka.gif created')"

# 转换 search_teresa.mp4
python -c "from moviepy.video.io.VideoFileClip import VideoFileClip; clip = VideoFileClip('My-Chat-LangChain/media/search_teresa.mp4'); clip.resized(width=640).write_gif('My-Chat-LangChain/media/search_teresa.gif', fps=10); clip.close(); print('search_teresa.gif created')"
```

**最终 Markdown 代码:**
```markdown
## 6. 运行演示

### 搜索 斋藤飞鸟 演示
![搜索 斋藤飞鸟 演示](media/search_asuka.gif)

### 搜索 池田瑛纱 演示
![搜索 池田瑛纱 演示](media/search_teresa.gif)
```

---

## 参考资料

*   [GitHub Docs - Attaching files](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)
*   [MoviePy Documentation](https://zulko.github.io/moviepy/)
