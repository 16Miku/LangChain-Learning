# 仓库拆分操作记录

> **操作日期**: 2025-12-30
> **操作目的**: 将 `My-Chat-LangChain` 从 `LangChain-Learning` 仓库中拆分为独立仓库

---

## 一、背景

原项目 `My-Chat-LangChain` 位于 `LangChain-Learning` 仓库的子目录中。随着项目发展到 V8.0 版本，需要将其拆分为独立仓库进行单独开发和维护。

**原仓库结构**:
```
LangChain-Learning/
├── My-Chat-LangChain/    ← 需要拆分的项目
├── other-projects/
└── ...
```

**目标**:
- 将 `My-Chat-LangChain` 拆分为独立 Git 仓库
- 保留与该目录相关的所有 commit 历史
- 推送到新的 GitHub 仓库

---

## 二、方案选择

| 方案 | 保留历史 | 复杂度 | 说明 |
|------|----------|--------|------|
| 直接复制 | ❌ | 低 | 简单但丢失历史 |
| git subtree split | ✅ | 中 | Git 内置命令 |
| **git filter-repo** | ✅ | 中 | 更现代、更快 ✔️ 选用 |

**选择 `git filter-repo`** 的原因:
1. 比 `git filter-branch` 快 10-100 倍
2. 语法简洁，操作安全
3. 完整保留相关 commit 历史

---

## 三、操作步骤

### 3.1 环境准备

```powershell
# 激活 conda 环境
conda activate My-Chat-LangChain

# 安装 git-filter-repo
pip install git-filter-repo
```

### 3.2 克隆原仓库

```powershell
# 切换到工作目录
cd A:\study\AI\LLM

# 克隆原仓库（避免修改原仓库）
git clone https://github.com/16Miku/LangChain-Learning.git
```

输出:
```
Cloning into 'LangChain-Learning'...
remote: Enumerating objects: 553, done.
remote: Total 553 (delta 47), reused 70 (delta 23), pack-reused 445 (from 1)
Receiving objects: 100% (553/553), 43.76 MiB | 3.48 MiB/s, done.
Resolving deltas: 100% (251/251), done.
```

### 3.3 执行拆分

```powershell
# 进入仓库目录
cd LangChain-Learning

# 使用 filter-repo 只保留 My-Chat-LangChain 子目录
git filter-repo --subdirectory-filter My-Chat-LangChain
```

输出:
```
NOTICE: Removing 'origin' remote; see 'Why is my origin removed?'
        in the manual if you want to push back there.
        (was https://github.com/16Miku/LangChain-Learning.git)
Parsed 81 commits
New history written in 1.04 seconds; now repacking/cleaning...
Repacking your repo and cleaning out old unneeded objects
HEAD is now at f1eb381 修正My-Chat-LangChain各个说明文档的图片路径
Enumerating objects: 357, done.
Counting objects: 100% (357/357), done.
Delta compression using up to 16 threads
Compressing objects: 100% (209/209), done.
Writing objects: 100% (357/357), done.
Total 357 (delta 144), reused 300 (delta 144), pack-reused 0 (from 0)
Completely finished after 2.99 seconds.
```

### 3.4 验证结果

```powershell
# 查看目录结构（应该是 My-Chat-LangChain 的内容直接在根目录）
ls

# 查看 commit 历史（应该只保留与 My-Chat-LangChain 相关的 81 个 commits）
git log --oneline
```

### 3.5 推送到新 GitHub 仓库

```powershell
# 1. 在 GitHub 上创建新仓库: https://github.com/new
#    仓库名建议: My-Chat-LangChain

# 2. 添加远程仓库
git remote add origin https://github.com/你的用户名/My-Chat-LangChain.git

# 3. 推送
git branch -M main
git push -u origin main
```

---

## 四、拆分结果

| 项目 | 拆分前 | 拆分后 |
|------|--------|--------|
| 仓库位置 | `LangChain-Learning/My-Chat-LangChain/` | 独立仓库根目录 |
| Commit 数量 | 553 (整个仓库) | 81 (仅相关历史) |
| 仓库大小 | 43.76 MB | 约 10 MB |
| 对象数量 | - | 357 |

**目录结构变化**:

```
# 拆分前
LangChain-Learning/
└── My-Chat-LangChain/
    ├── backend/
    ├── frontend/
    ├── Note/
    └── ...

# 拆分后 (新仓库)
My-Chat-LangChain/  (仓库根目录)
├── backend/
├── frontend/
├── Note/
└── ...
```

---

## 五、注意事项

### 5.1 关于 origin 被移除

`git filter-repo` 会自动移除 `origin` 远程仓库，这是故意设计的安全机制，防止意外推送到原仓库。

### 5.2 后续维护

1. **更新 .gitignore**: 确保包含必要的忽略规则
2. **更新 README.md**: 如有需要，更新项目描述和徽章链接
3. **配置 GitHub**: 设置 branch protection、secrets 等

### 5.3 原仓库处理

原 `LangChain-Learning` 仓库中的 `My-Chat-LangChain` 目录可以：
- 保留作为历史归档
- 删除并添加指向新仓库的链接
- 使用 git submodule 引用新仓库

---

## 六、相关命令参考

```bash
# 查看 filter-repo 帮助
git filter-repo --help

# 其他常用 filter-repo 操作
git filter-repo --path <dir>           # 只保留指定路径
git filter-repo --invert-paths --path <dir>  # 排除指定路径
git filter-repo --message-callback '...'      # 修改 commit message
```

---

## 七、参考链接

- [git-filter-repo GitHub](https://github.com/newren/git-filter-repo)
- [git-filter-repo 手册](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
- [GitHub: 拆分子目录为新仓库](https://docs.github.com/en/get-started/using-git/splitting-a-subfolder-out-into-a-new-repository)

---

> **文档版本**: 1.0
> **创建日期**: 2025-12-30



