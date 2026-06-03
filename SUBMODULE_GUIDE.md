# Ego-Video-to-SIM Submodule 使用指南

## 仓库结构

```
Ego-Video-to-SIM/              ← 主仓库
├── combination/                ← 子模块: 完整仿真管线
├── ReplicateAnyScene/          ← 子模块: 场景重建 (ananansmall/Ego-centric-Video-to-Simulation)
├── HaWoR/                      ← 子模块: 手部重建 (ThunderVVV/HaWoR)
├── pv_retargeting/             ← PV 重定向
├── libs/                       ← 本地依赖库
├── .gitmodules                 ← 子模块配置
└── README.md
```

## 一、克隆仓库（含子模块）

### 首次克隆

```bash
# 方式1: 一步到位，克隆时自动拉取所有子模块
git clone --recurse-submodules git@github.com:ananansmall/Ego-Video-to-SIM.git

# 方式2: 先克隆，再拉取子模块
git clone git@github.com:ananansmall/Ego-Video-to-SIM.git
cd Ego-Video-to-SIM
git submodule update --init --recursive
```

### 已有仓库但子模块为空

```bash
cd Ego-Video-to-SIM
git submodule update --init --recursive
```

## 二、子模块的日常操作

### 2.1 拉取子模块最新更新

```bash
# 拉取所有子模块的最新代码
git submodule update --remote

# 只拉取某个子模块
git submodule update --remote ReplicateAnyScene
git submodule update --remote HaWoR
```

### 2.2 在子模块中修改代码并推送

**关键概念**: 子模块是独立的 git 仓库，修改子模块内的代码需要在子模块目录内操作。

#### 修改 ReplicateAnyScene（你自己的仓库）

```bash
# 进入子模块目录
cd ReplicateAnyScene/

# 正常的 git 操作
git add <修改的文件>
git commit -m "描述你的修改"
git push origin main
```

#### 修改 HaWoR（上游仓库，你无推送权限）

如果你需要修改 HaWoR 的代码，有两种方式：

**方式A: Fork 到自己账号（推荐）**

1. 在 GitHub 上 fork `ThunderVVV/HaWoR` 到 `ananansmall/HaWoR`
2. 修改 `.gitmodules` 中的 URL：
   ```bash
   cd Ego-Video-to-SIM
   git config -f .gitmodules submodule.HaWoR.url git@github.com:ananansmall/HaWoR.git
   git submodule sync
   ```
3. 之后就可以在 HaWoR 子模块中推送自己的修改

**方式B: 本地修改但不推送**

```bash
cd HaWoR/
# 修改代码...
# 本地提交（不会推送到上游）
git add <修改的文件>
git commit -m "本地修改说明"
# 注意: git submodule update --remote 会丢失本地修改
```

### 2.3 更新主仓库中的子模块引用

当子模块有新提交后，主仓库需要更新引用：

```bash
cd Ego-Video-to-SIM/

# 1. 进入子模块拉取最新代码
cd ReplicateAnyScene/
git pull origin main
cd ..

# 2. 主仓库会检测到子模块指向了新的 commit
git add ReplicateAnyScene
git commit -m "Update ReplicateAnyScene to latest commit"
git push origin main
```

## 三、完整工作流示例

### 场景1: 修改 ReplicateAnyScene 代码并同步到主仓库

```bash
# Step 1: 在本地工作目录修改代码
cd /path/to/ReplicateAnyScene/
vim src/some_file.py

# Step 2: 在子模块内提交并推送
cd /path/to/ReplicateAnyScene/
git add src/some_file.py
git commit -m "Fix: 修复xxx问题"
git push origin main

# Step 3: 更新主仓库的子模块引用
cd /path/to/Ego-Video-to-SIM/
cd ReplicateAnyScene/
git pull origin main
cd ..
git add ReplicateAnyScene
git commit -m "Update ReplicateAnyScene: 修复xxx问题"
git push origin main
```

### 场景2: 从零开始在新机器上部署

```bash
# Step 1: 克隆主仓库及子模块
git clone --recurse-submodules git@github.com:ananansmall/Ego-Video-to-SIM.git
cd Ego-Video-to-SIM

# Step 2: 下载大文件（模型权重等，不在 git 中）
# ReplicateAnyScene 模型
# 需要单独下载 SAM3、VGGT 等模型到 ReplicateAnyScene/models/ 目录

# HaWoR 模型
# 需要单独下载权重到 HaWoR/weights/ 目录

# Step 3: 安装依赖
pip install -r requirements.txt  # 各子模块各自的 requirements
```

### 场景3: 添加新的子模块

```bash
cd Ego-Video-to-SIM/
git submodule add <仓库URL> <目录名>
git commit -m "Add <目录名> as submodule"
git push origin main
```

### 场景4: 删除子模块

```bash
cd Ego-Video-to-SIM/
git submodule deinit -f <子模块路径>
git rm -f <子模块路径>
rm -rf .git/modules/<子模块路径>
git commit -m "Remove <子模块名> submodule"
git push origin main
```

## 四、常见问题

### Q1: 子模块显示为 `-commit` 脏状态

```bash
# 子模块内有未提交的修改
cd <子模块路径>
git status
git stash  # 或 git checkout .
```

### Q2: `git submodule update --remote` 后主仓库显示子模块有变更

这是正常的——子模块指向了新的 commit。你需要提交这个变更：

```bash
git add <子模块路径>
git commit -m "Update submodule reference"
```

### Q3: 克隆后子模块目录为空

```bash
git submodule update --init --recursive
```

### Q4: 子模块 detached HEAD 状态

子模块默认处于 detached HEAD 状态。要切换到分支：

```bash
cd <子模块路径>
git checkout main
# 之后再做修改和提交
```

### Q5: 如何查看当前所有子模块的状态

```bash
git submodule status
# 输出格式: <commit-hash> <路径> <描述>
# 前缀含义:
#   空格 = 子模块已注册且已检出
#   -   = 子模块未初始化
#   +   = 子模块指向的 commit 与主仓库记录的不同
```

## 五、ReplicateAnyScene 的实时更新

ReplicateAnyScene 指向你自己的仓库 `ananansmall/Ego-centric-Video-to-Simulation`，所以：

1. **在本地 ReplicateAnyScene 目录修改代码后**，直接 `git push origin main` 即可推送到 GitHub
2. **主仓库 Ego-Video-to-SIM 不会自动更新引用**，需要手动操作：
   ```bash
   cd Ego-Video-to-SIM/ReplicateAnyScene/
   git pull origin main
   cd ..
   git add ReplicateAnyScene
   git commit -m "Update ReplicateAnyScene"
   git push origin main
   ```
3. **其他人克隆主仓库时**，会自动获取到主仓库记录的子模块 commit 版本

### 快速更新脚本

可以在主仓库中创建一个脚本 `update_submodules.sh`：

```bash
#!/bin/bash
echo "Updating all submodules..."
git submodule update --remote

echo "Checking submodule changes..."
git diff --exit-code --quiet
if [ $? -ne 0 ]; then
    echo "Submodules have been updated. Committing..."
    git add -A
    git commit -m "Update submodules to latest"
    git push origin main
    echo "Done!"
else
    echo "No submodule changes detected."
fi
```

## 六、大文件处理说明

子模块仓库中排除了以下大文件/目录（通过 .gitignore）：

### ReplicateAnyScene 排除项

| 目录/类型 | 大小 | 说明 |
|---|---|---|
| `models/` | 16G | SAM3、VGGT 等模型权重 |
| `output_v2/` | 2.3G | 输出数据（图片、深度图、点云、3D场景） |
| `assets/` | 3.3G | 输入视频等资源 |
| `outputs/` | 2.9G | 输出数据 |
| `*.pt, *.ckpt, *.safetensors` | - | 所有模型权重文件 |
| `*.ply, *.glb` | - | 3D 数据文件 |

### HaWoR 排除项（上游仓库已处理）

| 目录/类型 | 大小 | 说明 |
|---|---|---|
| `weights/` | 3.5G | 模型权重 |
| `thirdparty/` | 1.7G | 第三方依赖（DROID-SLAM等） |
| `hot3d/` | 7.3G | HOT3D 数据集相关 |
| `example/` | 619M | 示例视频和输出 |

这些大文件需要单独下载或生成，不在 git 仓库中管理。

## 七、调用关系

```
输入视频
    │
    ├──→ HaWoR/                    ← 手部重建
    │     ├── 手部 3D 关键点
    │     ├── 相机位姿 (SLAM)
    │     └── 手部 MANO 参数
    │
    ├──→ ReplicateAnyScene/        ← 场景重建
    │     ├── 3D 场景 (GLB)
    │     ├── 物体实例
    │     └── 物体位姿
    │
    └──→ combination/              ← 融合管线
          ├── 场景对齐
          ├── SAPIEN 物理仿真
          └── R1 机器人仿真
```
