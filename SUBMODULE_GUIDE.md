# Ego-Video-to-SIM Submodule 使用指南

## 仓库结构

```
Ego-Video-to-SIM/                          ← 主仓库
├── combination/                            ← 子模块: 完整仿真管线
├── ReplicateAnyScene/                      ← 子模块: 场景重建 (ananansmall/Ego-centric-Video-to-Simulation)
├── HaWoR/                                  ← 子模块: 手部重建 (ananansmall/HaWoR, forked from ThunderVVV/HaWoR)
├── pv_retargeting/                         ← PV 重定向
├── libs/                                   ← 本地依赖库
├── .gitmodules                             ← 子模块配置
├── SUBMODULE_GUIDE.md                      ← 本文档
└── README.md
```

## 一、克隆仓库（含子模块）

### 首次克隆

```bash
# 一步到位，克隆时自动拉取所有子模块
git clone --recurse-submodules git@github.com:ananansmall/Ego-Video-to-SIM.git

# 或者先克隆，再拉取子模块
git clone git@github.com:ananansmall/Ego-Video-to-SIM.git
cd Ego-Video-to-SIM
git submodule update --init --recursive
```

### 已有仓库但子模块为空

```bash
cd Ego-Video-to-SIM
git submodule update --init --recursive
```

## 二、日常修改与推送

### 2.1 修改 ReplicateAnyScene 代码

ReplicateAnyScene 指向你自己的仓库 `ananansmall/Ego-centric-Video-to-Simulation`，可以自由修改和推送。

```bash
# 在你本地的工作目录修改代码
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/
vim src/some_file.py

# 提交并推送到你自己的远程仓库
git add src/some_file.py
git commit -m "Fix: 修复xxx问题"
git push origin main
```

**推送后，还需要更新主仓库的子模块引用**（否则主仓库还指向旧版本）：

```bash
cd /tmp/Ego-Video-to-SIM/    # 或你克隆的主仓库目录
cd ReplicateAnyScene/
git pull origin main
cd ..
git add ReplicateAnyScene
git commit -m "Update ReplicateAnyScene: 修复xxx问题"
git push origin main
```

### 2.2 修改 HaWoR 代码

HaWoR 指向你自己的 fork `ananansmall/HaWoR`，同样可以自由修改和推送。

```bash
# 在你本地的工作目录修改代码
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/
vim demov2.py

# 提交并推送到你的 fork
git add demov2.py
git commit -m "Enhance demov2 visualization"
git push myfork main

# 同样需要更新主仓库的子模块引用
cd /tmp/Ego-Video-to-SIM/
cd HaWoR/
git pull origin main
cd ..
git add HaWoR
git commit -m "Update HaWoR: enhance demov2"
git push origin main
```

### 2.3 一键更新所有子模块引用

如果你同时修改了多个子模块，可以用这个快捷方式：

```bash
cd /tmp/Ego-Video-to-SIM/

# 拉取所有子模块的最新代码
git submodule update --remote

# 提交所有变更的子模块引用
git add ReplicateAnyScene HaWoR
git commit -m "Update submodules to latest"
git push origin main
```

## 三、完整工作流示例

### 场景1: 修改 ReplicateAnyScene 代码并同步到主仓库

```bash
# Step 1: 在本地工作目录修改代码
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/
vim src/some_file.py

# Step 2: 在子模块内提交并推送
git add src/some_file.py
git commit -m "Fix: 修复xxx问题"
git push origin main

# Step 3: 更新主仓库的子模块引用
cd /tmp/Ego-Video-to-SIM/
cd ReplicateAnyScene/ && git pull origin main && cd ..
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
# ReplicateAnyScene 模型 - 需要单独下载 SAM3、VGGT 等模型到 ReplicateAnyScene/models/ 目录
# HaWoR 模型 - 需要单独下载权重到 HaWoR/weights/ 目录
# 详见各子模块的 README

# Step 3: 安装依赖
pip install -r ReplicateAnyScene/requirements.txt   # 场景重建依赖
pip install -r HaWoR/requirements.txt               # 手部重建依赖
```

### 场景3: 添加新的子模块

```bash
cd /tmp/Ego-Video-to-SIM/
git submodule add <仓库URL> <目录名>
git commit -m "Add <目录名> as submodule"
git push origin main
```

### 场景4: 删除子模块

```bash
cd /tmp/Ego-Video-to-SIM/
git submodule deinit -f <子模块路径>
git rm -f <子模块路径>
rm -rf .git/modules/<子模块路径>
git commit -m "Remove <子模块名> submodule"
git push origin main
```

## 四、常见问题

### Q1: 子模块显示为 dirty 状态（有未提交的修改）

```bash
cd <子模块路径>
git status
git stash   # 暂存修改
# 或
git checkout .   # 丢弃修改
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

子模块默认处于 detached HEAD 状态。要切换到分支做修改：

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

### Q6: 推送时提示 "failed to push some refs"

```bash
# 先拉取最新代码再推送
cd <子模块路径>
git pull origin main
git push origin main
```

## 五、大文件处理说明

子模块仓库中排除了以下大文件/目录（通过 .gitignore）：

### ReplicateAnyScene 排除项

| 目录/类型 | 大小 | 说明 | 如何获取 |
|---|---|---|---|
| `models/` | 16G | SAM3、VGGT 等模型权重 | `hf download facebook/VGGT-1B --local-dir models/VGGT` 等 |
| `output_v2/` | 2.3G | 输出数据 | 运行 mainv2.py 生成 |
| `assets/` | 3.3G | 输入视频等资源 | 自行准备 |
| `*.pt, *.ckpt, *.safetensors` | - | 模型权重文件 | 下载或训练生成 |
| `*.ply, *.glb` | - | 3D 数据文件 | 运行管线生成 |

### HaWoR 排除项

| 目录/类型 | 大小 | 说明 | 如何获取 |
|---|---|---|---|
| `weights/` | 3.5G | 模型权重 | `wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/...` |
| `thirdparty/` | 1.7G | 第三方依赖（DROID-SLAM等） | `git clone --recursive` |
| `output/` | - | 输出数据 | 运行 demov2.py 生成 |
| `example/7/`, `example/beizi/` | 605M | 示例输出数据 | 运行 demov2.py 生成 |

## 六、调用关系

```
输入视频
    │
    ├──→ HaWoR/ (demov2.py)           ← 手部重建
    │     ├── 手部 3D 顶点 (MANO)
    │     ├── 相机位姿 (SLAM)
    │     ├── 手部 MANO 参数 (pred_trans, pred_rot, pred_hand_pose, pred_betas)
    │     └── 输出: reconstruction/hawor_results_*.npz
    │
    ├──→ ReplicateAnyScene/ (mainv2.py) ← 场景重建
    │     ├── 3D 场景 (GLB)
    │     ├── 物体实例
    │     ├── 物体位姿与关系
    │     └── 输出: final_scene.glb, all_instances.pkl
    │
    └──→ combination/                  ← 融合管线
          ├── 场景对齐 (HaWoR 手部 + RAS 场景)
          ├── SAPIEN 物理仿真
          └── R1 机器人仿真
```

## 七、本地工作目录 vs GitHub 克隆目录

你通常在两个地方操作：

| 位置 | 路径 | 用途 |
|---|---|---|
| **本地工作目录** | `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/` | 日常开发、修改代码、运行程序 |
| **GitHub 克隆目录** | `/tmp/Ego-Video-to-SIM/` | 更新子模块引用、推送到 GitHub |

**工作流程**：
1. 在本地工作目录修改代码 → `git push origin main`（ReplicateAnyScene）或 `git push myfork main`（HaWoR）
2. 在 GitHub 克隆目录更新子模块引用 → `git pull` + `git add` + `git commit` + `git push`
