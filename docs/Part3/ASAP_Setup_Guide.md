# ASAP 环境配置与Demo运行指南

> **ASAP**: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills
>
> 论文来源：RSS 2025 https://arxiv.org/abs/2502.01143 ｜ 仓库地址：https://github.com/LeCAR-Lab/ASAP ｜ 项目主页：https://agile.human2humanoid.com
>
> Edited by LKM from SSTIA，适用于 Ubuntu 24.04 + NVIDIA GPU 环境

---

> 💡 **遇到问题怎么办？**
>
> 环境配置过程中遇到报错是非常正常的事情，不同电脑的硬件、驱动、系统版本都可能带来差异，当你遇到问题时：
>
> 1. **先看本指南的 [附录 C：常见问题排查](#c-常见问题排查)**，其中记录了我实际遇到过的各种坑
> 2. **善用 LLM**（ChatGPT、Deepseek等）——把完整的报错信息贴给它，通常能快速定位原因
> 3. **在 GitHub、CSDN等平台搜索报错关键词**，大概率有人遇到过相同问题
> 4. 如果以上都无法解决，欢迎在本仓库的 [Discussions](https://github.com/UMJI-SSTIA/Deep-Learning-Workshop/discussions) 中提问

---

## 目录

- [背景知识：这些工具都是什么？](#背景知识这些工具都是什么)
- [方案一：从零开始配置](#方案一从零开始配置)
  - [0. 前置条件与显卡兼容性](#0-前置条件与显卡兼容性)
  - [1. 安装 NVIDIA 驱动](#1-安装-nvidia-驱动)
  - [2. 安装 CUDA Toolkit 12.8](#2-安装-cuda-toolkit-128)
  - [3. 安装 Miniconda 并创建环境](#3-安装-miniconda-并创建环境)
  - [4. 安装 PyTorch](#4-安装-pytorch)
  - [5. 克隆并安装 ASAP](#5-克隆并安装-asap)
  - [6. 安装 ROS2 Humble](#6-安装-ros2-humble)
  - [7. 安装 Unitree SDK](#7-安装-unitree-sdk)
  - [8. 安装 sim2real 剩余依赖](#8-安装-sim2real-剩余依赖)
  - [9. 运行 Demo](#9-运行-demo)
- [方案二：克隆预配置环境](#方案二克隆预配置环境)
  - [适用条件](#适用条件)
  - [配置步骤](#配置步骤)
- [进阶：本地训练](#进阶本地训练)
- [附录](#附录)
  - [A. 键盘控制说明](#a-键盘控制说明)
  - [B. 预训练动作列表](#b-预训练动作列表)
  - [C. 常见问题排查](#c-常见问题排查)

---

## 背景知识：这些工具都是什么？

在开始安装之前，先简单了解一下我们会用到的各种工具和库，方便你理解每一步在做什么：

**系统层**

| 工具 | 作用 |
|------|------|
| **NVIDIA 驱动** | 让操作系统能识别和使用你的 NVIDIA 显卡，是一切 GPU 计算的基础 |
| **CUDA Toolkit** | NVIDIA 的 GPU 并行计算平台，深度学习框架（如 PyTorch）依赖它来在 GPU 上运行计算 |
| **Miniconda** | Python 环境管理工具，可以创建隔离的虚拟环境，避免不同项目之间的依赖冲突 |

**核心框架**

| 工具 | 作用 |
|------|------|
| **PyTorch** | 主流深度学习框架，ASAP 用它来训练和运行强化学习策略 |
| **Genesis** | 轻量级物理仿真引擎，ASAP 用它模拟机器人在虚拟世界中的运动（替代原版教程中的 Isaac Gym） |
| **MuJoCo** | 另一个物理仿真引擎，ASAP 在 sim2sim 可视化 demo 中用它来渲染机器人 |

**通信与硬件**

| 工具 | 作用 |
|------|------|
| **ROS2 Humble** | 机器人操作系统，提供进程间通信机制。ASAP 的仿真环境和策略控制器通过 ROS2 话题通信 |
| **Unitree SDK** | Unitree（宇树）机器人的 Python SDK，用于与 G1 人形机器人通信 |
| **ONNX Runtime** | 模型推理引擎，将训练好的策略模型高效地部署运行 |

**ASAP 本体**

| 模块 | 作用 |
|------|------|
| **humanoidverse/** | 训练框架，包含环境定义、奖励函数、机器人模型等 |
| **sim2real/** | 部署模块，把训练好的策略在 MuJoCo 仿真（sim2sim）或真实机器人（sim2real）上运行 |
| **isaac_utils/** | 工具库，提供数学计算、旋转变换等辅助函数 |

---

## 方案一：从零开始配置

适用于全新安装的 Ubuntu 24.04 系统。

### 0. 前置条件与显卡兼容性

**操作系统**：Ubuntu 24.04 LTS

**显卡要求**：NVIDIA GPU，显存 ≥ 8GB，不同显卡的关键差异如下：

| 显卡系列 | 架构 | 驱动要求 | PyTorch 要求 |
|----------|------|----------|-------------|
| RTX 50 系（5060/5070/5080/5090） | Blackwell (sm_120) | **必须用 `nvidia-driver-570-open`（开源内核模块）** | **必须用 nightly cu128** |
| RTX 40 系（4060/4070/4080/4090） | Ada Lovelace (sm_89) | 闭源或开源驱动均可（≥535） | 稳定版 cu121/cu124 均可 |
| RTX 30 系（3060/3070/3080/3090） | Ampere (sm_86) | 闭源或开源驱动均可（≥525） | 稳定版 cu121/cu124 均可 |

> ⚠️ **RTX 50 系用户特别注意**：Blackwell 架构**必须**使用开源内核模块驱动，否则 `nvidia-smi` 会报 "No devices were found"；PyTorch 官方稳定版也不支持 sm_120，必须使用 nightly 版本。

> ⚠️ **关于 Isaac Gym**：ASAP 原版教程基于 Isaac Gym + Ubuntu 20.04。Isaac Gym 预编译二进制与 Ubuntu 24.04 的 glibc 2.39 存在 ABI 不兼容（直接段错误）。本指南统一使用 **Genesis** 仿真器，ASAP 同时支持两者，功能无差异。

> 💡 **关于网络问题**：本指南中多个步骤需要从国外服务器下载文件（PyTorch、pip 包、GitHub 仓库等）。如果你遇上下载速度很慢或连接超时等问题，建议**打开全局代理**，或使用清华镜像源（在 pip 命令末尾加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）。本指南后续不再逐一提醒。

---

### 1. 安装 NVIDIA 驱动

#### 1.1 清除旧驱动（如有）

如果系统中已有 NVIDIA 驱动残留，先彻底清除：

```bash
sudo dpkg --remove --force-remove-reinstreq \
  $(dpkg -l | grep -i nvidia | awk '{print $2}' | tr '\n' ' ')
sudo apt-get purge -y 'nvidia-*' 'libnvidia-*' 'cuda-*' 'libcuda-*'
sudo apt-get autoremove -y --purge
sudo dpkg --configure -a
```

#### 1.2 安装驱动

**RTX 50 系（必须使用开源内核模块）：**

```bash
sudo apt-get update
sudo apt-get install -y nvidia-kernel-open-570 nvidia-utils-570 libnvidia-gl-570
```

如果安装过程中遇到文件冲突（`15_nvidia_gbm.json`），执行：

```bash
sudo rm -f /usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json
sudo dpkg -i --force-overwrite /var/cache/apt/archives/libnvidia-gl-570_570.211.01-0ubuntu1_amd64.deb
sudo apt-get install -f -y
```

**RTX 30/40 系：**

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-570
```

#### 1.3 重启并验证

```bash
sudo reboot
```

重启后：

```bash
nvidia-smi
```

你应该看到类似以下的输出（具体数字因显卡不同而异）：

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...    On  |   00000000:02:00.0 Off |                  N/A |
| N/A   44C    P8              4W /   50W |      15MiB /   8151MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

关键确认项：**Driver Version** 显示 570.x，**CUDA Version** 显示 12.8，**GPU Name** 显示你的显卡型号。如果报错，参见 [附录 C](#c-常见问题排查)。

---

### 2. 安装 CUDA Toolkit 12.8

CUDA Toolkit 包含编译器（`nvcc`）和各种 GPU 计算库，是 PyTorch 等框架的底层依赖。

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

配置环境变量（告诉系统去哪里找 CUDA）：

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

验证：

```bash
nvcc --version
```

预期输出：

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

关键确认项：`release 12.8`。

安装 OpenGL 相关依赖（Genesis 的渲染功能需要）：

```bash
sudo apt-get install -y libgl1 libglu1-mesa mesa-utils xvfb
```

---

### 3. 安装 Miniconda 并创建环境

Miniconda 是一个轻量级的 Python 环境管理器，我们用它创建一个独立的虚拟环境，这样 ASAP 的依赖不会和你系统中的其他 Python 项目冲突。

如果你还没有安装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

按提示完成安装。安装过程中会问你安装路径，**默认路径是 `~/miniconda3`**（即 `/home/你的用户名/miniconda3`），直接回车即可。如果你改了路径，后面涉及 `miniconda3` 的命令都需要相应替换。安装完成后**关闭并重新打开终端**。

创建并激活环境：

```bash
conda create -n hvgen python=3.10 -y
conda activate hvgen
```

配置 conda 环境的动态库路径（某些库需要从 conda 环境目录加载 `.so` 文件）：

```bash
# ⚠️ 如果你的 Miniconda 不是装在默认路径 ~/miniconda3，
#    请将下面的路径替换为你的实际安装路径。
#    例如装在 ~/anaconda3 则改为 ~/anaconda3/envs/hvgen/lib
#    不确定路径？运行 `conda info` 查看 "active env location" 一行
echo "export LD_LIBRARY_PATH=$HOME/miniconda3/envs/hvgen/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
conda activate hvgen
```

---

### 4. 安装 PyTorch

PyTorch 是本项目的核心深度学习框架。根据你的显卡选择对应命令：

**RTX 50 系（必须使用 nightly cu128）：**

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

**RTX 30/40 系（可使用稳定版）：**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

验证：

```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(2, 64, 512, device='cuda', dtype=torch.float16)
print('Tensor shape:', x.shape)
"
```

预期输出（GPU 型号因人而异）：

```
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
Tensor shape: torch.Size([2, 64, 512])
```

关键确认项：`CUDA available: True`，且 `torch.randn` 在 GPU 上运行无报错。如果输出 `False`，参见 [附录 C](#c-常见问题排查)。

---

### 5. 克隆并安装 ASAP

```bash
cd ~
git clone https://github.com/LeCAR-Lab/ASAP.git
cd ASAP
```

**按顺序安装依赖（版本锁定非常重要，不要随意升级）：**

```bash
# genesis-world：轻量级物理仿真引擎，必须为 0.2.1（更新版本 API 不兼容）
pip install genesis-world==0.2.1

# libigl：几何计算库，Genesis 内部用它计算碰撞距离
# 必须为 2.4.1（2.6.1 的 signed_distance 函数返回值数量变化会导致报错）
pip install libigl==2.4.1

# numpy：数值计算库，ASAP 代码要求锁定此版本
pip install numpy==1.23.5

# 安装 ASAP 本体（--no-deps 防止自动安装的依赖覆盖上面锁定的版本）
pip install -e . --no-deps

# 安装 isaac_utils 工具库（提供旋转变换、数学辅助函数等）
pip install -e isaac_utils

# 安装 requirements.txt 中的其余依赖
pip install -r requirements.txt

# 补充安装 Genesis 和 Isaac 需要的库
pip install scipy imageio
```

---

### 6. 安装 ROS2 Humble

ROS2（Robot Operating System 2）是机器人领域的标准通信框架。ASAP 的仿真环境（终端 1）和策略控制器（终端 2）之间通过 ROS2 话题（topic）交换数据。通过 conda 的 robostack 渠道安装：

```bash
conda activate hvgen
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
conda install ros-humble-desktop -y
```

验证：

```bash
python -c "import rclpy; print('rclpy OK')"
```

预期输出：

```
rclpy OK
```

---

### 7. 安装 Unitree SDK

Unitree SDK 是宇树机器人的 Python 开发包，提供与 G1 人形机器人通信的接口（sim2sim 也需要它来模拟通信协议）。

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .

# 修复可能的版本冲突
pip install --upgrade numpy scipy
```

---

### 8. 安装 sim2real 剩余依赖

```bash
# mujoco：物理仿真引擎，用于 sim2sim 可视化
# onnxruntime：模型推理引擎，加载预训练的 .onnx 策略模型
# pygame：提供游戏开发基础功能，这里用于仿真环境的窗口和事件处理
# sshkeyboard：监听键盘按键，用于在终端中通过键盘控制机器人
pip install mujoco onnxruntime pygame sshkeyboard
```

---

### 9. 运行 Demo

#### 9.1 训练测试（验证环境安装是否正确）

这一步会在 Genesis 仿真器中启动 1024 个并行环境，训练 G1 机器人的行走策略。我们用它来验证整个环境是否安装正确。

```bash
conda activate hvgen
cd ~/ASAP

python humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_g1_locomotion \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=1024 \
  project_name=TestGenesisInstallation \
  experiment_name=G123dof_loco \
  headless=True
```

如果一切正常，你会看到 Genesis 初始化信息，然后开始输出训练日志：

```
[Genesis] [12:07:33] [INFO] 🚀 Genesis initialized. 🔖 version: 0.2.1
[Genesis] [12:07:33] [INFO] Running on [NVIDIA GeForce RTX 5060 Laptop GPU] with backend gs.cuda.
...
╭──────────────────────────────── Training Log ────────────────────────────────╮
│                       Learning iteration 0/1000000                          │
│                        Computation: 3579 steps/s                            │
│              Mean action noise std: 0.80                                    │
│  Mean episode rew_tracking_lin_vel: 0.0002                                  │
│                    Total timesteps: 24576                                   │
│                     Iteration time: 6.87s                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

看到 Training Log 就说明环境配置成功了。**按 `Ctrl+C` 停止训练即可**，不需要等它跑完（完整训练需要数小时到数天）。

如果 Genesis 报 EGL 错误，设置以下环境变量后重试：

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

#### 9.2 Sim2Sim 可视化 Demo（预训练模型）

这是最有趣的部分——你将在 MuJoCo 仿真中看到 G1 机器人执行各种动作。需要**两个终端**同时运行。

**终端 1 — 启动 MuJoCo 仿真环境：**

```bash
conda activate hvgen
cd ~/ASAP/sim2real
python sim_env/base_sim.py --config=config/g1_29dof_hist.yaml
```

等待 MuJoCo 可视化窗口弹出后，再开启终端 2。

**终端 2 — 启动策略控制：**

```bash
conda activate hvgen
cd ~/ASAP/sim2real
python rl_policy/deepmimic_dec_loco_height.py \
  --config=config/g1_29dof_hist.yaml \
  --loco_model_path=./models/dec_loco/20250109_231507-noDR_rand_history_loco_stand_height_noise-decoupled_locomotion-g1_29dof/model_6600.onnx \
  --mimic_model_paths=./models/mimic
```

#### 9.3 启动后的操作顺序

机器人初始状态下有一根隐形的"安全绳"将它悬挂在空中。你需要按照正确的顺序操作，否则机器人会直接瘫倒在地面。

**正确操作顺序：**

1. **在终端 2 中按 `]`** → 激活策略（此时机器人仍被安全绳悬挂）
2. **在终端 2 中按 `=`** → 开启行走模式（机器人的腿开始做行走动作，但仍在空中）
3. **在 MuJoCo 仿真窗口中按 `9`** → 释放安全绳，机器人落地并开始自主平衡行走

> ⚠️ **注意**：必须先激活策略并开启行走模式（步骤 1、2），再释放安全绳（步骤 3）。如果在策略未激活时直接释放，机器人无法保持平衡，会直接瘫倒。

4. 用 `w/a/s/d` 控制移动，`q/e` 控制转向
5. 按 `;` 或 `'` 切换动作，按 `[` 执行当前动作

详细按键说明见 [附录 A](#a-键盘控制说明)，可用动作列表见 [附录 B](#b-预训练动作列表)。

---

## 方案二：克隆预配置环境

我们已将完整的 ASAP 项目（含预训练模型和环境配置文件）上传到 GitHub 仓库，你可以直接克隆使用。

### 适用条件

| 条件 | 要求 |
|------|------|
| 操作系统 | Ubuntu 24.04 LTS |
| 显卡 | NVIDIA GPU，显存 ≥ 8GB |
| 驱动 | 已安装（RTX 50 系必须用 nvidia-driver-570-open） |
| CUDA | 12.8 已安装 |
| Miniconda | 已安装 |

> 驱动、CUDA、Miniconda 无法通过 GitHub 分发，必须提前装好。如果没装，请先按方案一的**第 1-3 步**完成（只需做到"创建并激活 conda 环境"即可，后续步骤按照方案二）。

### 配置步骤

#### 步骤 1：确认驱动和 CUDA

```bash
nvidia-smi
nvcc --version
```

确认 `nvidia-smi` 正常显示显卡信息，`nvcc --version` 显示 `release 12.8`。如有问题，参考方案一第 1、2 步。

#### 步骤 2：克隆仓库

```bash
cd ~
git clone https://github.com/UMJI-SSTIA/Deep-Learning-Workshop.git
cd Deep-Learning-Workshop/ASAP_prebuilt
```

仓库中的 `ASAP_prebuilt/` 目录结构如下：

```
ASAP_prebuilt/
├── ASAP/                     # 完整的 ASAP 项目代码 + 预训练模型
│   ├── humanoidverse/        # 训练框架
│   ├── sim2real/             # Sim2Sim / Sim2Real 部署
│   │   └── models/           # 预训练模型文件（.onnx）
│   ├── isaac_utils/          # 工具库
│   ├── requirements.txt
│   └── setup.py
│   └── README.md             # ASAP项目说明
├── environment.yml           # conda 环境导出（优先尝试）
├── pip_requirements.txt      # pip freeze 参考（不可直接使用）
└── setup.sh                  # 一键安装脚本
```

#### 步骤 3：运行一键安装脚本

```bash
chmod +x setup.sh
./setup.sh
```

脚本会自动完成以下工作：检查驱动/CUDA/conda → 从 `environment.yml` 恢复 conda 环境（失败则自动回退到手动安装全部依赖）→ 安装 Unitree SDK、ASAP、isaac_utils → 配置环境变量 → 安装系统依赖 → 验证所有关键库。

安装完成后：

```bash
source ~/.bashrc
conda activate hvgen
```

> **如果脚本报错**，可以改为手动安装：按方案一的第 3-8 步操作，但将所有路径中的 `~/ASAP` 替换为 `~/Deep-Learning-Workshop/ASAP_prebuilt/ASAP`。

#### 步骤 4：运行 Demo

参照 [方案一第 9 步](#9-运行-demo)，但注意路径：

```bash
conda activate hvgen

# 训练测试
cd ~/Deep-Learning-Workshop/ASAP_prebuilt/ASAP
python humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_g1_locomotion \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=1024 \
  project_name=TestGenesisInstallation \
  experiment_name=G123dof_loco \
  headless=True
```

Sim2Sim Demo：

```bash
# 终端 1
cd ~/Deep-Learning-Workshop/ASAP_prebuilt/ASAP/sim2real
python sim_env/base_sim.py --config=config/g1_29dof_hist.yaml

# 终端 2（等 MuJoCo 窗口弹出后）
cd ~/Deep-Learning-Workshop/ASAP_prebuilt/ASAP/sim2real
python rl_policy/deepmimic_dec_loco_height.py \
  --config=config/g1_29dof_hist.yaml \
  --loco_model_path=./models/dec_loco/20250109_231507-noDR_rand_history_loco_stand_height_noise-decoupled_locomotion-g1_29dof/model_6600.onnx \
  --mimic_model_paths=./models/mimic
```

启动后的操作顺序见 [9.3 节](#93-启动后的操作顺序非常重要)。

---

## 进阶：本地训练

ASAP 使用 PPO（Proximal Policy Optimization）算法训练机器人的运动策略。你可以在本地训练自己的模型，但请注意**完整训练需要数小时到数天**，取决于你的 GPU 性能和训练目标。本次 workshop 中我们直接使用预训练模型。

#### 训练命令

以训练 G1 机器人的行走策略为例（以下路径以方案二为准，方案一请自行替换）：

```bash
conda activate hvgen
cd ~/Deep-Learning-Workshop/ASAP_prebuilt/ASAP

python humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_g1_locomotion \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=1024 \
  project_name=MyTraining \
  experiment_name=G1_locomotion \
  headless=True
```

**命令参数说明：**

| 参数 | 含义 |
|------|------|
| `+simulator=genesis` | 使用 Genesis 仿真器 |
| `+exp=locomotion` | 实验类型：行走任务 |
| `+domain_rand=NO_domain_rand` | 不使用域随机化（简化训练） |
| `+rewards=loco/reward_g1_locomotion` | 奖励函数配置 |
| `+robot=g1/g1_29dof_anneal_23dof` | 机器人型号：G1，29 自由度 |
| `+terrain=terrain_locomotion_plane` | 地形：平地 |
| `num_envs=1024` | 并行环境数量（显存不够可减小，如 512 或 256） |
| `headless=True` | 无头模式（不渲染画面，训练更快） |

#### 训练日志示例

启动后，你会看到 Genesis 初始化、场景构建、内核编译等过程，然后开始输出训练迭代日志：

```
[Genesis] 🚀 Genesis initialized. 🔖 version: 0.2.1
[Genesis] Running on [NVIDIA GeForce RTX 5060 Laptop GPU] with backend gs.cuda. Device memory: 7.54 GB.
[Genesis] Building scene...
[Genesis] Compiling simulation kernels...       ← 首次运行会编译内核，耗时约 30 秒
[Genesis] Building visualizer...

╭──────────────────────────────── Training Log ────────────────────────────────╮
│                       Learning iteration 0/1000000                          │
│                        Computation: 3579 steps/s                            │
│              Mean action noise std: 0.80                                    │
│  Mean episode rew_tracking_lin_vel: 0.0002                                  │
│  Mean episode rew_tracking_ang_vel: 0.0001                                  │
│ Mean episode rew_penalty_ang_vel_xy: -0.0001                                │
│                    Total timesteps: 24576                                   │
│                     Iteration time: 6.87s                                   │
│                         Total time: 6.87s                                   │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────── Training Log ────────────────────────────────╮
│                       Learning iteration 1/1000000                          │
│                        Computation: 1844 steps/s                            │
│                        Mean reward: -38.55                                  │
│                Mean episode length: 47.55                                   │
│  Mean episode rew_tracking_lin_vel: 0.0039                                  │
│ Mean episode rew_penalty_ang_vel_xy: -0.3546                                │
│                    Total timesteps: 49152                                   │
│                     Iteration time: 13.32s                                  │
│                         Total time: 20.19s                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**日志解读：**

- **Mean reward**：平均奖励，训练目标是让它逐渐上升（刚开始为负数是正常的）
- **Mean episode length**：平均 episode 长度，越长说明机器人存活时间越久（即学会了保持平衡）
- **rew_tracking_lin_vel**：跟踪线速度的奖励，上升说明机器人学会了按指令移动
- **rew_penalty_\***：各种惩罚项，绝对值下降说明机器人的动作越来越自然
- **Computation: xxxx steps/s**：训练速度

训练产生的模型会保存在 `logs/` 目录下。

---

## 附录

### A. 键盘控制说明

Sim2Sim Demo 有两个按键区域：**终端 2（策略终端）** 和 **MuJoCo 仿真窗口**。

**终端 2 中的按键**（焦点需在终端 2 上）：

| 按键 | 功能 | 备注 |
|------|------|------|
| `]` | 激活策略 | **启动后第一步必按** |
| `=` | 切换行走模式（Stand command 0↔1） | 按了才能用 wasd |
| `w` / `s` | 前进 / 后退 | 需先开启行走模式 |
| `a` / `d` | 左平移 / 右平移 | |
| `q` / `e` | 左转 / 右转 | |
| `z` | 所有速度归零 | |
| `;` / `'` | 下一个 / 上一个 mimic 动作 | |
| `[` | 执行当前 mimic 动作（再按切回行走） | |
| `i` | 回到初始姿态 | |
| `o` | 紧急停止 | |
| `1` / `2` | 增加 / 减少基座高度 | |

**MuJoCo 仿真窗口中的按键**（焦点需在 MuJoCo 窗口上）：

| 按键 | 功能 | 备注 |
|------|------|------|
| `9` | 释放安全绳 | **必须先激活策略并开启行走模式后再按** |

### B. 预训练动作列表

通过 `;` / `'` 切换动作，按 `[` 执行：

| 动作名称 | 说明 |
|----------|------|
| APT_level1 | APT 舞蹈 |
| CR7_level1 | C 罗招牌动作（Siuuu） |
| jump_forward_level1/2/3 | 前跳（难度递增） |
| kick_level1/2/3 | 踢腿（难度递增） |
| Kobe_level1 | 科比投篮动作 |
| lebron_level1/2 | 勒布朗动作 |
| side_jump_level1/2/3 | 侧跳（难度递增） |

### C. 常见问题排查

#### C.1 nvidia-smi 报 "Driver/library version mismatch"

**原因**：系统中残留多版本驱动。

```bash
sudo dpkg --remove --force-remove-reinstreq \
  $(dpkg -l | grep -i nvidia | awk '{print $2}' | tr '\n' ' ')
sudo apt-get purge -y 'nvidia-*' 'libnvidia-*'
sudo apt-get autoremove -y --purge
sudo reboot
# 重启后重新安装驱动（参考第 1 步）
```

#### C.2 nvidia-smi 报 "No devices were found"

**原因**：RTX 50 系必须使用开源内核模块。

```bash
sudo dmesg | grep -i nvrm
# 如果看到 "requires use of the NVIDIA open kernel modules"：
sudo apt-get install -y nvidia-kernel-open-570
sudo reboot
```

#### C.3 PyTorch 报 sm_120 不兼容

**原因**：RTX 50 系需要 nightly 版本。

```bash
pip uninstall torch -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### C.4 Genesis 报 `GenesisException: Genesis hasn't been initialized`

**原因**：genesis-world 版本过新（>0.2.1）。

```bash
pip install genesis-world==0.2.1
```

#### C.5 `ValueError: too many values to unpack (expected 3)`

**原因**：libigl 版本过新（2.6.1 与 genesis-world 0.2.1 不兼容）。

```bash
pip install libigl==2.4.1
```

#### C.6 Genesis headless 模式报 EGL 错误

**原因**：NVIDIA EGL vendor 文件缺失或未被识别。

```bash
# 检查文件是否存在
ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 设置环境变量
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# 如果文件不存在，重装：
sudo apt-get install --reinstall libnvidia-gl-570
```

#### C.7 激活机器人后机器人抽搐

正常现象，弹性带（安全绳）托住了机器人，机器人无法达成平衡，重新启动demo后记得释放安全绳。

#### C.8 wasd 无反应

必须按正确顺序操作：先按 `]` 激活策略 → 再按 `=` 开启行走模式 → 然后 wasd 才会生效。

#### C.9 释放安全绳后机器人直接瘫倒

你没有先开启行走模式就释放了安全绳，请重新启动 demo，按照 [9.3 节](#93-启动后的操作顺序) 的顺序操作。

---

## 关键版本约束速查

| 包 | 必须版本 | 原因 |
|---|---------|------|
| genesis-world | 0.2.1 | 更新版本 API 不兼容 |
| libigl | 2.4.1 | 2.6.1 的 signed_distance 返回值数量变化 |
| numpy | 1.23.5 | ASAP 代码要求 |
| PyTorch | nightly cu128（RTX 50）/ 稳定版 cu124（RTX 30/40） | sm_120 计算能力支持 |

---

## 参考配置

### `sim2real/config/g1_29dof_hist.yaml` 关键字段

```yaml
ROBOT_SCENE: "../humanoidverse/data/robots/g1/scene_29dof.xml"  # 机器人场景文件
ENABLE_ELASTIC_BAND: True   # True=安全绳托起机器人；False=直接落地
DOMAIN_ID: 0                # ROS2 域 ID
INTERFACE: "lo"             # 网络接口。sim2sim 用 "lo"（本地回环）；sim2real 改为实际网卡
```
