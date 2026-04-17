#!/bin/bash
set -e

# =========================================
#   ASAP Prebuilt Environment Setup
#   适用于 Ubuntu 24.04 + NVIDIA GPU
# =========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASAP_DIR="${SCRIPT_DIR}/ASAP"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "  ASAP Prebuilt Environment Setup"
echo "========================================="
echo ""

# ------------------------------------------
# 1. 检查 NVIDIA 驱动
# ------------------------------------------
echo -e "[1/8] 检查 NVIDIA 驱动..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi 未找到${NC}"
    echo "   请先安装 NVIDIA 驱动（RTX 50 系必须使用 nvidia-driver-570-open）"
    echo "   参考 ASAP_Setup_Guide.md 方案一第 1 步"
    exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
echo -e "${GREEN}✅ 驱动正常: ${GPU_INFO}${NC}"

# ------------------------------------------
# 2. 检查 CUDA
# ------------------------------------------
echo -e "[2/8] 检查 CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}❌ nvcc 未找到${NC}"
    echo "   请先安装 CUDA Toolkit 12.8"
    echo "   参考 ASAP_Setup_Guide.md 方案一第 2 步"
    exit 1
fi
CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo -e "${GREEN}✅ CUDA ${CUDA_VER}${NC}"

# ------------------------------------------
# 3. 检查 conda
# ------------------------------------------
echo -e "[3/8] 检查 conda..."
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ conda 未找到${NC}"
    echo "   请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✅ conda 已安装${NC}"

# ------------------------------------------
# 4. 创建 conda 环境
# ------------------------------------------
echo -e "[4/8] 创建 conda 环境 hvgen..."
if conda info --envs | grep -q "hvgen"; then
    echo -e "${YELLOW}⚠️  hvgen 环境已存在，跳过创建${NC}"
    echo "   如需重建，请先运行: conda env remove -n hvgen"
else
    echo "   尝试从 environment.yml 恢复..."
    if [ -f "${SCRIPT_DIR}/environment.yml" ] && conda env create -f "${SCRIPT_DIR}/environment.yml" 2>/dev/null; then
        echo -e "${GREEN}✅ 从 environment.yml 恢复成功${NC}"
    else
        echo -e "${YELLOW}⚠️  environment.yml 恢复失败，改为手动安装...${NC}"

        conda create -n hvgen python=3.10 -y

        eval "$(conda shell.bash hook)"
        conda activate hvgen

        echo "   [4a] 安装 PyTorch nightly cu128..."
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

        echo "   [4b] 安装 genesis-world==0.2.1..."
        pip install genesis-world==0.2.1

        echo "   [4c] 安装 libigl==2.4.1..."
        pip install libigl==2.4.1

        echo "   [4d] 安装 numpy==1.23.5..."
        pip install numpy==1.23.5

        echo "   [4e] 安装 ASAP..."
        cd "${ASAP_DIR}"
        pip install -e . --no-deps
        pip install -e isaac_utils
        pip install -r requirements.txt
        pip install scipy imageio

        echo "   [4f] 安装 sim2real 依赖..."
        pip install mujoco onnxruntime pygame sshkeyboard

        echo "   [4g] 安装 ROS2 Humble..."
        conda config --env --add channels conda-forge
        conda config --env --add channels robostack-staging
        conda config --env --remove channels defaults 2>/dev/null || true
        conda install ros-humble-desktop -y

        cd "${SCRIPT_DIR}"
    fi
fi

# ------------------------------------------
# 5. 安装可编辑模式的包
# ------------------------------------------
echo -e "[5/8] 安装可编辑模式的包..."

eval "$(conda shell.bash hook)"
conda activate hvgen

# Unitree SDK
echo "   安装 Unitree SDK..."
cd "${SCRIPT_DIR}"
if [ ! -d "unitree_sdk2_python" ]; then
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
fi
cd unitree_sdk2_python
pip install -e .

# ASAP 本体和 isaac_utils
echo "   安装 ASAP 和 isaac_utils..."
cd "${ASAP_DIR}"
pip install -e . --no-deps 2>/dev/null || true
pip install -e isaac_utils 2>/dev/null || true

cd "${SCRIPT_DIR}"
echo -e "${GREEN}✅ 可编辑模式的包安装完成${NC}"

# ------------------------------------------
# 6. 配置环境变量
# ------------------------------------------
echo -e "[6/8] 配置环境变量..."

CONDA_PREFIX_PATH=$(conda run -n hvgen printenv CONDA_PREFIX 2>/dev/null)

BASHRC_MARKER="# >>> ASAP environment >>>"
if grep -q "${BASHRC_MARKER}" ~/.bashrc; then
    echo -e "${YELLOW}⚠️  环境变量已配置，跳过${NC}"
else
    cat >> ~/.bashrc << EOF

${BASHRC_MARKER}
export PATH=/usr/local/cuda-12.8/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${CONDA_PREFIX_PATH}/lib:\$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
# <<< ASAP environment <<<
EOF
    echo -e "${GREEN}✅ 环境变量已写入 ~/.bashrc${NC}"
fi

# ------------------------------------------
# 7. 检查系统依赖
# ------------------------------------------
echo -e "[7/8] 检查系统依赖..."
MISSING_PKGS=""
for pkg in libgl1 libglu1-mesa mesa-utils xvfb; do
    if ! dpkg -s "$pkg" &> /dev/null; then
        MISSING_PKGS="${MISSING_PKGS} ${pkg}"
    fi
done

if [ -n "${MISSING_PKGS}" ]; then
    echo "   安装缺失的系统包:${MISSING_PKGS}"
    sudo apt-get update
    sudo apt-get install -y ${MISSING_PKGS}
else
    echo -e "${GREEN}✅ 系统依赖已满足${NC}"
fi

# ------------------------------------------
# 8. 验证安装
# ------------------------------------------
echo -e "[8/8] 验证安装..."
echo ""

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${CONDA_PREFIX_PATH}/lib:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

conda run -n hvgen python -c "
import sys

# PyTorch + CUDA
try:
    import torch
    assert torch.cuda.is_available(), 'CUDA not available'
    print(f'  ✅ PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'  ❌ PyTorch/CUDA: {e}')
    sys.exit(1)

# Genesis
try:
    import genesis
    print(f'  ✅ Genesis {genesis.__version__}')
except Exception as e:
    print(f'  ❌ Genesis: {e}')

# ROS2
try:
    import rclpy
    print('  ✅ ROS2 (rclpy)')
except Exception as e:
    print(f'  ❌ ROS2: {e}')

# MuJoCo
try:
    import mujoco
    print(f'  ✅ MuJoCo {mujoco.__version__}')
except Exception as e:
    print(f'  ❌ MuJoCo: {e}')

# ONNX Runtime
try:
    import onnxruntime
    print(f'  ✅ ONNX Runtime {onnxruntime.__version__}')
except Exception as e:
    print(f'  ❌ ONNX Runtime: {e}')

# Unitree SDK
try:
    import unitree_sdk2py
    print('  ✅ Unitree SDK')
except Exception as e:
    print(f'  ❌ Unitree SDK: {e}')

print()
print('  🎉 验证完成！')
"

echo ""
echo "========================================="
echo "  安装完成！"
echo "========================================="
echo ""
echo "使用方法："
echo "  1. 重新打开终端（或执行 source ~/.bashrc）"
echo "  2. conda activate hvgen"
echo "  3. 运行 Sim2Sim Demo（需要两个终端）："
echo ""
echo "     终端 1:"
echo "       cd ${ASAP_DIR}/sim2real"
echo "       python sim_env/base_sim.py --config=config/g1_29dof_hist.yaml"
echo ""
echo "     终端 2（等 MuJoCo 窗口弹出后）:"
echo "       cd ${ASAP_DIR}/sim2real"
echo "       python rl_policy/deepmimic_dec_loco_height.py \\"
echo "         --config=config/g1_29dof_hist.yaml \\"
echo "         --loco_model_path=./models/dec_loco/20250109_231507-noDR_rand_history_loco_stand_height_noise-decoupled_locomotion-g1_29dof/model_6600.onnx \\"
echo "         --mimic_model_paths=./models/mimic"
echo ""
echo "  4. 终端 2 按 ] 激活策略 → 按 = 开启行走 → MuJoCo 窗口按 9 释放安全绳→进行操作"
echo ""
