#!/bin/bash
# Search-R1 主环境一键安装 (AutoDL / 国内云端)
# 使用国内 pip 镜像加速，适合在 AutoDL 等平台运行

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# 国内 pip 镜像 (可改为 aliyun 等)
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-searchr1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
# AutoDL 常见为 cu121，按实例选择
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
VLLM_VERSION="${VLLM_VERSION:-0.6.3}"

echo "=== Search-R1 环境安装 ==="
echo "项目路径: $PROJECT_ROOT"
echo "Conda 环境: $CONDA_ENV_NAME"
echo "Pip 镜像: $PIP_INDEX_URL"

# 创建 conda 环境
if conda env list | grep -q "^\s*${CONDA_ENV_NAME}\s"; then
    echo "环境 $CONDA_ENV_NAME 已存在，将复用并更新依赖"
else
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# PyTorch (CUDA 12.1，与 AutoDL 常见镜像一致)
pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_TAG} -q
pip install torchvision torchaudio -q

# vllm
pip install "vllm==${VLLM_VERSION}" -i "$PIP_INDEX_URL" -q

# 项目依赖 (requirements.txt)
pip install -r requirements.txt -i "$PIP_INDEX_URL" -q

# 以可编辑方式安装 verl
pip install -e . -i "$PIP_INDEX_URL" -q

# Flash Attention 2 (编译较慢，失败可暂时跳过)
pip install flash-attn --no-build-isolation -i "$PIP_INDEX_URL" -q || echo "flash-attn 安装失败，可稍后单独安装"

# wandb 等
pip install wandb -i "$PIP_INDEX_URL" -q

echo "=== Search-R1 环境安装完成 ==="
echo "激活环境: conda activate $CONDA_ENV_NAME"
echo "训练示例: bash train_ppo.sh"
echo "推理示例: python infer.py"
