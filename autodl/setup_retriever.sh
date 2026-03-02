#!/bin/bash
# Retriever 检索服务环境 (可选，用于本地检索)
# 建议与 searchr1 分开环境，在 AutoDL 上可单独开一个终端运行

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-retriever}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# 不启动 GPU 时设置 USE_GPU=0，使用 faiss-cpu 和 PyTorch CPU
USE_GPU="${USE_GPU:-1}"

echo "=== Retriever 环境安装 ==="
echo "GPU 模式: $([ "$USE_GPU" = "1" ] && echo '是 (faiss-gpu)' || echo '否 (faiss-cpu)')"

if conda env list | grep -q "^\s*${CONDA_ENV_NAME}\s"; then
    echo "环境 $CONDA_ENV_NAME 已存在"
else
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# PyTorch 与 faiss（USE_GPU=0 时用 CPU 版）
if [ "$USE_GPU" = "1" ]; then
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
  conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
else
  conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cpuonly -c pytorch -y
  pip install faiss-cpu -i "$PIP_INDEX_URL" -q
fi

pip install transformers datasets pyserini -i "$PIP_INDEX_URL" -q
pip install uvicorn fastapi -i "$PIP_INDEX_URL" -q

echo "=== Retriever 环境安装完成 ==="
echo "激活: conda activate $CONDA_ENV_NAME"
echo "启动检索: bash retrieval_launch.sh (需先修改其中的 file_path/index_file/corpus_file)"
