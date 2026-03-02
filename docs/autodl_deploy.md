# 在 AutoDL 上部署 Search-R1

本文档说明如何在 [AutoDL](https://www.autodl.com/) 上配置镜像环境并运行 Search-R1 项目，便于将项目上传至云端训练与推理。

## 一、推荐实例与镜像

- **GPU**：建议至少 1 张 RTX 3090 / 4090 或 A100（训练 3B 约需多卡，可按 `train_ppo.sh` 中 `CUDA_VISIBLE_DEVICES` 调整）。
- **系统镜像**：选择 **Ubuntu 20.04 / 22.04 + CUDA 12.1** 的 PyTorch 镜像（如「PyTorch 2.4.0 / Python 3.10 / CUDA 12.1」），与项目要求的 PyTorch 2.4 + CUDA 12.1 一致。
- **数据盘**：建议挂载足够大的数据盘，用于存放数据集、索引与 checkpoint。

## 二、方式一：在实例内一键安装（推荐）

开好 AutoDL 实例并进入终端后，在项目根目录执行：

```bash
# 1. 进入项目目录（若从本地上传，请先上传到 /root/autodl-tmp 或你的工作目录）
cd /root/autodl-tmp/Search-R1-wsy   # 按你实际上传路径修改

# 2. 一键安装 Search-R1 主环境（使用国内 pip 镜像）
bash autodl/setup_searchr1.sh

# 3. 激活环境并验证
conda activate searchr1
python -c "import torch; import vllm; print('OK')"
```

如需**本地检索服务**（NQ + E5 + Wikipedia 等），可再开一个终端安装并启动 Retriever 环境：

```bash
cd /root/autodl-tmp/Search-R1-wsy
bash autodl/setup_retriever.sh
conda activate retriever
# 修改 retrieval_launch.sh 中的 file_path / index_file / corpus_file 后执行
bash retrieval_launch.sh
```

## 三、方式二：使用 Conda 环境文件

若希望用同一份环境定义复现环境（例如在多台机器上保持一致）：

```bash
conda env create -f environment.yaml
conda activate searchr1
# 仍需在项目根目录安装 PyTorch(CUDA) 和 vllm，并执行: pip install -e .
# 建议直接使用 autodl/setup_searchr1.sh，已包含上述步骤
```

## 四、方式三：使用自定义 Docker 镜像

若希望把环境做成镜像，方便以后「选镜像即用」、减少每次安装时间：

### 4.1 在本地或可构建 Docker 的机器上

在**项目仓库根目录**执行：

```bash
docker build -f Dockerfile.autodl -t search-r1:autodl .
```

### 4.2 导出镜像并上传到 AutoDL

```bash
# 导出镜像（体积较大，可先压缩）
docker save search-r1:autodl | gzip > search-r1-autodl.tar.gz

# 将 search-r1-autodl.tar.gz 上传到 AutoDL 实例（或对象存储后下载到实例）
# 在 AutoDL 实例上加载镜像
gunzip -c search-r1-autodl.tar.gz | docker load
```

AutoDL 控制台若支持「自定义镜像」或「导入镜像」，可按平台说明将上述镜像导入后，创建实例时选择该镜像即可。

### 4.3 在容器内运行

容器内已安装好 `searchr1` 环境，默认 `WORKDIR` 为 `/workspace`。若将代码挂载到 `/workspace`：

```bash
docker run -it --gpus all -v /root/autodl-tmp/Search-R1-wsy:/workspace search-r1:autodl
conda activate searchr1
bash train_ppo.sh   # 或 python infer.py
```

## 五、环境变量与国内镜像（可选）

脚本已默认使用清华 pip 源。若需改用其他国内源，可在执行安装脚本前设置：

```bash
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com
bash autodl/setup_searchr1.sh
```

AutoDL 若已提供 conda 国内源，可无需改；未配置时可在 `~/.condarc` 中配置 channel 镜像。

## 六、快速跑通流程（NQ + 本地检索）

1. **下载索引与语料**（在项目根目录）：

   ```bash
   save_path=/root/autodl-tmp/data   # 建议放在数据盘
   python scripts/download.py --save_path $save_path
   cat $save_path/part_* > $save_path/e5_Flat.index
   gzip -d $save_path/wiki-18.jsonl.gz
   ```

2. **处理 NQ 数据**：

   ```bash
   python scripts/data_process/nq_search.py
   ```

3. **修改检索启动脚本**  
   编辑 `retrieval_launch.sh`，将 `file_path`、`index_file`、`corpus_file` 改为上面 `save_path` 下的路径。

4. **先启动检索服务**（在 retriever 环境中）：

   ```bash
   conda activate retriever
   bash retrieval_launch.sh
   ```

5. **再启动训练**（在 searchr1 环境中）：

   ```bash
   conda activate searchr1
   # 按 GPU 数量修改 train_ppo.sh 中的 CUDA_VISIBLE_DEVICES
   bash train_ppo.sh
   ```

## 七、常见问题

- **CUDA 版本不匹配**：AutoDL 实例若为 CUDA 11.8，可将 `autodl/setup_searchr1.sh` 中 `CUDA_TAG` 改为 `cu118`，并安装对应 PyTorch。
- **flash-attn 安装失败**：可暂时跳过，脚本已做容错；若需启用，在 `searchr1` 环境中单独执行：  
  `pip install flash-attn --no-build-isolation`
- **显存不足**：在 `train_ppo.sh` 中减小 `data.train_batch_size`、`actor_rollout_ref.actor.ppo_micro_batch_size` 等，或减少 `CUDA_VISIBLE_DEVICES` 的卡数并相应改 `trainer.n_gpus_per_node`。

按上述步骤即可在 AutoDL 上配置好 Search-R1 的镜像环境并上传至云端使用。
