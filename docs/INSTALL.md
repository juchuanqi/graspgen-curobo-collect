# 独立机器安装

当前运行栈为 Linux x86_64、NVIDIA CUDA GPU、Python 3.11、Isaac Sim 5.1.0 / Isaac Lab 2.3.0、cuRobo 0.7.8 和官方 GraspGen Panda 模型。Windows/macOS 未作为本项目运行平台验证；Windows 电脑可连接可运行 Isaac 的 Linux 工作站。需要 NVIDIA 驱动、Vulkan、对应 CUDA 编译工具和 C++ 编译器。共享环境已经安装好时，优先使用根目录 README 的共享盘入口。

三个 Python 环境分开，避免 Isaac 自带 PyTorch 与规划/推理依赖互相替换。启动器本身只用 Python 标准库，项目路径允许包含空格。

## 1. Isaac 环境

按 NVIDIA 对 Isaac Sim 5.1 / Isaac Lab 2.3 的要求安装系统依赖与驱动。以下为 Python 环境安装入口；平台已经安装 Isaac 时直接使用现有解释器。

```bash
conda create -n grasp-sim python=3.11 -y
conda activate grasp-sim
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
git clone --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git /你的依赖目录/IsaacLab
cd /你的依赖目录/IsaacLab
./isaaclab.sh --install
cd /你的代码目录/graspgen_curobo_collect
python -m pip install -r requirements-sim.txt
```

保持 Isaac 所需 PyTorch。原环境为 `torch 2.7.0+cu126`；Isaac Lab 发行元数据与 Git 标签字符串不同并不等同于安装错误。首次 Kit 启动或扩展下载可能较久，可以设置 `--startup-timeout 1200`。

## 2. cuRobo 环境

```bash
conda create -n grasp-planner python=3.11 -y
conda activate grasp-planner
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git /你的依赖目录/curobo
python -m pip install -r requirements-planner.txt
python -m pip install --no-build-isolation -e /你的依赖目录/curobo
```

PyTorch 2.11/CUDA 12.8 是现有规划环境使用的组合。扩展需用与 PyTorch 对应的 CUDA 工具链编译；设置正确的 `CUDA_HOME`。GPU 架构由本机工具链检测，启动器没有强制写死 A800 的 `sm_80`。复制其他显卡机器的已编译 `.so` 不等于完成本机编译。

## 3. GraspGen 环境

源码内的 `third_party/GraspGen` 保留当前使用版本及兼容修改，推理脚本会优先加载它。其原始 `requirements.txt` 含不适用于 Python 3.11 的 `pickle5`，本项目提供单独的依赖入口。

```bash
conda create -n grasp-model python=3.11 -y
conda activate grasp-model
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-graspgen.txt
python -m pip install --no-deps -e third_party/GraspGen
cd third_party/GraspGen/pointnet2_ops
python -m pip install --no-build-isolation .
```

`torch-scatter` 和 PointNet++ 包含 CUDA 扩展；如果当前 PyTorch 没有对应预编译 wheel，需要本机 CUDA 编译器。上面的 GraspGen 依赖固定了现有推理关键版本，其中旧 diffusers 需要对应的 transformers / huggingface-hub，避免只升级其中一个。当前整理已复用现有环境验证，未在一台全新电脑上重新完成以上安装。

## 4. 绑定解释器和权重

回到本仓库根目录，复制 `local_env.example.json` 为 `local_env.json`，将三个解释器路径改为本机实际路径。也可用环境变量：

```bash
export SIM_PYTHON=/你的/grasp-sim/bin/python
export PLANNER_PYTHON=/你的/grasp-planner/bin/python
export GRASPGEN_PYTHON=/你的/grasp-model/bin/python
```

下载脚本只下载官方 Panda 的生成器、判别器与模型配置，固定模型 revision 并逐文件校验 SHA256：

```bash
bash run.sh download-models
export GRASPGEN_CONFIG="$PWD/models/checkpoints/graspgen_franka_panda.yml"
bash run.sh check --gpus 0
bash run.sh prepare --gpu 0
bash run.sh collect --smoke --gpus 0
```

官方文件合计约 1.07 GB。也可将现有 `checkpoints/` 下三个文件复制到 `models/checkpoints/`，随后执行 `bash run.sh download-models --verify-only`。如使用镜像服务，可按 huggingface-hub 的方式配置 `HF_ENDPOINT`。

`check` 只检查文件与 CPU 依赖，不启动 Isaac 或 CUDA 计算。`prepare` 真正运行 GraspGen，一物品一次 200 候选，再按相同几何规则筛选；`collect` 真正启动 cuRobo 与 Isaac。已有候选的输入 mesh、模型、代码与哈希相同才复用；不同或不完整的目录会报错并保留，需要用新的 `--output-root`。自定义候选目录采集时通过 `--candidate-root` 指向同一位置。

运行缓存默认集中在本库 `.cache/`，可用 `COLLECT_CACHE_DIR` 调整；`TORCH_EXTENSIONS_DIR`、`CUDA_CACHE_PATH`、`TRITON_CACHE_DIR`、`XDG_CACHE_HOME`、`TMPDIR` 的显式设置仍优先。第三方 Isaac/Kit 也可能写自己的用户配置目录。
