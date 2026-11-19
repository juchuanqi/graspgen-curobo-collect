# GraspGen + cuRobo 数据采集

Panda 已知物体 mesh 抓取数采：官方 GraspGen 生成候选，cuRobo 规划，Isaac Lab / PhysX 执行，保存双相机、状态、动作和接触力，严格校验后导出 LeRobot v3。

当前交付范围为**抓取 → 抬升 → 稳持 → 数据验收**。输入使用已知 mesh 和仿真真实位姿；四类内置物体为方块、细长条、圆柱、圆盘。默认直线接近 **5 cm**，支持固定挡板避障。

运行平台为 Linux x86_64 + NVIDIA GPU；当前验证栈为 Isaac Sim 5.1.0 / Isaac Lab 2.3.0、cuRobo 0.7.8 和 Python 3.11。GraspGen、仿真、规划和 LeRobot 使用独立环境。权重、环境与候选需要在目标机器准备，目标机器先通过两条 smoke 再运行批量采集。

## 快速开始

```bash
git clone https://github.com/juchuanqi/graspgen-curobo-collect.git
cd graspgen-curobo-collect
git rev-parse HEAD
```

当前默认分支为 `beijing-d-collect`，请记录提交号用于复现。按机器选择准备路线：

| 环境 | 准备步骤 |
|---|---|
| 独立 Linux 工作站 / 新集群 | 按[独立安装](docs/INSTALL.md)安装三套运行环境与权重，复制 `local_env.example.json` 为 `local_env.json` 并填入本机绝对路径 |
| 北京 D 或挂载了原共享依赖的机器 | 按[共享环境复现](docs/REPRODUCE.md#2-北京-d复用已经准备的共享环境)绑定现有环境和验证权重；个人路径仅用于该环境 |

两条路线准备完成后，都在仓库根目录执行：

```bash
bash run.sh check --gpus 0

# 首先准备方块候选，并做两条物理和原始数据验收。
bash run.sh prepare --objects block55 --gpu 0
bash run.sh collect --smoke --gpus 0

# 四种物体的固定批次；每个预定初态一次物理尝试，失败保留。
bash run.sh prepare --objects all --gpu 0
bash run.sh collect --objects all --episodes-per-object 5 --gpus 0
```

候选默认每物体 200 个；显式使用 500／1000 见[候选数量](docs/CANDIDATE_COUNT.md)，准备与采集须使用同一个候选根目录。`check` 只做文件和 CPU 依赖预检；`prepare`、`collect` 真正使用 GPU。

多卡编号为当前 `CUDA_VISIBLE_DEVICES` 内的逻辑编号。四卡最多 3 个仿真 worker，共享第 4 张卡上的规划器；八卡最多 8 个 worker，每卡一个仿真及专属规划器。例如：

```bash
bash run.sh collect --objects all --episodes-per-object 5 \
  --gpus 0,1,2,3 --workers 3
```

单卡与双卡布局、显存和启动排查见[管线说明](docs/ISAAC_PIPELINE.md#入口与进程职责)。

## 采集与交付

| 命令 | 输出 |
|---|---|
| `bash run.sh collect ...` | 原始轨迹、双 RGB、初态与数据验证、失败记录 |
| `bash run.sh export --run RAW --output NEW --python LEROBOT_PYTHON` | 严格选择、LeRobot v3、官方完整读回及独立统计 |
| `bash run.sh pipeline --data-only ...` | 可恢复地串起采集和导出，结束于数据验收 |
| `bash run.sh COMMAND --help` | 各入口参数；查看帮助不需要 GPU 环境 |

```bash
# 独立 LeRobot 0.4.4 环境；只安装一次，勿混入仿真/规划环境。
python3.11 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install -r requirements-lerobot.txt
export LEROBOT_PYTHON="$PWD/.venv-lerobot/bin/python"

# 另起新批次，串联采集和数据交付；输出目录事先不能存在。
bash run.sh pipeline --data-only --output "$PWD/outputs/new_batch" --gpus 0 \
  --objects all --episodes-per-object 5 \
  --candidate-root "$PWD/generated/candidates"

# 按原登记配置恢复；保留已执行槽位，不自动补抓失败案例。
bash run.sh pipeline --data-only --output "$PWD/outputs/new_batch" --resume
```

已有 `collect` 原始批次用 `export --run <原始目录>`，无需再次采集。新 pipeline 可加 `--export-workers 4` 按物体并行生成 LeRobot 分片；独立 export 对应 `--workers 4`。仿真并发与导出并发分别配置，默认导出为单包。分片路径及联合读取见[导出说明](docs/LEROBOT.md)。

`pipeline` 请显式带 `--data-only`；兼容入口省略该参数仍会执行历史 ACT 读取检查。训练由下游单独负责。

默认输出 `outputs/<时间戳>/`；`collect` 目录含 `REPORT.md`、`SUMMARY.json`、逐回合 `trajectory.npz`、双路 `cameras/*.npy` 及验证回执。`pipeline` 将原始数据放在 `raw/`，导出与读回放在 `export/`，并冻结本次源码。N 个动作对应 N+1 帧观测；双 256×256 RGB、60 Hz 控制、480 Hz 物理、每控制间隔 8 子步接触记录。

验收时分别查看 `raw/RUN.json` 的槽位完成状态、`raw/SLOTS.json` 的成功/失败分类，以及 `export/SELECTION.json`、`export/AUDIT.json` 和 `export/COMMITTED.json` 的数据筛选、完整读回与封存。计划执行完成不代表全部抓取成功；原始成功数与最终合格导出数也可能因去重而不同。

| 配置 | `collect` 默认 | 新 `pipeline` 默认 |
|---|---|---|
| 执行优化 | `accelerated`；smoke/benchmark 为 `baseline` | `accelerated` |
| 关节速度 | `baseline` | `recommended` |
| 障碍物 | `none` | `none` |
| 夹爪对称派生 | `off` | `off` |

夹持力反馈、终点 IK 预筛及接触排序默认关闭，实验启用方式见[夹持质量](docs/GRASP_QUALITY.md)。桌面 1 mm、障碍物 6 mm 规划余量，完整臂／掌／指世界碰撞、规划自碰撞及独立几何复核均保留。

固定墙使用 `--obstacle wall`，须配合经过预检的有效初态；原始默认细长条带墙初态与墙相交，会拒绝整批启动。布局与显式初态设计见[固定墙说明](docs/OBSTACLES.md)和[恢复说明](docs/RECOVERY.md)。

## 当前验证范围

历史纯模型 500 候选带墙批次实际成功 **20/32（62.5%）**，该批使用 10 cm 接近。改为 5 cm 后，预选的 4 个恢复案例及 4 个成功对照 **8/8 抓起并稳持，挡板接触为 0**；这 8 例是定向回归，不能当作新版全量成功率。详见[当前基线](docs/BASELINE.md)。

本版收敛数采和数据核查流程。实验报告保留在文档和独立结果目录，RRT 对比与缓存路径执行未接入生产。搬运投篮、松手不属于当前采集任务。

## 开发检查

```bash
# 独立 CPU 测试环境，无需安装 Isaac / GraspGen / cuRobo。
python3.11 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-test.txt
.venv-cpu/bin/python -B -m unittest discover -s tests -v

# 源码交付包排除 .venv*、权重、候选、原始记录、日志和 local_env.json。
python3 -B tools/package.py --output ../graspgen_curobo_collect-source.tar.gz
```

CPU CI 自动执行几何、初态、数据筛选、恢复、调度及入口回归；cuRobo 专属和 LeRobot 专属测试需要各自环境，GPU 测试还需显式启用。

| 文档 | 内容 |
|---|---|
| [复现手册](docs/REPRODUCE.md) / [独立安装](docs/INSTALL.md) | 共享环境、新机器安装、smoke、资源排查 |
| [代码与扩展](docs/CODE.md) | 模块职责、坐标约定、新物品注册 |
| [Isaac 数采管线](docs/ISAAC_PIPELINE.md) | 进程分工、回合执行顺序、结果文件和故障定位 |
| [当前基线](docs/BASELINE.md) | 配置、物理验收、已验证范围、保留问题 |
| [恢复与固定初态](docs/RECOVERY.md) | 槽位账本、检查点、源码冻结、恢复边界 |
| [数据格式与导出](docs/LEROBOT.md) | 严格筛选、单包／分片、官方读回 |
| [验证记录](docs/VALIDATION.md) | 历史实验和各次增量验证 |
| [第三方来源](THIRD_PARTY_NOTICES.md) | 依赖与模型资产来源 |

仓库包含源码、四物体几何、Panda 模型和 GraspGen 源码。权重、Python 环境与实验产物单独管理；本地路径配置使用忽略版本管理的 `local_env.json`，环境变量优先。
