# GraspGen + cuRobo 数据采集

Panda 已知物体 mesh 抓取数采：官方 GraspGen 生成候选，cuRobo 规划，Isaac Lab / PhysX 执行，保存双相机、状态、动作和接触力，严格校验后导出 LeRobot v3。

当前交付范围为**抓取 → 抬升 → 稳持 → 数据验收**。输入使用已知 mesh 和仿真真实位姿；四类内置物体为方块、细长条、圆柱、圆盘。默认直线接近 **5 cm**，支持固定挡板避障。

## 快速开始

```bash
git clone --branch beijing-d-collect --single-branch \
  https://github.com/juchuanqi/graspgen-curobo-collect.git
cd graspgen-curobo-collect

# 北京 D 共享环境；其他机器先按 docs/INSTALL.md 安装并配置 local_env.json。
python3 -B tools/configure_shared.py --shared-root /mnt/public/juchuanqi
bash run.sh check --gpus 0

# 首先准备方块候选，并做两条物理和原始数据验收。
bash run.sh prepare --objects block55 --gpu 0
bash run.sh collect --smoke --gpus 0

# 四种物体的固定批次；每个预定初态一次物理尝试，失败保留。
bash run.sh prepare --objects all --gpu 0
bash run.sh collect --objects all --episodes-per-object 5 --gpus 0
```

候选默认每物体 200 个；显式使用 500／1000 见[候选数量](docs/CANDIDATE_COUNT.md)。`check` 只做文件和 CPU 依赖预检；`prepare`、`collect` 真正使用 GPU。目标机器必须先通过 smoke。

## 采集与交付

| 命令 | 输出 |
|---|---|
| `bash run.sh collect ...` | 原始轨迹、双 RGB、初态与数据验证、失败记录 |
| `bash run.sh export --run RAW --output NEW --python LEROBOT_PYTHON` | 严格选择、LeRobot v3、官方完整读回及独立统计 |
| `bash run.sh pipeline --data-only ...` | 可恢复地串起采集和导出，结束于数据验收 |
| `bash run.sh COMMAND --help` | 各入口参数；查看帮助不需要 GPU 环境 |

```bash
# 需要独立 LeRobot 0.4.4 环境；输出目录须为新的绝对路径。
bash run.sh pipeline --data-only --output /data/new_batch --gpus 0 \
  --objects all --episodes-per-object 5 \
  --candidate-root "$PWD/generated/candidates" \
  --lerobot-python /your/lerobot-venv/bin/python

# 按原登记配置恢复；保留已执行槽位，不自动补抓失败案例。
bash run.sh pipeline --data-only --output /data/new_batch --resume
```

`pipeline` 请显式带 `--data-only`；兼容入口省略该参数仍会执行历史 ACT 读取检查。训练由下游单独负责。

默认输出 `outputs/<时间戳>/`；`collect` 目录含 `REPORT.md`、`SUMMARY.json`、逐回合 `trajectory.npz`、双路 `cameras/*.npy` 及验证回执。`pipeline` 将原始数据放在 `raw/`，导出与读回放在 `export/`，并冻结本次源码。N 个动作对应 N+1 帧观测；双 256×256 RGB、60 Hz 控制、480 Hz 物理、每控制间隔 8 子步接触记录。

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
python3.11 -m venv /your/cpu-test-env
/your/cpu-test-env/bin/python -m pip install -r requirements-test.txt
/your/cpu-test-env/bin/python -B -m unittest discover -s tests -v

# 源码交付包排除权重、候选、原始记录、日志和 local_env.json。
python3 -B tools/package.py --output /data/graspgen_curobo_collect.tar.gz
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
