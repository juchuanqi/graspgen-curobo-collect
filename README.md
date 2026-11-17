# GraspGen + cuRobo / Isaac Lab 数据采集

来自**北京 D 集群**，原整理目录为 `/mnt/public/juchuanqi/project/graspgen_curobo_collect`。
同事首次使用请按 [复现手册](docs/REPRODUCE.md) 操作：包含北京 D 共享环境、新机器安装入口、两条验收、120 次采集及故障排查。
本项目范围是抓取数据采集、质量验收和数据交付。

GitHub：[juchuanqi/graspgen-curobo-collect](https://github.com/juchuanqi/graspgen-curobo-collect/tree/beijing-d-collect)，分支 `beijing-d-collect`。

```bash
git clone --branch beijing-d-collect --single-branch \
  https://github.com/juchuanqi/graspgen-curobo-collect.git
cd graspgen-curobo-collect
```

独立的 Panda 抓取数据采集代码，沿用当前仿真直接状态与 mesh 流程：

`物体 mesh → 2048 点云 → GraspGen 生成及评分 → 几何筛选 → cuRobo 规划 → Isaac Lab / PhysX 实执行 → 接触与抬升保持评分 → 保存和校验`

数据交付使用 `pipeline --data-only`，在采集后继续严格筛选、LeRobot v3 导出和官方完整读回，结束于数据验收。独立 `collect` 只保存原始采集数据，已有数据可用 `export` 后处理；具体命令见下文。

保留 480 Hz 物理、60 Hz 控制、每控制步 8 个物理子步、双路 256×256 RGB、120 控制步落稳、原始接触记录和当前成功判据。仿真位姿直接从 Isaac 读取，GraspGen 输入为 mesh 表面采样点。包含单环境常驻采集、四卡三环境共享规划器及八卡八环境专属规划器并发、冷/热/并发测速，以及首回合预热和初态核查。

本目录只交付代码、配置、Panda 模型和四种物体的几何资产。没有历史运行日志、图像、轨迹、实验结果或旧 GraspGen 候选。约 1 GB 的官方模型权重与 Python 环境单独使用；候选在同事自己的运行目录重新生成。

## 能访问现有共享目录时

在 Linux + NVIDIA GPU 的机器上运行，先确认下面目录可读：

```bash
cd /mnt/public/juchuanqi/project/graspgen_curobo_collect
python3 -B tools/configure_shared.py --shared-root /mnt/public/juchuanqi
bash run.sh check --gpus 0

# 为四种物体生成本次使用的 GraspGen 候选；生成后可复用。
bash run.sh prepare --gpu 0

# 单卡先采两条方块，校验实际初态、抓取及原始数据。
bash run.sh collect --smoke --gpus 0
```

`configure_shared.py` 只在本代码库写入忽略版本管理的 `local_env.json`，指向共享盘现有环境与权重；不会修改当前激活的 Python 环境。环境变量优先于该文件。单卡时仿真和规划是同卡的两个独立进程，需要容纳二者显存；GraspGen 准备进程先结束再启动采集。原先四卡实测经验不能直接作为单卡速度和显存保证。

如果同事电脑没有挂载这些目录，复制本目录或解压 `graspgen_curobo_collect.tar.gz`，按 [独立安装说明](docs/INSTALL.md) 配置三个环境即可，运行代码不依赖上述路径。

## 开始采集

```bash
# 单卡、方块连续 10 次；打开 Isaac Sim 窗口。
bash run.sh collect --objects block55 --episodes-per-object 10 --gpus 0 --gui

# 单卡、四物品各 10 次，无界面。
bash run.sh collect --objects all --episodes-per-object 10 --gpus 0

# 两卡：第一张仿真、第二张规划。
bash run.sh collect --objects all --episodes-per-object 10 --gpus 0,1

# 四卡：前三张仿真，末张共享规划；四物品各30次，共120次。
bash run.sh collect --objects all --episodes-per-object 30 --workers 3 --gpus 0,1,2,3

# 八卡：每张卡一个长驻仿真和专属规划器，使用已有执行优化与关节提速。
bash run.sh collect --objects all --episodes-per-object 32 --workers 8 \
  --gpus 0,1,2,3,4,5,6,7 --execution-profile accelerated --arm-speed-profile recommended

# 自动先验收2条；验收通过后才开始上述120次。
bash run_long_batch.sh 0,1,2,3

# 仅查看任务计划，不导入 Isaac、不运行 GPU。
bash run.sh collect --objects all --episodes-per-object 30 --workers 3 --gpus 0,1,2,3 --dry-run
```

`--gpus` 始终相对于当前 `CUDA_VISIBLE_DEVICES` 分配。默认只有一个仿真环境；多环境是独立进程，四卡最多三路共用一个规划器，八卡最多八路且每路有同卡专属规划器。GPU映射写入槽位账本，恢复拒绝改动。`--gui` 只用于一个环境。物品切换时重建仿真，同物品回合间复用环境与规划服务。`--seed` 控制默认初态采样：各物品参考位置附近 XY ±2 mm、yaw ±8°，位置总偏移不超过 ±15 mm，第一条使用参考初态。也可用 `coverage_plan.py` 生成四物体共 200 槽的显式分层清单，通过 `--plan-json` 输入并在每物体前 10 槽暂停检查；见[分批检查点](docs/RECOVERY.md#显式初态与分批检查点)。

普通采集的 `--execution-profile` 默认是 `accelerated`：复用历史力传感器环形调度、接触 CUDA Graph、相机显式调度、原生隐式执行器 CUDA Graph，以及闭合几何/抬升 RPC 重叠；当前 COM v3 评分和规划前静止证据保持。CPU两路验证与采集重叠，最多四条等待，每路首条验证完成后通过公共门禁进入测量。原速/原执行对照分别指定 `--arm-speed-profile baseline --execution-profile baseline`；旧 benchmark/smoke 默认使用原执行配置。

每个预定初态只做一次物理尝试，普通抓取失败保留，技术故障或初态核查失败会停止；不会补抓至成功。连续批次默认统计尝试完成情况，`--smoke` 还要求两条都通过抓取、初态与数据校验才返回成功。建议先完成 smoke 再扩大批次。

正式采集和 pipeline 登记前，会在 CPU 上检查整批请求的生成位姿和落稳位姿，覆盖完整机械臂、夹指、物体、桌面及障碍物。任一初态未通过既有几何标准即拒绝整批，不创建实验目录、不启动模拟器，也不自动筛掉或重采样该槽。pipeline 保存 `INITIAL_GEOMETRY_PREFLIGHT.json`；直接采集保存到 `ENVIRONMENT.json` 的 `checks.initial_geometry`。`--dry-run` 仅打印计划；实际落稳后的位姿、静止和几何核查仍保留。原始默认细长条带墙初态会被拒绝，须先显式设计有效初态。

可选关节速度配置（默认仍为原速度）：

```bash
bash run.sh collect --smoke --gpus 0 --arm-speed-profile recommended
bash run.sh collect --objects all --episodes-per-object 30 --workers 3 --gpus 0,1,2,3 --arm-speed-profile recommended
```

`recommended` 将接近段速度/路径进度加速度设为 1.0 rad/s、5.0 rad/s²，持物抬升段为 0.3 rad/s、2.0 rad/s²；下降、离桌直线、闭合和保持沿用原设置。重定时后的路径仍经过完整几何复核。配置写入 `PLAN.json` 和每条 `METADATA.json`，独立验证按保存配置检查各阶段动作增量，无需再设置环境变量。`--arm-speed-profile baseline` 显式恢复原速度；命令行优先于实验用的 `ARM_SPEED_PROFILE` 环境变量。2026-11-11四物品各30条验收：方块、长条、圆柱各30/30成功，圆盘17/30；120条初态与完整数据校验均通过，但物理成功门槛未通过。独立 `collect` 保留原速度兼容默认值，使用提速档位需显式选择；新 `pipeline` 默认采用提速档位，见后文。圆盘6对同初态同候选诊断：原速度4/6、新速度3/6，暂不作普遍质量等价结论。

## 夹爪对称朝向选择

通过 `--grasp-symmetry near-first` 开启当前夹爪朝向感知的候选配对：

```bash
bash run.sh collect --smoke --gpus 0 --grasp-symmetry near-first
# 只查看计划：
bash run.sh collect --objects all --grasp-symmetry near-first --dry-run
```

对每个已转换到 TCP 坐标约定的候选，生成 `R' = R @ diag(-1,-1,1)`，保持 TCP 位置与接近轴 Z，交换两指方向。用实时物体位姿转换到世界坐标，再计算与当前实测夹爪的完整三维转角。N 个源候选得到 N 个近向和 N 个远向版本；16 个源候选即两组各 16 个，不额外调用模型。夹角相同优先保留原朝向。

两组分别进行完整夹爪场景几何筛选，每组最多保留 50 个；随后对两组候选做批量 IK。完整规划优先近向组，组内沿用关节运动量排序；远向组作为剩余预算内的后备。两组共用最多 5 个完整规划候选，每个仍为 30 秒软预算，完整机械臂几何与自碰撞检查保持。此偏好减少的是目标朝向差，实际关节路程和轨迹转角仍须测量。

`PLAN.json`、任务及回合元数据保存模式；`CANDIDATE_REPORT.json` 保存配对、夹角、IK 和规划顺序。派生候选有独立 `candidate_id`，保留父候选索引和模型分数，派生 `score` 为空。默认 `--grasp-symmetry off` 沿用原流程。该功能已做 CPU 回归，尚未进行新增物理抓取验收。

## 当前支持的物体

| ID | 形状 | 尺寸 mm |
|---|---|---|
| `block55` | 长方体 | 55×38×32 |
| `slender95` | 细长条 | 95×26×28 |
| `cylinder36h52` | 圆柱 | 直径36，高52 |
| `disk56h24` | 圆盘 | 直径56，高24 |

这些是当前采集包支持的四物品。几何资产中的 `surface.npz` 是由物体模型生成的表面点、法向和面标记，属于运行资产，不是采集数据。增加物品需要同步物理 USD、mesh、表面点及完整几何分解，并注册初态；详见 [代码与扩展说明](docs/CODE.md)。

## 输出

默认新结果写在 `outputs/<时间戳>/`，可用 `--output /数据盘/新目录` 更换；已有输出目录不会被覆盖。运行后会产生：

- `REPORT.md`、`SUMMARY.json`、`episodes.csv`：每条结果、指定初态核查、数据验证及实际吞吐。
- 每条回合的 `trajectory.npz`：动作、关节与物体状态、末端位姿、8 子步接触、评分和时序。
- `cameras/base_camera.npy` 和 `cameras/wrist_camera.npy`：原始 RGB，N 个动作对应 N+1 个观测。
- 规划、初态、原始力校验、代码快照与本次运行日志，均位于这次输出目录。

模型下载写入 `models/`，新 GraspGen 候选写入 `generated/candidates/`，可配置缓存写入 `.cache/`。上述目录全部忽略版本管理及源码打包。约 825 控制步的双相机 RGB 每条约 0.32 GB，另有接触与状态数据；保存 120 次时可按 120 GB 预留原始采集空间以覆盖控制步上限。

## 导出 VLA 训练数据

`collect` 保存原始 NPZ 轨迹和双相机 NPY，不包含 LeRobot 转换；`export` 将已有轨迹转换为训练数据包，`pipeline` 串起采集、导出和读取检查，并支持 `--data-only`。

采集校验完成后，在独立的 LeRobot 0.4.4 环境里离线导出：

```bash
bash run.sh export --run outputs/原始采集目录 --output outputs/新导出目录 \
  --python /你的/lerobot-venv/bin/python
```

命令会筛选通过抓取、初态和数据校验的回合，写入 LeRobot v3，并用官方加载器逐行读回核对。见[导出和数据约定](docs/LEROBOT.md)。

并行后处理使用 `export --workers 4`：整批严格筛选一次，再按物体并行生成独立标准包，每包均完成完整读回和独立统计。训练检查自动联合读取这些包，并使用合并统计。

```bash
bash run.sh export --run outputs/原始采集目录 --output outputs/新分片目录 \
  --workers 4 --python /你的/lerobot-venv/bin/python
# 新采集批次自动接入并行后处理；仿真并发仍由原 --workers 控制。
bash run.sh pipeline --output outputs/新批次 --data-only --export-workers 4 \
  --candidate-root /你的/candidates --lerobot-python /你的/lerobot-venv/bin/python
```

默认仍为单包导出。分片数据位于 `portable/shards/<物体>/dataset`，合集与逐包审计绑定于 `portable/SHARDS.json`；联合训练方法见生成的 `DATA_USAGE.md`。分片后处理不启动新仿真，不改变筛选或质量门槛。

## 原测速模式与工具

```bash
# 保留原40次冷启动/热复用/三路并发对照；--full 为72次同场景对照。
bash run.sh benchmark --gpus 0,1,2,3

# 在规划环境中运行 CPU 测试。
"$PLANNER_PYTHON" -B -m unittest discover -s tests -v

# 从源码生成不含运行产物的交付包。
python3 -B tools/package.py --output /目标目录/graspgen_curobo_collect.tar.gz
```

本次迁移的验证边界见 [验证说明](docs/VALIDATION.md)。依赖与资产来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

基础数据验收可用 `bash run.sh pipeline --help`：固定采集 → 运动/静止证据校验 → 严格筛选 → 原生 LeRobot v3 导出 → 官方完整读取 → 短 ACT 训练。默认 4 种基础物体各 5 条，失败保留，不重试；详见 [验收说明](docs/VALIDATION.md)。

2026-11-16 起，新 `pipeline` 批次默认采用已有 `recommended` 关节加速档位和 `accelerated` 执行优化。支持八卡八路专属规划器布局；恢复按登记配置继续，旧冻结批次保持原配置。独立 `collect` 的速度兼容默认值仍为原速度，八卡加速采集需显式选 `recommended`。当前四物体的32条接入检查及128条正式复测登记于 [P2目录](../results/graspgen_accelerated_p2_20261116_01/PREREGISTRATION.json)，结论以真实验收结果为准。

P2 实测：接入检查32/32成功，正式125/128成功；三条失败均为细长物体135°朝向的稳持失败，原始记录保留。每物体八路首条验证后，四个有限测量窗口合计96槽、93成功，含在线验证7.650秒/成功条；完整采集含热身/切换/冷启动9.558秒/成功条。125条、71,302行严格导出和完整官方读回、20,160项独立统计及ACT100通过；扣除ACT的数据活跃总成本41.13秒/合格条。该批已知网格/固定初态结果见 [P2报告](../results/graspgen_accelerated_p2_20261116_01/REPORT.md)，与历史杯子/P1只作观测参考。后续优先优化CPU导出及回读；ACT检查不代表策略闭环能力。

新批次支持 `bash run.sh pipeline --output /原目录 --resume` 继续固定槽位及后处理，并可用 `--data-only` 单独交付数据、`--resume --with-training` 补训练检查。仅确定发生在执行前的初始化资源故障有限重启；已完成和执行进度不明的槽位不会自动重抓。状态、互斥、封存和数据使用方式见 [恢复与交付说明](docs/RECOVERY.md)。

2026-11-13 固定批次经一次初始化显存故障恢复后，20/20 条独立稳持确认通过，完成 16,836 行官方读回和 100 步 ACT 训练检查。结果、复现入口和资源恢复边界见 [基础管线验收](docs/PIPELINE_ACCEPTANCE_20261113.md)。

## 固定墙采集

`pipeline --obstacle wall` 将固定墙同时接入 Isaac 碰撞、cuRobo 规划和严格后处理；默认 `none`。新批次在登记前检查全部生成/落稳初态，任一不合格即拒绝整批。规划使用桌面 1 mm、障碍物 6 mm 余量，原控制器及物理阈值不变。

执行余量版本另跑新 256 槽：两种无墙布局均 64/64，近侧带墙 21/64、墙两侧带墙 19/64；最大禁止接触力 0 N，16 次物理失败及 72 次规划失败全部保留。全部初态合格，168 条不同成功起点完成严格导出及官方读回。近侧公共参考位置显式 +20 mm，每物体/布局使用 16 个不同 XY 起点；这是独立新批次，不能与历史修正汇总混用。带墙四物体可靠性门槛仍未通过；当前验收范围为抓取、抬升、稳持及数据交付。设计、耗时、失败与历史更正见 [固定墙说明](docs/OBSTACLES.md)。

可选的夹持力反馈、抓取终点预筛、接触排序和圆柱几何先验见 [夹持质量实验](docs/GRASP_QUALITY.md)。这些设置默认关闭，保留原物理判据；几何先验的候选来源单独记录，不能将其结果归为纯 GraspGen 提升。反馈动作可用独立工具从保存源码及上一帧接触力重建核查。
