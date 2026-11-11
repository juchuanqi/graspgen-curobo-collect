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

保留 480 Hz 物理、60 Hz 控制、每控制步 8 个物理子步、双路 256×256 RGB、120 控制步落稳、原始接触记录和当前成功判据。仿真位姿直接从 Isaac 读取，GraspGen 输入为 mesh 表面采样点。包含单环境常驻采集、四卡三环境并发、冷/热/并发测速，以及首回合预热和初态核查。

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

# 自动先验收2条；验收通过后才开始上述120次。
bash run_long_batch.sh 0,1,2,3

# 仅查看任务计划，不导入 Isaac、不运行 GPU。
bash run.sh collect --objects all --episodes-per-object 30 --workers 3 --gpus 0,1,2,3 --dry-run
```

`--gpus` 始终相对于当前 `CUDA_VISIBLE_DEVICES` 分配。默认只有一个仿真环境；多环境是独立进程，并且要求四卡。`--gui` 只用于一个环境。物品切换时重建仿真，同物品回合间复用环境与规划服务。`--seed` 控制初态采样；当前沿用各物品参考位置附近 XY ±2 mm、yaw ±8°的范围，位置总偏移不超过 ±15 mm。第一条使用该物品参考初态。

每个预定初态只做一次物理尝试，普通抓取失败保留，技术故障或初态核查失败会停止；不会补抓至成功。连续批次默认统计尝试完成情况，`--smoke` 还要求两条都通过抓取、初态与数据校验才返回成功。建议先完成 smoke 再扩大批次。

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

采集校验完成后，在独立的 LeRobot 0.4.4 环境里离线导出：

```bash
bash run.sh export --run outputs/原始采集目录 --output outputs/新导出目录 \
  --python /你的/lerobot-venv/bin/python
```

命令会筛选通过抓取、初态和数据校验的回合，写入 LeRobot v3，并用官方加载器逐行读回核对。见[导出和数据约定](docs/LEROBOT.md)。

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
