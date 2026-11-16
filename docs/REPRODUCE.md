# 从北京 D 环境复现 GraspGen + cuRobo 数据采集

本代码整理自**北京 D 集群**：`/mnt/public/juchuanqi/project/graspgen_curobo_collect`。
适用 Linux x86_64 + NVIDIA GPU，机器人为 Panda，默认四类物体。项目负责采集、质量验收和交付数据。

采集流程：物体 mesh 采样 2048 点 → GraspGen 生成 200 个候选并评分 → 完整几何筛选 → cuRobo 规划 → Isaac Lab / PhysX 实际抓取 → 抬升与稳定保持评分 → 保存和校验。
物体位姿和机器人状态直接读取仿真；双相机负责记录 RGB。当前初态是各物体参考位置附近的小范围扰动，具体范围见下文，不能视为广泛场景覆盖。

## 1. 拉取代码并选择环境路线

代码位于 [juchuanqi/graspgen-curobo-collect](https://github.com/juchuanqi/graspgen-curobo-collect/tree/beijing-d-collect) 的 `beijing-d-collect` 分支。直接克隆：

```bash
git clone --branch beijing-d-collect --single-branch \
  https://github.com/juchuanqi/graspgen-curobo-collect.git
cd graspgen-curobo-collect
git rev-parse HEAD
```

仓库已公开，可直接浏览和克隆，无需额外授权。记录提交号，便于后续对齐版本。
GraspGen 源码、Panda 模型与四种物体资产已包含在代码内，无需额外初始化 Git submodule。
模型权重、Python 环境、历史轨迹、日志和旧候选池不包含在仓库。

| 你的机器 | 使用方法 |
|---|---|
| 北京 D，能访问 `/mnt/public/juchuanqi` 下现有依赖 | 按第 2 节绑定共享环境，直接预检 |
| 其他集群，但已挂载同一共享目录 | 可尝试同样的绑定；目标机器驱动/Vulkan 仍须通过实际预跑 |
| 本地 Linux 工作站或没有上述共享盘的集群 | 先按 [INSTALL.md](INSTALL.md) 安装三套环境和权重，再从第 3 节开始 |

下面所有命令都在代码库根目录执行。首次安装或编译请用可联网的节点；真实采集使用平台分配的 GPU 节点。

## 2. 北京 D：复用已经准备的共享环境

共享资源为：

| 用途 | 路径 |
|---|---|
| Isaac 仿真解释器 | `/mnt/public/juchuanqi/venvs/graspbench_20261107/simulation/bin/python` |
| cuRobo 解释器 | `/mnt/public/juchuanqi/venvs/graspbench_20261107/planning/bin/python` |
| GraspGen 解释器 | `/mnt/public/juchuanqi/venvs/graspbench_20261107/graspgen/bin/python` |
| 官方 Panda 权重 | `/mnt/public/juchuanqi/models/graspgen/checkpoints/` |
| 共享 Isaac Lab 源码依赖 | `/mnt/public/juchuanqi/project/external/IsaacLab-2.3.0/` |

共享解释器通过自己的配置读取以上依赖；仅拷贝 `bin/python` 不能替代这些目录。
无需激活 Conda，也不要往当前 `(reason)` 或其他环境中混装 Isaac / cuRobo / GraspGen。

```bash
python3 -B tools/configure_shared.py --shared-root /mnt/public/juchuanqi

# 验证官方权重文件内容；只读现有权重，不下载、不执行 GPU。
bash run.sh download-models --verify-only --output /mnt/public/juchuanqi/models/graspgen

# 预检三套 Python、CPU 导入、Panda/物体资产和模型配置。
bash run.sh check --gpus 0
```

预期权重出现三行 `VERIFIED`，预检最终出现 `"status": "passed"`。
`configure_shared.py` 只写当前代码目录中被 Git 忽略的 `local_env.json`。
如果此前设置过 `SIM_PYTHON`、`PLANNER_PYTHON`、`GRASPGEN_PYTHON` 或 `GRASPGEN_CONFIG`，环境变量优先于该文件；请核对预检输出的实际解释器。

也可以使用同等步骤的一键入口。默认会生成候选并实际采两条：

```bash
# 仅绑定/校验，不运行 GraspGen 或 Isaac。
bash tools/reproduce_shared.sh 0 --check-only

# 四卡平台：前三张用于仿真、末张用于共享规划；此处仅运行两条预跑。
bash tools/reproduce_shared.sh 1,2,3,0
```

默认共享根目录是 `/mnt/public/juchuanqi`，可通过 `SHARED_ROOT` 更改。
这个脚本不自动启动 120 次批量采集。

## 3. 生成本机的 GraspGen 候选

```bash
# 先只准备 smoke 需要的方块，缩短首次排查链路。
bash run.sh prepare --objects block55 --gpu 0
```

预期出现 `GENERATED block55: 200 poses, ... geometry-qualified`。
候选写入 `generated/candidates/graspgen/block55/`，包含 `candidates.npz` 和 `REPORT.json`。
如已存在且 mesh/模型/代码/seed 等相同，显示 `CACHE_HIT block55`，不会重新抽样。
几何合格数应大于 0，但不要求不同 GPU 的浮点输出逐位一致。

完整四类准备：

```bash
bash run.sh prepare --objects all --gpu 0
```

这一阶段真正执行 GraspGen 的生成器与判别器；同一物体的后续初态复用本地候选，在线用实时物体位姿变换。
不是每条轨迹都重复推理。模型使用的 revision 和三个文件 SHA256 固定在 `runtime/config/models.json`。

## 4. 先完成两条真实验收

最容易启动的是单卡，无界面运行：

```bash
bash run.sh collect --smoke --gpus 0
```

如果有四张已分配的卡，推荐将仿真和规划分开：

```bash
bash run.sh collect --smoke --gpus 1,2,3,0
```

`--gpus` 是当前 `CUDA_VISIBLE_DEVICES` 分配中的逻辑编号。
例如平台已设置 `CUDA_VISIBLE_DEVICES=4,5,6,7`，应使用 `--gpus 0,1,2,3`，分别映射到分配的四张卡。
单卡会在同卡启动两个独立进程（仿真和规划），必须留出二者显存；未测定所有显卡的最小显存要求。
有显示服务且需要窗口时，可以在单 worker 命令上加 `--gui`；远程批量采集使用默认无界面模式。

每次运行开头打印 `Output: ...`，请记下该路径。检查该目录中的：

- `RUN.json`：状态为 `completed`。
- `SUMMARY.json`：两条均为 `requested_state_validated_successes`，合计 2。
- 每条 `INITIAL_STATE_AUDIT.json`：`status=passed`，实际 PhysX 与 cached 初态均符合请求值。
- 每条 `METADATA.json`：`success=true`、`reason=stable_grasp_hold`。
- 每条 `VALIDATION.json` 与 `RAW_FORCE_VALIDATION.json`：`status=passed`。

`--smoke` 仅在以上抓取、指定初态和数据验证均通过时返回 0；失败保留现场，不自动重试或补抓。
首次启动会初始化 Kit、CUDA 和扩展，可能明显比后续回合慢。Kit 很慢时可设置 `--startup-timeout 1200`。
原平台曾出现新进程首回合停在默认位姿的问题；当前版本包含预热及硬性初态检查，目标机器的两条验收仍是必需步骤。

## 5. 四类批量采集

当前物体 ID：`block55`、`slender95`、`cylinder36h52`、`disk56h24`。

```bash
# 先检查120个固定槽的计划，不启动 Isaac 或 GPU 计算。
bash run.sh collect --objects all --episodes-per-object 30 \
  --workers 3 --gpus 1,2,3,0 --dry-run

# 三路常驻采集：四类各30次，总120次固定尝试。
bash run.sh collect --objects all --episodes-per-object 30 \
  --workers 3 --gpus 1,2,3,0

# 或用自动门槛入口：先重新验收2条，通过后执行120次。
bash run_long_batch.sh 1,2,3,0
```

三张卡各运行一个独立 Isaac 进程，末张卡运行串行共享 cuRobo 服务；没有同进程多环境向量化。
同物体的 worker 连续做 10 条后切换物体，切换时重建仿真。
默认 `--seed 2026207000`；每类第一个初态固定在参考位置，其余在参考 XY ±2 mm、yaw ±8°内取样，XY 总偏移限制为 ±15 mm。
要扩大位置、姿态或物体范围，需要修改并实际验证相关配置；当前 120 次用于验证连续采集稳定性。

每个预定初态只有一次物理尝试。普通抓取失败会继续完成既定槽位，技术异常和错误初态会停止。
长批次的退出码 0 表示计划执行与数据校验完成，**不意味着 120/120 全部抓取成功**；请用报告核对成功数和原因。
请保留全部失败记录，不在统计中悄悄剔除。

要生成 VLA 输入，长批次结束后用 [LeRobot 导出指南](LEROBOT.md) 将已校验的原始批次转换为双 RGB 数据集。导出是独立的 CPU 步骤，可以先用 smoke 结果验证。

按控制步上限为 120 次预留 120 GB 数据盘空间。早期另一四卡 A800 平台测得约 100 秒/回合；
这是原平台观察值，不是此独立版本对当前机器的速度保证。排期可先预留约 90 分钟，再以实际输出为准。

## 6. 结果与验收标准

每次结果都写到新的 `outputs/<UTC时间戳>/`，或用 `--output /数据盘/新目录` 指定尚不存在的路径。
主要目录如下（worker 名按物体和 lane 变化）：

```text
outputs/<run>/
  PLAN.json / ENVIRONMENT.json / RUN.json
  REPORT.md / SUMMARY.json / EPISODES.json / episodes.csv
  collection/
    workers/<worker>/<case>/
      ENTRY.json / METADATA.json / SCENE.json
      INITIAL_STATE_AUDIT.json / RESET_READBACK.json
      VALIDATION.json / RAW_FORCE_VALIDATION.json
      trajectory.npz
      cameras/base_camera.npy
      cameras/wrist_camera.npy
```

`trajectory.npz` 保存 N 行 `action_targets` 和 N+1 行观测，含关节、物体、TCP、标定、原始 8 子步接触及评分信息。
物理 480Hz、控制和双路 256×256 RGB 为 60Hz，N+1 的最后一行是终态，不存在额外虚构动作。
动作的原始 9 维为 7 个绝对关节位置目标 rad + 左右指目标 m。

抓取成功要求抬升至少 60 mm、双侧指垫有效接触、保持 120 控制步，并满足运动、漂移、夹爪开口和禁止接触限制。
完整阈值见每条 `METADATA.json` 的 `success_contract`，验收通过不能仅凭截图或物体离桌判断。
汇总时同时报告固定槽完成数、抓取成功数、指定初态且验证成功数及完整墙钟吞吐。
采集器直接产出 NPZ / NPY；当前独立源码包没有集成 LeRobot 转换。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 提示 shared paths unavailable | 当前机器没有共享依赖；按 INSTALL 安装，配置 local_env.json |
| `isaaclab` / `isaacsim` 导入失败 | 检查预检使用的 SIM_PYTHON 和共享 Isaac Lab 目录，勿用普通 reason 环境启动 worker |
| 驱动、Vulkan、Kit 或显示错误 | 核对 Isaac Sim 5.1 的系统要求，先无界面预跑；CPU check 不能验证 Vulkan |
| cuRobo / PointNet++ 扩展加载失败 | 核对 torch/CUDA/编译器版本和 GPU 架构；新机器需要对应编译，不能直接复用异构卡二进制 |
| `No GraspGen pool` | 先对相同物体执行 prepare；自定义目录时 prepare 的 --output-root 与 collect 的 --candidate-root 指向同一路径 |
| incomplete/different pool | 保留原目录，选择新的候选目录并给采集器传同一目录；不要自动覆盖或反复抽样直到成功 |
| 首回合初态核查失败 | 停止扩大批次，保留 INITIAL_STATE_AUDIT、RESET_READBACK、ENTRY 和 worker 日志定位 |
| 已有输出目录报错 | 使用新输出路径，避免覆盖旧数据 |
| 远程服务器无法显示 GUI | 使用默认 headless；GUI 需要可用显示服务，且只支持单 worker |
| GitHub 404 或 clone 拒绝访问 | 核对仓库 URL、分支名及本机 GitHub 网络连接；此仓库已公开，无需单独授权 |

## 8. 软件版本与验证边界

| 组件 | 当前使用版本 |
|---|---|
| Python | 3.11.14 |
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.0 |
| 仿真 PyTorch | 2.7.0 + CUDA 12.6 |
| cuRobo / 规划 PyTorch | 0.7.8 / 2.11.0 + CUDA 12.8 |
| GraspGen | 仓库内源码，基于提交 `2dd8852e1be60f5f9d277fafcc621835cdf59110` 及兼容修改 |
| GraspGen 关键依赖 | diffusers 0.11.1、transformers 4.35.2、huggingface-hub 0.25.2、spconv-cu120 2.3.6 |

依赖分开安装；三种 torch / CUDA 组合分别属于不同解释器，不要合并成一个环境。
此独立版本已通过 CPU 可迁移性、资产、环境与权重检查；尚未在全新电脑重装并完成 GPU 端到端验收。
新目录中的新 GraspGen 推理、单卡显存、GUI 以及首回合预热的实际效果，应按第 3～4 节在目标机器验证。
详见 [VALIDATION.md](VALIDATION.md)，第三方来源及许可见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
