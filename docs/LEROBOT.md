# 从双 RGB 抓取采集生成 LeRobot v3 数据

当前默认导出还要求规划前静止证据和 `com_translation_pose_arc_v3` 的完整 120 控制间隔稳持证据，并独立复算导出数据的归一化统计。`collect` 保存原始 NPZ/NPY，`export` 转换已有轨迹；完整 `pipeline` 入口可以依次采集和交付数据。恢复入口与 `--data-only` / `--with-training` 用法见 [RECOVERY.md](RECOVERY.md)。

先按[复现手册](REPRODUCE.md)跑两条 smoke，再运行批量采集。记下采集命令打印的 `Output:` 路径。导出读取通过采集校验的原始批次，不需要启动 Isaac、cuRobo 或 GPU。

使用单独的 Python 3.11 环境安装 `requirements-lerobot.txt` 中的 `lerobot==0.4.4`，不要装进仿真或规划环境。

```bash
python3.11 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install -r requirements-lerobot.txt

# --run 指向采集输出；--output 指向尚不存在的导出目录。
bash run.sh export --run outputs/采集目录 --output outputs/导出目录 \
  --python .venv-lerobot/bin/python
```

也可以设 `LEROBOT_PYTHON` 代替 `--python`。可以用 `--per-object 1` 小批试导。普通采集和 smoke 自动选择唯一阶段；如果来源是多阶段 benchmark，需指定 `--stage parallel3` 等阶段名称。若采集使用自定义 runtime，导出时传同一 `--runtime`。

使用 `--workers 4` 开启并行分片：先对整批运行一次原严格筛选和去重，再按物体分片，最多同时处理 4 包。每个非空物体对应一包，`workers` 控制并发上限，不是分片条数。没有合格示范的物体仍在全局排除清单中，不生成空训练包。每包调用原导出器和审计器，重新从 0 编排回合索引，并完成官方加载器全量读回、终帧、padding 和独立统计。

```bash
bash run.sh export --run outputs/采集目录 --output outputs/分片导出目录 \
  --workers 4 --python .venv-lerobot/bin/python
```

分片模式的 `portable/SHARDS.json` 保存相对包路径、逐包选择/导出/审计回执哈希和总条数；父目录 `AUDIT.json` 只有在各包均通过且合集恰好等于整批选择结果时才发布。数据路径为 `portable/shards/<物体>/dataset`；终帧及物理证据在各包的 `sidecars/`。整体移动 portable 不影响内部路径。导出失败保留日志和部分文件；重跑使用新目录。pipeline 已封存的完整导出可恢复复用，尚未完成的后处理重新执行，不自动复用未封存的单个分片。

四物体一起训练时使用 `MultiLeRobotDataset`，并向官方预处理器传入 `data.stats`，不要只使用其中一包的统计：

```python
import json
from pathlib import Path
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

portable = Path('/absolute/path/to/portable')
manifest = json.loads((portable / 'SHARDS.json').read_text())
data = MultiLeRobotDataset(
    repo_ids=[f"{s['object_id']}/dataset" for s in manifest['shards']],
    root=portable / 'shards',
    delta_timestamps={'action': [i / 60 for i in range(16)]},
)
stats = data.stats
```

项目 `train_check.py --portable <portable> --audit <AUDIT.json> --output <新训练目录>` 自动识别单包或分片。`pipeline --export-workers 4` 也会接入联合读取；默认仍为单包。pipeline 的导出并发登记后固定，恢复沿用冻结配置。训练检查默认使用 CUDA，单独调试可用 `--device cpu`。

命令按固定计划检查每个槽位：要求物理抓取成功、两种原始数据校验通过、在线初态校验通过、规划前静止和完整稳持证据通过，且从原始记录重算的 PhysX 和缓存初态与计划相符。选择时按实测物体位姿和关节位置去重；失败或排除的槽仍列在 `SELECTION.json` 中。未启用原始校验的批次不能导出。历史记录缺少当前证据时不会自动获得资格；`--allow-legacy-unverified` 只供历史格式调试，不能通过当前 ACT 检查。

退出码为 0 时，默认单包训练数据在 `portable/dataset`，`AUDIT.json` 为 `passed`。读回会核对每行动作、状态、任务和时序，以及双路图像的逐像素一致性、终帧和回合边界。单包的完整原始轨迹、校验报告和终帧保存在 `portable/sidecars`；`EXPORT_MANIFEST.json` 记录来源、格式和字段映射。训练只需数据集目录；重新对照原始记录需保留原始采集批次。原始批次、导出数据和缓存都需要磁盘空间，建议先用小批估算。命令失败时查看输出目录里的阶段日志，重跑用新目录。

| 训练字段 | 含义 |
| --- | --- |
| `observation.images.base_camera`、`observation.images.wrist_camera` | 双路 256×256 RGB；无损 PNG 内嵌 Parquet |
| `observation.state` | 7 个实测臂关节角（rad）和两指总开口（m） |
| `action` | 7 个绝对关节位置目标（rad）和目标总开口（m） |

数据频率是 60 Hz。N 个动作与前 N 个观测配对；额外终态保存在附件中。训练配置只将双图像和 `observation.state` 作为输入，用 `action` 作目标；`aux.*`、`next.done` 和 `episode_success` 用于审核或表示回合边界。

历史参考批次使用四卡、四类物体、每类 30 个固定初态，采集和原始校验约 77 分钟；当时导出的 119 条轨迹通过格式读回（99,174 行、198,348 张训练图像）。后续可信度审计发现其中 20 条圆盘只有 119/120 个合格稳持间隔，全部旧记录也缺少当前规划前静止窗口；该旧包不能据格式通过获得当前静态训练资格，原标签和原件保留。

当前四物体验收：P1 正式导出 197 条 / 181,794 行，P2 八卡正式导出 125 条 / 71,302 行；均通过严格筛选、官方完整读回、独立统计和 ACT100。ACT 检查证明训练链路能消费数据，未评测闭环抓取能力。验证范围和失败说明见 [VALIDATION.md](VALIDATION.md)。
