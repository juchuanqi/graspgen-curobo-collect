# 从双 RGB 抓取采集生成 LeRobot v3 数据

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

命令按固定计划检查每个槽位：要求物理抓取成功、两种原始数据校验通过、在线初态校验通过，且从原始记录重算的 PhysX 和缓存初态与计划相符。选择时按实测物体位姿和关节位置去重；失败或排除的槽仍列在 `SELECTION.json` 中。未启用原始校验的批次不能导出。

退出码为 0 时，训练数据在 `portable/dataset`，`AUDIT.json` 为 `passed`。读回会核对每行动作、状态、任务和时序，以及双路图像的逐像素一致性、终帧和回合边界。完整的原始轨迹、校验报告和终帧保存在 `portable/sidecars`；`EXPORT_MANIFEST.json` 记录来源、格式和字段映射。训练只需数据集目录；重新对照原始记录需保留原始采集批次。原始批次、导出数据和缓存都需要磁盘空间，建议先用小批估算。命令失败时查看输出目录里的阶段日志，重跑用新目录。

| 训练字段 | 含义 |
| --- | --- |
| `observation.images.base_camera`、`observation.images.wrist_camera` | 双路 256×256 RGB；无损 PNG 内嵌 Parquet |
| `observation.state` | 7 个实测臂关节角（rad）和两指总开口（m） |
| `action` | 7 个绝对关节位置目标（rad）和目标总开口（m） |

数据频率是 60 Hz。N 个动作与前 N 个观测配对；额外终态保存在附件中。训练配置只将双图像和 `observation.state` 作为输入，用 `action` 作目标；`aux.*`、`next.done` 和 `episode_success` 用于审核或表示回合边界。

原平台参考批次使用四卡、四类物体、每类 30 个固定初态，采集和原始校验约 77 分钟，119 条合格轨迹导出后通过官方加载器全量读回（99,174 行、198,348 张训练图像）。该耗时不含导出或 LeRobot 读回；结果用于验证现有场景的长批采集和数据格式。
