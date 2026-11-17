# 固定 256 初态抓取采集复验（2026-11-17）

组合方案取得 **231/256（90.2%）** 合格回合，匹配基线为 168/256（65.6%）。物理执行 231 次，其中 231 次成功；规划失败 25 次、物理失败 0 次。

| 初态条件 | 方块 | 细长条 | 圆柱 | 圆盘 | 合计 |
|---|---:|---:|---:|---:|---:|
| 近侧／无墙 | 16/16（原 16/16） | 16/16（原 16/16） | 16/16（原 16/16） | 16/16（原 16/16） | 64/64 |
| 近侧／带墙 | 12/16（原 8/16） | 8/16（原 4/16） | 15/16（原 9/16） | 16/16（原 0/16） | 51/64 |
| 对侧／无墙 | 16/16（原 16/16） | 16/16（原 16/16） | 16/16（原 16/16） | 16/16（原 16/16） | 64/64 |
| 对侧／带墙 | 10/16（原 11/16） | 12/16（原 4/16） | 14/16（原 4/16） | 16/16（原 0/16） | 52/64 |

每格括号内为同一批实际初态的旧结果。每种物体、布局使用 16 个固定分层 XY 初态；长方体含四档 yaw，轴对称物体 yaw 为零。全部实际初态逐回合核对一致；未补抓、替换初态或放宽判据。

逐初态配对：原失败转成功 67 次，原成功转失败 4 次。对侧带墙方块从 11/16 降至 10/16，不能将整体提升理解为每个条件都改善。

## 方案及候选来源

闭合、抬升和保持启用 3 N 平均接触力反馈，沿用原生 PD；完整抓取终点 IK 和整臂几何预筛，优先尝试接触法向对齐较好的候选。三项均默认关闭，需在冻结的场景配置中显式开启。

**方块和细长条保持原始 GraspGen 缓存；圆柱和圆盘各使用独立的 200 姿态几何先验替换源池。**后两者没有模型推理或置信度，模型候选数记为零，不能作为纯 GraspGen 改进。多项干预同时变化，结果只能归于组合方案。

源池 200、场景候选 50、完整规划最多 5 个、每候选 30 秒软预算不变；额外终点预处理耗时另记。原生夹指对外碰撞保持开启；PhysX 机器人自碰撞沿用关闭，cuRobo 继续约束命令自碰撞。完整几何扫掠、1 mm 穿透上限、原始力及 480 Hz 位姿运动检查、120 控制周期（2 秒）稳定保持均保留。

## 数据验收与成本

- 231 条成功回合全部完成严格 LeRobot v3 导出、完整 RGB/Parquet/统计读回，共 **140,418 行**。
- 独立重建 **73,459 个**反馈控制命令，核对保存源码哈希、上一帧力、动作及实际下发目标；逐回合核对保持证据及全部源码/原始/导出封存。
- 全部回合禁止接触力峰值为 **0 N**；失败原因计数：`no_complete_candidate_within_budget` 9、`no_reachable_grasp_endpoint` 8、`no_scene_geometry_candidate` 8。
- 从采集启动到严格导出读回结束共 **3789.9 秒（63.2 分钟）**，每条合格回合 16.41 秒；基线为 3411.3 秒。包含初始化与失败成本，不含候选准备、独立汇总审计及视频编码。各条件导出可与下一条件采集重叠，一次本机计时不能代表生产吞吐。
- 104 项 CPU 回归和 4 项实际 GPU 回归通过；CPU 报告中另跳过 4 项 GPU 项及 1 项可选 LeRobot 单测，真实数据交付由专用 LeRobot 环境完成。

## 未采用方案和适用边界

18 次固定动作诊断表明减小挤压可缓解方块穿透和圆柱抖动，无法解决曲面偏边滑落；诊断不计正式成功。首轮 32 回合仅 23/32，含两次圆柱滑落和一次 3.4105215 N 非橡胶接触超限，原方案整批门槛未通过；原失败记录保留。两种模型候选平移细化探针未得到完整规划，未采用。改用圆柱几何先验后，另登记的 8 回合实验取得 8/8、禁止接触力 0 N，随后重新登记本轮 256；没有替换首轮失败后重算 23/32。

本轮仅覆盖四种已知几何、原生仿真位姿、一种固定墙和两种布局。规划失败仍属于失败；不能据此声称任意障碍、未知物体泛化或实际机器人成功率。范围结束于抓取采集、后处理和数据交付。

## 完整证据（共享盘绝对路径）

- 最终报告：`/mnt/public/juchuanqi/project/results/graspgen_quality_full256_20261117_02/REPORT.md`
- 实验登记：`/mnt/public/juchuanqi/project/results/graspgen_quality_full256_20261117_02/PREREGISTRATION.json`
- 汇总与逐回合数据：`/mnt/public/juchuanqi/project/results/graspgen_quality_full256_20261117_02/RESULT.json`、`/mnt/public/juchuanqi/project/results/graspgen_quality_full256_20261117_02/EPISODES.json`
- 溯源说明：冻结计划中沿用的 `scope` 描述仍写有“other3 keep official GraspGen sources”，是过时的说明文字；本轮登记、候选文件哈希及逐回合 `candidate_source_kind` 均明确圆柱和圆盘为几何先验。原封存文件保留，按后者解释来源。
- 文件哈希和最终验收：`/mnt/public/juchuanqi/project/results/graspgen_quality_full256_20261117_02/ARTIFACT_SHA256.json`、`/mnt/public/juchuanqi/project/results/graspgen_quality_full256_20261117_02/FINAL_VERIFICATION.json`
- 视频清单：`/mnt/public/juchuanqi/project/results/graspgen_quality_full256_20261117_02/preview/PREVIEW.json`，每条含完整绝对路径；全部帧完成解码核对，代表性回合另经 Codex 图像检查。
- 失败先导：`/mnt/public/juchuanqi/project/results/graspgen_quality_targeted32_20261117_01`
- 独立圆柱实验：`/mnt/public/juchuanqi/project/results/graspgen_cylinder_prior8_20261117_01`
- 基线：`/mnt/public/juchuanqi/project/results/graspgen_execution_clearance_full_20261117_01`
