# 代码入口与扩展

当前 Isaac 管线的进程分工、回合时序、结果文件和故障定位见 [ISAAC_PIPELINE.md](ISAAC_PIPELINE.md)；以下按源码模块索引。

| 路径 | 用途 |
|---|---|
| `cli.py` / `run.sh` | 选择准备、采集、预检、测速、导出和完整数据管线入口及对应解释器 |
| `generate_candidates.py` | mesh 均匀采样、GraspGen 单次生成与几何筛选 |
| `collect.py` / `common.py` | 固定初态计划、GPU 映射、进程调度、输出与退出处理 |
| `resumable_collect.py` / `recovery.py` | 逐槽账本、有限初始化恢复、互斥、指纹核验与有界 CPU 验证 |
| `coverage_plan.py` | 四物体分层初态清单及每物体前缀检查点 |
| `initial_geometry.py` / `tools/check_initial_geometry.py` | 登记前在独立 CPU 进程检查整批生成/落稳几何；任一不合格即拒绝整批 |
| `sim_worker.py` / `initial_state.py` | 常驻 Isaac 环境、连续 reset、独立 PhysX 初态核查 |
| `planner_service.py` | 本机回环规划服务；四卡最多三路共享，八卡每路同卡专属，每个服务串行更新自身场景 |
| `runtime/collector/episode_factory.py` | 接近、下降、闭合、抬升、保持、双相机与状态保存 |
| `runtime/collector/grip_control.py` | 可选、受限的对称夹指接触力反馈，实际命令在反馈后记录 |
| `runtime/collector/fast_env_v2.py` | 基础 I/O 优化与完整 8 子步接触记录 |
| `runtime/collector/fast_env_v3.py` 至 `fast_env_v6.py` | 逐层接入传感器环形调度、接触 Graph、相机调度和执行器 Graph；普通采集默认 accelerated 使用 v6 |
| `runtime/collector/overlap.py` | 闭合状态几何检查与 lift RPC 并行，二者通过后才允许执行 |
| `runtime/collector/generic_planner.py` | cuRobo 与完整目标、桌面、夹爪、机械臂检查 |
| `runtime/collector/target_provider.py` | 官方候选来源核验、TCP 世界变换、5 cm 接近和场景几何预筛 |
| `runtime/collector/planning_clearance.py` | 规划路径的桌面 1 mm、障碍物 6 mm 余量，由 cuRobo 世界和完整几何扫掠共同使用 |
| `runtime/collector/arm_speed.py` | 可选分阶段速度、路径重定时、记录配置及离线动作上限检查 |
| `runtime/collector/grasp_symmetry.py` | TCP 局部 Z 轴 180° 配对、当前世界朝向夹角、近向/远向分组及来源记录 |
| `runtime/geometry/` | TCP 适配、URDF FK、实体几何与连续路径检查 |
| `runtime/collector/*score.py` / `legacy_scoring.py` | 当前成功判据及保留的辅助遥测 |
| `runtime/validators/` | 时序、RGB、评分重放及原始接触力校验 |
| `tools/audit_grip_feedback.py` | 用回合保存的控制器源码和上一帧接触力，独立核对反馈动作及实际下发目标 |
| `runtime/config/scene.json` | 相机标定、初始关节、物体参考位置 |
| `runtime/assets/` / `runtime/robot/` | 物体、Panda USD / URDF 与碰撞模型 |
| `tools/pipeline.py` / `tools/recoverable_pipeline.py` | 冻结源码、采集、严格导出/读回及可选 ACT 检查；后处理阶段可恢复与复用 |
| `tools/guarded_process.py` | Linux 父进程死亡保护与继承锁，防止恢复时重叠执行 |
| `tools/lerobot_export/sharded_export.py` | 全局筛选后按物体并行导出与完整读回，验证分片合集 |
| `tools/lerobot_export/delivery.py` | 分片回执与相对路径验证、单包/多包统一训练加载和统计 |

## 坐标和执行约定

全部 mesh 为米，列向量齐次变换。模型输入点云减去均值，生成后仅恢复一次。固定夹爪 adapter 由 `geometry.robot.GRASPGEN_TO_TCP` 提供，转换为 Isaac Panda TCP 时仅应用一次。在线目标为 `T_world_object @ T_object_tcp`。四物品先各生成一份本地候选，同物品的新初态使用实时仿真物体位姿变换候选。

候选仅来自 GraspGen 模型，默认每物品 200 个，可显式生成并接入 500／1000 个，见 [候选数量说明](CANDIDATE_COUNT.md)。场景几何通过后最多 50 个，完整规划最多 5 个，每候选 30 秒软预算。普通失败有明确原因；不按抓取结果重新抽样。机器人自碰撞由 cuRobo 约束命令路径，PhysX 的 Panda 自碰撞设置沿用现有版本。完整物体、桌面、接触与抬升判据沿用当前实现。

可选 `--grasp-symmetry near-first` 在 `target_provider.screen_scene_candidates` 内、独立场景几何检查前生成对称表示。N 个输入得到 N 近向和 N 远向；每组各最多 50 个场景合格候选（总计最多 100），两个组共同使用 5 个完整规划预算。`generic_planner.select` 的近向组优先级在 IK 关节路程排序后仍保留，IK 或完整路径失败时在预算内回退远向组。夹角从本次请求的当前实测 TCP 与物体世界位姿计算；固定模型 adapter 不重复应用。翻转表示不继承父位姿的置信度或独立几何验收，两个版本均重新检查。默认关闭。

默认环境为桌面和一个目标，`--obstacle wall` 启用统一定义的固定墙；默认 `none`。整批初态预检在登记与 GPU 启动前完成，实际落稳后的位姿、静止及几何核查仍保留。预检沿用既有几何标准，规划路径另使用执行余量。增加障碍物必须同步仿真、规划世界、独立几何检查及接触评分，不能只加可视模型。

可选夹持力反馈、完整抓取终点预筛及曲面接触排序见 [夹持质量实验](GRASP_QUALITY.md)。默认配置不启用这些实验设置；候选来源、额外预处理时间和物理验收需要分别记录。手写几何候选生成器已移除，新采集拒绝加载历史手写候选池。

## 增加物品

当前没有任意 OBJ 自动导入并证明物理几何一致的入口。扩展时需要：

1. 准备同一局部坐标系、米单位的 `physics.usda`、`surface.obj` 和表面点/法向/面标记 `surface.npz`。
2. 在 `runtime/assets/MANIFEST.json` 注册相对路径、质量/摩擦、bounds、size、完整 box/cylinder 部件分解和候选 seed。
3. 在 `common.py` 的 `OBJECTS` 与 `REFERENCE` 注册物体和初态。支持的几何分解以当前 `geometry.primitives` 为准，不能用有孔物体的凸包替代完整分解。
4. 为新物体生成 GraspGen 候选，执行初态/碰撞/接触和数据完整性回归。

本版本交付范围为四物品抓取采集、后处理校验与数据交付；`pipeline --data-only` 结束于数据验收。新物体、额外障碍或不同机器人需要相应开发和实际验证。

当前配置与物理证据见 [BASELINE.md](BASELINE.md)。CPU 开发检查使用 `requirements-test.txt` 和 `python -B -m unittest discover -s tests -v`，无需安装 GPU 环境；依赖 cuRobo 或 LeRobot 的测试在对应环境运行。PR 的 CPU 工作流同时检查资产、shell 语法及源码打包，不替代物理 smoke。

`runtime/collector/linear_motion_score.py` 统一采集与重放使用的 COM 平移/转角评分；`static_state.py` 校验原有落稳过程中的末尾 0.5 秒证据。`tools/lerobot_export/` 负责默认严格筛选、原生 v3 导出、完整读取及 ACT 训练检查；`tools/pipeline.py` 串起本地验收各阶段并保存阶段日志。

固定墙几何由 `runtime/collector/obstacle_scene.py` 统一提供给仿真生成、规划场景及数据接收核对。`tools/recoverable_pipeline.py` 冻结并传递障碍选择，`resumable_collect.py` 接收 raw 和 `select_sources.py` 筛选时分别核对。新增布局需同时扩展这些证据约定，详见 [固定墙说明](OBSTACLES.md)。
