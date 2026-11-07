# 代码入口与扩展

| 路径 | 用途 |
|---|---|
| `cli.py` / `run.sh` | 选择准备、采集、预检、测速入口及对应解释器 |
| `generate_candidates.py` | mesh 均匀采样、GraspGen 单次生成与几何筛选 |
| `collect.py` / `common.py` | 固定初态计划、进程调度、共享规划服务、输出与退出处理 |
| `sim_worker.py` / `initial_state.py` | 常驻 Isaac 环境、连续 reset、独立 PhysX 初态核查 |
| `planner_service.py` | 本机回环地址上的共享规划服务；每次请求串行更新场景 |
| `runtime/collector/episode_factory.py` | 接近、下降、闭合、抬升、保持、双相机与状态保存 |
| `runtime/collector/fast_env_v2.py` | 当前 I/O 优化与完整 8 子步接触记录 |
| `runtime/collector/generic_planner.py` | cuRobo 与完整目标、桌面、夹爪、机械臂检查 |
| `runtime/geometry/` | TCP 适配、URDF FK、实体几何与连续路径检查 |
| `runtime/collector/*score.py` / `legacy_scoring.py` | 当前成功判据及保留的辅助遥测 |
| `runtime/validators/` | 时序、RGB、评分重放及原始接触力校验 |
| `runtime/config/scene.json` | 相机标定、初始关节、物体参考位置 |
| `runtime/assets/` / `runtime/robot/` | 物体、Panda USD / URDF 与碰撞模型 |

## 坐标和执行约定

全部 mesh 为米，列向量齐次变换。模型输入点云减去均值，生成后仅恢复一次。固定夹爪 adapter 由 `geometry.robot.GRASPGEN_TO_TCP` 提供，转换为 Isaac Panda TCP 时仅应用一次。在线目标为 `T_world_object @ T_object_tcp`。四物品先各生成一份本地候选，同物品的新初态使用实时仿真物体位姿变换候选。

候选保持原预算：每物品 200 个原始候选，场景几何通过后最多 50 个，完整规划最多 5 个，每候选 30 秒软预算。普通失败有明确原因；不按抓取结果重新抽样。机器人自碰撞由 cuRobo 约束命令路径，PhysX 的 Panda 自碰撞设置沿用现有版本。完整物体、桌面、接触与抬升判据沿用当前实现。

默认环境为桌面和一个目标。旧环境中预留的远处 wall 与遮挡辅助代码随环境实现保留，但当前入口没有加入新障碍物。增加障碍物必须同步仿真、规划世界、独立几何检查及接触评分，不能只加可视模型。

## 增加物品

当前没有任意 OBJ 自动导入并证明物理几何一致的入口。扩展时需要：

1. 准备同一局部坐标系、米单位的 `physics.usda`、`surface.obj` 和表面点/法向/面标记 `surface.npz`。
2. 在 `runtime/assets/MANIFEST.json` 注册相对路径、质量/摩擦、bounds、size、完整 box/cylinder 部件分解和候选 seed。
3. 在 `common.py` 的 `OBJECTS` 与 `REFERENCE` 注册物体和初态。支持的几何分解以当前 `geometry.primitives` 为准，不能用有孔物体的凸包替代完整分解。
4. 为新物体生成 GraspGen 候选，执行初态/碰撞/接触和数据完整性回归。

本版本交付范围与当前四物品采集器一致。新物体、额外障碍或不同机器人需要相应开发和实际验证。
