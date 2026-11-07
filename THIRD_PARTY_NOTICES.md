# 依赖和资产来源

- GraspGen：NVlabs/GraspGen，基于提交 `2dd8852e1be60f5f9d277fafcc621835cdf59110` 及当前工作区兼容修改。仅随附 Python/CUDA 源码、配置、Panda 相关模型资产，排除编译产物、训练数据、缓存与权重。许可原文见 `third_party/GraspGen/LICENSE`、`LICENSE_ASSETS.md` 及 PointNet++ 源文件声明。
- 官方 Panda 模型：`adithyamurali/GraspGenModels`，revision 与文件 SHA256 在 `runtime/config/models.json`。权重通过本机已有文件或下载工具取得。
- cuRobo：NVlabs/curobo v0.7.8，作为外部依赖安装，不随本库重新分发其整个资产库。
- Isaac Sim 5.1 / Isaac Lab 2.3：作为外部运行环境，按其安装和许可要求使用。
- `runtime/robot/`：现有部署采用的 ManiSkill Panda v2 URDF、碰撞/可视网格及其 Isaac 转换 USD，保持关节和 TCP 几何一致。没有附带历史演示数据。
- 四物品 USD、mesh、表面采样资产：由现有项目的参数化 box/cylinder 定义生成，作为运行资产随代码提供。

新采集代码没有另行改变以上第三方依赖与资产的许可。使用和再分发对应文件时保留原版权及许可声明。
