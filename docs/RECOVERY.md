# 固定槽位恢复与本地数据交付

新入口为每个预定槽位保存 `raw/SLOTS.json`。普通 `collect` 和 `pipeline` 使用恢复协议；旧 `--benchmark`、`--smoke` 入口保持原调度。旧验收目录不带该协议，不能直接使用新入口恢复，也不会迁移或重写旧标签。

当前数据交付使用 `pipeline --data-only`，结束于严格导出、官方完整读回及独立统计验收。省略此参数的新批次仍兼容历史 ACT 检查流程；需要时可显式用 `--with-training`。进程与结果文件索引见 [Isaac 数采管线](ISAAC_PIPELINE.md)。

```bash
# 四物体各一条，使用已有加速档位，完成严格数据交付。
bash run.sh pipeline --data-only --output /absolute/new-run --gpus 0,1 --workers 1 \
  --episodes-per-object 1 --arm-speed-profile recommended --candidate-root /absolute/candidates \
  --lerobot-python /path/to/lerobot/python

# 使用原来冻结的代码、配置和槽位，继续同一批次。
bash run.sh pipeline --data-only --output /absolute/new-run --resume

# 只交付经过完整读回和独立统计复算的数据包。
bash run.sh pipeline --output /absolute/new-data-run --data-only \
  --candidate-root /absolute/candidates --lerobot-python /path/to/lerobot/python

# 新批次按物体并行生成标准包，每包完整审核；训练检查联合读取各包。
bash run.sh pipeline --output /absolute/new-sharded-run --export-workers 4 --data-only \
  --candidate-root /absolute/candidates --lerobot-python /path/to/lerobot/python

# 已通过数据交付的批次，之后单独补ACT检查。
bash run.sh pipeline --output /absolute/new-data-run --resume --with-training

# 八卡加速：每卡一个仿真及专属规划器，四物体各32条。
bash run.sh pipeline --data-only --output /absolute/new-eight-gpu-run \
  --gpus 0,1,2,3,4,5,6,7 --workers 8 --episodes-per-object 32 \
  --execution-profile accelerated --arm-speed-profile recommended \
  --candidate-root /absolute/candidates --lerobot-python /path/to/lerobot/python
```

新 `pipeline` 批次默认使用已有 `recommended` 加速档位：接近 1.0 rad/s、持物抬升 0.3 rad/s，路径进度加速度分别为 5.0、2.0 rad/s²；下降、离桌、闭合、保持及质量规则沿用原实现。速度选择登记在 `PIPELINE.json` 并传给冻结采集子进程。需要原速对照时显式使用 `--arm-speed-profile baseline`。P2四物体正式复测125/128成功，热身后含验证7.650秒/成功条，完整采集9.558秒/成功条，含导出与完整读回的数据活跃成本41.13秒/合格条；有限固定设计及细长物体135°失败见[历史版本记录](https://github.com/juchuanqi/graspgen-curobo-collect/blob/73c6c31b2591dc6488608cee2f99c9c4566e1f18/README.md)。历史提速验收及圆盘失败记录继续保留。

恢复时省略原采集参数即可；如再次传入，则必须与登记值一致，包括速度档位。旧批次继续使用其冻结源码和原速度，不受新默认值影响。允许切换是否执行训练检查及本次调度的检查点限制。GPU和并发配置固定，调整资源需要另建批次。独立 `collect --resume` 需传原计划参数，包括物体、槽位数、种子、速度及候选路径，并指定原 `--output`。

八卡配置已接入恢复/交付入口：最多八路，每卡一个仿真及其专属规划器；四卡仍为最多三路共享规划器。规划服务并发启动，全部就绪后才派发物理槽位。映射登记在 `SLOTS.json` 的 `configuration.device_layout`，恢复拒绝改动。

新任务的 `--execution-profile accelerated` 复用已有传感器、接触/执行器 Graph、相机调度和闭合/抬升重叠模块；`--arm-speed-profile recommended` 独立控制关节提速。原执行和原速度对照需分别显式选 `baseline`。`accelerated` 下，两个CPU消费者验证已保存记录，四条可等待；接受回执位于轨迹目录外，原始记录在保存后保持不变。每个物体每路首条是热身，全部首条完成验证后打开公共测量门禁。门禁后的采集/验证窗口及含冷启动、热身的完整成本分别报告，当前任务是否接近约8秒/条由实测决定。

## 恢复规则

槽位区分待启动、初始化、规划、执行、校验、成功、规划失败、物理失败、初始化不合格、基础设施故障、数据不合格和中断未知。账本先登记启动，再创建子进程；worker 在任何单槽工作开始前先持久保存槽位目录及状态。工作进程、规划服务、验证器日志和每次启动记录均保留。

仅 worker 明确捕获的、发生在 READY 和所有单槽工作之前的初始化显存资源错误可以自动重启。每槽默认最多两次额外启动，可通过 `--initialization-retries 0/1/2` 收紧。普通导入错误、初始化超时、任意无法证明安全的异常均不作为该类重试。规划服务的资源故障重启只发生在派发槽位之前，也有同样的次数上限。

已成功、规划失败或物理失败的槽位均为终态，不为凑成功数补采。工作进程退出后，如果协议能确认后续槽位尚未进入，可继续这些未执行槽位；进入过槽位而没有完整保存标记的，标为 `interrupted_unknown` 并停止自动恢复。重新执行主命令不会清除这个状态。应人工检查现场，任何新物理实验另登记新槽位，不覆盖原证据。

每条完整记录先固定原始文件哈希，再运行状态/动作/RGB和原始子步力验证。恢复时复用校验结果之前核对这些哈希。数据验证失败也不触发重新抓取。运行开始及恢复时核对源码、配置、资产和候选；pipeline 恢复始终执行原 `source/` 快照，当前主目录后续编辑不会改变该批实现。

`flock` 防止同一目录被两个监督进程同时使用。Linux 子进程继承所有权并设置父进程死亡信号；监督进程遭 SIGKILL 后，其直接子进程终止，各级同样处理。子进程退出前锁保持占用，不允许恢复产生重叠执行。无需扫描或终止其他任务进程。

## 阶段与数据包

原始记录固定保存在 `raw/`，全部预定槽位有可审计去向后写入 `RAW_COMMITTED.json`。导出和 ACT 在 `work/` 中按次数创建新目录，失败或中断目录保留。只有严格筛选、官方完整读回和独立统计复算均通过，才将导出目录原子改名为 `export/`；训练检查通过后独立发布 `training/`。已发布产物带 `COMMITTED.json`，恢复时核对全部封存文件，拒绝变化。

导出或读回中断可重新从完整原始记录执行后处理；已完成的数据包直接复用。训练失败只重跑训练，不重新采集或重新导出。`PIPELINE.json` 分开保存 `data_status` 和 `training_status`；训练失败时已通过的数据交付仍有效。导出目录或源码受损时停止，不将它替换成另一份结果。

默认训练数据位于 `export/portable/dataset`。使用 `--export-workers 4` 时，数据位于 `export/portable/shards/<物体>/dataset`，`SHARDS.json` 绑定全部逐包审计；ACT 检查自动使用官方多数据集加载器及合并统计。仿真 `--workers` 与导出 `--export-workers` 分别登记，恢复不可更改。整批导出完成后照常原子发布和封存；失败分片不提前发布，后处理恢复从原始数据重新执行。关节顺序、单位、总开口、60 Hz 时间、双相机、动作对齐、末观测和 padding 见每个新包的 `export/portable/DATA_USAGE.md`。重新做原始数据独立审计还需要源记录，训练读取无需仿真环境。

时间累计各次监督进程互不重叠的实际运行区间，包含该入口的逐槽校验和退出开销；停机等待另列。SIGKILL 等缺少结束回执的区间不伪造时长：累计值标为下界、`timing_complete=false`，对应报告不输出吞吐率。恢复实验不是性能对照。
## 显式初态与分批检查点

`coverage_plan.py --output PLAN.json` 生成四物体各 50 槽的分层初态。使用
`tools/pipeline.py --plan-json PLAN.json --episodes-per-object 50 --seed 2026111510 --workers 3`
执行；其他 GPU、候选和解释器参数与普通入口相同。管线复制并核验固定清单。
`--stop-after-per-object 10` 只执行每物体前 10 槽，完整 200 槽仍保存在账本，
返回 `checkpoint`，此时不导出或训练。检查首批后使用同一输出目录 `--resume`
且省略该选项，继续剩余槽位。检查点限制只控制本次调度，不修改初态或终态记录。
已经完成的物理失败也不会重跑。显式计划不能改变物理频率、尝试预算或现有 XY/yaw 支持范围。


固定墙参数也随运行登记：`--obstacle wall` 同时传给冻结采集器和 worker，恢复不能切换。采集保存及独立导出均核对计划与场景几何，拒收静默降级为无墙的数据；旧批次保持原冻结源码。见 [固定墙说明](OBSTACLES.md)。
