# B3 EEF 数据转换与训练

当前 B3 训练使用 EEF observation.state 和 EEF action，二者均为 16 维。
训练数据适配类为 `LeRobotB3EEFDataConfig`。

## 数据格式转换

`convert_lerobot_v30_to_v21.py` 将本地 LeRobot v3.0 数据转换为当前 openpi
依赖支持的 v2.1 格式，不需要安装新版 LeRobot。输入数据不会修改。
已有 MRUI 导出的 EEF v2.1 数据时，可以直接使用，无需再次转换。

以下示例使用训练机路径；在其他机器上运行时，替换为实际数据目录。
在 `openpi` 目录执行：

```bash
uv run examples/B3/convert_lerobot_v30_to_v21.py \
  --data-dir /dataset-v2/haowenyan/project/data/MRUI-Dataset/stack_blocks_latest/data_v5_eef_lerobot_v30 \
  --repo-id local/stack_blocks_v5_eef_lerobot_v21
```

默认输出到 `HF_LEROBOT_HOME / local/stack_blocks_v5_eef_lerobot_v21`。
可添加 `--output-dir /其他输出目录` 指定位置；此时需要另行配置训练读取位置。
建议先加 `--max-episodes 1 --repo-id local/stack_blocks_v5_eef_v21_trial` 试转一个
episode，再使用不同输出目录进行完整转换。输出路径必须尚不存在，脚本不会
覆盖、删除或续写已有目录；失败后请将不完整目录移走，再重新运行。

转换保留三路相机的字段名和分辨率、state/action 的字段定义与数值、源数据 FPS 和任务
文本。这里的 v5 EEF 数据为 16 维 state、16 维 action、50 FPS。
按 v3 元数据中的视频时间偏移读取画面，重新生成每个 episode 的视频、parquet、任务表和
v2.1 统计量。视频会重新编码，像素可能有压缩差异，且需要额外磁盘空间和转换时间。
state/action 不做差分、归一化或坐标系转换。

## π₀.₅ EEF 训练配置

当前注册的 B3 配置均使用 `LeRobotB3EEFDataConfig`：

| 任务 | 配置名 | 数据集 repo_id |
| --- | --- | --- |
| 叠方块 | `pi05_b3_stack_blocks_v5_eef` | `local/stack_blocks_v5_eef_lerobot_v21` |
| 倒茶 | `pi05_b3_pour_tea_eef` | `local/pour_tea_eef_lerobot_v21` |

两项配置使用 `B3EEFInputs` / `B3EEFOutputs`。state 和 action 都是 16 维，
左右臂各占 8 维：`[x, y, z, qx, qy, qz, qw, gripper_open_fraction]`，位置单位为米。
保留绝对动作，不对四元数执行逐元素 `DeltaActions`。数据集的 `action` 字段用于读取
连续动作，随后改名为模型使用的 `actions`；任务表生成 `prompt`。

三路图像映射如下，均设置为有效图像：

| LeRobot 字段 | 模型图像槽位 |
| --- | --- |
| `observation.images.front_camera` | `base_0_rgb` |
| `observation.images.left_wrist_camera` | `left_wrist_0_rgb` |
| `observation.images.right_wrist_camera` | `right_wrist_0_rgb` |

CHW 浮点图像转为 HWC uint8，再由模型通用 transforms 调整到 224×224。
state/action 使用对应数据集的统计量归一化，再补齐到模型的 32 维；
推理输出在反归一化后裁回 16 维。`action_horizon=50`，数据为 50 Hz，
因此每个动作块覆盖约 1 秒。坐标系、四元数顺序和夹爪语义需要与执行端一致。

## 计算统计量与训练

以下以叠方块为例；倒茶使用对应配置名和数据集。
训练通过 `HF_LEROBOT_HOME / repo_id` 查找数据。例如，下面的训练机配置要求
`/dataset-v2/haowenyan/project/data/lerobot/local/stack_blocks_v5_eef_lerobot_v21`
包含完整数据集，或为指向数据目录的软链接。
每个训练或计算统计量的 shell 都需要设置 `HF_LEROBOT_HOME`。

下例使用训练机路径；在其他机器上运行时，需要同时调整数据路径和
`src/openpi/training/config.py` 中对应配置的 `weight_loader` 路径。

```bash
cd /dataset-v2/haowenyan/project/openpi
export HF_LEROBOT_HOME=/dataset-v2/haowenyan/project/data/lerobot
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

# 确认数据转换/复制完成：v5 EEF 叠方块数据应有 103 个 parquet 和每路相机各 103 个 mp4。
# 为当前数据集单独计算 EEF state/action 的统计量。
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_b3_stack_blocks_v5_eef

# 当前配置为 JAX 训练，加载配置中指定的 pi05_base/params。
# batch size / FSDP 设备数需按实际 GPU 资源设置。
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py \
  pi05_b3_stack_blocks_v5_eef --exp-name b3_v5_eef \
  --no-wandb-enabled --no-swanlab-enabled
```

统计量应生成在
`assets/pi05_b3_stack_blocks_v5_eef/local/stack_blocks_v5_eef_lerobot_v21/norm_stats.json`，
包含 16 维 `state` 和 `actions` 的均值、标准差及分位数。pi05 使用分位数归一化。
上述命令关闭 W&B 和 SwanLab 在线日志；需要时启用对应配置选项。

训练前还应核对四元数符号和夹爪语义。此前对 v5 EEF 叠方块数据 150,169 帧的检查发现，
左臂 state 有 30 处相邻四元数点积为负，action 则没有；`q` 与 `-q` 可以表示
同一姿态，建议为数据和在线观测约定一致的符号规则，再重新计算统计量。
state 的夹爪范围约为 `[0.593, 1]` / `[0.599, 1]`，action 的范围都是 `[0, 1]`，
需确认这是实际反馈与控制目标的差别。转换层保留这些原始值。

## MRUI 推理

训练侧 `B3EEFInputs` 使用 `observation.state`、三路 `observation.images.*` 和
`prompt`；推理时无需提供 action。MRUI 服务入口 `scripts/serve_b3_mrui.py`
将请求中的 `state`、`images/<相机名>` 和 `prompt` 映射为上述字段，随后使用同一套
EEF 数据变换及 checkpoint 中的统计量。

当前服务入口选择 `pi05_b3_pour_tea_eef`。评测叠方块时，应将其 `create_policy` 中的
`get_config` 改为 `pi05_b3_stack_blocks_v5_eef`，并加载对应任务的 checkpoint。
推理端需要提供同一坐标系下的 16 维 EEF state。模型输出的四元数不保证单位长度，
控制端应在使用前归一化并处理近零范数，夹爪输出按控制接口的合法范围处理。
输出 transform 本身只裁剪填充维度。
