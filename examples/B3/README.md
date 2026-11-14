# B3 数据格式转换

`convert_lerobot_v30_to_v21.py` 将本地 LeRobot v3.0 数据转换为当前 openpi
依赖支持的 v2.1 格式，不需要安装新版 LeRobot。输入数据不会修改。

在 `openpi` 目录执行：

```bash
uv run examples/B3/convert_lerobot_v30_to_v21.py \
  --data-dir /dataset-v2/haowenyan/project/data/MRUI-Dataset/stack_blocks_lerobot_v30 \
  --repo-id local/stack_blocks_lerobot_v21
```

默认输出到 `HF_LEROBOT_HOME / local/stack_blocks_lerobot_v21`，训练配置使用
`repo_id="local/stack_blocks_lerobot_v21"`。可添加 `--output-dir /其他输出目录`
指定位置；此时需要另行配置训练读取位置。

建议先加 `--max-episodes 1 --repo-id local/stack_blocks_v21_trial` 试转一个
episode，再使用不同输出目录进行完整转换。输出路径必须尚不存在，脚本不会
覆盖、删除或续写已有目录；失败后请将不完整目录移走，再重新运行。

转换保留三路相机的字段名和分辨率、14 维 state、16 维 action、30 FPS 和任务
文本。按 v3 元数据中的视频时间偏移读取画面，重新生成每个 episode 的视频、
parquet、任务表和 v2.1 统计量。视频会重新编码，像素可能有压缩差异，且需要
额外磁盘空间和转换时间。state/action 不做差分、归一化或坐标系转换。

## π₀.₅ 训练配置

`LeRobotB3DataConfig` 已接入 `pi05_b3_stack_blocks`，默认使用
`local/stack_blocks_lerobot_v21`。如果实际 repo_id 不同，需要修改训练配置。
数据集的 `action` 字段用于读取连续动作，随后改名为模型使用的 `actions`；
任务表生成 `prompt`。前视、左腕、右腕分别映射到模型的三个图像槽位，全部
设置为有效图像。LeRobot 的 CHW 浮点图像转为 HWC uint8 后，再由模型通用
transforms 调整到 224×224。

14 维关节 state 和 16 维末端位姿 action 保持原含义，不执行 `DeltaActions`：
关节角不能从末端位置/四元数中相减。state/action 使用 B3 自己的统计量归一化，
随后补齐到模型的 32 维；推理时仅返回前 16 维动作。该配置不会推断或改变原始
动作的坐标系、绝对/相对定义，执行端必须使用相同约定。

在 `openpi` 目录生成训练用归一化统计量：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_b3_stack_blocks
```

推理时使用与数据集相同的 `observation.state` 和三路 `observation.images.*`
字段，并额外提供 `prompt`；无需提供 action。

## v5 EEF 数据训练

`pi05_b3_stack_blocks_v5_eef` 使用 `LeRobotB3EEFDataConfig` 和
`B3EEFInputs` / `B3EEFOutputs`，适配
`data/MRUI-Dataset/stack_blocks_latest/data_v5_eef_lerobot_v21`。
此版本的 state 和 action 都是 16 维，左右臂各占 8 维：
`[x, y, z, qx, qy, qz, qw, gripper_open_fraction]`，位置单位为米。
保留绝对动作，不对四元数执行逐元素 `DeltaActions`。相机映射与上文相同。

配置保留 pi05 的 32 维模型输入输出，数据转换后由通用 transform 补零，
输出在反归一化后裁回 16 维。`action_horizon=50`，数据为 50 Hz，
因此每个动作块覆盖约 1 秒。任务文本从 `meta/tasks.jsonl` 读取为 `stack blocks`。

在当前机器上，已建立
`/dataset-v2/haowenyan/project/data/lerobot/local/stack_blocks_v5_eef_lerobot_v21`
到数据目录的软链接。每个训练或计算统计量的 shell 都需要设置 `HF_LEROBOT_HOME`：

```bash
cd /dataset-v2/haowenyan/project/openpi
export HF_LEROBOT_HOME=/dataset-v2/haowenyan/project/data/lerobot

# 等待数据转换/复制完成：应有 103 个 parquet 和每路相机各 103 个 mp4。
# 为这个 EEF 配置单独计算统计量，不复用 14 维关节 state 的旧统计量。
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_b3_stack_blocks_v5_eef

# 当前配置为 JAX 训练，加载本机 checkpoints/pi05_base/params。
# batch size / FSDP 设备数需按实际 GPU 资源设置。
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py \
  pi05_b3_stack_blocks_v5_eef --exp-name b3_v5_eef \
  --no-wandb-enabled --no-swanlab-enabled
```

统计量应生成在
`assets/pi05_b3_stack_blocks_v5_eef/local/stack_blocks_v5_eef_lerobot_v21/norm_stats.json`，
包含 16 维 `state` 和 `actions` 的均值、标准差及分位数。pi05 使用分位数归一化。
`main` 分支的 `scripts/train.py` 与 `scripts/train_pytorch.py` 使用 W&B，
`swanlab` 分支使用 SwanLab。上述命令关闭在线日志；若需开启，
在 `main` 分支去掉 `--no-wandb-enabled`，在 `swanlab` 分支去掉
`--no-swanlab-enabled`。

训练前还应核对四元数符号和夹爪语义。对此版本 150,169 帧的检查发现，
左臂 state 有 30 处相邻四元数点积为负，action 则没有；`q` 与 `-q` 可以表示
同一姿态，建议为数据和在线观测约定一致的符号规则，再重新计算统计量。
state 的夹爪范围约为 `[0.593, 1]` / `[0.599, 1]`，action 的范围都是 `[0, 1]`，
需确认这是实际反馈与控制目标的差别。转换层保留这些原始值。

推理端需要提供同一坐标系下的 16 维 EEF state，不能继续传旧的 14 维关节 state。
模型输出的四元数不保证单位长度，控制端应在使用前归一化并处理近零范数，
夹爪输出按控制接口的合法范围处理。输出 transform 本身只裁剪填充维度。
