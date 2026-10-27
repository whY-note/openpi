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
