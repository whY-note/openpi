"""Convert local LeRobot v3.0 data using openpi's pinned LeRobot v2.1 writer.

Run from the openpi directory::

    uv run examples/B3/convert_lerobot_v30_to_v21.py \
        --data-dir ../data/MRUI-Dataset/stack_blocks_lerobot_v30 \
        --repo-id local/stack_blocks_lerobot_v21

Output defaults to HF_LEROBOT_HOME / repo_id, where openpi can find it.
Use --output-dir to override, or --max-episodes 1 for a trial conversion.
The source is read without installing LeRobot v3. Videos are decoded and
re-encoded per episode; allow extra disk space and time. State/action values,
feature names, FPS and task text are preserved; no action conversion is applied.
Existing output directories are never overwritten. Failed output must be moved
aside before retrying. This script supports video and numeric features, not
image features embedded in parquet.
"""

from contextlib import ExitStack
import json
from pathlib import Path

import av
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import tyro

_GENERATED_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def video_path(root: Path, info: dict, episode: dict, key: str) -> Path:
    return root / info["video_path"].format(
        video_key=key,
        chunk_index=int(episode[f"videos/{key}/chunk_index"]),
        file_index=int(episode[f"videos/{key}/file_index"]),
    )


def video_frames(path: Path, timestamps: np.ndarray, fps: float):
    """Stream frames at requested file-relative times, checking alignment."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(float(timestamps[0]) / stream.time_base), stream=stream, backward=True)
        decoded = iter(container.decode(stream))
        frame = next(decoded, None)
        tolerance = 0.5 / fps + 1e-4
        for timestamp in timestamps:
            while frame is not None and (frame.time is None or frame.time < timestamp - tolerance):
                frame = next(decoded, None)
            if frame is None or abs(frame.time - timestamp) > tolerance:
                raise ValueError(f"Missing/alignment error in {path} at {timestamp:.6f}s")
            yield frame.to_ndarray(format="rgb24")


def main(
    data_dir: Path,
    *,
    repo_id: str = "local/stack_blocks_lerobot_v21",
    output_dir: Path | None = None,
    max_episodes: int | None = None,
    image_writer_threads: int = 4,
):
    """Read v3 parquet metadata, then add frames and save episodes with v2.1."""
    if CODEBASE_VERSION != "v2.1":
        raise ValueError(f"Run with openpi's LeRobot v2.1 dependency, got {CODEBASE_VERSION}")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if image_writer_threads < 0:
        raise ValueError("image_writer_threads must be nonnegative")
    source = data_dir.expanduser().resolve()
    target = (output_dir or HF_LEROBOT_HOME / repo_id).expanduser().resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Input and output directories must be separate, non-nested paths")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"Expected v3.0, got {info['codebase_version']}")
    features = {
        key: {"dtype": feature["dtype"], "shape": tuple(feature["shape"]), "names": feature.get("names")}
        for key, feature in info["features"].items()
        if key not in _GENERATED_KEYS
    }
    if any(feature["dtype"] == "image" for feature in features.values()):
        raise ValueError("Embedded image features are not supported; expected video features")
    cameras = [key for key, feature in features.items() if feature["dtype"] == "video"]
    numeric = [key for key in features if key not in cameras]
    tasks = {int(row["task_index"]): row["task"] for row in pq.read_table(source / "meta/tasks.parquet").to_pylist()}
    episodes = []
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda row: row["episode_index"])
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata count does not match info.json")
    if len({episode["episode_index"] for episode in episodes}) != len(episodes):
        raise ValueError("Duplicate episode indices")
    episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError("No episodes found")
    for episode in episodes:
        for key in cameras:
            path = video_path(source, info, episode, key)
            if not path.is_file():
                raise FileNotFoundError(path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=target,
        robot_type=info.get("robot_type"),
        fps=info["fps"],
        features=features,
        use_videos=True,
        image_writer_threads=image_writer_threads,
        video_backend="pyav",
    )
    # Cache only the current consolidated data file, not the whole dataset.
    cached_path = None
    table = None
    try:
        for episode in tqdm(episodes, desc="Converting episodes"):
            path = source / info["data_path"].format(
                chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"])
            )
            if path != cached_path:
                table = pq.read_table(path)
                cached_path = path
            rows = table.filter(np.asarray(table["episode_index"]) == episode["episode_index"]).to_pylist()
            rows.sort(key=lambda row: row["frame_index"])
            if len(rows) != episode["length"] or not rows:
                raise ValueError(f"Episode {episode['episode_index']}: unexpected frame count")
            if [row["frame_index"] for row in rows] != list(range(len(rows))):
                raise ValueError(f"Episode {episode['episode_index']}: noncontiguous frame indices")
            timestamps = np.array([row["timestamp"] for row in rows], dtype=np.float64)
            if not np.allclose(timestamps, np.arange(len(rows)) / info["fps"], rtol=0, atol=1e-4):
                raise ValueError(f"Episode {episode['episode_index']}: timestamps do not match FPS")
            with ExitStack() as stack:
                readers = {}
                for key in cameras:
                    reader = video_frames(
                        video_path(source, info, episode, key),
                        timestamps + float(episode[f"videos/{key}/from_timestamp"]),
                        info["fps"],
                    )
                    stack.callback(reader.close)
                    readers[key] = reader
                for row in rows:
                    frame = {}
                    for key in numeric:
                        value = np.asarray(row[key], dtype=features[key]["dtype"])
                        if value.shape != features[key]["shape"] or not np.isfinite(value).all():
                            raise ValueError(f"Invalid {key} in episode {episode['episode_index']}")
                        frame[key] = value
                    for key, reader in readers.items():
                        image = next(reader)
                        if image.shape != features[key]["shape"]:
                            raise ValueError(f"Unexpected image shape for {key}: {image.shape}")
                        frame[key] = image
                    frame["task"] = tasks[int(row["task_index"])]
                    # Source timestamps were checked against frame_index / fps above.
                    # Let the v2.1 writer generate this optional feature: its
                    # validator rejects Python floats supplied explicitly.
                    dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        dataset.stop_image_writer()
    expected_frames = sum(episode["length"] for episode in episodes)
    if dataset.meta.total_frames != expected_frames or dataset.meta.total_episodes != len(episodes):
        raise RuntimeError("Output counts do not match source")
    print(f"Saved v2.1 dataset: {target} ({len(episodes)} episodes, {expected_frames} frames)")


if __name__ == "__main__":
    tyro.cli(main)
