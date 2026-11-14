from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import swanlab

if TYPE_CHECKING:
    from openpi.training import config as _config


def init_swanlab(config: _config.TrainConfig, *, resuming: bool) -> None:
    """Initialize experiment tracking independently of model checkpoint restoration."""
    mode = os.environ.get("SWANLAB_MODE", "cloud")
    if not config.swanlab_enabled or mode == "disabled":
        swanlab.init(mode="disabled")
        return

    checkpoint_dir = config.checkpoint_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    run_id_path = checkpoint_dir / "swanlab_id.txt"
    run_id = None
    online = mode in ("cloud", "online")
    if resuming and online and run_id_path.exists():
        run_id = run_id_path.read_text().strip()
        if not run_id:
            raise ValueError(f"SwanLab run ID file {run_id_path} is empty.")
    elif resuming and online:
        logging.info("Checkpoint has no SwanLab run ID; creating a new experiment for the resumed training.")
    elif resuming:
        logging.info("SwanLab %s mode records resumed training as a new local experiment.", mode)

    # Training configs include paths, dtypes, and Flax filters that are not JSON values.
    run_config = json.loads(json.dumps(dataclasses.asdict(config), default=str))
    run = swanlab.init(
        project=config.project_name,
        experiment_name=config.exp_name,
        config=run_config,
        mode=mode,
        id=run_id,
        resume="must" if run_id is not None else None,
    )
    # Local experiment IDs cannot be resumed on the server. Preserve any online ID.
    if online:
        run_id_path.write_text(run.id)


def camera_image(image: np.ndarray) -> swanlab.Image:
    """Convert normalized [-1, 1] observation pixels to SwanLab's uint8 format."""
    return swanlab.Image(np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8))
