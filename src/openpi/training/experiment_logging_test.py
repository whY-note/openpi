import dataclasses
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from openpi.training import experiment_logging


@dataclasses.dataclass
class TrackingConfig:
    checkpoint_dir: pathlib.Path
    project_name: str = "openpi-test"
    exp_name: str = "test-run"
    swanlab_enabled: bool = True
    learning_rate: float = 0.001
    dtype: type = float


@pytest.fixture
def init_mock(monkeypatch):
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    init = mock.Mock(return_value=SimpleNamespace(id="new-swanlab-id"))
    monkeypatch.setattr(experiment_logging.swanlab, "init", init)
    return init


def test_new_experiment_replaces_stale_tracking_id(tmp_path, init_mock):
    id_path = tmp_path / "swanlab_id.txt"
    id_path.write_text("previous-experiment")

    experiment_logging.init_swanlab(TrackingConfig(tmp_path), resuming=False)

    assert init_mock.call_args.kwargs["id"] is None
    assert id_path.read_text() == "new-swanlab-id"
    config = init_mock.call_args.kwargs["config"]
    json.dumps(config)  # Configs with paths and dtype objects must be JSON serializable.
    assert config["learning_rate"] == 0.001
    assert config["checkpoint_dir"] == str(tmp_path)


@pytest.mark.parametrize("mode", [None, "cloud"])
def test_resume_reuses_swanlab_experiment(tmp_path, init_mock, monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv("SWANLAB_MODE", mode)
    (tmp_path / "swanlab_id.txt").write_text("existing-swanlab-id\n")
    init_mock.return_value.id = "existing-swanlab-id"

    experiment_logging.init_swanlab(TrackingConfig(tmp_path), resuming=True)

    assert init_mock.call_args.kwargs["id"] == "existing-swanlab-id"
    assert init_mock.call_args.kwargs["resume"] == "must"
    assert init_mock.call_args.kwargs["mode"] == "cloud"
    assert (tmp_path / "swanlab_id.txt").read_text() == "existing-swanlab-id"


@pytest.mark.parametrize("legacy_id", [None, "old-wandb-id"])
def test_resume_without_swanlab_id_creates_experiment(tmp_path, init_mock, legacy_id):
    if legacy_id is not None:
        (tmp_path / "wandb_id.txt").write_text(legacy_id)
    checkpoint = tmp_path / "1234"
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"existing model checkpoint")

    experiment_logging.init_swanlab(TrackingConfig(tmp_path), resuming=True)

    assert init_mock.call_args.kwargs["id"] is None
    assert init_mock.call_args.kwargs["resume"] is None
    assert (tmp_path / "swanlab_id.txt").read_text() == "new-swanlab-id"
    assert (checkpoint / "weights").read_bytes() == b"existing model checkpoint"
    if legacy_id is not None:
        assert (tmp_path / "wandb_id.txt").read_text() == legacy_id


def test_disabled_tracking_needs_no_checkpoint_directory(tmp_path, init_mock):
    checkpoint_dir = tmp_path / "does-not-exist"

    experiment_logging.init_swanlab(TrackingConfig(checkpoint_dir, swanlab_enabled=False), resuming=True)

    init_mock.assert_called_once_with(mode="disabled")
    assert not checkpoint_dir.exists()


@pytest.mark.parametrize("mode", ["offline", "local"])
def test_local_resume_preserves_online_experiment_id(tmp_path, init_mock, monkeypatch, mode):
    monkeypatch.setenv("SWANLAB_MODE", mode)
    id_path = tmp_path / "swanlab_id.txt"
    id_path.write_text("online-experiment")

    experiment_logging.init_swanlab(TrackingConfig(tmp_path), resuming=True)

    assert init_mock.call_args.kwargs["mode"] == mode
    assert init_mock.call_args.kwargs["id"] is None
    assert init_mock.call_args.kwargs["resume"] is None
    assert id_path.read_text() == "online-experiment"


def test_disabled_mode_needs_no_checkpoint_directory(tmp_path, init_mock, monkeypatch):
    monkeypatch.setenv("SWANLAB_MODE", "disabled")
    checkpoint_dir = tmp_path / "missing"

    experiment_logging.init_swanlab(TrackingConfig(checkpoint_dir), resuming=False)

    init_mock.assert_called_once_with(mode="disabled")
    assert not checkpoint_dir.exists()


def test_missing_checkpoint_directory(tmp_path, init_mock):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        experiment_logging.init_swanlab(TrackingConfig(tmp_path / "missing"), resuming=False)
    init_mock.assert_not_called()


def test_empty_tracking_id_is_not_silently_replaced(tmp_path, init_mock):
    (tmp_path / "swanlab_id.txt").write_text("\n")

    with pytest.raises(ValueError, match="is empty"):
        experiment_logging.init_swanlab(TrackingConfig(tmp_path), resuming=True)
    init_mock.assert_not_called()


def test_normalized_camera_pixels_retain_color():
    image = experiment_logging.camera_image(np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32))

    assert np.asarray(image.image_data).tolist() == [[[0, 127, 255]]]


def test_cloud_environment_is_valid_before_sdk_import(tmp_path):
    # The SDK validates this variable at import, before our initializer can run.
    env = {
        **os.environ,
        "SWANLAB_MODE": "cloud",
        "SWANLAB_SAVE_DIR": str(tmp_path / "settings"),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import openpi.training.experiment_logging; print('cloud import ok')"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cloud import ok" in result.stdout
