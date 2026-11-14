import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import b3_eef_policy
from openpi.training import config


def _observation():
    return {
        "observation.state": np.arange(16, dtype=np.float32),
        "observation.images.front_camera": np.ones((3, 8, 10), dtype=np.float32),
        "observation.images.left_wrist_camera": np.full((8, 10, 3), 127, dtype=np.uint8),
        "observation.images.right_wrist_camera": np.zeros((3, 8, 10), dtype=np.float32),
        "prompt": "stack blocks",
    }


def test_inference_observation():
    observation = _observation()
    result = b3_eef_policy.B3EEFInputs()(observation)
    np.testing.assert_array_equal(result["state"], observation["observation.state"])
    assert "actions" not in result
    assert result["prompt"] == "stack blocks"
    for key, value in zip(result["image"], [255, 127, 0], strict=True):
        assert result["image"][key].shape == (8, 10, 3)
        assert result["image"][key].dtype == np.uint8
        assert np.all(result["image"][key] == value)
        assert result["image_mask"][key]


@pytest.mark.parametrize("state_dim", [14, 32])
def test_rejects_incompatible_state(state_dim):
    observation = _observation()
    observation["observation.state"] = np.zeros(state_dim)
    with pytest.raises(ValueError, match="16-dimensional B3 EEF state"):
        b3_eef_policy.B3EEFInputs()(observation)


@pytest.mark.parametrize("shape", [(16,), (50, 14), (1, 50, 16)])
def test_rejects_incompatible_action_chunk(shape):
    with pytest.raises(ValueError, match="action chunk"):
        b3_eef_policy.B3EEFInputs()({**_observation(), "actions": np.zeros(shape)})


def test_training_and_output_pipeline(tmp_path, monkeypatch):
    # Keep the regression test offline; tokenizer behavior has its own tests.
    monkeypatch.setattr(
        config.ModelTransformFactory,
        "__call__",
        lambda self, model: transforms.Group(inputs=[transforms.PadStatesAndActions(model.action_dim)]),
    )
    train_config = config.get_config("pi05_b3_stack_blocks_v5_eef")
    assert isinstance(train_config.model, pi0_config.Pi0Config)
    assert train_config.model.pi05
    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.prompt_from_task
    assert data_config.use_quantile_norm

    actions = np.arange(50 * 16, dtype=np.float32).reshape(50, 16) / 1000
    sample = {**_observation(), "action": actions}
    inputs = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])(sample)
    # Absolute position, quaternion and gripper values must all survive unchanged.
    np.testing.assert_array_equal(inputs["actions"], actions)
    np.testing.assert_array_equal(inputs["state"], sample["observation.state"])

    stats = {
        key: transforms.NormStats(mean=np.zeros(16), std=np.ones(16), q01=-np.ones(16), q99=2 * np.ones(16))
        for key in ("state", "actions")
    }
    padded = transforms.compose(
        [transforms.Normalize(stats, use_quantiles=True), *data_config.model_transforms.inputs]
    )(inputs)
    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (50, 32)
    restored = transforms.compose(
        [transforms.Unnormalize(stats, use_quantiles=True), *data_config.data_transforms.outputs]
    )(padded)
    assert restored["actions"].shape == (50, 16)
    np.testing.assert_allclose(restored["actions"], actions, atol=1e-5)
