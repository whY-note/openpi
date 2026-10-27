"""B3 mappings for three cameras, joint observations and Cartesian actions."""

import dataclasses

import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class B3Inputs(transforms.DataTransformFn):
    """Inference expects the same observation keys as the LeRobot dataset."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation.state"])
        if state.shape != (14,):
            raise ValueError(f"Expected 14-dimensional B3 state, got {state.shape}")
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["observation.images.front_camera"]),
                "left_wrist_0_rgb": _parse_image(data["observation.images.left_wrist_camera"]),
                "right_wrist_0_rgb": _parse_image(data["observation.images.right_wrist_camera"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != 16:
                raise ValueError(f"Expected B3 action chunk (horizon, 16), got {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class B3Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Discard the model's padded action dimensions after unnormalization.
        return {"actions": np.asarray(data["actions"])[..., :16]}
