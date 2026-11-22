"""Offline regression for the MRUI request / B3 preprocessing boundary."""

import importlib
import importlib.util
import unittest
from unittest import mock

import numpy as np

from openpi import transforms
from openpi.policies import policy_config
from openpi.policies.b3_policy import B3Inputs


class MruiB3ServingTest(unittest.TestCase):
    def test_mrui_request_reaches_b3_inputs_without_changing_values(self):
        assert importlib.util.find_spec("scripts.serve_b3_mrui") is not None, (
            "The MRUI-compatible B3 server entry point is missing"
        )
        serving = importlib.import_module("scripts.serve_b3_mrui")

        def load_without_weights(train_config, checkpoint_dir, *, repack_transforms):
            assert train_config.name == "pi05_b3_stack_blocks"
            assert train_config.model.action_horizon == 16
            assert checkpoint_dir == "/offline/checkpoint"
            return transforms.compose([*repack_transforms.inputs, B3Inputs()])

        request = {
            "state": np.arange(14, dtype=np.float32),
            "images": {
                name: np.full((8, 12, 3), value, dtype=np.uint8)
                for name, value in (("front_camera", 11), ("left_wrist_camera", 22), ("right_wrist_camera", 33))
            },
            "prompt": "stack blocks",
        }
        with mock.patch.object(policy_config, "create_trained_policy", side_effect=load_without_weights):
            pipeline = serving.create_policy(serving.Args(checkpoint_dir="/offline/checkpoint"))
            output = pipeline(request)

        np.testing.assert_array_equal(output["state"], request["state"])
        for slot, source in (
            ("base_0_rgb", "front_camera"),
            ("left_wrist_0_rgb", "left_wrist_camera"),
            ("right_wrist_0_rgb", "right_wrist_camera"),
        ):
            np.testing.assert_array_equal(output["image"][slot], request["images"][source])
            assert output["image_mask"][slot]
        assert output["prompt"] == request["prompt"]
        assert "actions" not in output


if __name__ == "__main__":
    unittest.main()
