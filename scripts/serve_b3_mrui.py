"""Serve pi05_b3_stack_blocks with MRUI's state/images/prompt request format."""

import dataclasses
import logging

import tyro

from openpi import transforms
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Checkpoint directory containing params/ and assets/.
    checkpoint_dir: str
    # Port for the MRUI WebSocket client.
    port: int = 8000


def create_policy(args: Args) -> _policy.Policy:
    # Rename fields only. The checkpoint's existing transforms still perform
    # image preprocessing, normalization and action unnormalization.
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "observation.state": "state",
                    "observation.images.front_camera": "images/front_camera",
                    "observation.images.left_wrist_camera": "images/left_wrist_camera",
                    "observation.images.right_wrist_camera": "images/right_wrist_camera",
                    "prompt": "prompt",
                }
            )
        ]
    )
    return _policy_config.create_trained_policy(
        _config.get_config("pi05_b3_stack_blocks_eef"),
        # _config.get_config("pi05_b3_pour_tea_eef"),
        args.checkpoint_dir,
        repack_transforms=repack,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
