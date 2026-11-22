# MRUI B3 Serving Implementation Plan

> Execute inline in the current session. The user approved the earlier dedicated server-script design.

**Goal:** Serve the B3 stack_blocks checkpoint to MRUI without changing MRUI Python code.

**Architecture:** Add `scripts/serve_b3_mrui.py` using the existing
`create_trained_policy(..., repack_transforms=...)` extension. Repack MRUI's
`state/images/prompt` request into the literal `observation.*` keys consumed by
`B3Inputs`. Keep normalization, image preprocessing and action output transforms
in the existing policy. Add a separate MRUI evaluation YAML with the original
hardware connections, no XR device, 14-dimensional state, 16-dimensional action,
16 prediction steps and 30 Hz playback.

**Tech Stack:** Python, tyro, openpi transforms and WebSocket server, MRUI YAML.

## Steps

- [x] Add `scripts/serve_b3_mrui_test.py`: pass three distinguishable uint8 RGB
  views and a 14-element state through the server's configured repack and the
  real `B3Inputs`. Replace only checkpoint loading to avoid GPU/weight access.
- [x] Run the test before implementation and confirm the new script is missing.
- [x] Add `scripts/serve_b3_mrui.py` with required `--checkpoint-dir` and optional
  `--port` (8000). Use `pi05_b3_stack_blocks` and the previously agreed mapping.
- [x] Add `../MRUI/configs/b3_dual_dh_pi05_stack_blocks.yaml` derived from the
  supplied hardware YAML and existing policy layout; omit XR and teleop.
- [x] Document two-terminal startup and offline checks in `examples/B3/README.md`.
- [x] Run the regression test, server `--help`, MRUI `show`, a synthetic MRUI
  encoder-to-B3-input smoke check, syntax checks and `git diff --check`.

No robot connection, live movement, checkpoint loading, commit or publication is
part of this change. Real inference quality and hardware execution remain untested.

## Verification results

- Regression first failed with the missing-entry-point assertion, then passed.
- Actual MRUI encoder -> server repack -> B3Inputs preserved state and all three
  RGB views; B3Outputs -> MRUI decoder accepted a synthetic 16-step action chunk.
- MRUI `show` passed with layout 14 -> 16, 16 steps, 30 Hz and no XR device.
- Ruff check/format, server CLI help, Python compilation and diff whitespace
  checks passed. Hardware connections match the supplied collection config.
