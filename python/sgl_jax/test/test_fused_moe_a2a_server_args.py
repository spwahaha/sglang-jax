"""CLI contract for experimental fused-MoE communication modes."""

from __future__ import annotations

import argparse

import pytest

from sgl_jax.srt.server_args import ServerArgs


def _parse(*extra_args: str) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(["--model-path", "dummy", *extra_args])
    return ServerArgs.from_cli_args(namespace)


def test_fused_moe_a2a_mode_defaults_to_expert():
    assert _parse().fused_moe_a2a_mode == "expert"


def test_fused_moe_a2a_mode_parses_rank_broadcast():
    args = _parse(
        "--moe-backend",
        "fused_v2",
        "--fused-moe-a2a-mode",
        "rank_broadcast",
    )

    assert args.fused_moe_a2a_mode == "rank_broadcast"


def test_nondefault_fused_moe_a2a_mode_requires_v2_backend():
    args = _parse("--fused-moe-a2a-mode", "rank_broadcast")

    with pytest.raises(ValueError, match="requires --moe-backend fused_v2"):
        args.check_server_args()
