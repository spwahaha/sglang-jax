"""CPU-only tests for the MoeWorkloadRegime hint on the v2 tuned block-config
lookup. Validate that:
  (a) AUTO / None reproduce the pre-existing regime-agnostic lookup byte-identical
  (b) LATENCY vs THROUGHPUT pick distinct configs where a regime override is defined
  (c) an invalid regime string raises a clear error
"""

from unittest import mock

import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
    REGIME_TUNED_BLOCK_CONFIGS,
    MoeWorkloadRegime,
    get_tuned_fused_moe_v2_block_config,
)
from sgl_jax.srt.layers.fused_moe import fused_moe_workload_regime_from_forward_mode

_SHARED_SHAPE = dict(
    num_tokens=512,
    num_experts=384,
    top_k=8,
    hidden_size=6144,
    intermediate_size=2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.float8_e4m3fn,
    ep_size=32,
    use_shared_expert=False,
    use_grouped_topk=False,
)

_LEGACY_KEY = (
    "bfloat16",
    "float8_e4m3fn",
    512,
    384,
    8,
    6144,
    2048,
    32,
    False,
    False,
)


def _with_synthetic_regime_winners():
    return mock.patch.dict(
        REGIME_TUNED_BLOCK_CONFIGS,
        {
            MoeWorkloadRegime.LATENCY.value: {"TPU v7": {_LEGACY_KEY: (8, 512, 16, 256, 16)}},
            MoeWorkloadRegime.THROUGHPUT.value: {"TPU v7": {_LEGACY_KEY: (32, 1024, 32, 256, 32)}},
        },
        clear=True,
    )


def _with_device(device_name):
    return mock.patch(
        "sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs.get_device_name",
        return_value=device_name,
    )


def test_default_regime_matches_no_regime_argument():
    with _with_device("TPU v7"):
        implicit = get_tuned_fused_moe_v2_block_config(**_SHARED_SHAPE)
        explicit_none = get_tuned_fused_moe_v2_block_config(**_SHARED_SHAPE, workload_regime=None)
        explicit_auto = get_tuned_fused_moe_v2_block_config(
            **_SHARED_SHAPE, workload_regime=MoeWorkloadRegime.AUTO
        )
        explicit_auto_str = get_tuned_fused_moe_v2_block_config(
            **_SHARED_SHAPE, workload_regime="auto"
        )

    assert implicit == explicit_none == explicit_auto == explicit_auto_str
    assert implicit.bt == 16
    assert implicit.bf == 1024
    assert implicit.btc == 32
    assert implicit.bse == 256
    assert implicit.bts == 32


def test_default_regime_unaffected_by_regime_table_contents():
    """AUTO must never consult REGIME_TUNED_BLOCK_CONFIGS, so the pre-existing
    entry in TUNED_BLOCK_CONFIGS always wins regardless of what regime overrides
    exist for the same shape."""
    with _with_device("TPU v7"), _with_synthetic_regime_winners():
        cfg = get_tuned_fused_moe_v2_block_config(**_SHARED_SHAPE)
    assert (cfg.bt, cfg.bf, cfg.btc, cfg.bse, cfg.bts) == (16, 1024, 32, 256, 32)


def test_latency_and_throughput_regimes_pick_distinct_configs():
    with _with_device("TPU v7"), _with_synthetic_regime_winners():
        latency = get_tuned_fused_moe_v2_block_config(
            **_SHARED_SHAPE, workload_regime=MoeWorkloadRegime.LATENCY
        )
        throughput = get_tuned_fused_moe_v2_block_config(
            **_SHARED_SHAPE, workload_regime=MoeWorkloadRegime.THROUGHPUT
        )
        auto = get_tuned_fused_moe_v2_block_config(**_SHARED_SHAPE)

    assert latency != throughput
    assert (latency.bt, latency.bf, latency.btc, latency.bse, latency.bts) == (
        8,
        512,
        16,
        256,
        16,
    )
    assert (throughput.bt, throughput.bf, throughput.btc, throughput.bse, throughput.bts) == (
        32,
        1024,
        32,
        256,
        32,
    )
    assert latency != auto
    assert throughput != auto


def test_regime_falls_back_to_auto_table_when_no_override_defined():
    """A shape with no entry in REGIME_TUNED_BLOCK_CONFIGS for the requested
    regime must fall back to the regime-agnostic TUNED_BLOCK_CONFIGS/DEFAULT
    lookup rather than returning None or erroring."""
    shape = dict(_SHARED_SHAPE, num_tokens=64)
    with _with_device("TPU v7"):
        latency = get_tuned_fused_moe_v2_block_config(
            **shape, workload_regime=MoeWorkloadRegime.LATENCY
        )
        auto = get_tuned_fused_moe_v2_block_config(**shape)

    assert latency == auto


def test_unqualified_v6e_decode_shape_still_uses_production_fallback():
    shape = dict(
        num_tokens=128,
        num_experts=32,
        top_k=8,
        hidden_size=8192,
        intermediate_size=1024,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        ep_size=4,
    )
    with _with_device("TPU v6 lite"):
        auto = get_tuned_fused_moe_v2_block_config(**shape)
        latency = get_tuned_fused_moe_v2_block_config(
            **shape, workload_regime=MoeWorkloadRegime.LATENCY
        )
        throughput = get_tuned_fused_moe_v2_block_config(
            **shape, workload_regime=MoeWorkloadRegime.THROUGHPUT
        )

    assert latency == throughput == auto
    assert (auto.bt, auto.bf, auto.btc, auto.bse, auto.bts) == (32, 512, 32, 256, None)


def test_regime_accepts_plain_string_equivalent_to_enum():
    with _with_device("TPU v7"), _with_synthetic_regime_winners():
        via_enum = get_tuned_fused_moe_v2_block_config(
            **_SHARED_SHAPE, workload_regime=MoeWorkloadRegime.LATENCY
        )
        via_str = get_tuned_fused_moe_v2_block_config(**_SHARED_SHAPE, workload_regime="latency")
    assert via_enum == via_str


def test_invalid_regime_raises_clear_error():
    with _with_device("TPU v7"), pytest.raises(ValueError, match="Unknown MoeWorkloadRegime"):
        get_tuned_fused_moe_v2_block_config(**_SHARED_SHAPE, workload_regime="turbo")


@pytest.mark.parametrize(
    ("mode_name", "expected"),
    [
        ("DECODE", "latency"),
        ("EXTEND", "throughput"),
        ("MIXED", "auto"),
        ("TARGET_VERIFY", "auto"),
        ("DRAFT_EXTEND", "auto"),
    ],
)
def test_scheduler_forward_mode_mapping_is_conservative(mode_name, expected):
    forward_mode = mock.Mock(name=mode_name)
    forward_mode.name = mode_name
    assert fused_moe_workload_regime_from_forward_mode(forward_mode) == expected
