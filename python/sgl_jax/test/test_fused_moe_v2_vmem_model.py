"""CPU-only tests for fused MoE v2 scratch geometry and VMEM accounting."""

from __future__ import annotations

import pytest

from sgl_jax.srt.kernels.fused_moe.v2.vmem_model import (
    FUSED_MOE_V2_VMEM_LIMIT_BYTES,
    build_fused_moe_v2_scratch_plan,
    estimate_fused_moe_v2_vmem,
    validate_fused_moe_v2_vmem_budget,
)


def _plan(**overrides):
    args = dict(
        num_tokens=512,
        num_experts=32,
        top_k=8,
        hidden_size=8192,
        ep_size=4,
        bt=32,
        bf=512,
        bts=32,
        input_bytes=2,
        enable_act_quant=False,
        has_weight_scales=False,
        quant_block_k=128,
        enable_bt_scatter_overlap=True,
        fixed_gather_banks=2,
        a2a_mode="expert",
    )
    args.update(overrides)
    return build_fused_moe_v2_scratch_plan(**args)


def test_fixed_gather_banks_match_kernel_geometry():
    plan = _plan()

    assert plan.local_num_tokens == 128
    assert plan.num_bt == 4
    assert plan.use_fixed_gather_banks
    assert plan.gather_bank_count == 2
    assert plan.num_bt_banks == 2
    assert plan.smem_banks == 3  # K payload banks plus one metadata lookahead bank.

    components = estimate_fused_moe_v2_vmem(plan).as_dict()
    assert components["output_banks"] == 2 * 32 * 8192 * 2
    assert components["topk_weights"] == 3 * 32 * 128 * 4
    assert components["topk_ids"] == 3 * 32 * 128 * 4


def test_single_bt_uses_one_payload_bank_and_two_metadata_banks():
    plan = _plan(num_tokens=128)

    assert plan.local_num_tokens == 32
    assert plan.num_bt == 1
    assert not plan.use_gather_bank
    assert plan.num_bt_banks == 1
    assert plan.smem_banks == 2


def test_activation_quantization_matches_scale_and_stage_shapes():
    plan = _plan(
        enable_act_quant=True,
        has_weight_scales=True,
        quant_block_k=128,
    )
    components = estimate_fused_moe_v2_vmem(
        plan,
        weight_bytes=1,
        direct_scaled_dot=True,
    ).as_dict()

    assert plan.internal_packing == 4
    assert plan.h_per_internal == 2048
    assert plan.scale_groups_w13 == 16
    assert plan.scale_groups_w2 == 4
    assert components["token_stage"] == 32 * 8192
    assert components["activation_scales"] == 32 * 128 * 4
    assert components["w1_scale_double_buffer"] == 2 * 4 * 16 * 512 * 4
    assert components["w2_scale_double_buffer"] == 2 * 2 * 4 * 4096 * 4
    assert "w1_dequant" not in components


def test_dequant_scratch_is_counted_only_for_non_direct_scaled_dot():
    plan = _plan(has_weight_scales=True)
    direct = estimate_fused_moe_v2_vmem(
        plan,
        weight_bytes=1,
        direct_scaled_dot=True,
    ).as_dict()
    dequant = estimate_fused_moe_v2_vmem(
        plan,
        weight_bytes=1,
        direct_scaled_dot=False,
    ).as_dict()

    assert "w1_dequant" not in direct
    assert dequant["w1_dequant"] == 8192 * 512 * 2
    assert dequant["w2_dequant"] == 8192 * 512 * 2
    assert dequant["w3_dequant"] == 8192 * 512 * 2


def test_rank_combine_adds_rank_transport_vmem():
    expert = estimate_fused_moe_v2_vmem(_plan()).as_dict()
    rank = estimate_fused_moe_v2_vmem(_plan(a2a_mode="rank_dispatch_combine")).as_dict()

    assert "rank_pack" not in expert
    assert "rank_inbox" not in expert
    assert "rank_partial" not in expert
    assert rank["route_mailbox"] == 2 * 4 * 32 * 16 * 4
    assert rank["rank_pack"] == 2 * 5 * 32 * 8192 * 2
    assert rank["rank_inbox"] == 2 * 4 * 32 * 8192 * 2
    assert rank["rank_partial"] == 2 * 4 * 32 * 8192 * 4
    assert rank["rank_return"] == 2 * 4 * 32 * 8192 * 4


def test_total_is_sum_of_named_vmem_components():
    estimate = estimate_fused_moe_v2_vmem(_plan())
    assert estimate.total_bytes == sum(estimate.as_dict().values())


def test_rank_broadcast_avoids_source_pack_and_fits_production_shape():
    estimate = estimate_fused_moe_v2_vmem(_plan(a2a_mode="rank_broadcast"))

    assert "rank_pack" not in estimate.as_dict()
    assert estimate.total_bytes == 62_107_648
    validate_fused_moe_v2_vmem_budget(estimate)


def test_packed_rank_production_shape_is_rejected_before_lowering():
    estimate = estimate_fused_moe_v2_vmem(_plan(a2a_mode="rank_dispatch"))

    assert estimate.as_dict()["rank_pack"] == 5_242_880
    assert estimate.total_bytes == 67_350_528
    assert estimate.total_bytes > FUSED_MOE_V2_VMEM_LIMIT_BYTES
    with pytest.raises(ValueError, match="estimated scoped VMEM exceeds"):
        validate_fused_moe_v2_vmem_budget(estimate)


def test_expert_production_shape_passes_logical_vmem_preflight():
    estimate = estimate_fused_moe_v2_vmem(_plan(a2a_mode="expert"))

    assert estimate.total_bytes == 57_905_152
    assert estimate.compiler_budget_bytes > estimate.total_bytes
    validate_fused_moe_v2_vmem_budget(estimate)


@pytest.mark.parametrize(
    ("bt", "bf", "bts"),
    [
        (256, 128, 512),
        (256, 256, 320),
        (256, 256, 256),
        (128, 256, 256),
        (64, 512, 80),
    ],
)
def test_calibrated_budget_rejects_observed_mosaic_scoped_ooms(bt, bf, bts):
    estimate = estimate_fused_moe_v2_vmem(_plan(num_tokens=4096, bt=bt, bf=bf, bts=bts))

    assert estimate.total_bytes < FUSED_MOE_V2_VMEM_LIMIT_BYTES
    assert estimate.compiler_budget_bytes > FUSED_MOE_V2_VMEM_LIMIT_BYTES
    with pytest.raises(ValueError, match="estimated scoped VMEM exceeds"):
        validate_fused_moe_v2_vmem_budget(estimate)


@pytest.mark.parametrize(
    ("bt", "bf", "bts"),
    [
        (128, 256, 160),  # Best measured prefill candidate.
        (128, 256, 128),
        (64, 512, 64),
    ],
)
def test_calibrated_budget_retains_observed_successes(bt, bf, bts):
    estimate = estimate_fused_moe_v2_vmem(_plan(num_tokens=4096, bt=bt, bf=bf, bts=bts))

    assert estimate.compiler_budget_bytes < FUSED_MOE_V2_VMEM_LIMIT_BYTES
    validate_fused_moe_v2_vmem_budget(estimate)
