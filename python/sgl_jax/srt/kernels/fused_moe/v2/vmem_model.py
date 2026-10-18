"""Pure scratch-layout and VMEM accounting for fused MoE v2.

This module intentionally has no JAX/Pallas dependency.  The kernel uses the
derived layout fields when constructing scratch refs; the offline tuner uses
the same plan to reject configurations that exceed the configured VMEM budget.

The logical byte estimate is the exact sum of the requested VMEM array shapes.
The compiler-budget estimate additionally models the scoped copy/alignment of
the expert-row working set observed in Mosaic lowering.  Successful lowering
remains the final feasibility check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FUSED_MOE_V2_VMEM_LIMIT_BYTES = 64 * 1024 * 1024

# On TPU v6e with JAX 0.11.1, Mosaic scoped allocation was consistently the
# logical request plus approximately 1.15x one expert-row working set.  Use a
# 1.20x bound so tuning rejects all five observed false-positive candidates
# while retaining every observed successful candidate.  Keep this correction
# separate from ``total_bytes`` so callers can distinguish exact requested
# shapes from compiler-budget calibration.
_MOSAIC_WORKING_SET_HEADROOM_NUMERATOR = 6
_MOSAIC_WORKING_SET_HEADROOM_DENOMINATOR = 5
_EXPERT_WORKING_SET_COMPONENTS = frozenset(
    {
        "gate_accumulator",
        "up_accumulator",
        "token_stage",
        "output_accumulator",
        "output_stage",
        "activation_scales",
    }
)

_RANK_A2A_MODES = frozenset(
    {"rank_direct", "rank_broadcast", "rank_dispatch", "rank_dispatch_combine"}
)
_SUPPORTED_A2A_MODES = frozenset({"expert", *_RANK_A2A_MODES})


def _align_to(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"Expected {alignment=} > 0.")
    return ((value + alignment - 1) // alignment) * alignment


def _packing_for_bytes(item_bytes: int) -> int:
    if item_bytes not in (1, 2, 4):
        raise ValueError(f"Expected 1-, 2-, or 4-byte values, got {item_bytes=}.")
    return 4 // item_bytes


@dataclass(frozen=True)
class FusedMoEV2ScratchPlan:
    """Static dimensions shared by kernel scratch construction and tuning."""

    ep_size: int
    top_k: int
    hidden_size: int
    bt: int
    bf: int
    bts: int
    input_bytes: int
    internal_bytes: int
    output_bytes: int
    enable_act_quant: bool
    has_weight_scales: bool
    local_num_tokens: int
    local_num_experts: int
    padded_num_experts: int
    padded_top_k: int
    routing_smem_top_k: int
    route_fields: int
    input_packing: int
    internal_packing: int
    output_packing: int
    h_per_input: int
    h_per_internal: int
    h_per_output: int
    acc_bt: int
    num_bt: int
    use_bt_scatter_bank: bool
    use_gather_bank: bool
    use_bt_banking: bool
    use_fixed_gather_banks: bool
    gather_bank_count: int
    num_bt_banks: int
    smem_banks: int
    a2a_max_tokens: int
    rank_dispatch_enabled: bool
    rank_pack_enabled: bool
    rank_combine_enabled: bool
    scale_groups_w13: int
    scale_groups_w2: int


def build_fused_moe_v2_scratch_plan(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    ep_size: int,
    bt: int,
    bf: int,
    bts: int,
    input_bytes: int = 2,
    enable_act_quant: bool = False,
    has_weight_scales: bool = False,
    quant_block_k: int | None = None,
    enable_bt_scatter_overlap: bool = True,
    fixed_gather_banks: int | None = 2,
    a2a_mode: str = "expert",
    pad_topk_to_128: bool = True,
    route_smem_topk_only: bool = False,
) -> FusedMoEV2ScratchPlan:
    """Derive the static scratch geometry used by ``fused_ep_moe_v2``."""

    positive = {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "ep_size": ep_size,
        "bt": bt,
        "bf": bf,
        "bts": bts,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Expected positive scratch dimensions, got {invalid}.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} divisible by {ep_size=}.")
    if num_experts % ep_size != 0:
        raise ValueError(f"Expected {num_experts=} divisible by {ep_size=}.")
    local_num_tokens = num_tokens // ep_size
    if local_num_tokens % bt != 0:
        raise ValueError(f"Expected {local_num_tokens=} divisible by {bt=}.")
    if a2a_mode not in _SUPPORTED_A2A_MODES:
        raise ValueError(
            f"Unsupported {a2a_mode=}; expected one of {sorted(_SUPPORTED_A2A_MODES)}."
        )

    input_packing = _packing_for_bytes(input_bytes)
    internal_bytes = 1 if enable_act_quant else input_bytes
    internal_packing = _packing_for_bytes(internal_bytes)
    output_packing = input_packing
    for label, packing in (
        ("input", input_packing),
        ("internal", internal_packing),
        ("output", output_packing),
    ):
        if hidden_size % packing != 0:
            raise ValueError(f"Expected {hidden_size=} divisible by {label} {packing=}.")

    num_bt = local_num_tokens // bt
    use_bt_scatter_bank = enable_bt_scatter_overlap and num_bt > 1
    use_gather_bank = num_bt > 1
    use_bt_banking = use_bt_scatter_bank or use_gather_bank
    use_fixed_gather_banks = (
        use_gather_bank and fixed_gather_banks is not None and fixed_gather_banks < num_bt
    )
    if use_fixed_gather_banks and fixed_gather_banks < 2:
        raise ValueError("fixed_gather_banks must be >= 2 when the fixed-bank path is active.")
    gather_bank_count = int(fixed_gather_banks) if use_fixed_gather_banks else num_bt
    num_bt_banks = gather_bank_count if use_gather_bank else (2 if use_bt_scatter_bank else 1)
    smem_banks = gather_bank_count + (1 if use_fixed_gather_banks else 0) if use_gather_bank else 2

    rank_dispatch_enabled = a2a_mode in _RANK_A2A_MODES
    # Broadcast can DMA the original dense HBM token block directly into each
    # destination inbox. Only direct and packed-rank modes need a source-side
    # VMEM staging/packing buffer.
    rank_pack_enabled = a2a_mode in {
        "rank_direct",
        "rank_dispatch",
        "rank_dispatch_combine",
    }
    rank_combine_enabled = a2a_mode == "rank_dispatch_combine"
    route_fields = 2 * top_k if rank_combine_enabled else top_k

    h_per_internal = hidden_size // internal_packing
    h_per_output = hidden_size // output_packing
    if enable_act_quant and not has_weight_scales:
        raise ValueError("enable_act_quant requires quantized weights with scales.")
    if quant_block_k is not None:
        if quant_block_k <= 0:
            raise ValueError(f"Expected {quant_block_k=} > 0.")
        if h_per_internal % quant_block_k != 0:
            raise ValueError(
                f"Expected h_per_internal={h_per_internal} divisible by {quant_block_k=}."
            )
        if bf % quant_block_k != 0:
            raise ValueError(f"Expected {bf=} divisible by {quant_block_k=}.")
    per_channel = has_weight_scales and quant_block_k is None
    ffn1_qbk = h_per_internal if per_channel else quant_block_k
    ffn2_qbk = bf if per_channel else quant_block_k
    scale_groups_w13 = h_per_internal // ffn1_qbk if ffn1_qbk is not None else 1
    scale_groups_w2 = bf // ffn2_qbk if ffn2_qbk is not None else 1

    return FusedMoEV2ScratchPlan(
        ep_size=ep_size,
        top_k=top_k,
        hidden_size=hidden_size,
        bt=bt,
        bf=bf,
        bts=bts,
        input_bytes=input_bytes,
        internal_bytes=internal_bytes,
        output_bytes=input_bytes,
        enable_act_quant=enable_act_quant,
        has_weight_scales=has_weight_scales,
        local_num_tokens=local_num_tokens,
        local_num_experts=num_experts // ep_size,
        padded_num_experts=_align_to(num_experts, 128),
        padded_top_k=_align_to(top_k, 128) if pad_topk_to_128 else top_k,
        routing_smem_top_k=(
            top_k if route_smem_topk_only else (_align_to(top_k, 128) if pad_topk_to_128 else top_k)
        ),
        route_fields=route_fields,
        input_packing=input_packing,
        internal_packing=internal_packing,
        output_packing=output_packing,
        h_per_input=hidden_size // input_packing,
        h_per_internal=h_per_internal,
        h_per_output=h_per_output,
        acc_bt=math.gcd(bt, 16),
        num_bt=num_bt,
        use_bt_scatter_bank=use_bt_scatter_bank,
        use_gather_bank=use_gather_bank,
        use_bt_banking=use_bt_banking,
        use_fixed_gather_banks=use_fixed_gather_banks,
        gather_bank_count=gather_bank_count,
        num_bt_banks=num_bt_banks,
        smem_banks=smem_banks,
        a2a_max_tokens=_align_to(bt * ep_size, bts),
        rank_dispatch_enabled=rank_dispatch_enabled,
        rank_pack_enabled=rank_pack_enabled,
        rank_combine_enabled=rank_combine_enabled,
        scale_groups_w13=scale_groups_w13,
        scale_groups_w2=scale_groups_w2,
    )


@dataclass(frozen=True)
class FusedMoEV2VmemEstimate:
    """Logical requested VMEM bytes, broken down by scratch allocation."""

    components: tuple[tuple[str, int], ...]

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.components)

    def as_dict(self) -> dict[str, int]:
        return dict(self.components)

    @property
    def expert_working_set_bytes(self) -> int:
        return sum(size for name, size in self.components if name in _EXPERT_WORKING_SET_COMPONENTS)

    @property
    def mosaic_scoped_overhead_bytes(self) -> int:
        working_set = self.expert_working_set_bytes
        return (
            working_set * _MOSAIC_WORKING_SET_HEADROOM_NUMERATOR
            + _MOSAIC_WORKING_SET_HEADROOM_DENOMINATOR
            - 1
        ) // _MOSAIC_WORKING_SET_HEADROOM_DENOMINATOR

    @property
    def compiler_budget_bytes(self) -> int:
        return self.total_bytes + self.mosaic_scoped_overhead_bytes


def estimate_fused_moe_v2_vmem(
    plan: FusedMoEV2ScratchPlan,
    *,
    weight_bytes: int = 2,
    direct_scaled_dot: bool = True,
) -> FusedMoEV2VmemEstimate:
    """Sum the VMEM shapes requested by the kernel for ``plan``.

    SMEM arrays and semaphore storage are deliberately excluded: they do not
    consume the compiler's VMEM budget.  The returned components mirror the
    VMEM section of ``kernel.py`` in allocation order.
    """

    for name, value in {"weight_bytes": weight_bytes}.items():
        if value <= 0:
            raise ValueError(f"Expected {name} > 0, got {value}.")

    fp32 = 4
    int32 = 4
    wb_slots = 2
    h = plan.hidden_size
    bt = plan.bt
    bf = plan.bf
    bts = plan.bts

    components: list[tuple[str, int]] = [
        ("metadata_mailbox", 2 * plan.ep_size * plan.padded_num_experts * int32),
    ]
    if plan.rank_dispatch_enabled:
        components.append(
            (
                "route_mailbox",
                2 * plan.ep_size * bt * plan.route_fields * int32,
            )
        )
        if plan.rank_pack_enabled:
            components.append(
                (
                    "rank_pack",
                    plan.num_bt_banks * (plan.ep_size + 1) * bt * h * plan.internal_bytes,
                )
            )
        components.append(
            (
                "rank_inbox",
                plan.num_bt_banks * plan.ep_size * bt * h * plan.internal_bytes,
            )
        )
    if plan.rank_combine_enabled:
        rank_acc_bytes = plan.num_bt_banks * plan.ep_size * bt * h * fp32
        components.extend([("rank_partial", rank_acc_bytes), ("rank_return", rank_acc_bytes)])

    components.extend(
        [
            (
                "gather_accumulator",
                2 * plan.top_k * plan.acc_bt * h * plan.output_bytes,
            ),
            ("topk_weights", plan.smem_banks * bt * plan.padded_top_k * fp32),
            ("topk_ids", plan.smem_banks * bt * plan.padded_top_k * int32),
            (
                "output_banks",
                plan.num_bt_banks * bt * h * plan.output_bytes,
            ),
            ("w1_double_buffer", wb_slots * h * bf * weight_bytes),
            ("w3_double_buffer", wb_slots * h * bf * weight_bytes),
            ("w2_double_buffer", wb_slots * h * bf * weight_bytes),
        ]
    )

    if plan.has_weight_scales:
        w13_scale_bytes = wb_slots * plan.internal_packing * plan.scale_groups_w13 * bf * fp32
        w2_scale_bytes = (
            wb_slots * plan.output_packing * plan.scale_groups_w2 * plan.h_per_output * fp32
        )
        components.extend(
            [
                ("w1_scale_double_buffer", w13_scale_bytes),
                ("w3_scale_double_buffer", w13_scale_bytes),
                ("w2_scale_double_buffer", w2_scale_bytes),
            ]
        )
        if not direct_scaled_dot:
            dequant_bytes = h * bf * 2  # bfloat16
            components.extend(
                [
                    ("w1_dequant", dequant_bytes),
                    ("w3_dequant", dequant_bytes),
                    ("w2_dequant", dequant_bytes),
                ]
            )

    components.extend(
        [
            ("gate_accumulator", bts * bf * fp32),
            ("up_accumulator", bts * bf * fp32),
            ("token_stage", bts * h * plan.internal_bytes),
            ("output_accumulator", bts * h * fp32),
            ("output_stage", bts * h * plan.output_bytes),
        ]
    )
    if plan.enable_act_quant:
        components.append(("activation_scales", bts * 128 * fp32))

    return FusedMoEV2VmemEstimate(tuple(components))


def validate_fused_moe_v2_vmem_budget(
    estimate: FusedMoEV2VmemEstimate | int,
    *,
    limit_bytes: int = FUSED_MOE_V2_VMEM_LIMIT_BYTES,
) -> None:
    """Reject a scratch plan whose logical VMEM request exceeds ``limit_bytes``.

    This is intentionally a necessary, not sufficient, feasibility check:
    Mosaic may add layout/alignment overhead beyond the logical array sizes.
    Successful lowering therefore remains the final authority for plans that
    pass this check.
    """

    if limit_bytes <= 0:
        raise ValueError(f"Expected {limit_bytes=} > 0.")
    estimated_bytes = estimate if isinstance(estimate, int) else estimate.compiler_budget_bytes
    if estimated_bytes <= limit_bytes:
        return

    excess_bytes = estimated_bytes - limit_bytes
    raise ValueError(
        "Fused MoE v2 estimated scoped VMEM exceeds the TPU VMEM limit: "
        f"estimated_vmem_bytes={estimated_bytes}, "
        f"vmem_limit_bytes={limit_bytes}, excess_bytes={excess_bytes}."
    )
