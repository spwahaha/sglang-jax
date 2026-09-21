"""CPU reference utilities for rank-coalesced MoE routing experiments.

This module is deliberately independent of Pallas and TPU-only APIs.  It is a
test/benchmark oracle for the proposed rank-coalesced fused-MoE protocol; the
serving path must not call it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Literal, cast

import numpy as np

RoutingPattern = Literal["one_rank", "two_ranks", "four_ranks", "all_ranks", "random"]


@dataclasses.dataclass(frozen=True)
class RankRoutingPlan:
    """Fixed-capacity CPU representation of expert routes grouped by rank."""

    expert_ids: np.ndarray
    local_expert_ids: np.ndarray
    topk_slot_ids: np.ndarray
    router_weights: np.ndarray
    valid_mask: np.ndarray
    counts: np.ndarray
    unique_rank_mask: np.ndarray

    @property
    def valid_expert_routes(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def unique_rank_routes(self) -> int:
        return int(self.unique_rank_mask.sum())


@dataclasses.dataclass(frozen=True)
class RankRoutingMetrics:
    """Logical payload metrics; these are not physical ICI byte counters."""

    num_tokens: int
    valid_expert_routes: int
    unique_rank_routes: int
    mean_unique_ranks_per_token: float
    rank_overlap_ratio: float
    expert_dispatch_activation_bytes: int
    rank_dispatch_activation_bytes: int
    expert_return_activation_bytes: int
    rank_combine_return_activation_bytes: int
    remote_expert_routes: int | None = None
    unique_remote_rank_routes: int | None = None
    expert_dispatch_remote_activation_bytes: int | None = None
    rank_dispatch_remote_activation_bytes: int | None = None

    @property
    def dispatch_payload_reduction_ratio(self) -> float:
        if self.expert_dispatch_activation_bytes == 0:
            return 0.0
        return 1.0 - (self.rank_dispatch_activation_bytes / self.expert_dispatch_activation_bytes)

    @property
    def remote_dispatch_payload_reduction_ratio(self) -> float | None:
        baseline = self.expert_dispatch_remote_activation_bytes
        optimized = self.rank_dispatch_remote_activation_bytes
        if baseline is None or optimized is None:
            return None
        if baseline == 0:
            return 0.0
        return 1.0 - optimized / baseline

    def as_dict(self) -> dict[str, int | float | None]:
        result = dataclasses.asdict(self)
        # The current v2 kernel scatters and gathers once per valid
        # (token, Top-K slot).  The candidate transfers once per unique
        # destination rank for that token.
        result["current_v2_dispatch_activation_copies"] = self.valid_expert_routes
        result["candidate_rank_dispatch_activation_copies"] = self.unique_rank_routes
        result["current_v2_return_activation_copies"] = self.valid_expert_routes
        result["candidate_rank_combine_return_activation_copies"] = self.unique_rank_routes
        result["current_v2_remote_dispatch_activation_copies"] = self.remote_expert_routes
        result["candidate_remote_rank_dispatch_activation_copies"] = self.unique_remote_rank_routes
        result["dispatch_payload_reduction_ratio"] = self.dispatch_payload_reduction_ratio
        result["remote_dispatch_payload_reduction_ratio"] = (
            self.remote_dispatch_payload_reduction_ratio
        )
        return result


def _validate_layout(*, num_experts: int, ep_size: int) -> int:
    if num_experts <= 0:
        raise ValueError(f"Expected num_experts > 0, got {num_experts}.")
    if ep_size <= 0:
        raise ValueError(f"Expected ep_size > 0, got {ep_size}.")
    if num_experts % ep_size != 0:
        raise ValueError(f"Expected num_experts={num_experts} divisible by ep_size={ep_size}.")
    return num_experts // ep_size


def reference_rank_pack(
    topk_ids: np.ndarray,
    topk_weights: np.ndarray,
    *,
    num_experts: int,
    ep_size: int,
) -> RankRoutingPlan:
    """Group Top-K routes by destination rank without changing route order.

    Invalid expert IDs (negative or greater than/equal to ``num_experts``) are
    ignored. Duplicate expert IDs are preserved as distinct Top-K routes; only
    their activation destination rank is deduplicated.
    """

    experts_per_rank = _validate_layout(num_experts=num_experts, ep_size=ep_size)
    ids = np.asarray(topk_ids)
    weights = np.asarray(topk_weights)
    if ids.ndim != 2:
        raise ValueError(f"Expected rank-2 topk_ids, got shape={ids.shape}.")
    if weights.shape != ids.shape:
        raise ValueError(f"Expected topk_weights.shape={ids.shape}, got {weights.shape}.")

    num_tokens, top_k = ids.shape
    shape = (num_tokens, ep_size, top_k)
    expert_ids = np.full(shape, -1, dtype=np.int32)
    local_expert_ids = np.full(shape, -1, dtype=np.int32)
    topk_slot_ids = np.full(shape, -1, dtype=np.int32)
    packed_weights = np.zeros(shape, dtype=weights.dtype)
    valid_mask = np.zeros(shape, dtype=np.bool_)
    counts = np.zeros((num_tokens, ep_size), dtype=np.int32)

    for token_id in range(num_tokens):
        for topk_slot in range(top_k):
            expert_id = int(ids[token_id, topk_slot])
            if expert_id < 0 or expert_id >= num_experts:
                continue
            rank_id = expert_id // experts_per_rank
            rank_slot = int(counts[token_id, rank_id])
            expert_ids[token_id, rank_id, rank_slot] = expert_id
            local_expert_ids[token_id, rank_id, rank_slot] = expert_id % experts_per_rank
            topk_slot_ids[token_id, rank_id, rank_slot] = topk_slot
            packed_weights[token_id, rank_id, rank_slot] = weights[token_id, topk_slot]
            valid_mask[token_id, rank_id, rank_slot] = True
            counts[token_id, rank_id] += 1

    return RankRoutingPlan(
        expert_ids=expert_ids,
        local_expert_ids=local_expert_ids,
        topk_slot_ids=topk_slot_ids,
        router_weights=packed_weights,
        valid_mask=valid_mask,
        counts=counts,
        unique_rank_mask=counts > 0,
    )


def infer_contiguous_source_ranks(num_tokens: int, ep_size: int) -> np.ndarray:
    """Return source ranks for a token array sharded contiguously over EP."""

    if num_tokens < 0:
        raise ValueError(f"Expected num_tokens >= 0, got {num_tokens}.")
    if ep_size <= 0:
        raise ValueError(f"Expected ep_size > 0, got {ep_size}.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected num_tokens={num_tokens} divisible by ep_size={ep_size}.")
    return np.repeat(np.arange(ep_size, dtype=np.int32), num_tokens // ep_size)


def calculate_rank_routing_metrics(
    topk_ids: np.ndarray,
    *,
    num_experts: int,
    ep_size: int,
    hidden_size: int,
    activation_dtype_bytes: int,
    source_ranks: np.ndarray | None = None,
) -> RankRoutingMetrics:
    """Calculate logical expert- and rank-oriented activation payloads."""

    experts_per_rank = _validate_layout(num_experts=num_experts, ep_size=ep_size)
    ids = np.asarray(topk_ids)
    if ids.ndim != 2:
        raise ValueError(f"Expected rank-2 topk_ids, got shape={ids.shape}.")
    if hidden_size <= 0:
        raise ValueError(f"Expected hidden_size > 0, got {hidden_size}.")
    if activation_dtype_bytes <= 0:
        raise ValueError(f"Expected activation_dtype_bytes > 0, got {activation_dtype_bytes}.")

    valid = (ids >= 0) & (ids < num_experts)
    destination_ranks = np.where(valid, ids // experts_per_rank, -1)
    num_tokens = ids.shape[0]
    rank_mask = np.zeros((num_tokens, ep_size), dtype=np.bool_)
    for token_id in range(num_tokens):
        valid_ranks = destination_ranks[token_id, valid[token_id]]
        rank_mask[token_id, valid_ranks] = True

    valid_routes = int(valid.sum())
    unique_rank_routes = int(rank_mask.sum())
    bytes_per_activation = hidden_size * activation_dtype_bytes
    overlap = 0.0 if valid_routes == 0 else 1.0 - unique_rank_routes / valid_routes

    remote_expert_routes = None
    unique_remote_rank_routes = None
    expert_remote_bytes = None
    rank_remote_bytes = None
    if source_ranks is not None:
        sources = np.asarray(source_ranks)
        if sources.shape != (num_tokens,):
            raise ValueError(f"Expected source_ranks.shape={(num_tokens,)}, got {sources.shape}.")
        if np.any((sources < 0) | (sources >= ep_size)):
            raise ValueError(f"source_ranks must be in [0, {ep_size}), got {sources}.")
        remote = valid & (destination_ranks != sources[:, None])
        remote_rank_mask = rank_mask.copy()
        remote_rank_mask[np.arange(num_tokens), sources] = False
        remote_expert_routes = int(remote.sum())
        unique_remote_rank_routes = int(remote_rank_mask.sum())
        expert_remote_bytes = remote_expert_routes * bytes_per_activation
        rank_remote_bytes = unique_remote_rank_routes * bytes_per_activation

    return RankRoutingMetrics(
        num_tokens=num_tokens,
        valid_expert_routes=valid_routes,
        unique_rank_routes=unique_rank_routes,
        mean_unique_ranks_per_token=(0.0 if num_tokens == 0 else unique_rank_routes / num_tokens),
        rank_overlap_ratio=overlap,
        expert_dispatch_activation_bytes=valid_routes * bytes_per_activation,
        rank_dispatch_activation_bytes=unique_rank_routes * bytes_per_activation,
        expert_return_activation_bytes=valid_routes * bytes_per_activation,
        rank_combine_return_activation_bytes=unique_rank_routes * bytes_per_activation,
        remote_expert_routes=remote_expert_routes,
        unique_remote_rank_routes=unique_remote_rank_routes,
        expert_dispatch_remote_activation_bytes=expert_remote_bytes,
        rank_dispatch_remote_activation_bytes=rank_remote_bytes,
    )


def reference_rank_combine(
    expert_outputs: np.ndarray,
    topk_ids: np.ndarray,
    topk_weights: np.ndarray,
    *,
    num_experts: int,
    ep_size: int,
) -> np.ndarray:
    """Return FP32 weighted expert-output partials grouped by destination rank."""

    experts_per_rank = _validate_layout(num_experts=num_experts, ep_size=ep_size)
    outputs = np.asarray(expert_outputs)
    ids = np.asarray(topk_ids)
    weights = np.asarray(topk_weights)
    if outputs.ndim != 3:
        raise ValueError(
            f"Expected expert_outputs with shape (tokens, top_k, hidden), got {outputs.shape}."
        )
    if ids.shape != outputs.shape[:2]:
        raise ValueError(f"Expected topk_ids.shape={outputs.shape[:2]}, got {ids.shape}.")
    if weights.shape != ids.shape:
        raise ValueError(f"Expected topk_weights.shape={ids.shape}, got {weights.shape}.")

    num_tokens, top_k, hidden_size = outputs.shape
    partials = np.zeros((num_tokens, ep_size, hidden_size), dtype=np.float32)
    outputs_f32 = outputs.astype(np.float32)
    weights_f32 = weights.astype(np.float32)
    for token_id in range(num_tokens):
        for topk_slot in range(top_k):
            expert_id = int(ids[token_id, topk_slot])
            if expert_id < 0 or expert_id >= num_experts:
                continue
            rank_id = expert_id // experts_per_rank
            partials[token_id, rank_id] += (
                outputs_f32[token_id, topk_slot] * weights_f32[token_id, topk_slot]
            )
    return partials


def make_controlled_topk(
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    pattern: RoutingPattern,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic Top-K IDs with controlled destination fanout."""

    experts_per_rank = _validate_layout(num_experts=num_experts, ep_size=ep_size)
    if num_tokens < 0:
        raise ValueError(f"Expected num_tokens >= 0, got {num_tokens}.")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"Expected top_k in [1, {num_experts}], got {top_k}.")

    if pattern == "random":
        rng = np.random.default_rng(seed)
        if num_tokens == 0:
            ids = np.empty((0, top_k), dtype=np.int32)
        else:
            ids = np.stack(
                [rng.choice(num_experts, size=top_k, replace=False) for _ in range(num_tokens)],
                axis=0,
            ).astype(np.int32)
    else:
        requested_ranks = {
            "one_rank": 1,
            "two_ranks": 2,
            "four_ranks": 4,
            "all_ranks": ep_size,
        }.get(pattern)
        if requested_ranks is None:
            raise ValueError(f"Unsupported routing pattern {pattern!r}.")
        if requested_ranks > ep_size:
            raise ValueError(
                f"Pattern {pattern!r} requires {requested_ranks} ranks, got ep_size={ep_size}."
            )
        max_routes_on_one_rank = (top_k + requested_ranks - 1) // requested_ranks
        if max_routes_on_one_rank > experts_per_rank:
            raise ValueError(
                f"Pattern {pattern!r} needs {max_routes_on_one_rank} distinct local experts "
                f"per selected rank, but experts_per_rank={experts_per_rank}."
            )

        ids = np.empty((num_tokens, top_k), dtype=np.int32)
        for token_id in range(num_tokens):
            base_rank = token_id % ep_size
            selected_ranks = [
                (base_rank + rank_offset) % ep_size for rank_offset in range(requested_ranks)
            ]
            for topk_slot in range(top_k):
                rank_position = topk_slot % requested_ranks
                local_round = topk_slot // requested_ranks
                rank_id = selected_ranks[rank_position]
                local_expert_id = (token_id + local_round) % experts_per_rank
                ids[token_id, topk_slot] = rank_id * experts_per_rank + local_expert_id

    descending = np.arange(top_k, 0, -1, dtype=np.float32)
    normalized = descending / descending.sum(dtype=np.float32)
    weights = np.broadcast_to(normalized, (num_tokens, top_k)).copy()
    return ids, weights


def generate_cpu_baseline_records(
    *,
    patterns: list[RoutingPattern],
    token_counts: list[int],
    top_k: int,
    num_experts: int,
    ep_size: int,
    hidden_size: int,
    activation_dtype_bytes: int,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Generate JSON-serializable logical-payload records without JAX or TPU."""

    records: list[dict[str, object]] = []
    for num_tokens in token_counts:
        source_ranks = infer_contiguous_source_ranks(num_tokens, ep_size)
        for pattern in patterns:
            topk_ids, _ = make_controlled_topk(
                num_tokens=num_tokens,
                top_k=top_k,
                num_experts=num_experts,
                ep_size=ep_size,
                pattern=pattern,
                seed=seed,
            )
            metrics = calculate_rank_routing_metrics(
                topk_ids,
                num_experts=num_experts,
                ep_size=ep_size,
                hidden_size=hidden_size,
                activation_dtype_bytes=activation_dtype_bytes,
                source_ranks=source_ranks,
            )
            records.append(
                {
                    "evidence_type": "predicted_logical_payload",
                    "baseline_protocol": "current_v2_expert_route_scatter_and_gather",
                    "candidate_protocol": "rank_coalesced_scatter_and_weighted_rank_combine",
                    "measurement_scope": "cpu_route_trace_not_physical_ici_or_latency",
                    "routing_pattern": pattern,
                    "seed": seed,
                    "num_tokens": num_tokens,
                    "num_experts": num_experts,
                    "ep_size": ep_size,
                    "top_k": top_k,
                    "hidden_size": hidden_size,
                    "activation_dtype_bytes": activation_dtype_bytes,
                    **metrics.as_dict(),
                }
            )
    return records


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_patterns(value: str) -> list[RoutingPattern]:
    supported = {"one_rank", "two_ranks", "four_ranks", "all_ranks", "random"}
    patterns = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(patterns) - supported)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported patterns {invalid}; expected values from {sorted(supported)}"
        )
    return cast(list[RoutingPattern], patterns)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CPU-only rank-coalesced MoE logical-payload evidence."
    )
    parser.add_argument(
        "--patterns",
        type=_parse_patterns,
        default=_parse_patterns("one_rank,two_ranks,four_ranks,random"),
    )
    parser.add_argument("--token-counts", type=_parse_csv_ints, default=[128, 512])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--activation-dtype-bytes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Write JSONL to this path; omit to print JSONL to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    records = generate_cpu_baseline_records(
        patterns=args.patterns,
        token_counts=args.token_counts,
        top_k=args.top_k,
        num_experts=args.num_experts,
        ep_size=args.ep_size,
        hidden_size=args.hidden_size,
        activation_dtype_bytes=args.activation_dtype_bytes,
        seed=args.seed,
    )
    lines = [json.dumps(record, sort_keys=True) for record in records]
    if args.output is None:
        sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
