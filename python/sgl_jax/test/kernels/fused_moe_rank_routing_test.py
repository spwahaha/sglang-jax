"""CPU tests for rank-coalesced fused-MoE routing semantics."""

from __future__ import annotations

import unittest

import numpy as np

from sgl_jax.tools.moe_rank_routing import (
    calculate_rank_routing_metrics,
    generate_cpu_baseline_records,
    infer_contiguous_source_ranks,
    make_controlled_topk,
    reference_rank_combine,
    reference_rank_pack,
)


class RankRoutingReferenceTest(unittest.TestCase):
    def test_controlled_patterns_have_expected_fanout_and_payload(self):
        expected = {
            "one_rank": (1, 0.875),
            "two_ranks": (2, 0.75),
            "four_ranks": (4, 0.5),
        }
        for pattern, (expected_ranks, expected_reduction) in expected.items():
            with self.subTest(pattern=pattern):
                ids, weights = make_controlled_topk(
                    num_tokens=4,
                    top_k=8,
                    num_experts=32,
                    ep_size=4,
                    pattern=pattern,
                )
                plan = reference_rank_pack(
                    ids,
                    weights,
                    num_experts=32,
                    ep_size=4,
                )
                metrics = calculate_rank_routing_metrics(
                    ids,
                    num_experts=32,
                    ep_size=4,
                    hidden_size=1024,
                    activation_dtype_bytes=2,
                )

                np.testing.assert_array_equal(
                    plan.unique_rank_mask.sum(axis=1),
                    np.full(4, expected_ranks),
                )
                self.assertEqual(metrics.valid_expert_routes, 4 * 8)
                self.assertEqual(metrics.unique_rank_routes, 4 * expected_ranks)
                self.assertEqual(metrics.expert_dispatch_activation_bytes, 4 * 8 * 1024 * 2)
                self.assertEqual(
                    metrics.rank_dispatch_activation_bytes,
                    4 * expected_ranks * 1024 * 2,
                )
                self.assertAlmostEqual(
                    metrics.dispatch_payload_reduction_ratio,
                    expected_reduction,
                )

    def test_pack_preserves_experts_slots_and_weights(self):
        ids = np.array([[0, 1, 8, 9, 16, 17, 24, 25]], dtype=np.int32)
        weights = np.array([[0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04]])
        plan = reference_rank_pack(
            ids,
            weights,
            num_experts=32,
            ep_size=4,
        )

        reconstructed_ids = np.full_like(ids, -1)
        reconstructed_weights = np.zeros_like(weights)
        for rank_id in range(4):
            for rank_slot in range(plan.counts[0, rank_id]):
                topk_slot = plan.topk_slot_ids[0, rank_id, rank_slot]
                reconstructed_ids[0, topk_slot] = plan.expert_ids[0, rank_id, rank_slot]
                reconstructed_weights[0, topk_slot] = plan.router_weights[0, rank_id, rank_slot]

        np.testing.assert_array_equal(reconstructed_ids, ids)
        np.testing.assert_allclose(reconstructed_weights, weights)

    def test_invalid_routes_are_ignored_without_renormalizing_weights(self):
        ids = np.array([[0, -1, 8, 32]], dtype=np.int32)
        weights = np.array([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
        plan = reference_rank_pack(
            ids,
            weights,
            num_experts=32,
            ep_size=4,
        )

        self.assertEqual(plan.valid_expert_routes, 2)
        self.assertEqual(plan.unique_rank_routes, 2)
        self.assertAlmostEqual(plan.router_weights[0, 0, 0], 0.4)
        self.assertAlmostEqual(plan.router_weights[0, 1, 0], 0.2)

    def test_duplicate_experts_remain_distinct_routes(self):
        ids = np.array([[3, 3, 9]], dtype=np.int32)
        weights = np.array([[0.5, 0.25, 0.25]], dtype=np.float32)
        plan = reference_rank_pack(
            ids,
            weights,
            num_experts=32,
            ep_size=4,
        )

        self.assertEqual(plan.valid_expert_routes, 3)
        self.assertEqual(plan.unique_rank_routes, 2)
        np.testing.assert_array_equal(plan.topk_slot_ids[0, 0, :2], [0, 1])
        np.testing.assert_allclose(plan.router_weights[0, 0, :2], [0.5, 0.25])

    def test_rank_partials_equal_ordinary_weighted_topk_combine(self):
        ids = np.array(
            [
                [0, 1, 8, 9, 16, 17, 24, 25],
                [7, 15, 23, 31, 6, 14, 22, 30],
            ],
            dtype=np.int32,
        )
        weights = np.array(
            [
                [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04],
                [0.22, 0.18, 0.16, 0.14, 0.12, 0.08, 0.06, 0.04],
            ],
            dtype=np.float32,
        )
        outputs = np.arange(2 * 8 * 5, dtype=np.float32).reshape(2, 8, 5) / 10

        ordinary = np.sum(outputs * weights[..., None], axis=1, dtype=np.float32)
        rank_partials = reference_rank_combine(
            outputs,
            ids,
            weights,
            num_experts=32,
            ep_size=4,
        )

        np.testing.assert_allclose(rank_partials.sum(axis=1), ordinary, rtol=1e-6, atol=1e-6)

    def test_remote_payload_excludes_source_rank(self):
        ids = np.array(
            [
                [0, 1, 8, 9, 16, 17, 24, 25],
                [8, 9, 16, 17, 24, 25, 0, 1],
                [16, 17, 24, 25, 0, 1, 8, 9],
                [24, 25, 0, 1, 8, 9, 16, 17],
            ],
            dtype=np.int32,
        )
        metrics = calculate_rank_routing_metrics(
            ids,
            num_experts=32,
            ep_size=4,
            hidden_size=1024,
            activation_dtype_bytes=2,
            source_ranks=infer_contiguous_source_ranks(4, 4),
        )

        self.assertEqual(metrics.remote_expert_routes, 24)
        self.assertEqual(metrics.unique_remote_rank_routes, 12)
        self.assertAlmostEqual(metrics.remote_dispatch_payload_reduction_ratio, 0.5)

        evidence = metrics.as_dict()
        self.assertEqual(evidence["current_v2_dispatch_activation_copies"], 32)
        self.assertEqual(evidence["candidate_rank_dispatch_activation_copies"], 16)
        self.assertEqual(evidence["current_v2_return_activation_copies"], 32)
        self.assertEqual(evidence["candidate_rank_combine_return_activation_copies"], 16)
        self.assertEqual(evidence["current_v2_remote_dispatch_activation_copies"], 24)
        self.assertEqual(evidence["candidate_remote_rank_dispatch_activation_copies"], 12)

    def test_random_pattern_is_reproducible_and_has_unique_experts(self):
        first_ids, first_weights = make_controlled_topk(
            num_tokens=16,
            top_k=8,
            num_experts=32,
            ep_size=4,
            pattern="random",
            seed=7,
        )
        second_ids, second_weights = make_controlled_topk(
            num_tokens=16,
            top_k=8,
            num_experts=32,
            ep_size=4,
            pattern="random",
            seed=7,
        )

        np.testing.assert_array_equal(first_ids, second_ids)
        np.testing.assert_allclose(first_weights, second_weights)
        for row in first_ids:
            self.assertEqual(len(np.unique(row)), 8)
        np.testing.assert_allclose(first_weights.sum(axis=1), 1.0)

    def test_invalid_layout_and_pattern_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_controlled_topk(
                num_tokens=1,
                top_k=2,
                num_experts=10,
                ep_size=4,
                pattern="one_rank",
            )
        with self.assertRaisesRegex(ValueError, "requires 4 ranks"):
            make_controlled_topk(
                num_tokens=1,
                top_k=2,
                num_experts=8,
                ep_size=2,
                pattern="four_ranks",
            )

    def test_empty_random_routing_has_zero_payload(self):
        ids, weights = make_controlled_topk(
            num_tokens=0,
            top_k=8,
            num_experts=32,
            ep_size=4,
            pattern="random",
        )
        metrics = calculate_rank_routing_metrics(
            ids,
            num_experts=32,
            ep_size=4,
            hidden_size=1024,
            activation_dtype_bytes=2,
        )

        self.assertEqual(ids.shape, (0, 8))
        self.assertEqual(weights.shape, (0, 8))
        self.assertEqual(metrics.valid_expert_routes, 0)
        self.assertEqual(metrics.unique_rank_routes, 0)
        self.assertEqual(metrics.dispatch_payload_reduction_ratio, 0.0)

    def test_cpu_baseline_records_are_json_serializable_and_labelled_predicted(self):
        records = generate_cpu_baseline_records(
            patterns=["one_rank", "four_ranks"],
            token_counts=[4],
            top_k=8,
            num_experts=32,
            ep_size=4,
            hidden_size=1024,
            activation_dtype_bytes=2,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["evidence_type"], "predicted_logical_payload")
        self.assertEqual(
            records[0]["baseline_protocol"],
            "current_v2_expert_route_scatter_and_gather",
        )
        self.assertEqual(
            records[0]["measurement_scope"],
            "cpu_route_trace_not_physical_ici_or_latency",
        )
        self.assertEqual(records[0]["current_v2_dispatch_activation_copies"], 32)
        self.assertEqual(records[0]["candidate_rank_dispatch_activation_copies"], 4)
        self.assertEqual(records[0]["dispatch_payload_reduction_ratio"], 0.875)
        self.assertEqual(records[1]["current_v2_dispatch_activation_copies"], 32)
        self.assertEqual(records[1]["candidate_rank_dispatch_activation_copies"], 16)
        self.assertEqual(records[1]["dispatch_payload_reduction_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
