"""CPU-only tests for deterministic fused MoE v2 tuning selection."""

from __future__ import annotations

from sgl_jax.srt.kernels.fused_moe.v2.tune_select import select_tuned_configs


def _record(
    *,
    variant: str,
    run_id: str,
    samples: list[float],
    config: tuple[int, int, int, int, int],
    correctness: str = "passed",
    tail_aware_btc: bool = False,
):
    bt, bf, btc, bse, bts = config
    return {
        "variant": variant,
        "run_id": run_id,
        "execution_status": "success",
        "correctness_status": correctness,
        "git_revision": "0123456789abcdef",
        "git_dirty": False,
        "samples_ms": samples,
        "block_config": f"bt={bt},bf={bf},btc={btc},bse={bse},bts={bts}",
        "resolved_block_config": {"bt": bt, "bf": bf, "btc": btc, "bse": bse, "bts": bts},
        "device_kind": "TPU v6e",
        "device_count": 4,
        "process_count": 1,
        "activation_dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "quant_block_k": 128,
        "tokens": 512,
        "num_experts": 32,
        "top_k": 8,
        "hidden_size": 8192,
        "intermediate_size": 1024,
        "ep_size": 4,
        "use_shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "enable_act_quant": False,
        "enable_bt_scatter_overlap": True,
        "fixed_gather_banks": 2,
        "direct_scaled_dot": False,
        "a2a_mode": "expert",
        "timing_method": "trace",
        "swiglu_limit": None,
        "shared_swiglu_limit": None,
        "cross_expert_prefetch_mode": "full",
        "tail_aware_btc": tail_aware_btc,
        "config_source": (
            "production_lookup" if variant == "fused_moe_v2_baseline" else "tune_candidate"
        ),
        "production_resolved_block_config": {
            "bt": bt,
            "bf": bf,
            "btc": btc,
            "bse": bse,
            "bts": bts,
        },
        "matches_production_block_config": True,
        "forward_mode": "DECODE",
        "workload_regime": "latency",
        "routing_mode": "random",
    }


def _three_runs(variant, value, config, correctness="passed"):
    return [
        _record(
            variant=variant,
            run_id=f"run-{index}",
            samples=[value] * 10,
            config=config,
            correctness=correctness,
        )
        for index in range(3)
    ]


def test_stable_candidate_passes_performance_and_promotion_gates():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    records += _three_runs("candidate", 0.9, (64, 512, 32, 256, 64))

    [report] = select_tuned_configs(records)

    assert report["winner"]["variant"] == "candidate"
    assert report["p50_improvement"] > 0.09
    assert report["performance_gate_passed"]
    assert report["promotion_ready"]


def test_unknown_correctness_can_win_performance_but_cannot_be_promoted():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    records += _three_runs(
        "candidate",
        0.9,
        (64, 512, 32, 256, 64),
        correctness="not_checked_per_config",
    )

    [report] = select_tuned_configs(records)

    assert report["performance_gate_passed"]
    assert not report["promotion_ready"]


def test_under_sampled_candidate_is_rejected():
    baseline = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    candidate = [
        _record(
            variant="candidate",
            run_id="only-run",
            samples=[0.8] * 10,
            config=(64, 512, 32, 256, 64),
        )
    ]

    [report] = select_tuned_configs(baseline + candidate)

    assert report["winner"] is None
    assert not report["performance_gate_passed"]
    candidate_summary = next(
        summary for summary in report["all_config_summaries"] if summary["variant"] == "candidate"
    )
    assert any("samples=" in reason for reason in candidate_summary["rejection_reasons"])
    assert any("independent_runs=" in reason for reason in candidate_summary["rejection_reasons"])


def test_p95_regression_blocks_a_low_p50_candidate():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    candidate_samples = [0.8] * 9 + [1.5]
    records += [
        _record(
            variant="candidate",
            run_id=f"run-{index}",
            samples=candidate_samples,
            config=(64, 512, 32, 256, 64),
        )
        for index in range(3)
    ]

    [report] = select_tuned_configs(records, max_cv=1.0)

    assert report["p50_improvement"] > 0.03
    assert report["p95_regression"] > 0.02
    assert not report["performance_gate_passed"]


def test_dirty_evidence_is_provisional_not_promotion_ready():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    records += _three_runs("candidate", 0.9, (64, 512, 32, 256, 64))
    for record in records:
        record["git_dirty"] = True

    [report] = select_tuned_configs(records)

    assert report["performance_gate_passed"]
    assert not report["promotion_ready"]


def test_nonproduction_baseline_blocks_promotion_but_not_performance_result():
    baseline = _three_runs("fused_moe_v2_baseline", 1.0, (32, 256, 32, 256, 32))
    candidate = _three_runs("candidate", 0.8, (32, 256, 64, 256, 64))
    for record in baseline:
        record["config_source"] = "explicit"
        record["matches_production_block_config"] = False
        record["production_resolved_block_config"] = {
            "bt": 32,
            "bf": 512,
            "btc": 32,
            "bse": 256,
            "bts": 32,
        }

    [report] = select_tuned_configs(baseline + candidate)

    assert report["performance_gate_passed"]
    assert not report["promotion_ready"]
    assert any("production-resolved" in reason for reason in report["promotion_gate_reasons"])


def test_normalized_configs_repair_stale_false_production_match_flag():
    baseline = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    candidate = _three_runs("candidate", 0.8, (64, 512, 64, 256, 64))
    for record in baseline:
        # Direct bench execution used to load two equivalent dataclasses under
        # different module names, making object equality false despite these
        # two normalized dictionaries being identical.
        record["matches_production_block_config"] = False

    [report] = select_tuned_configs(baseline + candidate)

    assert report["baseline"]["matches_production_block_config"]
    assert report["promotion_ready"]


def test_each_run_has_equal_weight_even_when_iteration_counts_differ():
    baseline = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    candidate = [
        _record(
            variant="candidate",
            run_id="short-fast-run",
            samples=[0.8] * 100,
            config=(64, 512, 32, 256, 64),
        ),
        _record(
            variant="candidate",
            run_id="normal-run-1",
            samples=[1.1] * 10,
            config=(64, 512, 32, 256, 64),
        ),
        _record(
            variant="candidate",
            run_id="normal-run-2",
            samples=[1.1] * 10,
            config=(64, 512, 32, 256, 64),
        ),
    ]

    [report] = select_tuned_configs(baseline + candidate, max_cv=1.0)

    assert report["winner"]["p50_ms"] == 1.1
    assert not report["performance_gate_passed"]


def test_cross_run_and_within_run_dispersion_are_reported_separately():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    records += [
        _record(
            variant="candidate",
            run_id="run-0",
            samples=[0.8, 1.0] * 5,
            config=(64, 512, 32, 256, 64),
        ),
        _record(
            variant="candidate",
            run_id="run-1",
            samples=[0.9] * 10,
            config=(64, 512, 32, 256, 64),
        ),
        _record(
            variant="candidate",
            run_id="run-2",
            samples=[0.9] * 10,
            config=(64, 512, 32, 256, 64),
        ),
    ]

    [report] = select_tuned_configs(records, max_cv=0.05)
    candidate = next(
        summary for summary in report["all_config_summaries"] if summary["variant"] == "candidate"
    )

    assert candidate["run_p50_cv"] == 0.0
    assert candidate["max_within_run_cv"] > 0.05
    assert any("max_within_run_cv" in reason for reason in candidate["rejection_reasons"])


def test_explicit_correctness_record_qualifies_timing_records():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    candidate = _three_runs(
        "candidate",
        0.9,
        (64, 512, 32, 256, 64),
        correctness="not_checked_per_config",
    )
    correctness = dict(candidate[0])
    correctness.update(
        execution_status="correctness_only",
        correctness_status="passed",
        samples_ms=[],
    )

    [report] = select_tuned_configs(records + candidate + [correctness])

    assert report["winner"]["correctness_status"] == "passed"
    assert report["promotion_ready"]


def test_execution_failure_rejects_otherwise_stable_config():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    candidate = _three_runs("candidate", 0.9, (64, 512, 32, 256, 64))
    failed = dict(candidate[0])
    failed.update(execution_status="failed", samples_ms=[])

    [report] = select_tuned_configs(records + candidate + [failed])

    summary = next(
        item for item in report["all_config_summaries"] if item["variant"] == "candidate"
    )
    assert not summary["stable"]
    assert any("execution_failures=" in reason for reason in summary["rejection_reasons"])


def test_one_bad_run_p95_cannot_hide_behind_median_run():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    candidate = _three_runs("candidate", 0.9, (64, 512, 32, 256, 64))
    candidate[0]["samples_ms"] = [0.8] * 9 + [1.5]

    [report] = select_tuned_configs(records + candidate, max_cv=1.0)

    assert report["p95_regression"] < 0.02
    assert report["worst_run_p95_regression"] > 0.02
    assert not report["performance_gate_passed"]


def test_prefetch_modes_are_not_pooled_as_one_configuration():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 32, 256, 32))
    full = _three_runs("candidate", 0.9, (64, 512, 32, 256, 64))
    none = _three_runs("candidate", 1.2, (64, 512, 32, 256, 64))
    for record in none:
        record["cross_expert_prefetch_mode"] = "none"

    [report] = select_tuned_configs(records + full + none)

    candidates = [
        summary for summary in report["all_config_summaries"] if summary["variant"] == "candidate"
    ]
    assert len(candidates) == 2
    assert report["winner"]["p50_ms"] == 0.9


def test_tail_aware_modes_are_not_pooled_as_one_configuration():
    records = _three_runs("fused_moe_v2_baseline", 1.0, (32, 512, 8, 256, 32))
    static = _three_runs("candidate", 1.1, (32, 512, 8, 256, 32))
    dynamic = [
        _record(
            variant="candidate",
            run_id=f"run-{index}",
            samples=[0.8] * 10,
            config=(32, 512, 8, 256, 32),
            tail_aware_btc=True,
        )
        for index in range(3)
    ]

    [report] = select_tuned_configs(records + static + dynamic)

    candidates = [
        summary for summary in report["all_config_summaries"] if summary["variant"] == "candidate"
    ]
    assert len(candidates) == 2
    assert report["winner"]["tail_aware_btc"] is True
    assert report["winner"]["p50_ms"] == 0.8
