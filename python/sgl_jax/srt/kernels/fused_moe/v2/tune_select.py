"""Offline winner selection for fused MoE v2 JSONL benchmark evidence.

This tool never edits ``tuned_block_configs.py``.  It produces a reviewable
report that distinguishes a provisional performance winner from a configuration
that has enough independent samples and correctness evidence to be promoted.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SELECTION_KEY_FIELDS = (
    "git_revision",
    "device_kind",
    "device_count",
    "process_count",
    "activation_dtype",
    "weight_dtype",
    "quant_block_k",
    "tokens",
    "num_experts",
    "top_k",
    "hidden_size",
    "intermediate_size",
    "ep_size",
    "use_shared_expert",
    "shared_expert_intermediate_size",
    "enable_act_quant",
    "enable_bt_scatter_overlap",
    "fixed_gather_banks",
    "direct_scaled_dot",
    "a2a_mode",
    "timing_method",
    "swiglu_limit",
    "shared_swiglu_limit",
    "forward_mode",
    "workload_regime",
    "routing_mode",
)

_PASS_STATUSES = frozenset({"pass", "passed", "correct"})
_FAIL_STATUSES = frozenset({"fail", "failed", "incorrect", "mismatch"})
_EXECUTION_FAILURE_STATUSES = frozenset({"failed", "config_rejected", "no_timing_data"})
_BLOCK_CONFIG_FIELDS = ("bt", "bf", "btc", "bse", "bts")


def load_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                record["_evidence_path"] = str(path)
                records.append(record)
    return records


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _selection_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in SELECTION_KEY_FIELDS)


def _config_key(record: dict[str, Any]) -> tuple[Any, ...]:
    resolved = record.get("resolved_block_config")
    if isinstance(resolved, dict):
        return (
            *(resolved.get(field) for field in ("bt", "bf", "btc", "bse", "bts")),
            record.get("cross_expert_prefetch_mode"),
            bool(record.get("tail_aware_btc", False)),
        )
    return (record.get("block_config"),)


def _matches_production_block_config(record: dict[str, Any]) -> bool:
    """Prefer normalized evidence fields over a possibly stale recorded flag."""
    resolved = record.get("resolved_block_config")
    production = record.get("production_resolved_block_config")
    if isinstance(resolved, dict) and isinstance(production, dict):
        return tuple(resolved.get(field) for field in _BLOCK_CONFIG_FIELDS) == tuple(
            production.get(field) for field in _BLOCK_CONFIG_FIELDS
        )
    return record.get("matches_production_block_config") is True


def _correctness_status(records: list[dict[str, Any]]) -> str:
    statuses = {str(record.get("correctness_status", "unknown")).lower() for record in records}
    if statuses & _FAIL_STATUSES:
        return "failed"
    if statuses & _PASS_STATUSES:
        return "passed"
    return "unknown"


def _summarize_config(
    records: list[dict[str, Any]],
    *,
    min_samples: int,
    min_runs: int,
    max_cv: float,
) -> dict[str, Any]:
    samples_by_run: dict[str, list[float]] = defaultdict(list)
    successful_records = [
        record for record in records if record.get("execution_status", "success") == "success"
    ]
    for record in successful_records:
        run_id = record.get("run_id")
        if run_id is None:
            continue
        samples_by_run[str(run_id)].extend(float(sample) for sample in record.get("samples_ms", []))

    run_summaries = []
    for run_id, run_samples in sorted(samples_by_run.items()):
        if not run_samples:
            continue
        run_mean = statistics.fmean(run_samples)
        run_std = statistics.pstdev(run_samples)
        run_summaries.append(
            {
                "run_id": run_id,
                "sample_count": len(run_samples),
                "mean_ms": run_mean,
                "p50_ms": _percentile(run_samples, 50),
                "p95_ms": _percentile(run_samples, 95),
                "std_ms": run_std,
                "cv": run_std / run_mean if run_mean else math.inf,
            }
        )

    samples = [sample for run_samples in samples_by_run.values() for sample in run_samples]
    run_ids = set(samples_by_run)
    reasons: list[str] = []
    failed_execution_count = sum(
        str(record.get("execution_status", "success")).lower() in _EXECUTION_FAILURE_STATUSES
        for record in records
    )
    if failed_execution_count:
        reasons.append(f"execution_failures={failed_execution_count}")
    if any(record.get("run_id") is None for record in successful_records):
        reasons.append("successful evidence is missing run_id")
    if len(samples) < min_samples:
        reasons.append(f"samples={len(samples)} < {min_samples}")
    if len(run_ids) < min_runs:
        reasons.append(f"independent_runs={len(run_ids)} < {min_runs}")

    if run_summaries:
        run_p50s = [run["p50_ms"] for run in run_summaries]
        run_p95s = [run["p95_ms"] for run in run_summaries]
        mean = statistics.fmean(run["mean_ms"] for run in run_summaries)
        p50 = _percentile(run_p50s, 50)
        p95 = _percentile(run_p95s, 50)
        worst_run_p95 = max(run_p95s)
        std = statistics.pstdev(run_p50s)
        cv = std / statistics.fmean(run_p50s) if statistics.fmean(run_p50s) else math.inf
        max_within_run_cv = max(run["cv"] for run in run_summaries)
        if cv > max_cv:
            reasons.append(f"run_p50_cv={cv:.4f} > {max_cv:.4f}")
        if max_within_run_cv > max_cv:
            reasons.append(f"max_within_run_cv={max_within_run_cv:.4f} > {max_cv:.4f}")
    else:
        mean = std = p50 = p95 = None
        worst_run_p95 = None
        cv = None
        max_within_run_cv = None
        reasons.append("no successful timing samples")

    correctness = _correctness_status(records)
    if correctness == "failed":
        reasons.append("correctness failed")

    revisions = {record.get("git_revision") for record in records}
    provenance_clean = (
        len(revisions) == 1
        and None not in revisions
        and all(record.get("git_dirty") is False for record in records)
    )

    first = records[0]
    production_match_values = {
        _matches_production_block_config(record) for record in successful_records
    }
    matches_production_block_config = production_match_values == {True}
    evidence_paths = sorted(
        {str(record["_evidence_path"]) for record in records if "_evidence_path" in record}
    )
    return {
        "variant": first.get("variant"),
        "block_config": first.get("block_config"),
        "resolved_block_config": first.get("resolved_block_config"),
        "cross_expert_prefetch_mode": first.get("cross_expert_prefetch_mode"),
        "tail_aware_btc": bool(first.get("tail_aware_btc", False)),
        "config_source": first.get("config_source"),
        "production_resolved_block_config": first.get("production_resolved_block_config"),
        "matches_production_block_config": matches_production_block_config,
        "sample_count": len(samples),
        "independent_run_count": len(run_ids),
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "worst_run_p95_ms": worst_run_p95,
        "std_ms": std,
        "run_p50_std_ms": std,
        "cv": cv,
        "run_p50_cv": cv,
        "max_within_run_cv": max_within_run_cv,
        "aggregation": "median_of_independent_run_percentiles",
        "run_summaries": run_summaries,
        "correctness_status": correctness,
        "provenance_clean": provenance_clean,
        "stable": not reasons,
        "rejection_reasons": reasons,
        "evidence_paths": evidence_paths,
    }


def select_tuned_configs(
    records: Iterable[dict[str, Any]],
    *,
    baseline_variant: str = "fused_moe_v2_baseline",
    min_samples: int = 20,
    min_runs: int = 3,
    max_cv: float = 0.05,
    min_p50_improvement: float = 0.03,
    max_p95_regression: float = 0.02,
) -> list[dict[str, Any]]:
    """Build one deterministic selection report per complete workload key."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_selection_key(record)].append(record)

    reports: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(v) for v in value)):
        workload_records = grouped[key]
        by_config: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
        for record in workload_records:
            identity = (str(record.get("variant")), _config_key(record))
            by_config[identity].append(record)

        summaries = [
            _summarize_config(
                config_records,
                min_samples=min_samples,
                min_runs=min_runs,
                max_cv=max_cv,
            )
            for _, config_records in sorted(by_config.items(), key=lambda item: str(item[0]))
        ]
        stable = [summary for summary in summaries if summary["stable"]]
        baselines = [summary for summary in stable if summary["variant"] == baseline_variant]
        candidates = [summary for summary in stable if summary["variant"] != baseline_variant]
        baseline = min(baselines, key=lambda item: item["p50_ms"], default=None)
        winner = min(candidates, key=lambda item: item["p50_ms"], default=None)

        reasons: list[str] = []
        p50_improvement = None
        p95_regression = None
        worst_run_p95_regression = None
        if baseline is None:
            reasons.append(f"no stable baseline variant {baseline_variant!r}")
        if winner is None:
            reasons.append("no stable candidate")
        if baseline is not None and winner is not None:
            p50_improvement = (baseline["p50_ms"] - winner["p50_ms"]) / baseline["p50_ms"]
            p95_regression = (winner["p95_ms"] - baseline["p95_ms"]) / baseline["p95_ms"]
            worst_run_p95_regression = (
                winner["worst_run_p95_ms"] - baseline["worst_run_p95_ms"]
            ) / baseline["worst_run_p95_ms"]
            if p50_improvement < min_p50_improvement:
                reasons.append(f"p50_improvement={p50_improvement:.4f} < {min_p50_improvement:.4f}")
            if p95_regression > max_p95_regression:
                reasons.append(f"p95_regression={p95_regression:.4f} > {max_p95_regression:.4f}")
            if worst_run_p95_regression > max_p95_regression:
                reasons.append(
                    "worst_run_p95_regression="
                    f"{worst_run_p95_regression:.4f} > {max_p95_regression:.4f}"
                )

        performance_gate_passed = not reasons
        correctness_passed = winner is not None and winner["correctness_status"] == "passed"
        provenance_clean = (
            baseline is not None
            and winner is not None
            and baseline["provenance_clean"]
            and winner["provenance_clean"]
        )
        promotion_reasons: list[str] = []
        if baseline is not None and not baseline["matches_production_block_config"]:
            promotion_reasons.append(
                "baseline is not verified as the production-resolved block config"
            )
        if not correctness_passed:
            promotion_reasons.append("winner correctness is not explicitly passed")
        if not provenance_clean:
            promotion_reasons.append("baseline or winner provenance is not clean")
        workload = dict(zip(SELECTION_KEY_FIELDS, key, strict=True))
        reports.append(
            {
                "workload": workload,
                "baseline": baseline,
                "winner": winner,
                "p50_improvement": p50_improvement,
                "p95_regression": p95_regression,
                "worst_run_p95_regression": worst_run_p95_regression,
                "performance_gate_passed": performance_gate_passed,
                "promotion_ready": (
                    performance_gate_passed
                    and correctness_passed
                    and provenance_clean
                    and not promotion_reasons
                ),
                "gate_reasons": reasons,
                "promotion_gate_reasons": promotion_reasons,
                "all_config_summaries": sorted(
                    summaries,
                    key=lambda item: (
                        item["p50_ms"] is None,
                        item["p50_ms"] if item["p50_ms"] is not None else math.inf,
                    ),
                ),
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", help="One or more bench_v2 JSONL evidence files")
    parser.add_argument("--baseline-variant", default="fused_moe_v2_baseline")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--min-p50-improvement", type=float, default=0.03)
    parser.add_argument("--max-p95-regression", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = select_tuned_configs(
        load_jsonl(args.jsonl),
        baseline_variant=args.baseline_variant,
        min_samples=args.min_samples,
        min_runs=args.min_runs,
        max_cv=args.max_cv,
        min_p50_improvement=args.min_p50_improvement,
        max_p95_regression=args.max_p95_regression,
    )
    rendered = json.dumps(reports, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
