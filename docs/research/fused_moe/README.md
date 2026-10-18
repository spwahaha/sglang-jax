# TPU fused MoE experiments and analysis

Start with the [communication report](COMMUNICATION.md) for rank deduplication,
direct-row dispatch, dense rank broadcast, and destination-side combine. It
explains why we tried each design, what we implemented, what the measurements
show, and which explanations still need independent experiments.

The [broader research report](REPORT.md) also covers block tuning, VMEM search,
and other kernel findings.

This branch preserves experimental Pallas fused-MoE v2 code, controlled-routing
experiments, tuning tools, and the reasoning behind the resulting design
choices. It is a research snapshot; it has not been promoted upstream or
qualified for checkpoint-backed serving.

## Questions explored

- Can per-destination-rank token deduplication reduce expert-parallel dispatch
  costs without losing expert-major placement and fine-grained compute overlap?
- How do rank-packed dispatch, direct rank broadcast, and destination-side
  combine change payload, local rearrangement, synchronization, and kernel time?
- Which existing block configurations work better on TPU v6e, and how can VMEM
  accounting and repeatable p50/p95 measurements bound the search?
- Why do GPU and TPU implementations respond differently to similar routing
  ideas? The GPU comparison is source-based architectural analysis, not a GPU
  benchmark conducted in this branch.

## Code and detailed analysis

- [Consolidated analysis](../../developer_guide/fused_moe_v2_optimization_deep_dive.md)
- [Rank-dispatch design and experiments](../../developer_guide/fused_moe_rank_coalesced_a2a_plan.md)
- [Rank-dispatch consultation and numerical caveats](../../developer_guide/fused_moe_rank_dispatch_consultation_brief.md)
- [Tuning experiments and qualification requirements](../../developer_guide/fused_moe_workload_regime_plan.md)
- [Pallas kernel and CPU reference](../../../python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py)
- [Benchmark and candidate generation](../../../python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py)
- [VMEM accounting](../../../python/sgl_jax/srt/kernels/fused_moe/v2/vmem_model.py)
- [Offline evidence selection](../../../python/sgl_jax/srt/kernels/fused_moe/v2/tune_select.py)
- [Routing reference and payload models](../../../python/sgl_jax/tools/moe_rank_routing.py)
- [Single-host TPU runner](../../../scripts/run_fused_moe_rank_a2a_tpu.sh)

## Findings and evidence boundaries

Static tuning uses the repository's existing shape/device lookup mechanism.
It is not a new kernel architecture. Experimental regime overrides remain
empty, the default communication mode remains `expert`, and tail-aware
execution remains disabled by default.

For a MiMo-V2-Flash-shaped FP8 workload on 16 TPU v6e chips across four hosts,
three independent pairs measured median-of-run p50 latency of 1.381356 ms for
the production-resolved fallback and 0.526846 ms for the tuned candidate,
approximately a 2.62x kernel speedup. This is a synthetic-input microbenchmark,
not a model-serving speedup. The historical runs came from a dirty experimental
revision and lack an explicit per-configuration multi-host numerical oracle;
the retained selector correctly reports `promotion_ready=false`.

Rank-packed dispatch and destination-side combine were prototyped and checked
against numerical references. Completed controlled cases were slower than the
expert path. Rank broadcast showed a small-token crossover in the retained
experiments. These results guide subsequent layout and synchronization work;
logical payload models do not establish physical ICI traffic or serving gains.

The [evidence directory](evidence/) includes unmodified JSONL records for the
six MiMo timing runs, rank-broadcast crossover/repeat experiments, and clean
v6e-4 decode/prefill comparisons, plus their historical selector reports.
The [SHA256 manifest](evidence/manifest.json)
records every copied evidence file. Files have descriptive names instead of
host identifiers; record contents and provenance fields are unchanged.
Full historical archives referenced in the older developer notes remain local;
this compact packet does not contain every earlier experiment or device trace.

## Reproduce the retained MiMo selection

From the repository root, with the project's dependencies installed:

```bash
PYTHONPATH=python python -m sgl_jax.srt.kernels.fused_moe.v2.tune_select \
  docs/research/fused_moe/evidence/mimo-v6e16-*.jsonl \
  --baseline-variant mimo_v2_flash_v6e16_prefill_production_lookup
```

This recomputes the research selection from historical measurements. It does
not rerun a TPU kernel or make the historical source revision clean.

## Snapshot validation

On 2026-10-18, the existing local CPU environment (JAX 0.10.2) passed
63 tests and 3 subtests covering routing references, workload-regime defaults,
grouped numerical reference math, selector gates, VMEM accounting, and CLI
contracts. The TPU runner passed `bash -n`. No new TPU or serving run was
performed while preparing this snapshot.

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu python -m pytest -q \
  python/sgl_jax/test/kernels/fused_moe_rank_routing_test.py \
  python/sgl_jax/test/kernels/fused_moe_workload_regime_test.py \
  python/sgl_jax/test/kernels/fused_moe_v2_reference_test.py \
  python/sgl_jax/test/test_fused_moe_v2_tune_select.py \
  python/sgl_jax/test/test_fused_moe_v2_vmem_model.py \
  python/sgl_jax/test/test_fused_moe_a2a_server_args.py
```
