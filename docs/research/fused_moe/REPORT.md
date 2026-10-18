# TPU Fused MoE Optimization Research

Experiments, results, and engineering conclusions

Research snapshot prepared on 2026-10-18. Implementation snapshot:
[`5448acf1`](https://github.com/spwahaha/sglang-jax/commit/5448acf11e0826efa36b65191ada1590ecb60336).

## Purpose and main findings

We investigated how to make the fused mixture-of-experts layer in SGLang-JAX
more efficient on TPU. The work covered expert-parallel communication, the
layout consumed by expert matrix multiplication, block tiling, and VMEM
feasibility. The objective was to identify useful changes through prototypes
and controlled experiments, and explain the tradeoffs well enough to guide
subsequent implementation.

The strongest measured improvement came from tuning the kernel's existing
block configurations. A MiMo-V2-Flash-shaped FP8 microbenchmark on 16 TPU v6e
chips improved from 1.381356 ms to 0.526846 ms in median-of-run p50 latency,
approximately a 2.62x kernel speedup. This experiment used synthetic inputs and
has outstanding correctness and provenance requirements before promotion.

Communication experiments produced a different lesson. Sending fewer logical
activation rows did not consistently reduce total kernel time. The existing
expert path already deposits incoming rows in the layout needed by compute
and exposes readiness separately for each expert. Rank-oriented alternatives
changed those properties as well as the payload.

| Investigation | What we learned | Current decision |
|---|---|---|
| Rank-packed dispatch | Fewer logical rows; slower completed kernel cases | Keep as research |
| Destination-side combine | Coupled prototype correct but slower; precision changes the byte tradeoff | Isolate before revisiting |
| Direct rank broadcast | Small-token benefit in matched-tile experiments; larger-token regression | Qualify the narrow region |
| Static block tuning | Large repeated improvements over the resolved fallback | Prioritize model qualification |
| Tail-aware execution | Helps a padded layout; small extra gain over the best static layout | Keep disabled |
| Regime selection and VMEM tools | Useful experimental plumbing; no independent serving gain demonstrated | Retain tools and conservative defaults |

All performance figures in this report are synthetic-input kernel experiments
on real TPUs. We did not collect a checkpoint-backed serving A/B for this MoE
work or run a GPU benchmark. Historical measurements are analyzed here;
preparing this report did not rerun the accelerator experiments.

<!-- pagebreak -->

## The baseline and what we built

An MoE router selects multiple experts for each token. An expert-parallel rank
owns a subset of those experts. Several selected experts may live on the same
rank, creating an opportunity to send a token's activation once to that rank
and reuse it locally.

We use `T` for global token rows in one layer invocation, `H` for hidden width,
`I` for expert intermediate width, `E` for expert count, and `K` for selected
experts per token. `T` is not sequence length or a count of serving requests.
EP is the number of expert-parallel ranks.

### Why the existing expert path is a strong baseline

The [Pallas kernel](../../../python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py)
combines routing-related movement and expert computation in one pipeline.

| Stage | Existing behavior | Performance significance |
|---|---|---|
| Dispatch | One row transfer for each token/expert route | Repeats a row when several experts need it |
| Destination placement | DMA writes into final expert-major queue slots | No separate rank-to-expert rearrangement |
| Readiness | Per-expert receive synchronization | One expert can proceed while another receives |
| Expert FFN | Contiguous row tiles and streamed weight slices | Regular inputs for the matrix pipeline |
| Return and combine | Expert outputs return to the source and are weighted | Retains expert-level pipeline opportunities |

The baseline does not minimize modeled transfer starts. Its advantage is that
data movement also creates the final compute layout. Rotating scratch banks
and completion waits protect buffer reuse.

### Implementation and evaluation work

We added rank-packed dispatch, rank-direct diagnostics, dense rank broadcast,
and destination-side combine prototypes. We built CPU routing/combination
references, controlled routing patterns, logical payload accounting, and
benchmark modes to compare the alternatives. Further work added a shared VMEM
model, bounded configuration generation, failure records, and an offline
selector for repeated timing evidence.

The experiments used one-rank, multi-rank, random, and skewed routing patterns
to vary fan-out and imbalance. Comparisons were interpreted within their own
hardware, shape, precision, and block-configuration controls. Timings from
different experiment families are not interchangeable baselines.

The [benchmark](../../../python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py),
[routing references](../../../python/sgl_jax/tools/moe_rank_routing.py), and
[single-host runner](../../../scripts/run_fused_moe_rank_a2a_tpu.sh) preserve
these controls. The initial packed-dispatch results below are single-run
means; later repeated experiments explicitly summarize run p50 values.

<!-- pagebreak -->

## Rank dispatch and destination combine

### Hypothesis and prototype

Rank-packed dispatch deduplicates destination ranks per token, packs the
selected rows into rank buckets, transfers those buckets, then reconstructs
the local expert inputs. Under controlled one-rank routing with `K=8`, this
reduces logical activation copies by 87.5 percent. This is an algorithmic
count, not a measurement of ICI traffic or exposed communication time.

The prototype receives a rank-major inbox. Before expert computation, it must
recover expert membership and gather indexed rows into contiguous compute
tiles. This introduces a local layout conversion that the baseline avoids.

### Observed results

The initial experiments used a single v6e-4, BF16 activations, `E=32`, `K=8`,
`I=1024`, and controlled one-rank routing. The optimized cells passed the
reported numerical comparisons. These are historical single-run mean timings
from an experimental working revision.

| T | H | Expert baseline ms | Rank packed ms | Latency change |
|---:|---:|---:|---:|---:|
| 128 | 1024 | 0.071 | 0.106 | 49.3 percent higher |
| 128 | 8192 | 0.328 | 0.337 | 2.7 percent higher |
| 512 | 8192 | 1.258 | 1.334 | 6.0 percent higher |

The `T=512` case is especially useful: substantially fewer logical rows still
produced a slower whole kernel. Reduced payload alone therefore did not
identify the best execution strategy for these cases.

### What can explain the regression

Code inspection establishes added source packing, expert-to-inbox metadata
reconstruction, indexed expert-row loads, and coarser readiness. These are
credible sources of extra local work and lost overlap. Aggregate timings do
not identify their individual contributions or prove which one dominates.

The rank-direct diagnostic removed ragged source packing while retaining
destination reconstruction. Historical notes say it did not recover the
baseline, but a standalone timing table was not retained. This argues against
source packing being the sole explanation; it is not a controlled causal
attribution of the remaining costs.

### Destination-side combine

The coupled combine prototype weights expert outputs at the destination and
returns FP32 partial sums per token/rank pair. Its completed small-shape case
measured 0.108 ms against 0.071 ms for the expert baseline, 52.1 percent higher
latency. The reported numerical errors passed the experiment's reference.

FP32 partials use twice the bytes of BF16 rows. At `K=8`, return bytes decrease
only when mean destination-rank fan-out is below four. The experiment couples
combine with packed dispatch, so it cannot establish that destination combine
alone is slow. An independent combine ablation would be needed.

Sources: [consultation brief](../../developer_guide/fused_moe_rank_dispatch_consultation_brief.md)
and [detailed communication analysis](../../developer_guide/fused_moe_v2_optimization_deep_dive.md).

<!-- pagebreak -->

## Rank broadcast and the GPU comparison

### Why we tried sending more regular data

Direct-HBM rank broadcast sends an unchanged dense source block to participating
ranks and lets routing metadata identify relevant rows. It avoids ragged source
packing and uses regular transfers, at the cost of sending rows that some
destinations do not use. It also removed an unused source staging buffer.

On v6e-4 with BF16, random routing, `H=8192`, `I=1024`, `E=32`, `K=8`, EP=4:

| T | Expert p50 ms | Broadcast p50 ms | Comparison |
|---:|---:|---:|---|
| 32 | 0.303173 | 0.288712 | 4.77 percent lower latency |
| 64 | 0.313143 | 0.300163 | 4.15 percent lower latency |
| 128 | 0.448801 | 0.445009 | 0.84 percent lower latency |
| 256 | 0.949149 | 0.953810 | 0.49 percent higher latency |
| 512 | 2.006105 | 2.037165 | 1.55 percent higher latency |

The first two rows use the median of run p50s from three alternating A/B pairs.
The last three are a single crossover sweep. Modes used matched explicit tiles,
but the records mark them as not matching the production-resolved configuration.
Small-token timing records do not include per-configuration numerical checks;
separate checks were reported at `T=128` and `512`.

The crossover is consistent with a startup/regularity advantage at small token
counts being overtaken by extra payload and inbox work at larger counts. This
is a plausible explanation, not a measured decomposition of those costs.
The positive region still needs a production-baseline and serving comparison.

### Why a GPU analogy does not determine the TPU outcome

The source comparison examined [DeepEP](https://github.com/deepseek-ai/DeepEP)
and [CUTLASS grouped GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html).
DeepEP's expanded dispatch representation separates rank-level transport from
aligned expert segments. CUTLASS's contiguous grouped-GEMM interface consumes
segments identified by expert offsets. Local expert expansion remains part of
the design; communication deduplication does not remove that work.

Our TPU prototype retained a rank inbox and performed indexed consumption near
expert computation. The useful design question is how to produce expert-ready
tiles and readiness events efficiently after deduplicated transport. It is not
whether a GPU technique can be copied without its compute-layout contract.

TPUs support asynchronous remote DMA and pipelining, as documented in the
[JAX distributed Pallas guide](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html)
and [pipelining guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).
We ran no GPU A/B. The comparison motivates tiled expert expansion and finer
readiness experiments; it does not prove a fundamental GPU/TPU limitation.

<!-- pagebreak -->

## Static tuning and VMEM feasibility

### The capability audit changed the direction

The repository already has device- and shape-keyed block configuration lookup.
At the audited snapshot, fused-MoE v2 has TPU v7 entries and a generic fallback,
but no TPU v6e table. The actionable gap is measured v6e configurations and
qualification. Adding a new model-specific tuning architecture was unnecessary.

The tuple is `(bt, bf, btc, bse, bts)`: outer source-token block, intermediate
weight slice, expert compute-row group, shared-expert block, and resident
expert-row window. The experiments changed these existing parameters while
retaining expert dispatch.

| Experiment | Baseline p50 ms | Candidate p50 ms | Latency reduction |
|---|---:|---:|---:|
| v6e-4 BF16, T=128 | 0.448601 | 0.336544 | 24.98 percent |
| v6e-4 BF16, T=4096 | 18.734134 | 3.982071 | 78.74 percent |
| v6e-16 MiMo shape, T=2048 | 1.381356 | 0.526846 | 61.86 percent |

These are medians of run p50s from three independent comparisons against the
production-resolved fallback `(32,512,32,256,32)`. The v6e-4 shape uses
`H=8192`, `I=1024`, `E=32`, `K=8`, EP=4; its winners are
`(32,256,64,256,64)` and `(128,256,160,256,160)` respectively.

The MiMo-shaped test uses `E=256`, `H=4096`, `I=2048`, `K=8`, EP=16,
block-wise FP8 weights, activation quantization, and BF16 output. The winning
tuple is `(128,1024,80,256,80)`. No checkpoint was loaded.

Larger useful row windows and weight slices can amortize loop/control and
movement costs. The measured tuple comparison establishes a benefit for the
combined configuration, not an isolated causal contribution from each knob.
The legal non-power-of-two value 80 also illustrates why measurement and layout
constraints should choose a tile rather than a power-of-two convention.

### Correctness and provenance remain separate gates

The clean v6e-4 records pass the retained selector's synthetic correctness and
provenance gates. The MiMo records pass its performance gate but lack explicit
multi-host per-configuration oracle results and came from a dirty revision.
Its `promotion_ready=false` remains correct. Committing this snapshot does not
retroactively change the provenance of a historical run.

### The VMEM estimator is a pruning tool

A logical-under-64-MiB check admitted five candidates whose Mosaic scoped
allocations were 65.63, 68.44, 72.82, 81.62, and 94.60 MiB. We added a calibrated
1.20x allowance on relevant expert working-set components. CPU checks reject
those observed failures while retaining observed successful candidates.

The 64-MiB value is this kernel's compiler budget, not the TPU's advertised
physical memory capacity. The empirical allowance needs confirmation on TPU
and may change with compiler, precision, shape, or communication mode.
Successful lowering and execution remain the final feasibility checks.

<!-- pagebreak -->

## Tail handling and workload selection

### Tail-aware execution

The tail-aware experiment keeps a fixed expert compute-row group but shortens
the loop when the final resident tile contains entirely empty groups. It does
not remove padding inside a nonempty matrix group.

| MiMo routing | Matched padded layout ms | Tail-aware ms | Best static ms |
|---|---:|---:|---:|
| Random | 0.705 | 0.519 | 0.527 |
| Hot expert | 3.240 | 3.083 | 3.117 |

Against the same padded layout, the reductions are approximately 26.4 percent
and 4.8 percent. Against the best static layout, the extra benefit is only
about 1.5 percent and 1.1 percent. These rounded historical comparison points
are exploratory measurements, not serving qualification or the three-run
MiMo static result above.

The experiment demonstrates useful work avoidance for particular layouts.
It does not establish that enabling the mechanism is better than selecting
the best static tuple. Tail awareness remains disabled by default.

### Workload-regime plumbing

We added a static hint mapping plain decode to a latency objective, extend to
throughput, and ambiguous modes to `auto`. Compatibility tests cover the
mapping and lookup behavior. The override table remains empty and `auto`
preserves the existing lookup.

The existing key already includes token count and model/runtime dimensions.
A smaller decode-shaped experiment and a larger prefill-shaped experiment
therefore do not prove that an extra phase key is needed. A useful override
requires the same complete existing key to have different measured winners
because of routing distribution or serving objective.

### Engineering lesson

The audit separated three types of work: missing measurements for an existing
configuration mechanism, experimental execution changes, and selection
plumbing. That distinction prevented us from crediting static tuning gains to
an unproven new architecture.

For subsequent optimization work, first trace the production selection path,
record its resolved baseline, and determine whether existing configuration
can express the idea. Prototype a new mechanism only where that audit exposes
a real limitation.

Sources: [workload investigation](../../developer_guide/fused_moe_workload_regime_plan.md),
[configuration lookup](../../../python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py),
and [VMEM model](../../../python/sgl_jax/srt/kernels/fused_moe/v2/vmem_model.py).

<!-- pagebreak -->

## What the results establish and what comes next

### Confidence in the explanations

| Statement | Evidence level | Remaining uncertainty |
|---|---|---|
| Some rank variants are slower despite lower logical payload | Observed in completed controlled kernel cases | Generality across real routing and hardware |
| Rank prototypes add packing, indexed loads, and different readiness | Direct code observation | Their individual critical-path costs |
| The static candidate improves the measured MiMo-shaped kernel | Three independent timing pairs | Clean-source numerical oracle and serving impact |
| Broadcast regularity explains the small-token crossover | Plausible interpretation of code and timing | Stage-level attribution and production baseline |
| GPU-style expert expansion could improve the TPU design | Source-informed design hypothesis | Efficient implementation and measured benefit |

The central causal limit is the absence of a stage-resolved trace that
attributes the regression. Code inspection identifies credible mechanisms,
but summing nominal stage times would also be misleading if stages overlap.
We need to identify work exposed on the critical path.

### Decisions and reopening conditions

Keep expert dispatch as the default, leave regime overrides empty, and keep
tail awareness off. Retain prototypes, negative results, and benchmark tools
as the basis for targeted follow-up.

| Next experiment | Question it resolves | Decision condition |
|---|---|---|
| Clean MiMo-shaped rerun with explicit oracle | Is the static tuple correct and repeatable from a clean source? | Correctness, compile feasibility, and stable timing pass together |
| Checkpoint-backed fallback versus tuned tuple | Does the kernel result improve actual serving? | Natural buckets and routing; TTFT, TPOT/ITL, throughput, and memory checked |
| Stage-resolved transport/layout ablations | Where are packing, expansion, waits, and FFN exposed? | Optimize only a material measured bottleneck |
| Small-token broadcast with resolved production tiles | Does the narrow positive region survive the real baseline? | Repeated correct comparisons and serving benefit |
| Tiled expert expansion or independent destination combine | Can deduplication preserve expert-ready layout and overlap? | Whole-kernel gain over the same expert baseline |

No accelerator resources were launched for this report. These are proposed
qualification experiments, not completed results.

<!-- pagebreak -->

## Evidence and reproducibility

### Published records

The [evidence packet](evidence/) preserves unmodified JSONL records for the
three MiMo baseline/candidate pairs, rank-broadcast crossover and repeated
small-token runs, and clean v6e-4 decode/prefill comparisons. Selector reports
preserve the distinction between a performance winner and a qualified
candidate. The [SHA256 manifest](evidence/manifest.json) identifies every
copied file; descriptive filenames replace host-based names without changing
record contents.

The packed-dispatch, combine, hot-expert, and tail-aware tables are reported
from the historical [detailed analysis](../../developer_guide/fused_moe_v2_optimization_deep_dive.md)
and [consultation record](../../developer_guide/fused_moe_rank_dispatch_consultation_brief.md).
The compact packet does not contain all original logs or device traces. Their
coverage should not be assumed equivalent to the repeated, directly retained
MiMo or broadcast records.

### Recompute the MiMo selection

From the repository root with project dependencies installed:

```bash
PYTHONPATH=python python -m sgl_jax.srt.kernels.fused_moe.v2.tune_select \
  docs/research/fused_moe/evidence/mimo-v6e16-*.jsonl \
  --baseline-variant mimo_v2_flash_v6e16_prefill_production_lookup
```

This summarizes historical timing records. It does not rerun a TPU kernel.
The result should retain `performance_gate_passed=true` and
`promotion_ready=false` for the reasons discussed above. Statistics in this
report use median-of-run p50; earlier notes also quote an across-run mean,
which explains the small difference between 61.86 and 61.82 percent.

### Validation boundaries

The research snapshot passed 63 CPU tests and 3 subtests in the existing local
JAX 0.10.2 environment. Tests cover routing references, grouped reference
math, default/configuration contracts, evidence-selection gates, and VMEM
accounting. Commit hooks passed after formatting. CPU validation does not
establish TPU lowering, race freedom across all device schedules, physical
traffic, numerical parity for untested configurations, or serving performance.

### Reference guide

- [Research code and entry points](README.md)
- [Full kernel and GPU comparison notes](../../developer_guide/fused_moe_v2_optimization_deep_dive.md)
- [Tuning methodology and experiment history](../../developer_guide/fused_moe_workload_regime_plan.md)
- [Pallas remote DMA and synchronization](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html)
- [Pallas pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
- [DeepEP source and examples](https://github.com/deepseek-ai/DeepEP)
- [CUTLASS contiguous grouped GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html)
