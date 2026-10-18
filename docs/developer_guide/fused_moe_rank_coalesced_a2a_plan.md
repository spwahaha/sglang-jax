# Fused MoE Rank-Coalesced A2A Implementation Plan

## Status

**Last updated:** 2026-09-27 (America/Los_Angeles)

This document is the live execution record for a private proof of concept. It
is not yet a proposal to change the public SGLang-JAX production default.

Current private implementation status on
`codex/fused-moe-rank-coalesced-a2a`:

- consultation snapshot and TPU-native redesign options:
  `docs/developer_guide/fused_moe_rank_dispatch_consultation_brief.md`;
- workload-regime selection, compute-layout experiments, current/proposed
  diagrams, and acceptance gates:
  `docs/developer_guide/fused_moe_workload_regime_plan.md`;
- complete: NumPy reference rank packing and payload accounting;
- complete: controlled one-, two-, four-, all-, and random-rank generators;
- complete: CPU tests for metadata preservation, exact payload formulas,
  invalid routes, duplicate experts, remote-only accounting, and weighted
  rank-partial equivalence;
- complete: CPU JSONL evidence generation, with logical activation rows and
  modeled DMA launches labelled separately from physical ICI measurements;
- complete: controlled-routing and JSONL payload reporting wired into
  `bench_v2.py`;
- complete for the small synthetic shape: one current-v2 TPU baseline run for
  every routing-pattern and token-count cell, including numerical reference
  checks per routing mode;
- complete: A2 static `a2a_mode` API, benchmark plumbing, and JSONL labelling;
- implemented and TPU-correct: fixed-capacity packed `rank_dispatch`, including
  route metadata all-gather, per-destination VMEM packing, bulk rank DMA, and
  destination expert-queue reconstruction;
- implemented and TPU-correct at the small shape: experimental
  `rank_dispatch_combine`, with destination-side FP32 weighted partials and
  packed rank-partial return;
- complete locally: modeled packed-DMA launch accounting keyed by the resolved
  local `bt`; 36 focused CPU tests plus 3 subtests pass;
- implemented and TPU-correct: two source-layout diagnostics,
  `rank_direct` (exact deduplicated rows, no ragged pack) and `rank_broadcast`
  (one unchanged dense source block per participating rank). Both retain the
  current rank-inbox consumer so their A/B isolates source packing, DMA starts,
  and payload bytes;
- implemented and TPU-correct: direct-HBM `rank_broadcast`, which removes the
  unused source-side VMEM pack/stage and its semaphore. It recovers 5,242,880
  bytes at the production `T=512, bf=512` shape and passes 128- and 512-token
  reference checks;
- complete: exact VMEM preflight is now enforced for explicit benchmark
  configurations and in the kernel. Requested token buckets with no successful
  timing make the benchmark exit nonzero after preserving failure evidence;
- complete locally: dense-broadcast launch/row accounting and payload-only
  bandwidth-time arithmetic. The consultation brief now includes DeepSeek-V3
  and MiMo-V2-Flash per-layer and model-wide ceilings;
- complete locally: a resumable single-host TPU runner records Git provenance,
  validates the four-device TPU topology, launches every communication mode in
  an independent process, and preserves JSONL plus stdout for smoke, repeated,
  and bank-reuse stress phases;
- complete for the current v6e-4 synthetic gate: random-routing crossover and
  three independent alternating runs. Direct-HBM `rank_broadcast` is 4.77%
  faster at 32 global tokens and 4.15% faster at 64, but crosses to a regression
  between 128 and 256 tokens. It is therefore a latency/decode specialization,
  not a universal replacement;
- complete: explicit default-off `--fused-moe-a2a-mode` propagation through
  server args, model config, all four `FusedEPMoEV2` model families, and the
  static kernel argument. Non-default modes require `--moe-backend fused_v2`;
- complete: exact production-relative decode tile qualification on a clean
  v6e-4. The unchanged `expert` transport improves from 0.448601 ms to
  0.336544 ms p50 (24.98%) through workload-specific compute layout alone;
- complete: first TPU tail-aware-loop ablation and bounded prefill tile sweep.
  Tail awareness helps padded shapes. The prefill winner qualifies at
  3.982 ms versus 18.734 ms for the production fallback across three clean
  runs, with correctness and the exact-baseline selector gate passing;
- pending: real-model A/B on hardware large enough for a supported
  `FusedEPMoEV2` checkpoint, followed by larger EP/multi-host qualification.
  Packed rank dispatch and rank-combine remain research variants because
  neither has passed a performance gate.

Verified TPU development environment:

```text
SSH target:       moe
Checkout:         /home/gcpuser/sky_workdir/sglang-jax-moe-test
Remote branch:    codex/fused-moe-rank-coalesced-a2a-test
Python:           .venv/bin/python
JAX:              0.11.1
Backend:          TPU
Local devices:    4
Latest remote base: de68656974b5b4aeb8f418e4d2451b60cbee4fc5
Evidence:         /tmp/fused_moe_rank_a2a/*.jsonl
Local evidence:   mtp-run/fused-moe-rank-a2a-20260927/
Latest VM:        moe-2-6fe92778-head-5j2zsdqa-tpu, v6e-4, 35.239.175.30
Final VM state:   Sky cluster `moe` absent after qualification; verified with
                  `sky status moe` on 2026-09-27
```

The later qualification VM is an externally provided ephemeral target reached
through `ssh moe_2`. It runs a clean isolated checkout at commit
`477b8b326eb3496f76326aac88350534965c408b`, reuses the original checkout's
virtual environment, and has four TPU v6 lite devices. Every completed JSONL
and log is copied immediately to
`mtp-run/fused-moe-tuning-20260927/moe_2-exact-477b8b32/`; no result is treated
as durable while it exists only on that VM.

The final prefill qualification used clean follow-up commit
`2a7ab5ef65989bbc5a9483b92fbee7d472c3f884`. A complete 52-file remote evidence
snapshot is retained at
`mtp-run/fused-moe-tuning-20260927/moe_2-remote-final/`. After confirming no
benchmark process remained, `sky down -y moe_2` completed and `sky status
moe_2` reported the cluster absent on 2026-09-27.

TPU resource lifecycle for follow-up work:

1. Check `sky status moe` before a pending TPU test.
2. If the saved cluster is stopped or unreachable, relaunch it only when the
   requested test can run on its topology. Do not relaunch v6e-4 for the
   blocked MiMo-V2-Flash model A/B, which needs v6e-16 or v7x-8.
3. Copy raw JSONL/log evidence back to the local evidence directory before
   releasing the resource.
4. Once all runnable verification is complete, run `sky down -y moe` and
   confirm that `sky status moe` reports the cluster absent.

The isolated TPU checkout intentionally reuses the main checkout's virtual
environment. The runner prepends its own `python/` directory to `PYTHONPATH` so
editable-install metadata cannot silently import the clean base checkout. The
working tree is dirty because this remains a private proof of concept; raw
evidence records the base revision plus `git_dirty=true` and therefore cannot
yet be promoted as clean production tuning data.

### Latest TPU decision data

BF16, v6e-4, `H=8192`, `I=1024`, `E=32`, Top-K 8, EP=4, `bf=512`, random
routing, trace device time:

| Global tokens | Expert p50 (ms) | Direct-HBM rank broadcast p50 (ms) | Change |
|---:|---:|---:|---:|
| 32 | 0.303375 | 0.288794 | 4.81% faster |
| 64 | 0.313474 | 0.300118 | 4.26% faster |
| 128 | 0.448801 | 0.445009 | 0.84% faster |
| 256 | 0.949149 | 0.953810 | 0.49% slower |
| 512 | 2.006105 | 2.037165 | 1.55% slower |

The qualification gate uses independent processes rather than only the sweep:

| Global tokens | Expert median of run p50s | Broadcast median of run p50s | Improvement |
|---:|---:|---:|---:|
| 32 | 0.303173 ms | 0.288712 ms | 4.77% |
| 64 | 0.313143 ms | 0.300163 ms | 4.15% |

Each row has three alternating-order processes and ten samples per process.
The direct-HBM broadcast path also passed reference correctness at `T=128`
(`max_abs_err=0.0020`, relative error `0.006289`) and `T=512`
(`max_abs_err=0.0020`, relative error `0.005714`). At controlled four-rank
`T=512`, it compiles after the VMEM reduction but remains slower than expert
(1.288 ms versus 1.269 ms), reinforcing the small-token specialization.

### Real-model A/B readiness and current blocker

The code path is ready for a controlled serving comparison. The two launch
commands must be identical except for:

```text
--moe-backend fused_v2 --fused-moe-a2a-mode expert
--moe-backend fused_v2 --fused-moe-a2a-mode rank_broadcast
```

The available `moe_2` VM is a v6e-4 with no target model checkpoint. It cannot
run the MiMo-V2-Flash qualification: the checked-in
recipe requires v6e-16 or v7x-8, and the 309B-parameter checkpoint is much
larger than the available disk. Gemma 31B/27B and MiMo-7B fit v6e-4 but are dense
models, so they do not exercise this MoE kernel. DeepSeek-V2-Lite fits v6e-4,
but its supported recipe uses `epmoe`, not `FusedEPMoEV2`; using it would test a
different implementation.

Therefore the next real-model run requires one of:

1. a v6e-16 MiMo-V2-Flash slice with the checkpoint available through a model
   volume/cache;
2. a v7x-8 MiMo-V2-Flash host with the checkpoint available; or
3. a smaller supported checkpoint whose architecture actually constructs
   `FusedEPMoEV2`.

For each mode, record identical prompt/output distributions, warmup and compile
policy, concurrency 32/64/128 decode buckets, TTFT, TPOT/ITL, output-token
throughput, routing fanout, compiler/cache time, and deterministic output or
logprob agreement. Keep `expert` for prefill and buckets at or above the
measured crossover until a separate policy is justified. Do not download a
large checkpoint to this spot VM: even if storage were expanded, v6e-4
does not have the required accelerator capacity for the target model.

The public RFC should be opened only after the evidence gate in this document is
met. A baseline alone can demonstrate duplicated payload and establish a
theoretical opportunity, but only a TPU proof-of-concept can demonstrate an
actual latency or throughput improvement.

## Section-by-section status

| Section | Status | Latest evidence or next action |
|---|---|---|
| Decision, terminology, repository map | Settled for private POC | V2-first, static mode, no public flag before A2 passes. |
| A1.1 CPU reference/accounting | Complete | 12/12 CPU tests pass locally; payload rows and modeled DMA launches are now separate evidence. |
| A1.2 controlled routes | Complete | One-, two-, four-, all-, and seeded-random generators implemented. |
| A1.3 small TPU baseline | Complete for current v6e-4 shape | Controlled routes, random crossover, correctness, and three alternating small-token repetitions are recorded. |
| A1.3 realistic TPU stress shape | Not started | Run a fitting E=256 shape after baseline repetition automation is recorded. |
| A1 baseline checklist | Complete for current BF16 scope | CPU oracle, device-time evidence, correctness, crossover, and independent repetitions pass. |
| A2 rank-dispatch POC | Implemented, correct, not yet faster | Packed one-way transport passes 128- and 512-token correctness; best measured cells still trail `expert`. |
| A2 rank-dispatch-combine POC | Implemented, correct, not faster | H=1024 and H=8192 small-shape checks pass; no measured performance win. |
| A2 `rank_direct` diagnostic | TPU-correct, not faster | Exact deduplicated rows from an unchanged source tile; it matched packed dispatch latency at T=128, showing source packing was not the main loss. |
| A2 `rank_broadcast` diagnostic | Small-token performance gate passed | Direct HBM-to-inbox transport passes correctness; +4.77% at T=32 and +4.15% at T=64 across three runs, but regresses at T>=256. |
| TPU benchmark automation | Complete | `smoke`, configurable alternating `repeat`, and `stress` phases emit one log/JSONL pair per process and fail closed on missing token buckets. |
| A2 semaphore/stress work | In progress | Four-BT one-way run passes correctness; two-way bank-reuse and routing matrix remain. |
| A2 performance gate | Passed only for small-token broadcast | Random routing: +4.77% at T=32 and +4.15% at T=64. Packed dispatch/combine still fail; broadcast regresses beyond the crossover. |
| E0 opportunity/attribution gate | Not started | Trace existing regions, compare local versus shifted-remote identical workloads, and sweep bytes versus DMA descriptors before another transport rewrite. |
| Model byte/time gate | Formula and reference table complete | Use observed remote routes; DeepSeek/MiMo `T=128` payload-only ceilings are microseconds/layer, not automatic kernel wins. |
| Decode/prefill workload specialization | Synthetic decode and prefill gates passed | Exact decode tile tuning wins 24.98%; prefill tile tuning wins 78.74% across three clean runs with correctness. Broadcast belongs only on measured small-token latency buckets. Real-model A/B remains. |
| Serving flag | Implemented, default off | `--fused-moe-a2a-mode` reaches MiMo, Bailing, Bailing Linear, and GLM v2 layers; `expert` remains the default. |
| Stage B public RFC | Blocked on real-model gate | Synthetic small-token broadcast passes; do not open until target-model TPOT/throughput evidence exists. |
| Public PR 1 | Private equivalent complete | Clean up only after A1/A2 evidence justifies upstream discussion. |
| Public PRs 2-4 | Private PR2 equivalent implemented | Broadcast kernel/flag exists privately; combine remains separate and negative, qualification pending. |
| v6e-4 qualification | Synthetic complete | Current hardware cannot host a compatible `FusedEPMoEV2` checkpoint. |
| v7x/model/multi-host qualification | Externally blocked | Provision v6e-16 or v7x-8 plus MiMo-V2-Flash weights. |
| Current go/no-go state | Ready, externally blocked for real-model A/B | Current v6e-4 has no compatible checkpoint; provision v6e-16 for MiMo-V2-Flash or v7x-8, then compare decode TPOT plus prefill/throughput. |

## Decision summary

**Status: settled for the private proof of concept; a default-off private
server argument exists, but no upstream/public production default has changed.**

The proposed feature has two separately reviewable optimizations:

1. **Rank-coalesced dispatch**: transmit one token activation per unique
   destination rank rather than one activation per selected expert.
2. **Destination-side partial combine**: on each destination rank, apply the
   original router weights and sum that rank's selected expert outputs before
   returning one partial output per source token.

Implementation should start in fused MoE v2. The feature should be selected by
a static communication-mode argument, not by adding a public `fused_v3`
backend. V1 can be considered after v2 demonstrates a repeatable benefit.

Proposed modes:

```text
expert                 Existing expert-granular dispatch and return
rank_direct            Exact rank-dedup rows, direct row DMAs, existing return
rank_broadcast         Dense HBM source block per participating rank, existing return
rank_dispatch           Rank-coalesced dispatch; existing expert-granular return
rank_dispatch_combine   Rank-coalesced dispatch and rank-partial return
```

During the private proof-of-concept, the mode is exposed through the v2
microbenchmark and a default-off serving flag for real-model A/B. Any public
implementation should retain `expert` as the default until model qualification
is complete.

## Terminology and invariants

**Status: implemented in the CPU oracle and covered by unit tests; the Pallas
candidate must still demonstrate these invariants on TPU.**

For token `t`, let the router choose `K` expert IDs and weights:

```text
topk_ids[t]     = [e_0, ..., e_(K-1)]
topk_weights[t] = [w_0, ..., w_(K-1)]
```

With contiguous expert placement:

```text
experts_per_rank = num_experts / ep_size
destination_rank = expert_id // experts_per_rank
local_expert_id  = expert_id % experts_per_rank
```

The existing expert-granular payload is proportional to `K * H` per token,
where `H` is the hidden size. Rank-coalesced dispatch is proportional to
`U_t * H`, where `U_t` is the number of unique destination ranks selected for
token `t`.

The optimization must preserve these invariants:

- The router and Top-K interface do not change.
- Expert placement and weight sharding do not change.
- Router weights are transported unchanged; they are not renormalized per rank.
- Every selected expert contributes exactly once.
- Invalid or padded routes contribute zero.
- The public MoE input and output shapes do not change.
- Shared-expert behavior is unchanged.
- The transport remains the fused Pallas HBM-buffer A2A; this work does not
  replace it with a separate JAX `ragged_all_to_all` operation.

## Repository map

**Status: evidence/test files exist; the v2 kernel has only the defaulted static
mode contract so far. Other listed production files remain planned A2 or later
touch points.**

### Evidence and test code

**Status: implemented and exercised on CPU and TPU benchmark paths.**

- `python/sgl_jax/tools/moe_rank_routing.py` (new): controlled routing
  generators, a NumPy reference rank packer, logical payload accounting, and
  result summaries. This is benchmark/test tooling and is not imported by the
  serving path.
- `python/sgl_jax/test/kernels/fused_moe_rank_routing_test.py` (new): CPU tests
  for rank mapping, metadata, and mathematical equivalence.
- `python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py`: synthetic TPU timing,
  controlled routing selection, and A/B mode comparison.
- `scripts/run_fused_moe_rank_a2a_tpu.sh`: fail-fast v6e-4 preflight and
  resumable smoke/repetition/stress orchestration, with one Python process and
  one JSONL/log pair per mode and repetition.
- `python/sgl_jax/test/kernels/fused_moe_v2_test.py`: TPU correctness and stress
  coverage once production kernel behavior changes.

The NumPy rank packer is a test oracle only. It must not be called by the
serving path.

The repository already uses synthetic MoE-layer inputs for this purpose.
`fused_moe_v2_test.py::gen_moe_inputs` constructs random token activations,
expert W1/W2/W3 arrays, router logits, and Top-K routes, then compares
`fused_ep_moe_v2` with `ref_moe`. `bench_v2.py` similarly creates sharded
synthetic weights and tokens directly on devices. The evidence work extends
this established model-free approach; it does not introduce a fake transformer
model.

### Production code

**Status: private kernel behavior is implemented; serving remains on the
default `expert` mode.** `v2/kernel.py` implements expert, direct-rank,
dense-broadcast, packed-rank, and packed-rank-combine protocols behind a static
argument. No public server argument or automatic production selection exists
yet, so ordinary model execution remains unchanged.

- `python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py`
  - `fused_ep_moe_v2`: validate and forward the static communication mode.
  - `_fused_ep_moe_kernel`: allocate mode-specific scratch and run the selected
    communication protocol.
  - `start_expert_a2a_scatter_batch_range`: current expert-granular scatter.
  - `start_rank_a2a_scatter_batch`: direct, dense-broadcast, and packed-rank
    dispatch plus destination queue reconstruction.
  - `start_a2a_gather`: expert return or destination-side rank partial return.
  - `acc_and_store_output`: consume expert results in the baseline modes and
    rank partials in the combined mode.
- `python/sgl_jax/srt/layers/fused_moe.py`
  - `FusedEPMoEV2`: hold and pass the selected communication mode.
- `python/sgl_jax/srt/server_args.py`: define the public flag.
- `python/sgl_jax/srt/model_executor/model_runner.py`: propagate the selected
  mode into the model configuration.
- Supported MiMo, Bailing, and GLM model constructors: pass the mode to
  `FusedEPMoEV2`.
- `python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py`: include
  the communication mode in tuned-config selection if measurements show that
  the optimal tiles differ.

## Stage A: private evidence branch

**Status: synthetic v6e-4 gate complete for small-token broadcast.** CPU
evidence, controlled routing, random-route crossover, independent repetitions,
VMEM feasibility, and 128/512 correctness are complete. The next stage is
guarded production-mode plumbing and a real-model A/B.

The private evidence branch is intentionally allowed to contain diagnostic
code that would be reduced or reorganized before an upstream PR. It has two
milestones. Both are required before opening the public RFC.

### A1. Controlled-routing baseline

**Status: in progress.** The controlled test harness is complete; the baseline
needs two additional independent runs and a realistic stress shape.

#### Objective

**Status: partially achieved.** Duplication is quantified and a first current-v2
latency baseline exists; repeatability and the realistic shape remain.

Quantify the duplication opportunity and establish a reproducible current-v2
latency baseline. This milestone does not change production kernel behavior.

#### A1.1 CPU reference and payload accounting

**Status: complete.** The implementation is in
`python/sgl_jax/tools/moe_rank_routing.py`; all 12 routing tests pass locally,
along with the VMEM and deterministic-selection tests.

Implement a NumPy test oracle with an interface equivalent to:

```python
reference_rank_pack(
    topk_ids,
    topk_weights,
    *,
    num_experts,
    ep_size,
)
```

It should produce, per source token and destination rank:

- destination rank;
- selected local expert IDs;
- original Top-K slot IDs;
- original router weights;
- a valid-route mask;
- the number of unique destination ranks.

It should also report:

```text
valid_expert_routes
unique_rank_routes
mean_unique_ranks_per_token
rank_overlap_ratio = 1 - unique_rank_routes / valid_expert_routes
expert_dispatch_activation_bytes
ideal_rank_dispatch_activation_bytes
ideal_rank_combine_return_bytes
metadata_route_entries
```

Do not report metadata bytes until A2 selects a concrete fixed-capacity encoding
and field widths. A1 can report the number of route entries that the eventual
encoding must represent.

Generate the initial CPU-only JSONL evidence from the repository root:

```bash
PYTHONPATH=python UV_CACHE_DIR=/tmp/sgljax-rank-a2a-uv-cache \
  uv run --with numpy==2.2.6 -- python -m sgl_jax.tools.moe_rank_routing \
  --patterns one_rank,two_ranks,four_ranks,random \
  --token-counts 128,512 \
  --top-k 8 --num-experts 32 --ep-size 4 \
  --hidden-size 1024 --activation-dtype-bytes 2 \
  --output /tmp/fused_moe_rank_routing_cpu.jsonl
```

This command requires NumPy but does not import JAX, initialize a TPU runtime,
or allocate model weights. Each record is explicitly labelled
`predicted_logical_payload` to distinguish analytical bytes from physical ICI
counters and measured kernel latency.

The CPU baseline mirrors the checked-in v2 protocol, rather than assuming a
generic MoE implementation:

- `start_a2a_scatter_batch_range` in `v2/kernel.py` iterates every token and
  every Top-K slot, then starts one local or remote activation copy for that
  expert route;
- `start_a2a_gather` sends each local expert's output back to every source rank
  that routed tokens to it;
- therefore `logical_current_v2_*_activation_rows` is the exact logical
  payload-row count implied by the current routing loops;
- `logical_rank_*_activation_rows` deduplicates those same routes only by
  destination rank. No expert route is removed from expert computation;
- `modeled_*_dma_launches` is separate and depends on resolved local `bt`;
  neither family is labelled as a physical TPU hardware counter.

This validates a real redundancy in the current algorithm on CPU. It does not
prove a wall-clock speedup: physical ICI bytes, DMA granularity, added metadata
cost, Pallas scheduling, and kernel latency still require a TPU run.

The recorded `E=32, EP=4, K=8, H=1024, BF16` CPU results are:

| Routing | Current-v2 copies/token | Candidate copies/token | Logical reduction |
|---|---:|---:|---:|
| one rank | 8 | 1 | 87.5% |
| two ranks | 8 | 2 | 75.0% |
| four ranks | 8 | 4 | 50.0% |
| seeded random, 128 tokens | 8 | 3.758 mean | 53.0% |

Artifact on the VM:
`/tmp/fused_moe_rank_a2a/cpu_logical_payload.jsonl` (eight records).

Payload numbers in A1 are analytical protocol bytes. They must be labelled
`predicted_payload_bytes`, not presented as physical ICI counter measurements.
DMA alignment, packetization, and implementation overhead may make physical
traffic differ.

#### A1.2 Controlled routing cases

**Status: complete for the planned generators and deterministic fanout tests.**

The minimum deterministic matrix is `E=32`, `EP=4`, `K=8`, with eight experts
per rank:

```text
one_rank:
  [0, 1, 2, 3, 4, 5, 6, 7]
  expected unique ranks: 1

two_ranks:
  [0, 1, 2, 3, 8, 9, 10, 11]
  expected unique ranks: 2

four_ranks:
  [0, 1, 8, 9, 16, 17, 24, 25]
  expected unique ranks: 4

distributed:
  seeded random valid Top-K routes
```

`K=8, EP=4` is intentionally the first TPU topology. Every token has eight
expert routes but at most four unique destination ranks, so rank coalescing
guarantees at least 50% lower logical activation payload. Under uniform random
expert placement, the expected unique-rank count is:

```text
E[unique ranks] = 4 * (1 - (3/4)^8) ~= 3.60
expected logical reduction = 1 - 3.60/8 ~= 55%
```

Local routes use local HBM copies rather than ICI. Report both total logical
payload and remote-only logical payload so the analysis does not imply that
every expert-oriented copy crosses ICI.

Additional CPU cases:

- `K=1`;
- all local routes;
- all remote routes;
- invalid or padded expert IDs;
- zero router weights;
- nonuniform weights;
- multiple tokens with different fanout;
- expert IDs at each rank boundary;
- duplicate expert IDs as a defensive input case.

#### A1.3 TPU baseline

**Status: partially complete.** The small synthetic shape has one full timing
run. Each routing-mode benchmark process reported numerical `PASS`; the
one-rank 128-token cell was also checked separately. Two more independent
timing runs and the realistic stress shape remain.

Use the repository's existing `bench_v2.py` synthetic-input path. It already
creates random, sharded tokens and W1/W2/W3 expert tensors without a model
checkpoint. Extend it with controlled routing and payload reporting:

```text
BENCH_ROUTING_MODE=one_rank|two_ranks|four_ranks|all_ranks|random
BENCH_ROUTING_SEED=<integer>
```

`BENCH_A2A_MODE` is added in A2 together with the kernel
mode; A1 measures only the existing expert path.

Start with a small correctness shape on v6e-4:

```text
E=32, EP=4, K=8, H=1024, I=1024, BF16
```

From the repository root, run each controlled mode with a command equivalent
to:

```bash
PYTHONPATH=python \
BENCH_SINGLE_HOST=1 \
BENCH_D=1024 BENCH_F=1024 BENCH_E=32 BENCH_TOPK=8 \
BENCH_TOKENS=128,512 BENCH_CHECK=1 \
BENCH_ROUTING_MODE=one_rank \
BENCH_JSONL=/tmp/fused_moe_v2_one_rank.jsonl \
python3 python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py
```

On the verified VM, use `.venv/bin/python` and explicitly select a valid local
BT for the small shape:

```bash
ssh moe
cd /home/gcpuser/sky_workdir/sglang-jax
PYTHONPATH=python \
BENCH_SINGLE_HOST=1 \
BENCH_D=1024 BENCH_F=1024 BENCH_E=32 BENCH_TOPK=8 \
BENCH_TOKENS=128,512 BENCH_BT=32 BENCH_BF=256 BENCH_BTC=32 \
BENCH_WARMUP=2 BENCH_ITERS=5 BENCH_CHECK=1 \
BENCH_ROUTING_MODE=four_ranks \
BENCH_JSONL=/tmp/fused_moe_rank_a2a/current_v2_four_ranks.jsonl \
.venv/bin/python python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py
```

First-run current-v2 results (`bt=32, bf=256, btc=32, bts=32`, five device
trace samples after two warmups):

| Routing | Tokens | Current -> candidate payload rows | Remote payload rows | Predicted reduction | Device p50 |
|---|---:|---:|---:|---:|---:|
| one rank | 128 | 1024 -> 128 | 768 -> 96 | 87.5% | 0.071078 ms |
| one rank | 512 | 4096 -> 512 | 3072 -> 384 | 87.5% | 0.269938 ms |
| two ranks | 128 | 1024 -> 256 | 768 -> 192 | 75.0% | 0.071431 ms |
| two ranks | 512 | 4096 -> 1024 | 3072 -> 768 | 75.0% | 0.268586 ms |
| four ranks | 128 | 1024 -> 512 | 768 -> 384 | 50.0% | 0.071301 ms |
| four ranks | 512 | 4096 -> 2048 | 3072 -> 1536 | 50.0% | 0.268500 ms |
| seeded random | 128 | 1024 -> 481 | 776 -> 363 | 53.0% total, 53.2% remote | 0.094126 ms |
| seeded random | 512 | 4096 -> 1919 | 3081 -> 1442 | 53.1% total, 53.2% remote | 0.411591 ms |

These are baseline latencies for the unchanged `expert` protocol, not an A/B
speedup result. The random-routing cells have different expert load and queue
shapes, so their latency must not be attributed solely to rank fanout. Compare
the same routes between modes. The A2 section records initial `expert` versus
`rank_dispatch` runs on
identical one-rank routes; the remaining routing-pattern A/B cells are pending.

Saved TPU artifacts:

```text
/tmp/fused_moe_rank_a2a/current_v2_one_rank.jsonl
/tmp/fused_moe_rank_a2a/current_v2_two_ranks.jsonl
/tmp/fused_moe_rank_a2a/current_v2_four_ranks.jsonl
/tmp/fused_moe_rank_a2a/current_v2_random.jsonl
```

Repeat with `two_ranks`, `four_ranks`, and `random`. Preserve the JSONL files
and the full console output as A1 evidence. These commands exercise the current
`expert` kernel; A2 later adds the optimized A/B mode.

Then measure a realistic stress shape that fits the available devices:

```text
E=256, EP=4, K=8, H=8192, I=2048
BF16 activations, checkpoint-relevant weight dtype
```

Measure at least one decode-like and one prefill-like token bucket. Record:

- compilation success and compilation time;
- device kernel latency from the TPU trace;
- host-observed latency separately;
- predicted dispatch and return payload bytes;
- current A2A/synchronization contribution if the existing ablation tooling
  can isolate it;
- block configuration and scratch-memory estimate.

Do not use host wall time as the only kernel-performance number.

#### A1 baseline checklist (non-blocking for A2 development)

**Status: met for the current BF16 v6e-4 synthetic scope and no longer a
prerequisite for continuing A2.**

A1 does not modify production kernel behavior, so it does not need to gate A2
development. This checklist instead freezes a trustworthy control: identical
routes, numerical reference, device-time method, payload accounting, and raw
configuration. The public RFC remains gated on A2 results; A1 repetitions may
run in parallel with A2 implementation.

- Met: CPU tests pass without TPU initialization.
- Met: deterministic predicted bytes match hand calculations.
- Met: current expert kernel produces numerical-reference `PASS` for every
  controlled routing mode; broadcast passes at 128 and 512 tokens.
- Met for the promotion cells: three independent alternating benchmark
  processes per mode after warmup at random-routed T=32 and T=64.
- Met: trace device time is recorded separately from process initialization and
  host/reference correctness time.
- Pending outside the current scope: target checkpoint dtype and model-level
  stress data on a large enough accelerator slice.

- CPU tests pass without a TPU runtime.
- Predicted bytes match hand-calculated values for every deterministic case.
- The existing `expert` kernel matches the existing numerical reference.
- TPU baseline results are reproducible across at least three independent
  benchmark repetitions after warmup.
- Results distinguish predicted bytes, measured device time, and measured host
  time.

A1 proves the opportunity size. It does not prove that local fanout and extra
metadata are cheaper than the avoided ICI payload.

### A2. Minimal rank-dispatch TPU proof-of-concept

**Status: implemented; only direct-HBM `rank_broadcast` passed a performance
gate.** `expert` remains the default. Packed `rank_dispatch` and
`rank_dispatch_combine` execute correctly but are slower. All modes are now
selectable through an explicit experimental server flag for real-model A/B.

Verified on the TPU VM:

- focused five-mode `test_a2a_mode_contract`: pass on CPU;
- `rank_dispatch`, one-rank routing: numerical reference passes at H=1024 and
  H=8192 for 128 tokens, plus H=8192 for 512 tokens/four local BT blocks;
- `rank_dispatch_combine`, one-rank routing, H=1024/T=128: numerical reference
  passes (`max_abs_err=0.0001`, `rel_err=0.003623`);
- JSONL records identify the static mode and separately report logical payload
  rows, modeled DMA starts, and measured device time;
- the modeled DMA counts are explicitly labelled as an algorithmic model, not
  TPU hardware counters;
- direct-HBM `rank_broadcast` removes its unused 5 MiB source staging buffer,
  passes T=128/T=512 correctness, and wins three independent random-routing
  runs at T=32/T=64.

Current one-rank results on v6e-4 (BF16, E=32, top-k=8, I=1024):

| Global tokens | H | Local `bt` | Mode | Device time | Versus expert | Correctness |
|---:|---:|---:|---|---:|---:|---|
| 128 | 1024 | 32 | `expert` | 0.071 ms | control | pass |
| 128 | 1024 | 32 | packed `rank_dispatch` | 0.106 ms | +49.3% | pass |
| 128 | 1024 | 32 | `rank_dispatch_combine` | 0.108 ms | +52.1% | pass |
| 128 | 8192 | 32 | `expert` | 0.328 ms | control | pass |
| 128 | 8192 | 32 | packed `rank_dispatch` | 0.337 ms | +2.7% | pass |
| 512 | 8192 | 32 | `expert` | 1.258 ms | control | existing path |
| 512 | 8192 | 32 | packed `rank_dispatch` | 1.334 ms | +6.0% | pass |

Positive percentages are regressions. These are initial single-run means from
the saved JSONL samples, not the three-run evidence required by the public RFC
gate. The H=8192 `rank_dispatch_combine` cell remains useful for completeness,
but it is no longer the highest-value next action. Before another kernel
rewrite, E0 measures whether remote bytes or DMA descriptor startup are exposed
on the critical path. `expert_dispatch_combine` is now conditional on E0 showing
meaningful gather/return opportunity; otherwise its additional route-mailbox
and FP32 accumulation work is unjustified.

The detailed diagnosis, TPU constraints, alternative designs, and questions
for external reviewers are recorded in
`fused_moe_rank_dispatch_consultation_brief.md`.

#### Objective

**Status: correctness achieved; performance objective not yet achieved.**

Demonstrate that rank-coalesced dispatch produces a real device-time benefit on
at least one meaningful controlled-routing workload before requesting public
project adoption.

#### Scope

**Status: implemented for private benchmark use.**

Add a kernel-only static mode to `fused_ep_moe_v2`:

```python
a2a_mode: Literal[
    "expert",
    "rank_direct",
    "rank_broadcast",
    "rank_dispatch",
    "rank_dispatch_combine",
] = "expert"
```

Expose the mode through `bench_v2.py`, but do not yet add a production server
flag or modify model constructors.

#### Dispatch metadata

**Status: CPU contract and TPU encoding implemented.** The NumPy oracle proves
which expert IDs, slots, and router weights are preserved. TPU metadata uses a
fixed-capacity route mailbox containing expert IDs and bitcast FP32 router
weights. Evidence still distinguishes logical/modelled counts from physical
hardware counters.

The next implementation slice will use fixed-capacity, banked structures:

```text
route mailbox: [bank, source_rank, bt, top_k] IDs (one-way), or
               [bank, source_rank, bt, 2 * top_k] IDs + weight bits (two-way)
rank pack:     [bank, source-token block + destination buckets, hidden]
rank inbox:    [bank, source_rank, bt, hidden]
expert map:    [bank, local_expert, max_routes] inbox index + weight
```

The source loads one local token block, packs one contiguous bucket per
destination, and emits one bulk rank transfer for every non-empty
source-block/destination pair. The destination reconstructs the same per-expert
order with indexed VMEM reads. `rank_dispatch` keeps the existing gather;
`rank_dispatch_combine` applies router weights on the destination and returns
one FP32 rank partial per unique token/rank route.

For each source token, the Pallas implementation needs enough information to:

1. identify each unique destination rank;
2. send the activation once to that rank;
3. preserve the source token identity;
4. carry the destination-local expert IDs selected on that rank;
5. carry the corresponding Top-K slot IDs or router weights;
6. reconstruct expert-local token queues on the destination.

Use fixed-capacity, statically shaped buffers and masks. Do not introduce a
Python- or host-managed variable-length structure into the serving path.

The initial implementation should favor correctness and inspectability over
maximum metadata compression. Metadata compaction can be evaluated only after
the activation-payload benefit is demonstrated.

#### Kernel work sequence

**Status: steps 1-9 are implemented for one-way dispatch; the two-way extension
is implemented and has passed its first TPU correctness case.** The current
expert branch remains the A/B control.

1. Add `a2a_mode` to the v2 JIT static arguments and validation. **Complete.**
2. Preserve the existing `expert` branch byte-for-byte where practical.
3. Derive destination rank and local expert ID for every valid route.
4. Build a per-token destination mask and per-destination expert metadata.
5. Allocate rank-oriented scatter slots and semaphore counts.
6. Issue one activation DMA per unique destination rank.
7. On the destination, fan the received activation into the existing
   expert-oriented compute queues or provide equivalent indexed reads.
8. Keep the existing expert-granular gather and source-side weighted combine.
9. Compare every output with the existing v2 reference. **Complete for the
   one-rank 128/512-token cells listed above; broader matrix pending.**

#### Semaphore correctness

**Status: test contract defined; TPU implementation and stress execution are
pending.**

There is no single runtime `semaphore_leak_count`. Correctness is established
by static accounting plus repeated TPU execution.

The reference accounting should prove, for every routing case:

```text
scatter sends issued == scatter-send waits
scatter receives expected == scatter receive signals
gather sends issued == gather-send waits
gather receives expected == gather receive signals
```

TPU stress tests should run under a bounded test timeout and cover:

- all-local and all-remote routes;
- one, two, and all available destination ranks;
- empty destination ranks;
- alternating sparse and dense patterns across invocations;
- padded decode inputs;
- multiple BT tiles;
- enough iterations to reuse every rotating scratch/semaphore bank;
- a trivial TPU operation and another MoE call after the stress loop.

A timeout, device halt, failed post-stress operation, or failure on a later bank
reuse is a correctness failure even if the first invocation succeeded.

#### A2 benchmark protocol

**Status: active.** Identical controlled routes, device-trace timing, numerical
reference checking, static mode labels, and payload/launch accounting are
wired. The next TPU session must complete the two-way realistic-shape A/B and
the routing/stress matrix.

For every shape and routing pattern:

1. Compile `expert`, `rank_direct`, `rank_broadcast`, `rank_dispatch`, and
   `rank_dispatch_combine` separately. Treat the two new modes as transport
   diagnostics, not production candidates, until they win.
2. Verify both against the same numerical reference.
3. Warm each compiled variant.
4. Randomize or alternate measurement order to reduce drift.
5. Collect multiple device-trace samples.
6. Report median and dispersion, not only the best sample.
7. Report metadata and scratch overhead alongside latency.

Report decode-like and prefill-like regimes separately:

- decode-like: small local token counts and latency-focused batch sizes;
- prefill-like: larger local token counts, multiple BT blocks, and throughput;
- do not infer a production phase policy from `BENCH_TOKENS` alone;
- evaluate `jax.lax.ragged_all_to_all` as a separate prefill-oriented baseline,
  since leaving the fused Pallas call may trade simpler packing for extra HBM
  materialization and synchronization.

Completeness run after a new v6e-4 VM is attached and the draft is synced:

```bash
BENCH_TOKENS=128 BENCH_D=8192 BENCH_F=1024 BENCH_E=32 BENCH_TOPK=8 \
BENCH_BT=32 BENCH_BF=256 BENCH_BTC=32 BENCH_BTS=32 \
BENCH_A2A_MODE=rank_dispatch_combine BENCH_ROUTING_MODE=one_rank \
BENCH_WARMUP=1 BENCH_ITERS=5 BENCH_CHECK=1 \
BENCH_JSONL=/tmp/fused_moe_rank_a2a/h8192_one_rank_rank_dispatch_combine.jsonl \
.venv/bin/python python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py
```

First compile/correctness A/B for the new source-layout diagnostics:

```bash
for mode in expert rank_direct rank_broadcast rank_dispatch; do
  BENCH_TOKENS=128 BENCH_D=8192 BENCH_F=1024 BENCH_E=32 BENCH_TOPK=8 \
  BENCH_BT=32 BENCH_BF=256 BENCH_BTC=32 BENCH_BTS=32 \
  BENCH_A2A_MODE="$mode" BENCH_ROUTING_MODE=one_rank \
  BENCH_WARMUP=1 BENCH_ITERS=5 BENCH_CHECK=1 \
  BENCH_JSONL="/tmp/fused_moe_rank_a2a/h8192_one_rank_${mode}.jsonl" \
  .venv/bin/python python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py
done
```

Interpretation:

- `rank_direct` versus `rank_dispatch` isolates exact-row descriptor cost
  versus ragged source packing;
- `rank_broadcast` versus `rank_dispatch` isolates extra dense rows versus
  ragged source packing while keeping one transfer per active rank block;
- every rank mode versus `expert` still includes the shared rank-inbox
  reconstruction, indexed expert input load, and whole-inbox wait.

Regardless of that cell, run the E0 opportunity/attribution gate before more
tightly coupling dispatch and combine:

1. trace existing named regions for `expert`, `rank_dispatch`, and combine;
2. compare `local_one_rank` with topology-shifted `shifted_one_rank` routes that
   preserve queue sizes and descriptor counts while changing local copies to
   remote copies;
3. use a transport-only microbenchmark to vary bytes and descriptor count
   independently;
4. stop rank-payload work if projected removable exposed cost is below 5%;
5. if descriptors dominate, try source-side expert-sorted expansion while
   preserving the current destination consumer;
6. if bytes dominate, benchmark indexed, pre-expanded, and rank-token-major
   consumers with a prebuilt rank inbox before changing source packing.

The full S0-S4 matrix and branch-specific gates are in
`fused_moe_rank_dispatch_consultation_brief.md`.

#### A2 evidence gate

**Status: evaluated and not yet met.** One-way rank dispatch is correct but has
not beaten expert mode on the tested v6e-4 cells. The realistic two-way result
is still pending after spot-VM preemption.

Open the public RFC only when all of the following hold:

- Numerical parity passes for controlled and seeded-random routing.
- Stress tests complete without timeout or device-health failure.
- `rank_dispatch` reduces predicted activation payload exactly in proportion
  to unique destination ranks.
- On v6e-4, at least one high-overlap realistic-shape cell improves median
  device kernel time by at least 5% across three independent runs.
- No tested dense-fanout cell regresses by more than 3%, or the data clearly
  demonstrates that an explicit selection/fallback policy can avoid it.
- The improvement is not solely host-launch noise.
- The result includes enough metadata and scratch accounting for reviewers to
  evaluate the tradeoff.

The 5% and 3% thresholds are initial decision thresholds, not permanent API
promises. If measurement variance is close to either threshold, collect more
samples before deciding.

If A2 fails the gate, retain the benchmark evidence and stop. Do not add a
public production flag for an optimization that has not demonstrated a device
benefit.

## Stage B: public RFC

**Status: blocked on the A2 evidence gate. Do not open the public issue yet.**

After A2 passes, open an upstream SGLang-JAX issue before the production PR.

Suggested title:

```text
[RFC][TPU][MoE] Rank-coalesced payload dispatch and destination-side partial combine
```

The RFC should include:

- current expert-granular behavior;
- controlled routing diagrams;
- predicted payload formulas;
- v6e-4 A/B device-time data with raw configuration details;
- metadata and scratch-memory costs;
- correctness and stress-test coverage;
- dense-fanout results and fallback discussion;
- the proposed static modes;
- explicit non-goals;
- a request for maintainer feedback on v2-first scope and flag naming.

Reference adjacent work rather than claiming duplication:

- SGLang-JAX fused MoE v2 integration PR #1307;
- SGLang-JAX metadata barrier removal PR #1497;
- SGLang-JAX fixed gather banks PR #1525;
- SGLang-JAX routing-sharding issue #1494;
- SGLang sparse destination-aware MoE issue #36820;
- SGLang MoE refactor roadmap issue #8715.

## Stage C: upstream implementation PRs

**Status: not started upstream.** The current branch is private evidence work.

Public PRs should be smaller and cleaner than the private proof-of-concept.

### Public PR 1: reproducible baseline and reference contract

**Status: private equivalent implemented and tested.** Defer upstream opening
until A2 shows value, then clean the diagnostic branch into this bounded PR.

Contents:

- controlled-routing generators;
- CPU reference rank packer and payload accounting;
- v2 benchmark controls;
- baseline documentation;
- no production behavior change.

This PR makes the RFC evidence reproducible and gives later kernel PRs a stable
oracle.

### Public PR 2: small-token active-rank broadcast

**Status: private equivalent implemented; public opening waits for real-model
evidence.**

Contents:

- production v2 `expert` and `rank_broadcast` modes;
- direct HBM-to-rank-inbox DMA without source-side VMEM packing;
- `--fused-moe-a2a-mode=expert|rank_broadcast` with `expert` as default;
- validation that the experimental mode is initially supported only by
  `fused_v2`;
- CPU metadata tests;
- TPU numerical, timeout, bank-reuse, and performance tests;
- documentation stating that `expert` remains the default.

The private branch already contains the flag so the real-model A/B can run.
For upstream review, land the baseline/reference PR first, then include the
flag with the winning broadcast kernel rather than exposing negative packed
dispatch variants as supported user-facing modes.

### Public PR 3: destination-side partial combine

**Status: not started and intentionally separated from dispatch.**

Extend the mode set with:

```text
rank_dispatch_combine
```

For source token `t` and destination rank `r`, compute on the destination:

```text
p[t, r] = sum(weight[t, e] * expert_output[t, e]
              for selected expert e owned by rank r)
```

Return one `H`-element partial per participating destination rank, then sum the
rank partials on the source. Accumulate destination partials in FP32 and cast at
the same logical output boundary as the existing path. Test tolerances must
account for the changed floating-point reduction order.

This remains separate from Public PR 2 because it changes reduction ownership,
gather layout, numerical ordering, scratch requirements, and semaphore counts.

### Public PR 4: qualification, tuning, and selection policy

**Status: not started; blocked on PR 2 performance evidence.**

Contents:

- tuned block configurations keyed by communication mode if required;
- v7x EP=8 MiMo-V2-Flash end-to-end comparison;
- multi-host qualification when resources are available;
- documentation and troubleshooting guidance;
- an optional `auto` policy only if a stable selection rule is supported by
  measurements.

Do not make the optimized mode the default solely because a synthetic
high-overlap case wins. Default selection requires representative model routing
and end-to-end evidence.

## End-to-end qualification matrix

**Status: CPU and v6e-4 small-token synthetic qualification pass for
direct-HBM rank broadcast; model, v7x, and multi-host cells remain.**

### CPU

**Status: pass (36 focused tests plus 3 subtests locally).**

- expert-to-rank mapping;
- reference rank packing and unpacking;
- metadata preservation;
- payload formulas;
- ordinary Top-K result equals sum of destination-rank partials;
- invalid and padded routing cases.

### v6e-4 synthetic TPU

**Status: pass for BF16 small-token broadcast.** Controlled one/two/four-rank
and seeded-random routes, token crossover, 128/512 correctness, four-block
bank reuse, VMEM preflight, and three independent T=32/64 repetitions are
complete. FP8/quantized model shapes remain for model qualification.

- v2 compile and numerical correctness;
- controlled fanout A/B performance;
- BF16 and the target checkpoint weight dtype;
- decode-like and prefill-like token buckets;
- repeated semaphore-bank reuse;
- scratch-memory and VMEM validation.

### v7x EP=8 synthetic TPU

**Status: not started; no v7x resource is currently assigned.**

- eight-rank fanout behavior;
- comparison against both current v2 and EPMoE where relevant;
- dense-fanout fallback behavior.

### v7x EP=8 model

**Status: code ready; waiting for a v7x-8 or v6e-16 resource and checkpoint.**

- MiMo-V2-Flash;
- identical request set and serving configuration per mode;
- output token throughput, TTFT, TPOT/ITL, and compilation cost;
- deterministic token/logprob qualification where supported;
- routing-fanout distribution collected from the same workload.

### Multi-host

**Status: not started by design.**

- deferred until single-host correctness and performance gates pass;
- explicitly revalidate semaphore protocol and latency under cross-host ICI;
- do not infer multi-host benefit solely from single-host payload reduction.

## Go/no-go outcomes

**Current outcome: advance the broadcast-only small-token candidate to a
guarded real-model A/B.** Packed dispatch and rank-combine remain private
negative/inconclusive experiments. Direct-HBM broadcast has passed the v6e-4
synthetic performance and correctness gate only for small token buckets.

### Go

**Status: reached for a private real-model experiment, not for a public RFC or
default change.**

Open the public RFC and proceed with the public PR sequence when A2 passes its
evidence gate.

### Iterate privately

**Status: active.** Wire an explicit default-off serving flag, run identical
real-model requests in `expert` and `rank_broadcast`, and record TTFT, TPOT/ITL,
output throughput, route fanout, compile cost, and output agreement. Keep
prefill/large-token execution on `expert` unless end-to-end evidence supports a
separate selector.

Continue private work when correctness passes but performance is inconclusive,
especially if metadata format, local fanout copies, or buffer layout dominate.
Change one variable at a time and retain the expert baseline.

### Stop

**Status: triggered for universal rank dispatch, not for small-token
broadcast.** Do not continue optimizing packed rank dispatch or make any rank
mode universal based on byte counts alone. Continue only the measured
small-token broadcast specialization and model qualification.

Stop implementation and publish only findings if:

- local fanout costs consistently exceed avoided communication;
- wins occur only in artificial routing distributions not observed in target
  models;
- dense-fanout regressions cannot be selected away safely;
- correctness requires a synchronization boundary that removes the fused
  kernel advantage;
- scratch growth makes important production shapes infeasible.

## Out of scope for the first feature

**Status: unchanged.** TPU VM availability does not expand the feature scope.

- Changing expert placement or adding expert replication.
- Replacing Top-K routing.
- Moving model weights between ranks at inference time.
- Replacing the production Pallas transport with a separate collective in the
  first feature. A `ragged_all_to_all` microbenchmark remains in scope as a
  prefill-oriented comparison, not as an assumed implementation choice.
- Porting v1 before v2 evidence exists.
- Adding per-token runtime `auto` switching before the cost of compiling and
  maintaining both protocols is understood.
- Claiming physical ICI-byte savings from analytical payload counts alone.
