# Fused MoE Rank-Coalesced A2A Implementation Plan

## Status

**Last updated:** 2026-09-20 (America/Los_Angeles)

This document is the live execution record for a private proof of concept. It
is not yet a proposal to change the public SGLang-JAX production default.

Current private implementation status on
`codex/fused-moe-rank-coalesced-a2a`:

- complete: NumPy reference rank packing and payload accounting;
- complete: controlled one-, two-, four-, all-, and random-rank generators;
- complete: CPU tests for metadata preservation, exact payload formulas,
  invalid routes, duplicate experts, remote-only accounting, and weighted
  rank-partial equivalence;
- complete: CPU JSONL evidence generation, with current-v2 and candidate copy
  counts labelled separately from physical ICI measurements;
- complete: controlled-routing and JSONL payload reporting wired into
  `bench_v2.py`;
- complete for the small synthetic shape: one current-v2 TPU baseline run for
  every routing-pattern and token-count cell, including numerical reference
  checks per routing mode;
- complete: A2 static `a2a_mode` API and benchmark plumbing, JSONL labelling,
  validation test, and fail-fast unimplemented candidate mode;
- pending: two more independent A1 repetitions, the realistic stress shape,
  and the A2 rank-oriented Pallas transport.

Verified TPU development environment:

```text
SSH target:       moe
Checkout:         /home/gcpuser/sky_workdir/sglang-jax
Remote branch:    codex/fused-moe-rank-coalesced-a2a
Python:           .venv/bin/python
JAX:              0.11.1
Backend:          TPU
Local devices:    4
Remote base:      ae890f34 (upstream main at VM setup time)
Remote commits:   cbbc1a1d, ccbeb906 (replayed evidence commits)
Evidence:         /tmp/fused_moe_rank_a2a/*.jsonl
```

The local and TPU commit hashes differ because the two evidence commits were
replayed onto the VM's newer clean `main`. The content is equivalent.

The public RFC should be opened only after the evidence gate in this document is
met. A baseline alone can demonstrate duplicated payload and establish a
theoretical opportunity, but only a TPU proof-of-concept can demonstrate an
actual latency or throughput improvement.

## Section-by-section status

| Section | Status | Latest evidence or next action |
|---|---|---|
| Decision, terminology, repository map | Settled for private POC | V2-first, static mode, no public flag before A2 passes. |
| A1.1 CPU reference/accounting | Complete | 10/10 CPU tests pass locally and on the VM; eight JSONL records generated. |
| A1.2 controlled routes | Complete | One-, two-, four-, all-, and seeded-random generators implemented. |
| A1.3 small TPU baseline | In progress | First full eight-cell run complete; collect two more independent runs. |
| A1.3 realistic TPU stress shape | Not started | Run a fitting E=256 shape after baseline repetition automation is recorded. |
| A1 baseline checklist | Partial, non-blocking | CPU and first TPU correctness run pass; repetitions and stress data can continue alongside A2. |
| A2 rank-dispatch POC | In progress | Static API/benchmark mode contract is implemented; Pallas transport is next. |
| A2 semaphore/stress work | Not started | Depends on rank-oriented Pallas buffers and DMA implementation. |
| A2 performance gate | Blocked on implementation | Requires expert-versus-rank_dispatch device-time A/B data. |
| Stage B public RFC | Blocked on A2 gate | Do not open yet. |
| Public PR 1 | Private equivalent complete | Clean up only after A1/A2 evidence justifies upstream discussion. |
| Public PRs 2-4 | Not started | Dispatch, combine, then qualification remain separate changes. |
| v6e-4 qualification | In progress | Current expert path baseline exists; optimized A/B does not. |
| v7x/model/multi-host qualification | Not started | Deferred until the single-host A2 gate passes. |
| Current go/no-go state | Iterate privately | Logical opportunity is proven; device speedup is not. |

## Decision summary

**Status: settled for the private proof of concept; no public server API or
production behavior change has been made.**

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
rank_dispatch           Rank-coalesced dispatch; existing expert-granular return
rank_dispatch_combine   Rank-coalesced dispatch and rank-partial return
```

During the private proof-of-concept, the mode may be exposed only through the
v2 microbenchmark. The public implementation should expose it as
`--fused-moe-a2a-mode`, defaulting to `expert` until qualification is complete.

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

**Status: communication behavior is not modified.** `v2/kernel.py` accepts the
defaulted static mode contract, but still runs only the expert-granular
protocol; requesting `rank_dispatch` fails before compilation.

- `python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py`
  - `fused_ep_moe_v2`: validate and forward the static communication mode.
  - `_fused_ep_moe_kernel`: allocate mode-specific scratch and run the selected
    communication protocol.
  - `start_a2a_scatter_batch_range`: current expert-granular scatter entry
    point; rank-coalesced dispatch branches here or in a nearby isolated helper.
  - `start_a2a_gather`: change only for destination-side partial combine.
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

**Status: in progress.** A1 CPU evidence and the first small-shape TPU baseline
are complete. A1 repetition/stress work can proceed alongside the A2 Pallas
transport implementation.

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
`python/sgl_jax/tools/moe_rank_routing.py`; all 10 CPU tests pass both locally
and in the VM checkout.

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
- therefore `current_v2_*_activation_copies` is the exact logical element-copy
  count implied by the current routing loops;
- `candidate_rank_*_activation_copies` deduplicates those same routes only by
  destination rank. No expert route is removed from expert computation.

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

`BENCH_A2A_MODE=expert|rank_dispatch` is added in A2 together with the kernel
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

| Routing | Tokens | Current -> candidate copies | Remote copies | Predicted reduction | Device p50 |
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
`expert` and `rank_dispatch` on identical routes only after A2 exists.

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

**Status: partially met and no longer a prerequisite for starting A2.**

A1 does not modify production kernel behavior, so it does not need to gate A2
development. This checklist instead freezes a trustworthy control: identical
routes, numerical reference, device-time method, payload accounting, and raw
configuration. The public RFC remains gated on A2 results; A1 repetitions may
run in parallel with A2 implementation.

- Met: CPU tests pass without TPU initialization.
- Met: deterministic predicted bytes match hand calculations.
- Met for the first run: current expert kernel produces numerical-reference
  `PASS` for every routing mode; independently check both token buckets in the
  repetition script before closing A1.
- Pending: three independent benchmark processes per cell after warmup.
- Pending: realistic stress-shape data and device/host-time separation review.

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

**Status: in progress.** The static `a2a_mode` contract and
`BENCH_A2A_MODE` plumbing are implemented. `expert` remains the default and is
unchanged. `rank_dispatch` currently fails early with `NotImplementedError` so
an expert-path run cannot be mislabeled as optimized. Rank-oriented Pallas
transport is the next slice.

#### Objective

**Status: unchanged.** This remains the next implementation objective.

Demonstrate that rank-coalesced dispatch produces a real device-time benefit on
at least one meaningful controlled-routing workload before requesting public
project adoption.

#### Scope

**Status: planned, not implemented.**

Add a kernel-only static mode to `fused_ep_moe_v2`:

```python
a2a_mode: Literal["expert", "rank_dispatch"] = "expert"
```

Expose the mode through `bench_v2.py`, but do not yet add a production server
flag or modify model constructors.

#### Dispatch metadata

**Status: CPU contract implemented; TPU encoding not selected.** The NumPy
oracle proves which expert IDs, slots, and router weights must be preserved,
but it intentionally does not claim metadata byte counts.

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

**Status: step 1 plumbing is complete; steps 2-9 remain.** The current expert
branch remains the A/B control.

1. Add `a2a_mode` to the v2 JIT static arguments and validation. **Complete:**
   API default, benchmark environment variable, JSONL field, validation test,
   and fail-fast placeholder are implemented.
2. Preserve the existing `expert` branch byte-for-byte where practical.
3. Derive destination rank and local expert ID for every valid route.
4. Build a per-token destination mask and per-destination expert metadata.
5. Allocate rank-oriented scatter slots and semaphore counts.
6. Issue one activation DMA per unique destination rank.
7. On the destination, fan the received activation into the existing
   expert-oriented compute queues or provide equivalent indexed reads.
8. Keep the existing expert-granular gather and source-side weighted combine.
9. Compare every output with the existing v2 reference.

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

**Status: protocol defined; blocked on the rank-dispatch kernel mode.** The
current benchmark already supplies identical controlled routes and JSONL
output for the future A/B comparison.

For every shape and routing pattern:

1. Compile `expert` and `rank_dispatch` separately.
2. Verify both against the same numerical reference.
3. Warm each compiled variant.
4. Randomize or alternate measurement order to reduce drift.
5. Collect multiple device-trace samples.
6. Report median and dispersion, not only the best sample.
7. Report metadata and scratch overhead alongside latency.

#### A2 evidence gate

**Status: not evaluated.** CPU evidence and current-v2 latency do not satisfy
this gate; only a working `rank_dispatch` TPU path can do so.

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

### Public PR 2: rank-coalesced dispatch

**Status: not started; this corresponds to A2.**

Contents:

- production v2 `expert` and `rank_dispatch` modes;
- `--fused-moe-a2a-mode=expert|rank_dispatch`;
- validation that the experimental mode is initially supported only by
  `fused_v2`;
- CPU metadata tests;
- TPU numerical, timeout, bank-reuse, and performance tests;
- documentation stating that `expert` remains the default.

The server flag belongs in this PR, because this is the first PR containing a
production behavior that must be selectable for end-to-end testing.

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

**Status: CPU qualification passes; v6e-4 current-path qualification is in
progress; all optimized, model, v7x, and multi-host cells remain.**

### CPU

**Status: pass (10/10 tests locally and on the VM).**

- expert-to-rank mapping;
- reference rank packing and unpacking;
- metadata preservation;
- payload formulas;
- ordinary Top-K result equals sum of destination-rank partials;
- invalid and padded routing cases.

### v6e-4 synthetic TPU

**Status: current expert-path small-shape baseline complete for one run;
optimized A/B, repeatability, dtype expansion, and stress tests remain.**

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

**Status: not started; deferred until synthetic A2 passes.**

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

**Current outcome: iterate privately.** CPU evidence proves logical copy
redundancy and the VM baseline is healthy, but no optimized device-time result
exists yet.

### Go

**Status: not reached.**

Open the public RFC and proceed with the public PR sequence when A2 passes its
evidence gate.

### Iterate privately

**Status: active.** Continue A1 repetition/stress evidence in parallel with the
minimal benchmark-only A2 rank-dispatch path.

Continue private work when correctness passes but performance is inconclusive,
especially if metadata format, local fanout copies, or buffer layout dominate.
Change one variable at a time and retain the expert baseline.

### Stop

**Status: not triggered.** No candidate kernel exists yet, so none of the stop
conditions can be evaluated.

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
- Replacing the Pallas transport with a separate collective.
- Porting v1 before v2 evidence exists.
- Adding per-token runtime `auto` switching before the cost of compiling and
  maintaining both protocols is understood.
- Claiming physical ICI-byte savings from analytical payload counts alone.
