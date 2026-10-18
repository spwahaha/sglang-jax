# TPU Fused-MoE Rank Dispatch: Consultation Brief

## Question

Can SGLang-JAX fused MoE v2 reduce expert-parallel communication by sending one
activation per destination rank, instead of one activation per selected expert,
without losing more time to local packing, routing, and unpacking than it saves
on ICI communication?

This document is a self-contained snapshot intended for review by another
engineer or AI. The work is private proof-of-concept code on branch
`codex/fused-moe-rank-coalesced-a2a`; it is not a proposed production default.

## Short answer as of 2026-09-21

No measured optimized configuration is faster yet. All completed measurements
are on a single `tpu-v6e-4`, BF16 activations, 32 experts, Top-K 8, FFN
intermediate size 1024, with controlled routing in which all eight selected
experts are on one destination rank.

| Global tokens | Hidden size | Mode | Device time | Versus current `expert` |
|---:|---:|---|---:|---:|
| 128 | 1024 | current `expert` | 0.071 ms | control |
| 128 | 1024 | packed `rank_dispatch` | 0.106 ms | 49.3% slower |
| 128 | 1024 | `rank_dispatch_combine` | 0.108 ms | 52.1% slower |
| 128 | 8192 | current `expert` | 0.328 ms | control |
| 128 | 8192 | packed `rank_dispatch` | 0.337 ms | 2.7% slower |
| 512 | 8192 | current `expert` | 1.258 ms | control |
| 512 | 8192 | packed `rank_dispatch` | 1.334 ms | 6.0% slower |

These are initial single-run means, not publication-quality repeated results.
All listed optimized cells passed the numerical reference. The TPU VM was
preempted before the H=8192 two-way-combine cell could run.

The closest result, H=8192/T=128, is only 2.7% behind. That is encouraging
enough to isolate costs, but it is not evidence of an optimization win.

## Corrections to the external reviews

The Claude and Gemini reviews agree on the important qualitative point: the
prototype exchanges less data but introduces serial/data-dependent local
rearrangement. Several quantitative claims must be corrected before choosing a
design:

- The measured cells used controlled `one_rank` routing, which reduces logical
  activation rows by 87.5% at Top-K 8. The often-quoted 55% reduction is the
  uniform-random EP=4 expectation (`3.6 / 8` rows retained), not the routing
  pattern used for the optimized timings.
- The current dispatch does not use one bulk DMA per expert. It starts a row
  transfer for each token/expert route, but that DMA deposits the row directly
  into the final expert-major queue. Its advantage is placement and overlap,
  not a low dispatch-descriptor count.
- Official JAX documentation lists 128 MiB of VMEM and 1 MiB of SMEM per v6e
  TensorCore, not 16 MiB of VMEM. Capacity alone does not make an additional
  expert queue free: all banks, weight tiles, accumulators, alignment padding,
  and compiler allocation must fit simultaneously.
- Google publishes 800 GB/s bidirectional aggregate ICI bandwidth per v6e
  chip. Dividing payload by an assumed 100 GB/s does not predict exposed kernel
  time: direction, topology, descriptor startup, contention, and overlap with
  weight streaming all matter.
- The proposed `rank_inbox[tile_indices]` expression is a useful compile probe,
  not an established vector-gather primitive. Mosaic may reject it, scalarize
  it, or generate an expensive gather. Generated code and device time decide.
- Pre-expanding into contiguous expert queues moves the same fan-out row copies
  out of `bts_body`; it does not eliminate them. It is viable only if the full
  unpack-plus-contiguous-load path is faster than indexed consumption.
- The T=512 one-rank experiment already moved 87.5% fewer logical activation
  rows and was still 6% slower. Therefore “prefill volume will easily amortize
  packing” is a hypothesis, not a conclusion; `BENCH_TOKENS` also does not by
  itself identify a serving phase.
- `pl.when` should not automatically be described as a hardware branch. The
  safer diagnosis is scalar predicates and loop-carried counters surrounding
  row movement, which inhibit regular vectorization and pipelining.

For uniform Top-K 8 routing, the approximate logical rank-dedup reduction
shrinks as EP grows: 34% at EP=8, 19% at EP=16, 10% at EP=32, 5% at EP=64, and
3% at EP=128. Real routers can correlate experts by rank and move those values
in either direction, so qualification needs measured model route histograms.

References: [JAX TPU hardware](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html),
[Google Cloud TPU v6e](https://docs.cloud.google.com/tpu/docs/v6e), and
[distributed Pallas collectives](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html).

## Model-size byte and time gate

The dispatch arithmetic for BF16 activations is:

```text
row_bytes = H * 2
expert_bytes = T * K * row_bytes
direct_or_packed_rank_bytes = T * mean_unique_destination_ranks * row_bytes
saved_bytes = T * (K - mean_unique_destination_ranks) * row_bytes
```

`T` is the global token count in one MoE-layer invocation, not the sequence
length and not a number of decode iterations. For ICI opportunity, calculate
the same expression using only remote expert routes and remote unique ranks.
Local rows do not consume ICI. The current `rank_direct`, `rank_broadcast`, and
`rank_dispatch` modes change dispatch only.

The existing expert return sends BF16 rows, while the current experimental
`rank_dispatch_combine` returns FP32 partials to preserve accumulation quality.
Its return-byte saving is therefore:

```text
expert_return_bytes = T * K * H * 2
fp32_rank_return_bytes = T * mean_unique_destination_ranks * H * 4
return_saved_bytes = T * H * (2 * K - 4 * mean_unique_destination_ranks)
```

At Top-K 8, FP32 combine reduces return bytes only when the mean rank fan-out
is below four; at four it is byte-neutral, and above four it sends more bytes.
For one-rank routing it saves 75% of return bytes, not the dispatch path's
87.5%. Any future BF16-partial variant needs a separate numerical-accuracy gate.

The official model configurations give these relevant dimensions:

| Model | H | K | BF16 row | MoE layers used here |
|---|---:|---:|---:|---:|
| DeepSeek-V3 | 7,168 | 8 | 14 KiB | 58 (61 layers minus first 3 dense) |
| MiMo-V2-Flash | 4,096 | 8 | 8 KiB | 47 (48 layers; layer 0 dense) |

Sources: [DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)
and [MiMo-V2-Flash config](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/main/config.json).

For `T=128`, the one-way logical dispatch saving is:

| Model | Mean ranks/token | Global bytes saved/layer | Balanced bytes/chip on EP=4 |
|---|---:|---:|---:|
| DeepSeek-V3 | 1 | 12.25 MiB | 3.06 MiB |
| DeepSeek-V3 | 4 | 7.00 MiB | 1.75 MiB |
| MiMo-V2-Flash | 1 | 7.00 MiB | 1.75 MiB |
| MiMo-V2-Flash | 4 | 4.00 MiB | 1.00 MiB |

These are global logical bytes, not necessarily remote bytes. As a neutral
occupancy reference, eight uniformly sampled distinct experts among 256 on
EP=4 touch 3.614 ranks/token on average. They save 3.289 *remote* rows/token
after excluding the source rank. At `T=128`, that is about 5.76 MiB remote for
DeepSeek-V3 and 3.29 MiB for MiMo-V2-Flash across the slice. Real routing,
expert placement, and DeepSeek's group restriction can differ substantially;
the benchmark JSONL must therefore use the observed remote-rank histogram.

Translate bytes to only the bandwidth component with:

```text
payload_time_ms = remote_bytes_on_critical_chip / (effective_GBps * 1e9) * 1e3
T_transport ~= alpha * descriptors + payload_time
```

Google specifies 800 GB/s bidirectional peak ICI bandwidth per v6e chip. It
does not specify that a one-way application stream achieves 800 GB/s. Until
the transport microbenchmark measures `beta`, the table below uses 100 and
400 GB/s as explicit scenarios, not measured claims:

| Remote bytes on one chip | 100 GB/s | 400 GB/s |
|---:|---:|---:|
| 10 KiB | 0.000102 ms | 0.000026 ms |
| 1 MiB | 0.0105 ms | 0.00262 ms |
| 10 MiB | 0.1049 ms | 0.0262 ms |

For the uniform EP=4 occupancy reference at `T=128`, balanced remote savings
produce these payload-only ceilings:

| Model | Per MoE layer at 100-400 GB/s | Across all MoE layers if fully exposed |
|---|---:|---:|
| DeepSeek-V3 | 0.0151-0.00377 ms | 0.875-0.219 ms |
| MiMo-V2-Flash | 0.00862-0.00216 ms | 0.405-0.101 ms |

The model-wide multiplication is an optimistic bound: every layer would need
balanced traffic and a fully exposed transfer. If communication overlaps weight
streaming or MXU work, actual latency saving is smaller and may be zero. This
explains the decision threshold: a 10 KiB saving is not worth a new layout;
multi-MiB per layer merits measurement, but still does not justify packing or
unpacking overhead without a whole-kernel win.

Generate route-specific JSONL, including aggregate saved bytes, the maximum
source/destination-chip saving, FP32-combine bytes, and 100/400 GB/s
payload-only scenarios, without a TPU:

```bash
.venv/bin/python -m sgl_jax.tools.moe_rank_routing \
  --patterns one_rank,two_ranks,four_ranks,random \
  --token-counts 4,32,128,512 --top-k 8 --num-experts 256 --ep-size 4 \
  --hidden-size 7168 --activation-dtype-bytes 2 \
  --output /tmp/deepseek_v3_rank_payload.jsonl

.venv/bin/python -m sgl_jax.tools.moe_rank_routing \
  --patterns one_rank,two_ranks,four_ranks,random \
  --token-counts 4,32,128,512 --top-k 8 --num-experts 256 --ep-size 4 \
  --hidden-size 4096 --activation-dtype-bytes 2 \
  --output /tmp/mimo_v2_flash_rank_payload.jsonl
```

## What the current and experimental algorithms do

For each token, the router selects `K` experts. With Top-K 8 and four expert
parallel ranks, multiple selected experts can reside on the same rank.

### Current `expert` path

The source issues one activation transfer for each selected expert. The
transfer lands directly in that expert's queue on the destination. The
destination compute loop later loads a contiguous expert tile with one async
copy. Expert outputs are returned separately and weighted/summed on the source.

The source activation exists once in the token input buffer. When a token
selects multiple experts, the current path reads that same row once per route
and transmits it into each selected expert's destination slot. “Duplicate” here
means duplicate communication payload, not duplicate source storage.

Those destination slots are already expert-major. For example, routes
`t0 -> {e1,e3}`, `t1 -> {e3}`, and `t2 -> {e1}` must produce queues
`e1:[t0,t2]` and `e3:[t0,t1]`. The current metadata computes each queue offset,
and each row DMA writes directly to that final offset. The DMA is still an
ordinary contiguous row copy; the collection of destination addresses performs
the grouping without a separate local permutation pass.

### Experimental `rank_dispatch`

The current prototype scans routes, deduplicates destination ranks, copies each
selected token into a contiguous per-rank VMEM bucket, and sends one
variable-size block per source-block/destination pair. That source bucket is an
implementation choice made to reduce DMA starts; it is not required by the
rank-dedup protocol.

An alternative can issue one DMA directly from the original token row for each
unique token/rank pair and send the expert IDs/weights as metadata. This avoids
source packing but uses more DMA descriptors. A single bulk DMA cannot normally
read an arbitrary list of non-contiguous source rows merely because metadata
describes them: either issue separate row DMAs, pack first, or transmit a fixed
tile including unused rows.

The destination receives one copy of the activation and reconstructs an
expert-to-inbox index queue. Reconstructing the integer indices is cheap. The
expensive part can be consuming them: the existing expert GEMM needs a
contiguous `bts x H` input tile, while a shared rank inbox is token-major. The
current prototype therefore copies each indexed H-wide row into `b_x_vmem`.
Local experts execute sequentially in the current one-TensorCore expert loop,
not as parallel expert consumers, so this row gathering is on the compute path.
Expert outputs still use the current expert-granular return path.

### Experimental transport diagnostics added on 2026-09-21

- `rank_direct` stages the original dense source block once in VMEM, then sends
  one row for each unique token/destination-rank pair into a fixed
  `[source_rank, token]` inbox slot. It performs no ragged source packing. Its
  payload equals packed rank dispatch, but its DMA-start count is much higher.
- `rank_broadcast` sends the unchanged dense source block once to each rank
  selected by at least one token in that block. It also performs no ragged
  source packing and has the same low descriptor count as packed dispatch, but
  it includes unused token/rank rows. Metadata filters which experts consume
  each row.
- `rank_dispatch` remains the exact-row, low-descriptor packed variant.

All three currently reuse the same destination expert-index reconstruction and
indexed expert input consumer. This is intentional: their A/B comparison
isolates source packing, descriptor count, and transferred bytes. It does not
yet solve the destination gather or whole-inbox wait.

### Experimental `rank_dispatch_combine`

Dispatch is the same as `rank_dispatch`. The destination additionally applies
the original router weights, sums the selected local experts into one FP32
partial per source-token/rank pair, returns those partials, and sums rank
partials on the source.

### Where communication overlaps computation

Pallas async DMA `start()` launches work without immediately waiting for it.
The v2 schedule exploits that in several places:

- scatter for a block starts before its expert loop; each expert waits only for
  its own receive immediately before consuming that queue;
- when block-scatter overlap is enabled, scatter for block `N+1` starts while
  block `N` is still in its expert loop;
- after one expert finishes, its return gather starts before later local experts
  compute;
- weight tiles for a following FFN tile/expert are prefetched while current
  expert work proceeds;
- when configured, shared-expert computation occupies the scatter-receive
  window before routed-expert computation.

Overlap does not make communication free. A semaphore wait becomes visible
whenever the DMA has not completed before its consumer reaches the wait. This
is why peak ICI bandwidth or logical byte counts cannot tell us how much kernel
latency rank dedup can remove; E0 traces and the local-versus-remote control are
intended to measure the uncovered portion.

The current packed rank prototype weakens this pipeline: it waits for the whole
rank inbox before entering the local expert loop, whereas expert mode waits on
one expert queue immediately before computing that expert. A production rank
protocol needs finer readiness, such as source-segment/rank-tile semaphores and
double buffering, so tile `N+1` can arrive while tile `N` computes.

## What was saved, and why it still lost

For the T=128, Top-K 8, EP=4, one-destination-rank case, the logical/modelled
counts across the four devices are:

| Dispatch transport | Rows | DMA starts | Ragged source pack |
|---|---:|---:|---|
| Current `expert` | 1,024 | 1,024 = 768 remote + 256 local | no; DMA writes expert-major |
| `rank_direct` | 128 | 128 = 96 remote + 32 local | no |
| `rank_broadcast` | 512 | 16 = 12 remote + 4 local | no |
| Packed `rank_dispatch` | 128 | 16 = 12 remote + 4 local | yes, 128 row copies |

The broadcast row count is 512 because each 32-token source block contains
tokens targeting all four ranks, so it sends `32 * 4` rows per source rank and
there are four source ranks. It is not always 512: the CPU traffic oracle
derives it from the actual destinations present in each source block.

For return, current expert mode sends 1,024 BF16 expert-output rows but groups
them into 128 source-block/expert transfers (96 remote and 32 local). Packed
combine returns 128 FP32 rank partial rows in 16 transfers. Those row counts
cannot be compared as equal-width bytes: the FP32 partial is twice the BF16 row
size.

The DMA counts are an algorithmic model derived from the code and routes, not
hardware performance counters. The timing proves that fewer logical rows and
fewer starts are not sufficient: local work and layout quality dominate at
these shapes.

The experimental path adds four data-dependent stages:

1. **Source packing.** `_scatter_one_token` scans every token, destination, and
   Top-K slot, then copies a full hidden row inside VMEM into a per-rank bucket.
2. **Destination route reconstruction.** `_fanout_source_token` scans every
   source token and Top-K slot and writes scalar SMEM indices and weights into
   expert queues.
3. **Expert input gather.** `_load_rank_token` performs one dynamically indexed
   VMEM row copy per expert input. The current path instead performs one
   contiguous async copy for a whole expert tile.
4. **Two-way accumulation.** The combine mode performs dynamically addressed
   FP32 read-modify-write accumulation for each expert result, then reconstructs
   source-token order with more scalar routing loops.

The H=1024 regression shrinking from about 49% to 2.7% at H=8192 is consistent
with a substantial fixed control/layout cost that is amortized by larger matrix
work. It does not identify one stage conclusively; stage-level profiling or
ablation is still required.

## Why this shape is awkward on a TPU

The relevant TPU execution properties are:

- Pallas TPU computation happens in VMEM after explicit transfers from HBM.
  Good kernels pipeline large regular transfers with VPU/MXU work.
- TPU vector registers and common VMEM layouts are tiled. On v6, 32-bit vector
  registers are effectively 8 by 128 elements, and Pallas blocks commonly need
  alignment in their last two dimensions.
- SMEM supports scalar random access, but is small; VMEM is large and vector
  oriented. Dynamic scalar control is useful for descriptors and counters, not
  for repeatedly moving hidden-size rows.
- Layout-changing operations and arbitrary gathers over the tiled dimensions
  can be expensive or unsupported. A Python/JAX expression that resembles a
  cheap array assignment can lower to real vector work and VMEM traffic.
- DMA engines are valuable not only for bandwidth. In the existing path they
  also place rows directly into the final expert-major layout. The packed path
  removes network duplication but introduces a second local layout conversion.

Official references:

- [Pallas TPU hardware](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
- [Pallas TPU details and layouts](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [Pallas TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
- [Pallas TPU quickstart](https://docs.jax.dev/en/latest/pallas/tpu/quickstart.html)
- [OpenXLA Mosaic alignment error E2003](https://openxla.org/xla/errors/error_2003)

This is not a proof that rank coalescing cannot work on TPU. It means the
algorithm should preserve static, aligned tiles and avoid converting network
duplication into serial VMEM gather/scatter work.

## Feasible investigation plan

The next step is not another full transport rewrite. First determine whether
bytes or DMA descriptors are exposed on the critical path.

### E0: opportunity and attribution gate

#### E0a: trace existing regions

Capture device traces for the existing `expert`, `rank_dispatch`, and
`rank_dispatch_combine` modes. The kernel already has named scopes for metadata,
scatter, expert input load, FFN, expert store, gather, and output accumulation.
Use identical routes and report both elapsed scope time and overlap; adding
scope durations is not valid when transfers overlap compute.

Run at least these shapes:

| Cell | E / K / H / I | Global tokens / `bt` | Purpose |
|---|---|---|---|
| S0 | 32 / 8 / 1024 / 1024 | 128 / 32 | fixed-overhead stress |
| S1 | 32 / 8 / 8192 / 1024 | 128 / 32 | closest current result |
| S2 | 32 / 8 / 8192 / 1024 | 512 / 32 | multiple block generations |
| S3 | 256 / 8 / 8192 / 2048 | 128 and 512 / 32 | fitting realistic expert count |
| S4 | 256 / 8 / 8192 / 2048 | 2048 / 128 | large-token/tiled hypothesis |

S4 is conditional on compilation and scratch-memory fit; it is not required to
pass E0. Record compiler scratch allocation, not only the hand-calculated
payload size.

#### E0b: measure exposed remote cost without changing the kernel

Add two controlled routing patterns to benchmark/test tooling:

- `local_one_rank`: source rank `r` selects Top-K experts owned by `r`;
- `shifted_one_rank`: source rank `r` selects the same local expert indices on
  rank `(r + 1) % EP`.

They have the same route count, per-expert queue sizes, compute, and modeled DMA
start count. The difference is local versus remote dispatch and return. Record
the actual physical destination mapping and hop geometry; a numeric rank shift
is not by itself proof of a one-hop route.
Run the unchanged `expert` kernel for S0-S4. The whole-kernel latency delta is
an empirical upper bound on currently exposed remote communication. For Top-K
8 to one rank, dispatch dedup can remove at most seven eighths of the activation
rows, so even that delta overstates the achievable benefit.

**Stop gate:** if the projected removable exposed byte cost is below 5% for
every meaningful cell, stop byte-motivated rank dedup and combine work. Retain
descriptor-reduction or return-path experiments only when E0a/E0c attributes a
separate exposed cost to them.

#### E0c: bytes-versus-descriptors transport microbenchmark

Build a small Pallas remote-DMA benchmark independent of MoE:

1. hold total bytes fixed and sweep 1, 4, 8, 32, 128, and 256 descriptors;
2. hold descriptor count fixed and sweep total rows/bytes;
3. run local-copy and one-hop-remote controls;
4. use aligned BF16 rows corresponding to H=1024 and H=8192;
5. report start-to-wait device time over three independent processes.

This decides the branch:

- descriptor-sensitive, byte-insensitive -> keep expert-major layout and reduce
  starts;
- byte-sensitive -> rank dedup remains plausible;
- neither exposed -> communication is sufficiently hidden; stop this track.

Fit the transport measurements to a first-order model:

```text
T_transport(bytes, descriptors) ~= alpha * descriptors + bytes / beta
```

where `alpha` is measured DMA startup cost and `beta` is measured effective
bandwidth for the topology. For the overlapped expert pipeline, use:

```text
T_stage ~= T_serial + max(T_transport, T_weight_stream, T_mxu)
```

instead of summing the three terms. The ideal communication benefit is:

```text
max(T_transport_old, T_other) - max(T_transport_new, T_other)
```

with `T_other = max(T_weight_stream, T_mxu)`. The robust empirical counterpart
is the unchanged-kernel `shifted_one_rank - local_one_rank` delta. Record route
rows, unique token/rank pairs, active experts, number of `bts` tiles, model
weight bytes loaded, activation/output bytes, and modeled descriptors in JSONL.
This exposes the step behavior where weight loads are roughly fixed while an
expert fits in one `bts` tile, then repeat when its token count crosses the tile
capacity.

#### E0d: optimistic `ideal_rank` ceiling

Run only if E0b's predicted opportunity is within two percentage points of the
5% gate or the trace is ambiguous. Prebuild source rank buckets and destination
expert queues outside the measured region, then execute representative bulk
rank DMA concurrently with the unchanged FFN/gather over prebuilt queues. This
is deliberately not a correct algorithm: it makes packing and fan-out free and
therefore supplies an optimistic end-to-end ceiling.

If even `ideal_rank` cannot improve 5%, no real rank-packing implementation can
justify its additional work.

### E1: descriptor-bound branch

The first production candidate is **source-side expert-sorted expansion**, not
rank dedup:

- permute the existing `bt * K` rows into a fixed expert-major staging buffer;
- issue one DMA per non-empty `(source block, expert)` instead of one per route;
- keep destination queues, FFN consumer, gather, and source accumulation
  unchanged;
- include the local permutation in timed results.

This does not reduce activation bytes. It tests whether fewer DMA starts are
worth one regular local permutation. It is useful only when `bt * K` is
materially larger than the number of active experts; for MiMo-like decode with
more experts than routes, it may remove few or no starts.

**Go gate:** at least 5% median device-time improvement across three processes,
with no unavoidable regression above 3% on dense or random routes.

An XLA `all_gather`/dense-broadcast row is also allowed as an EP=4/8 diagnostic.
A ring all-gather makes every token available on every rank and uses regular
communication, but it is not a general replacement for A2A. Its per-token
remote replication is `EP - 1`, versus approximately
`K * (EP - 1) / EP` remote expert routes under uniform routing; it moves no more
bytes than expert dispatch only when `EP <= K`. Exact topology, route
correlation, local filtering, HBM materialization, and overlap still require
measurement. Do not implement a custom in-kernel ring before the XLA baseline.

### E2: byte-bound rank-dedup branch

First solve the destination consumer in isolation with a prebuilt rank inbox.
Compare four complete consumer paths:

1. current scalar indexed row loads;
2. coalesce each expert's monotonic indices into contiguous source runs and
   issue one aligned copy per run;
3. pre-expansion into contiguous per-expert queues followed by the current
   async expert-tile load, timing both unpack and compute;
4. rank-token-major masked compute, reusing one aligned activation tile across
   active local experts.

A batched `jnp.take`-style gather is a compile/code-generation probe, not the
plan. Accept it only if Mosaic produces a supported aligned implementation and
the device measurement wins. Likewise, pre-expansion is not accepted based on
VMEM capacity alone; it must include both queue banks and the extra row copies
in the timed path. A new VMEM expert queue consumes capacity proportional to
`num_devices * bt * top_k * H`; using the existing `a2a_s` HBM expert queues
instead adds a VMEM-to-HBM write and a later HBM-to-VMEM read. Benchmark these
as distinct variants rather than calling either one a free unpack phase.

Also benchmark `rank_broadcast_masked`: send the original contiguous `bt x H`
source block once to each participating rank, then use a
`[source, token, local_expert]` mask. This removes source packing and indexed
gathers at the cost of transmitting unused token/rank rows and running FFNs on
masked rows. A mask prevents unwanted outputs but does not make the MXU skip
those matrix multiplies. The design is most plausible when rank fan-out and
per-rank expert density are already high.

Only after a consumer wins should source packing change to fixed-capacity,
aligned rank tiles with vector mask/prefix construction. Then compare full
`rank_dispatch` against `expert`. Direct one-row deduplicated DMA and 8/16-row
buckets remain granularity diagnostics.

### E3: return-combine branch

`expert_dispatch_combine` remains a useful independent candidate, but build it
only if E0 attributes meaningful exposed time to gather/output accumulation or
E0b demonstrates return bytes on the critical path. It preserves existing
expert dispatch and compute and changes only weighted destination accumulation
and return. Its route-mailbox all-gather and complete FP32 accumulation cost
must be timed.

### E4: separate baselines

- `jax.lax.ragged_all_to_all`: prefill-oriented architectural baseline; it
  still requires contiguous packing, offsets/sizes, and a destination consumer.
- EPMoE/replicate-and-reduce: production control at small EP and a diagnostic
  for the cost of irregular routing; not evidence that replication scales.
- `bts` sweep: test 32 versus larger fitting tiles for random routing so an
  overloaded expert does not trigger a second weight pass. This is independent
  of rank dedup and may be upstreamable even if the dedup track stops.

Reference: [JAX `ragged_all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html).

## Phase policy

Do not use one protocol for every inference phase merely for API simplicity.

- **Decode/small token blocks:** keep `expert` or EPMoE unless an optimized mode
  wins consistently. Fixed packing and synchronization overhead is hardest to
  amortize here.
- **Prefill/large token blocks:** evaluate tiled rank dispatch and
  `ragged_all_to_all`, because more payload is available to amortize setup.
- **Large EP or cross-host:** do not assume rank coalescing becomes better. At
  fixed Top-K, expected duplicate ranks decrease as EP grows, while each saved
  cross-host byte can become more valuable. Re-evaluate only with measured
  route histograms and topology-specific timing.

Selection should ultimately depend on at least phase, hidden size, token count,
EP size, and observed rank fan-out. It must be compile-static or selected at a
coarse serving boundary; per-token runtime switching would add more control
cost.

## Immediate implementation plan

1. Restore a v6e-4 TPU VM and repeat the current `expert` controls so the new
   environment is comparable.
2. Add the CPU-tested `local_one_rank` and `shifted_one_rank` route generators;
   this is the only kernel-independent code required for E0b.
3. Add the transport-only bytes/descriptors microbenchmark and trace capture;
   do not alter production mode selection.
4. Run E0a-E0c on S0-S3, with three independent timing processes and retained
   trace/JSONL artifacts. Run S4 only if its scratch allocation fits.
5. Apply the E0 decision gate. If communication is hidden, stop rank dedup. If
   descriptors dominate, implement expert-sorted expansion. If bytes dominate,
   build the three destination-consumer ablations before revisiting packing.
6. Implement `expert_dispatch_combine` only when E0 shows an exposed return-path
   opportunity; do not couple it to rank dispatch for its first A/B test.
7. Run XLA `all_gather`, `ragged_all_to_all`, EPMoE, and `bts` tuning as separate
   labelled baselines, not as silent substitutions for the fused v2 protocol.
8. Keep current `expert` as the default and do not open a public feature RFC
   until at least one representative cell improves by 5% with no unselectable
   regression above 3%.

## Evidence still missing

- Repeated runs and variance; current optimized timing cells are single-run
  means.
- Stage-level TPU timing for pack, route reconstruction, gather, and combine.
- H=8192 `rank_dispatch_combine` timing.
- Two-, four-, and random-rank optimized timing.
- Decode-sized local token blocks smaller than 32.
- Large-EP, multi-host, and real model routing distributions.
- End-to-end serving metrics and compilation/scratch-memory costs.

The absence of this evidence is why the current conclusion is “iterate
privately,” not “the idea is impossible” and not “ready for a model test.”

## Questions for an external reviewer

1. Is there a Mosaic/Pallas idiom for an aligned batched gather from VMEM that
   avoids per-row dynamic accesses, and what generated layout should be
   inspected to verify it?
2. Can a rank-token-major expert schedule reuse one activation tile across all
   local selected experts without destroying weight prefetch overlap?
3. For Top-K equal to local experts per rank, is padded dense per-rank compute
   likely cheaper than expert compaction on v6e/v7?
4. Can destination-side weighted combine be implemented as a regular tiled
   reduction rather than dynamic FP32 scatter-add?
5. What tile granularity best trades source pack bandwidth against DMA startup:
   one row, 8/16 rows, or a full `bt` bucket?
6. Does `ragged_all_to_all` on TPU remain fused enough to avoid an extra HBM
   round trip for this use case, and how should that be verified in HLO/profile
   output?
7. At what EP size or cross-host topology should reduced ICI payload plausibly
   dominate these local layout costs?
8. Which profiler counters or trace events can separate VPU scalar/vector work,
   VMEM movement, DMA wait, MXU utilization, and ICI transfer time here?

## Relevant implementation locations

- Source rank packing: `python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py`,
  `start_rank_a2a_scatter_batch`, around lines 881-965.
- Destination route reconstruction: same file, around lines 971-1040.
- Dynamic expert input load: same file, around lines 1573-1601.
- Destination rank partial accumulation: same file, around lines 2045-2103.
- Source rank-partial reconstruction: same file, around lines 2129-2184.
- CPU route/payload oracle:
  `python/sgl_jax/tools/moe_rank_routing.py`.
- Benchmark and JSONL evidence:
  `python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py`.
- Full execution record:
  `docs/developer_guide/fused_moe_rank_coalesced_a2a_plan.md`.

## Reproduction caveat

`BENCH_TOKENS=128` means one MoE layer invocation containing 128 token
activations globally. It is not 128 sequential decode iterations. Depending on
the serving batch, it can resemble one decode step across 128 sequences or a
small prefill chunk. Decode and prefill must therefore be labelled by the
actual serving workload, not inferred from this variable alone.
