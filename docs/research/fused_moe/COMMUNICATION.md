# TPU MoE Communication Experiments

Rank deduplication broadcast and destination combine

Research report prepared on 2026-10-18. Code preserved in
[implementation snapshot 5448acf1](https://github.com/spwahaha/sglang-jax/commit/5448acf11e0826efa36b65191ada1590ecb60336).

## Purpose and main finding

We explored how to reduce expert-parallel all-to-all communication in SGLang-JAX's fused
mixture-of-experts kernel on TPU. We built rank-deduplicated dispatch,
direct-row diagnostics, dense rank broadcast, and destination-side combine
prototypes, then compared their numerical behavior and whole-kernel latency.

Broadcast asked a specific question: could a few regular dense transfers beat
precise rank deduplication once packing and local rearrangement were included?
Sending complete blocks directly from HBM gave about 4–5 percent lower latency
at global token counts of 32 and 64 in repeated synthetic experiments. The
benefit disappeared at larger counts. Packed rank dispatch moved fewer logical
rows but was slower in the completed cases. These results support continued
experimentation with layout and readiness; they do not establish a serving gain.

This report covers dispatch, return traffic, data placement, and synchronization.
Block tuning, VMEM search, and other kernel findings remain in the
[broader research report](REPORT.md).

### The baseline already performs useful work during communication

The router selects `K` experts for each token. An expert-parallel rank owns
several experts, so one token may select multiple experts on the same rank.
The existing `expert` path sends one activation row per selected expert.

Each DMA writes directly into that expert's final input queue. Its destination
address performs the routing permutation, so compute can load contiguous rows
without a separate rank-to-expert rearrangement. Receive readiness is tracked
per expert: one expert can begin computing while another is still receiving.
Rotating scratch banks and completion waits protect buffer reuse.

Deduplicating transport must therefore be evaluated together with destination
layout and readiness. A flat rank inbox saves copies of a vector but still has
to make that vector available to each selected expert.

We use `T` for global token rows in one MoE invocation, `H` for hidden width,
`I` for expert intermediate width, `E` for expert count, and EP for the number
of ranks. `T` is not sequence length or a count of decode iterations. All
latencies below are synthetic-input whole-kernel times on real TPU hardware.

<!-- pagebreak -->

## Rank deduplication and the direct row diagnostic

### Why we tried deduplicating by rank

If a Top-K 8 token selects eight experts on one rank, the baseline sends eight
copies of the same activation to that rank. Sending one copy and reusing it
locally reduces logical activation rows by 87.5 percent. If the eight experts
span three ranks, the count falls from eight rows to three. The benefit depends
on actual route fan-out and expert placement.

For BF16, the global logical dispatch payload is `2 * T * K * H` bytes.
Exact rank deduplication changes this to `2 * T * R * H`, where `R` is the mean
number of distinct destination ranks per token. An ICI estimate must exclude
local routes and local rank copies. Neither formula measures physical link
traffic, metadata overhead, or communication exposed on the critical path.

### What we implemented

In `rank_dispatch`, we scan routes, deduplicate destinations, and pack selected
activation rows into contiguous per-rank VMEM buckets. One variable-size
transfer carries each nonempty source-block and destination-rank bucket.
Routing metadata lets the receiver reconstruct expert-to-inbox indices and
load the selected rows into contiguous expert compute tiles.

The prototype waits for the rank inbox before its expert loop. It therefore
changes source work, destination consumption, and readiness as well as payload.
The baseline's direct placement and per-expert receive events are meaningful
parts of the comparison.

### What the completed experiments showed

Historical single-run means on v6e-4 used BF16, `E=32`, `K=8`, `I=1024`, and
controlled routing with one destination rank per token. Reported numerical
comparisons passed. These runs came from an experimental working revision.

| Global T | H | Expert ms | Rank packed ms | Latency change |
|---:|---:|---:|---:|---:|
| 128 | 1024 | 0.071 | 0.106 | 49.3 percent higher |
| 128 | 8192 | 0.328 | 0.337 | 2.7 percent higher |
| 512 | 8192 | 1.258 | 1.334 | 6.0 percent higher |

The `rank_direct` diagnostic kept exact token/rank deduplication but removed
ragged source packing. It staged the dense source block, then sent individual
rows into fixed rank-inbox slots. Historical notes report correct output and
latency similar to packed dispatch at `T=128`; a separate timing table was not
retained. This argues against packing being the sole cause of the regression.
It does not isolate the cost of destination reconstruction or lost overlap.

Source: [dispatch experiments and numerical notes](../../developer_guide/fused_moe_rank_dispatch_consultation_brief.md).

<!-- pagebreak -->

## Why we tried dense rank broadcast

### The question after exact deduplication

Packed dispatch reduced rows but introduced irregular work. Rank-direct removed
packing but still issued one transfer for every selected token/rank pair.
We tried `rank_broadcast` to retain few regular transfers while also removing
source packing. It deliberately sends extra rows, allowing us to explore the
tradeoff between payload volume and transfer regularity.

### What the broadcast path does

For each dense `bt * H` source block, the kernel finds every destination rank
selected by at least one token in that block. It sends the unchanged block once
to each participating rank. The destination uses routing metadata to identify
which rows belong to its local experts. This is a block-level fan-out protocol,
not a transfer of only the exactly deduplicated useful rows.

For example, suppose a two-row block contains `t0` for rank A and `t1` for
rank B. Packed dispatch sends `[t0]` to A and `[t1]` to B. Broadcast sends
`[t0, t1]` to both, avoiding source packing but transmitting four rows instead
of two. Each receiver filters the row it needs. The actual amplification
depends on which ranks occur within each source block.

The later direct-HBM implementation reads the already dense input block from
HBM as the DMA source. It removed an unused approximately 5 MiB source VMEM
staging buffer and its semaphore. No new packing step in HBM is involved.
Destination reconstruction, indexed expert input loads, and inbox readiness
remain; direct HBM does not restore the baseline's expert-major placement.

### A controlled comparison of the transport choices

For `T=128`, `K=8`, EP=4, and `bt=32`, the controlled one-rank-per-token pattern
touches all four ranks within each source block. Counts below are global
algorithmic activation-transfer counts, including local copies. They exclude
metadata and source staging and are not hardware counters.

| Dispatch mode | Rows | Transfer starts | Source treatment |
|---|---:|---:|---|
| Expert | 1024 | 1024 | Per expert route |
| Rank direct | 128 | 128 | Stage block then send rows |
| Rank packed | 128 | 16 | Pack selected rows |
| Rank broadcast | 512 | 16 | Send original dense block |

Broadcast transfers `4 source ranks * 4 destinations * 32 rows = 512` rows
in this pattern. The 16 starts include 12 remote and four local transfers.
The count is route-dependent, not a fixed property of broadcast. It sends four
times the packed payload here, while retaining the same modeled start count.

Sources: [scatter implementation](../../../python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py)
and [routing and traffic models](../../../python/sgl_jax/tools/moe_rank_routing.py).

<!-- pagebreak -->

## What the broadcast measurements show

The random-routing comparisons used v6e-4, BF16 activations and weights,
`H=8192`, `I=1024`, `E=32`, `K=8`, and EP=4. Each pair used identical explicit
block configurations. Different token counts used different outer block sizes.

| Global T | Expert p50 ms | Broadcast p50 ms | Latency change | Evidence |
|---:|---:|---:|---:|---|
| 32 | 0.303173 | 0.288712 | 4.77 percent lower | Three A/B pairs |
| 64 | 0.313143 | 0.300163 | 4.15 percent lower | Three A/B pairs |
| 128 | 0.448801 | 0.445009 | 0.85 percent lower | Single sweep |
| 256 | 0.949149 | 0.953810 | 0.49 percent higher | Single sweep |
| 512 | 2.006105 | 2.037165 | 1.55 percent higher | Single sweep |

At `T=32` and `64`, each value is the median of three run p50s from alternating
expert/broadcast pairs. Each repeat record retains ten timing samples. At `T=128`,
`256`, and `512`, values come from the initial crossover sweep with five samples
per record. Latency change
is `100 * (broadcast / expert - 1)`. The smaller single-sweep differences do
not establish statistical significance.

### What was controlled and what remains unqualified

The records identify a dirty experimental revision, synthetic inputs, and
`matches_production_block_config=false`. Matching explicit tiles between modes
helps compare communication paths, but the baseline is not the production
configuration. For example, the `T=32` run used `btc=bts=32`, while the resolved
production configuration used `btc=bts=8`.

The retained timing records say `correctness_status=not_checked_per_config`.
Separate numerical checks were reported at `T=128` and `512`; those checks do
not certify every small-token configuration. No checkpoint-backed serving A/B
was collected. The table therefore describes a measured experimental region,
not a validated decode optimization.

### Why the result may change with token count

Code inspection shows fewer modeled starts and no ragged source pack, alongside
extra payload and a rank inbox that still needs expert reconstruction. A
plausible explanation is that transfer startup and regularity matter more at
small `T`, while extra rows and destination work absorb that saving as `T`
grows. Whole-kernel timings do not separate those costs or prove this mechanism.

Broadcast remains experimental, with `expert` as the default. A useful next
comparison must use production-resolved tiles, observed model routing, repeated
timings, and numerical checks for every tested configuration before measuring
TTFT, TPOT or ITL, throughput, and memory in a real server.

Sources: [repeat pair one expert](evidence/rank-broadcast-repeat1_random_t32_64_expert.jsonl),
[pair one broadcast](evidence/rank-broadcast-repeat1_random_t32_64_rank_broadcast.jsonl),
[crossover expert](evidence/rank-broadcast-crossover-smoke_random_t32_64_128_256_512_expert.jsonl),
and [crossover broadcast](evidence/rank-broadcast-crossover-smoke_random_t32_64_128_256_512_rank_broadcast.jsonl).

<!-- pagebreak -->

## Return communication and destination combine

### Why we explored combining before the return transfer

The baseline returns a BF16 output row for every selected expert route, then
applies the router-weighted sum on the source. Several outputs may come from
experts on the same rank. Combining those contributions at the destination
could reduce return traffic as well as source accumulation work.

In `rank_dispatch_combine`, router weights travel with dispatch metadata. The
destination weights each local expert output, accumulates an FP32 partial for
each source token, and returns one partial per token/destination-rank pair.
The source then sums the rank partials. Changing where accumulation happens
also changes floating-point ordering and requires numerical validation.

### Precision changes the byte tradeoff

For BF16 expert outputs, global logical return bytes are `2 * T * K * H`.
For FP32 rank partials, they are `4 * T * R * H`. Thus the FP32 design reduces
return bytes only when `R < K / 2`. At Top-K 8, it saves bytes below four ranks
per token, is neutral at four, and increases bytes above four. For ICI, apply
the comparison to remote routes and remote rank partials only.

With one destination rank per token, the modeled saving is 75 percent for
FP32 return partials, versus 87.5 percent for BF16 dispatch deduplication.
Counting only rows would miss this difference.

### What we measured and what it tells us

The completed controlled `T=128`, `H=1024`, BF16, one-rank case measured
0.108 ms for the coupled dispatch/combine prototype versus 0.071 ms for the
expert baseline, 52.1 percent higher latency. Reported maximum absolute error
was 0.0001 and relative error was 0.003623; the experiment's numerical check
passed. These are historical single-run means.

This version inherits packed-dispatch overhead and adds dynamically addressed
FP32 accumulation. It does not isolate the cost of combine itself. To answer
that question, an independent destination-combine variant would have to preserve
expert-major dispatch and readiness while changing only the return path.

### What the communication experiments established

We demonstrated correct prototypes in the reported checked cases, quantified
how rank overlap changes logical payload, and measured cases where fewer rows
still produced higher latency. The remaining design problem is how to retain
expert-ready contiguous inputs and fine-grained readiness after rank transport.
The experiments do not identify a single dominant overhead without stage traces
or independent ablations.

Source: [combine design and numerical evidence](../../developer_guide/fused_moe_rank_dispatch_consultation_brief.md).

<!-- pagebreak -->

## Evidence and next experiments

### What we would test next

1. Measure actual per-token rank fan-out, destinations per source block, and
   expert imbalance for a target checkpoint and serving batch distribution.
   These determine the useful deduplication and broadcast amplification.
2. Profile dispatch, local reconstruction, expert-input loads, exposed receive
   waits, and output return under the same shape and resolved configuration.
   Logical byte savings are an opportunity model, not a measured time budget.
3. Recheck the small-token broadcast region against production-resolved expert
   tiles. Use a clean revision, per-configuration correctness checks, repeated
   alternating pairs, and p50/p95 reporting. Include routing patterns where
   broadcast sends many unused rows.
4. If communication is exposed, test a receiver that produces contiguous expert
   tiles with segment or tile readiness. Preserve buffer ownership and wait for
   outstanding transfers before reusing storage. Test destination combine
   independently only if output return is a material bottleneck.
5. Qualify a surviving candidate in checkpoint-backed serving. Any selector
   needs an observed operating region and an `expert` fallback.

These are proposed experiments. Preparing this report did not run new TPU jobs
or change the production default.

### Where the work and measurements are preserved

The [Pallas kernel](../../../python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py)
contains all four experimental paths. The
[CPU routing and traffic reference](../../../python/sgl_jax/tools/moe_rank_routing.py)
and [routing tests](../../../python/sgl_jax/test/kernels/fused_moe_rank_routing_test.py)
encode deduplication, source/rank accounting, and combine math. The
[benchmark](../../../python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py) and
[TPU runner](../../../scripts/run_fused_moe_rank_a2a_tpu.sh) preserve controlled
routing and numerical-check entry points.

The [evidence directory](evidence/) contains the six small-token repeat files
and two crossover files. For the remaining repeat pairs, see
[pair two expert](evidence/rank-broadcast-repeat2_random_t32_64_expert.jsonl),
[pair two broadcast](evidence/rank-broadcast-repeat2_random_t32_64_rank_broadcast.jsonl),
[pair three expert](evidence/rank-broadcast-repeat3_random_t32_64_expert.jsonl),
and [pair three broadcast](evidence/rank-broadcast-repeat3_random_t32_64_rank_broadcast.jsonl).
The [SHA256 manifest](evidence/manifest.json) records the unmodified evidence.

The [consultation brief](../../developer_guide/fused_moe_rank_dispatch_consultation_brief.md)
retains initial packed-dispatch and combine results and the qualitative
rank-direct diagnostic. Those early measurements do not have a standalone raw
timing packet in this report's evidence directory. The
[communication sections of the deep dive](../../developer_guide/fused_moe_v2_optimization_deep_dive.md)
provide additional implementation context.

To reproduce the broadcast table from the retained files, group repeat records
by `a2a_mode` and `tokens`, take the median of `latency_p50_ms` across repeats
for `T=32/64`, and use the crossover record for `T=128/256/512`. Keep those two
evidence strengths distinct. Recomputing this summary does not rerun the device
experiment or make the historical working revision clean.
