# TPU MoE Communication Experiments

Rank deduplication broadcast and destination combine

Research report revision 3 prepared on 2026-10-18. Code preserved in
[implementation snapshot 5448acf1](https://github.com/spwahaha/sglang-jax/commit/5448acf11e0826efa36b65191ada1590ecb60336).

## TLDR

This work began with a communication opportunity in SGLang-JAX's fused MoE
kernel on TPU. The existing path dispatches a token's activation once for each
selected expert and returns expert outputs for a weighted sum. If several of
those experts share a rank, the same activation is delivered repeatedly to that
rank. We investigated sending it once per destination rank and reusing it
locally, while preserving the expert computations and routing semantics.

We examined related GPU work, including [DeepEP](https://github.com/deepseek-ai/DeepEP/blob/b5a33eefda39485f438dc764695b71ba7b9249b6/README.md),
which shares the high-level idea of rank-oriented MoE dispatch and combine.
Our TPU work explored three optimization directions: exact rank deduplication,
regular dense block broadcast, and destination-side combine. A direct-row
variant helped test whether source packing alone explained the cost.

Controlled prototypes and TPU experiments showed that fewer logical bytes did
not guarantee lower kernel latency. Packed dispatch was slower in the completed
cases. Broadcast reduced kernel p50 by about 4–5 percent at small token counts,
but the benefit disappeared at larger counts. The main engineering question is
how to preserve efficient expert input layout and communication/compute overlap
after reducing transfers. These are synthetic kernel results, not a qualified
serving improvement. The DeepEP comparison is based on source inspection.

## Contents

- [Existing MoE communication flow](#existing-moe-communication-flow)
- [Motivation and the optimization directions](#motivation-and-the-optimization-directions)
- [GPU precedent and DeepEP](#gpu-precedent-and-deepep)
- [Direction 1 Rank deduplication](#direction-1-rank-deduplication)
- [Direction 2 Dense rank broadcast](#direction-2-dense-rank-broadcast)
- [Broadcast measurements and limits](#broadcast-measurements-and-limits)
- [Direction 3 Destination combine](#direction-3-destination-combine)
- [Findings evidence and next experiments](#findings-evidence-and-next-experiments)

<!-- pagebreak -->

## Existing MoE communication flow

In expert-parallel MoE, a router selects `K` experts and weights per token.
Each rank owns a subset of experts. Ranks exchange routing counts, dispatch
activations, run the selected experts, return their outputs, and form a weighted
sum on the token's source rank. Figure 1 shows the logical stages; the kernel
can overlap parts of these stages rather than executing global barriers
between all of them.

![MoE flow from routing metadata through expert dispatch computation return and weighted combine](figures/communication_flow.svg)

Figure 1. Baseline MoE flow. Dispatch places rows directly into expert input
queues, and receive readiness is tracked per expert.

### What happens in the existing expert path

1. The router produces expert IDs and router weights for every token before
   the fused kernel runs. Each expert-parallel rank owns a subset of experts.
   The token's source rank retains the information needed to associate returned
   outputs with the original token and selected routes.
2. Ranks exchange per-expert token counts. The kernel derives total expert
   sizes and source-prefix offsets, giving every routed row a distinct input
   queue position. In the preserved implementation, the `all_reduce_metadata`
   scope gathers histogram rows from peers and reduces them locally. This
   metadata exchange is separate from activation payload transfers.
3. Dispatch sends one activation row per token/expert route. A remote DMA
   writes directly into the destination expert's final queue slot; a local
   route uses a local copy. The destination address performs the routing
   permutation, so no separate rank-to-expert activation rearrangement is
   required. Only remote transfers consume ICI link traffic.
4. An expert waits for its own receive completion, loads contiguous input rows,
   and executes its feed-forward network. Expert-level readiness lets it begin
   while another expert is still receiving. Dispatch for a later block and
   output return for a completed expert can overlap other work, subject to
   buffer availability and completion waits.
5. Expert outputs return to the source rank, where router weights are applied
   and contributions are summed for each original token. The BF16 experiments
   return one logical output row per expert route. Return transfers can group
   rows by expert and source block, so output-row counts and transfer-start
   counts are different quantities.

Rotating scratch banks and DMA send/receive semaphores protect outstanding
transfers. Reusing a buffer requires completion of its previous use. The
baseline's useful properties are therefore expert-ready placement and fine
readiness as well as moving the required data.

We use `T` for global token rows in one MoE invocation, `H` for hidden width,
`I` for expert intermediate width, `E` for expert count, and EP for the number
of ranks. `T` is not sequence length or a count of decode iterations. All
latencies in this report are synthetic-input whole-kernel times on real TPUs.

<!-- pagebreak -->

## Motivation and the optimization directions

### Repeated delivery to the same rank

A token's activation vector is identical across its selected expert routes,
but the experts use different weights. If several selected experts share one
rank, dispatch repeats delivery of the same vector. We treated this as an
opportunity to investigate, not proof that communication dominated kernel time.
Every selected expert must still compute its own output, and expert IDs,
Top-K slots, and router weights must remain associated with the original token.

![Four expert copies compared with two rank deliveries followed by local expert fanout](figures/rank_deduplication.svg)

Figure 2. Illustrative Top-K 4 routing to two ranks. Exact rank deduplication
reduces activation copies from four to two. It does not remove any expert
computation. Copies to the source rank are local and must be excluded from an
ICI traffic estimate.

### Questions we tested

| Direction | Prototype | Question |
|---|---|---|
| Exact deduplication | `rank_dispatch` and `rank_direct` | Can fewer rows repay packing and receiver work? |
| Regular transfers | `rank_broadcast` | Can simpler block DMAs repay extra rows? |
| Return reduction | `rank_dispatch_combine` | Can local weighted sums reduce return cost? |

We built routing references, controlled routing patterns, logical payload and
start-count models, and TPU comparisons. Rank-direct removes ragged source
packing while preserving exact deduplication, helping test one part of the
packed design. Broadcast removes packing and keeps block transfers, accepting
unused rows. Destination combine changes both weighting location and return
precision; its coupled prototype needs an independent ablation.

A rank inbox still has to become usable expert input. Reconstruction, local
fan-out, and readiness are part of the cost we tested. Block tuning and VMEM
search remain in the [broader research report](REPORT.md). Metadata exchange
is included as baseline context; no independent metadata-collective speedup is
claimed in this report.

<!-- pagebreak -->

## GPU precedent and DeepEP

### The shared idea

GPU work provides a concrete precedent for rank-oriented expert communication.
DeepEP's [EPHandle source](https://github.com/deepseek-ai/DeepEP/blob/b5a33eefda39485f438dc764695b71ba7b9249b6/deep_ep/buffers/ep.py)
describes received-token accounting that counts a token once per scale-up rank,
even when several selected experts are on that rank. This matches the input
reuse opportunity behind our exact rank-deduplication prototypes.

Its [recommended integration](https://github.com/deepseek-ai/DeepEP/blob/b5a33eefda39485f438dc764695b71ba7b9249b6/README.md)
also requests `do_expand=True` and expert alignment for grouped expert GEMMs.
The transport representation and the compute representation serve different
purposes: a vector can be delivered once to a rank, then appear in several
local expert segments. Local fan-out remains necessary.

![DeepEP expert expansion compared with indexed consumption in the TPU rank prototype](figures/gpu_tpu_layout.svg)

Figure 3. The shared transport idea leads to different compute inputs. The
DeepEP path shown is its recommended expanded layout; the TPU row describes
our experimental rank-inbox consumer, not every possible TPU implementation.

### What we learned from the comparison

Our TPU prototype retained a flat inbox, reconstructed expert membership,
waited for the expected inbox payload, and dynamically copied rows into expert
tiles. The baseline instead writes directly into expert queues and exposes
per-expert readiness. Comparing only rank-deduplicated bytes hides this
consumer-side difference.

DeepEP's asynchronous event/stream integration and Pallas remote DMA both
permit overlap. The [Pallas documentation](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html)
explicitly supports starting a remote DMA and doing other work before waiting.
The question is how a particular layout and readiness protocol uses that
capability. GPU precedent does not establish a TPU speedup, and these
experiments do not show that rank deduplication is unsuitable for TPU.

DeepEP sources were inspected at revision `b5a33eef` on 2026-10-18. We did not
run a GPU benchmark or port DeepEP's implementation.

<!-- pagebreak -->

## Direction 1 Rank deduplication

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

## Direction 2 Dense rank broadcast

### The question after exact deduplication

Packed dispatch reduced rows but introduced irregular work. Rank-direct removed
packing but still issued one transfer for every selected token/rank pair.
We tried `rank_broadcast` to retain few regular transfers while also removing
source packing. It deliberately sends extra rows, allowing us to explore the
tradeoff between payload volume and transfer regularity.

### What the broadcast path does

For each dense `bt * H` source block, `rank_broadcast` sends the unchanged
block once to every rank selected by at least one token in that block.
Metadata identifies the useful rows and experts at the receiver. It therefore
trades extra activation rows for regular block transfers.

![Packed dispatch sends only selected rows while broadcast sends the full block to each participating rank](figures/packed_vs_broadcast.svg)

Figure 4. Two tokens target different ranks. Packing sends two rows in total;
broadcast sends four. The amplification depends on the ranks touched within
each source block.

The later direct-HBM version uses the original dense input as the DMA source.
It removed an unused approximately 5 MiB source VMEM staging buffer and its
semaphore. It still needs destination reconstruction, indexed expert loads,
and inbox readiness; it does not recreate the baseline expert-major layout.

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

## Broadcast measurements and limits

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

## Direction 3 Destination combine

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

![Expert outputs are weighted and accumulated into FP32 rank partials before return to the source](figures/destination_combine.svg)

Figure 5. Destination combine returns one weighted partial per token/rank pair.
All expert computations remain; moving the sum changes accumulation order and
return precision.

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

## Findings evidence and next experiments

### Main finding

The experiments tested complete communication paths, including their local
consumers. Reduced payload alone did not make the measured packed path faster.
Broadcast exposed a narrow region where regular transfers helped. A next
candidate should preserve efficient expert input layout and useful readiness
while reducing communication; traces and independent ablations are needed to
identify the exposed cost before claiming a general optimization.

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
