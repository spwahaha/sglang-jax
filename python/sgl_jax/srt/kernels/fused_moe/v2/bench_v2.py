"""Standalone micro-bench for fused_ep_moe_v2.

Trace-based timing (device_duration_ps) matches v1 bench methodology.
Supports config sweeping via comma-separated env vars.

Env vars:
  BENCH_TOKENS  — comma-separated token counts (default: 4096)
  BENCH_BT      — comma-separated bt candidates (default: 128)
  BENCH_BF      — comma-separated bf candidates (default: 256)
  BENCH_BTC     — comma-separated btc candidates (default: 128)
  BENCH_BTS     — comma-separated bts candidates (default: auto)
  BENCH_BSE     — bse value (default: 256)
  BENCH_FP8     — 1 to enable fp8 weights
  BENCH_QBK     — quant_block_k for fp8 (default: 128)
  BENCH_DIRECT_SCALED_DOT — 1 to use direct-scaled-dot for both FFN1/FFN2
  BENCH_TAIL_AWARE_BTC — 0, 1, or 0,1 to compare static versus tail-aware btc
                         loop bounds (default: 0)
  BENCH_GATHER_BANKS — fixed rotating gather banks K (default: 2); gather-compute
                       overlap is unconditional for num_bt > 1
  BENCH_VARIANT — label written to BENCH_JSONL for ablation tracking
  BENCH_RUN_ID  — independent-run label written to BENCH_JSONL
  BENCH_TUNE    — 1 to auto-generate bt/bf candidates
  BENCH_MAX_CONFIGS — maximum auto-tune candidates per token shape (default: 48)
  BENCH_TUNE_VMEM_HEADROOM — VMEM budget ratio used by the tuner (default: 0.95)
  BENCH_JSONL   — optional path for rank-0 structured measurements
  BENCH_WARMUP  — warmup iterations (default: 2)
  BENCH_ITERS   — timed iterations (default: 5)
  BENCH_CHECK   — 1 to run correctness check (single-host only)
  BENCH_A2A_MODE — expert, rank_direct, rank_broadcast, rank_dispatch, or
                   rank_dispatch_combine
                   (default: expert)
  BENCH_WORKLOAD_REGIME — auto, latency, or throughput (default: auto)
  BENCH_FORWARD_MODE — scheduler stage label stored in evidence (default: SYNTHETIC)
  BENCH_ROUTING_MODE — random, deterministic, hot_expert, one_rank,
                       two_ranks, four_ranks, or all_ranks (default: random)
  BENCH_ROUTING_SEED — seed for controlled/random CPU routing data (default: 42)
  BENCH_SWIGLU_LIMIT — SwiGLU clamp limit for routed experts (default: off)
  BENCH_SHARED_SWIGLU_LIMIT — SwiGLU clamp limit for the shared expert (default: off)
  BENCH_D/F/E/TOPK — model dims (default: MiMo V2 Pro)
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from sgl_jax.tools.moe_rank_routing import (
    calculate_expert_btc_work,
    calculate_expert_gather_launches,
    calculate_packed_rank_dispatch_launches,
    calculate_rank_broadcast_traffic,
    calculate_rank_routing_metrics,
    infer_contiguous_source_ranks,
    make_controlled_topk,
)

t0 = time.time()
KERNEL_NAME_RE = re.compile(r"fused-moe-v2-k_.*")
TRACE_ROOT = "/tmp/tpu_logs/v2_trace"


def log(msg):
    print(f"[{time.time() - t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


def _git_provenance() -> tuple[str | None, bool | None]:
    """Return revision and dirty state without making benchmark startup fatal."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


# ---------------------------------------------------------------------------
# Trace timing (mirrors benchmark/utils.py)
# ---------------------------------------------------------------------------


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")
    combined: dict[str, Any] = {"traceEvents": []}
    for tf in sorted(trace_files):
        with gzip.open(tf, "rb") as fh:
            shard = json.load(fh)
        events = shard.get("traceEvents", [])
        if isinstance(events, list):
            combined["traceEvents"].extend(events)
    return combined


def _extract_durations_ms(trace: dict[str, Any]) -> list[float]:
    """Extract per-iteration device durations for the v2 kernel from trace.

    Matches events by kernel name regex (same approach as v1 bench),
    extracts device_duration_ps for accurate on-device timing.
    """
    matched = [
        e for e in trace.get("traceEvents", []) if "name" in e and KERNEL_NAME_RE.match(e["name"])
    ]
    if not matched:
        return []
    by_pid: dict[int, list[dict[str, Any]]] = {}
    for e in matched:
        pid = e.get("pid")
        if isinstance(pid, int):
            by_pid.setdefault(pid, []).append(e)
    durations: dict[int, list[float]] = {}
    for pid, evts in by_pid.items():
        evts.sort(key=lambda x: float(x.get("ts", 0)))
        d: list[float] = []
        for e in evts:
            args = e.get("args", {})
            if args.get("device_duration_ps"):
                d.append(float(args["device_duration_ps"]) / 1e9)
            elif "dur" in e:
                d.append(float(e["dur"]) / 1e3)
        if d:
            durations[pid] = d
    if not durations:
        return []
    return max(sorted(durations.items()), key=lambda kv: len(kv[1]))[1]


def _build_xprof_counter_options():
    """Periodic TPU TensorCore perf-counter sampling for xprof (notebook-style).

    Enabled by BENCH_XPROF_COUNTERS=1. Requires jax>=0.9 / Ironwood runtime and
    the LLO trace flags in LIBTPU_INIT_ARGS:
        --xla_enable_custom_call_region_trace=true
        --xla_xprof_register_llo_debug_info=true
    Returns None (-> plain trace) when disabled.
    """
    if os.environ.get("BENCH_XPROF_COUNTERS", "0") != "1":
        return None
    opts = jax.profiler.ProfileOptions()
    opts.advanced_configuration = {
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": (
            "interval_us:1 scaling:0 counter_size_bits:1 "
            "indices:1 indices:3 indices:4 indices:10 indices:11 "
            "indices:31 indices:32 indices:33 indices:34 indices:35 "
            "indices:37 indices:38 indices:56 indices:57 indices:58 "
            "indices:73 indices:74 indices:75 indices:105"
        ),
        "num_tensor_cores_to_trace_per_device": 1,
    }
    return opts


def trace_timeit(run_fn, warmup: int, iters: int) -> list[float]:
    """Warmup then profile *iters* calls, return per-iter device durations (ms)."""
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)

    tag = f"{os.getpid()}_{int(time.time())}"
    trace_dir = os.path.join(TRACE_ROOT, f"run_{tag}")
    os.makedirs(trace_dir, exist_ok=True)

    prof_opts = _build_xprof_counter_options()
    trace_cm = (
        jax.profiler.trace(trace_dir, profiler_options=prof_opts)
        if prof_opts is not None
        else jax.profiler.trace(trace_dir)
    )
    with trace_cm:
        for i in range(iters):
            out = run_fn()
            jax.block_until_ready(out)

    try:
        trace = _load_trace(trace_dir)
        return _extract_durations_ms(trace)
    except FileNotFoundError:
        return []
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


def wall_timeit(run_fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(iters):
        t_start = time.monotonic()
        out = run_fn()
        jax.block_until_ready(out)
        times.append((time.monotonic() - t_start) * 1e3)
    return times


def split_timeit(run_fn, warmup: int, iters: int) -> tuple[list[float], list[float]]:
    """Measure dispatch and block_until_ready separately."""
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)
    dispatch_times = []
    wait_times = []
    for _ in range(iters):
        t0 = time.monotonic()
        out = run_fn()
        t1 = time.monotonic()
        jax.block_until_ready(out)
        t2 = time.monotonic()
        dispatch_times.append((t1 - t0) * 1e3)
        wait_times.append((t2 - t1) * 1e3)
    return dispatch_times, wait_times


# ---------------------------------------------------------------------------
# Env parsing helpers
# ---------------------------------------------------------------------------


def parse_csv_int(env_key: str, default: list[int]) -> list[int]:
    v = os.environ.get(env_key)
    if v is None:
        return default
    return [int(x.strip()) for x in v.split(",")]


def parse_csv_str(env_key: str, default: list[str]) -> list[str]:
    v = os.environ.get(env_key)
    if v is None:
        return default
    return [x.strip() for x in v.split(",")]


def parse_csv_int_or_none(env_key: str) -> list[int | None]:
    v = os.environ.get(env_key)
    if v is None:
        return [None]
    return [int(x.strip()) for x in v.split(",")]


def parse_csv_bool(env_key: str, default: list[bool]) -> list[bool]:
    v = os.environ.get(env_key)
    if v is None:
        return default
    out = []
    for raw in v.split(","):
        item = raw.strip().lower()
        if item in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif item in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"Unsupported boolean value {raw!r} for {env_key}")
    return out


def align_local_tokens_for_v2(local_num_tokens: int) -> int:
    if local_num_tokens <= 8:
        return 8
    return ((local_num_tokens + 7) // 8) * 8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if os.environ.get("BENCH_SINGLE_HOST", "0") != "1":
    jax.distributed.initialize()
log(f"initialized: {jax.device_count()} devices, {jax.process_count()} procs")

from kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe_v2,
    ref_moe,
    validate_fused_moe_a2a_mode,
)
from tune_prune import prune_candidates
from vmem_model import (
    FUSED_MOE_V2_VMEM_LIMIT_BYTES,
    build_fused_moe_v2_scratch_plan,
    estimate_fused_moe_v2_vmem,
    validate_fused_moe_v2_vmem_budget,
)

P = jax.sharding.PartitionSpec
num_devices = jax.device_count()
devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_size = num_devices
git_revision, git_dirty = _git_provenance()
device_kind = jax.devices()[0].device_kind

d = int(os.environ.get("BENCH_D", "6144"))
f = int(os.environ.get("BENCH_F", "2048"))
E = int(os.environ.get("BENCH_E", "384"))
top_k = int(os.environ.get("BENCH_TOPK", "8"))
a2a_mode = validate_fused_moe_a2a_mode(os.environ.get("BENCH_A2A_MODE", "expert"))
routing_mode = os.environ.get("BENCH_ROUTING_MODE", "random")
routing_seed = int(os.environ.get("BENCH_ROUTING_SEED", "42"))
bse = int(os.environ.get("BENCH_BSE", "256"))
use_shared_expert = os.environ.get("BENCH_SHARED", "0") == "1"
se_inter = int(os.environ.get("BENCH_SE_INTER", str(f)))
warmup = int(os.environ.get("BENCH_WARMUP", "2"))
iters = int(os.environ.get("BENCH_ITERS", "5"))
check_correctness = os.environ.get("BENCH_CHECK", "0") == "1"
SWIGLU_LIMIT = (
    float(os.environ["BENCH_SWIGLU_LIMIT"]) if os.environ.get("BENCH_SWIGLU_LIMIT") else None
)
SHARED_SWIGLU_LIMIT = (
    float(os.environ["BENCH_SHARED_SWIGLU_LIMIT"])
    if os.environ.get("BENCH_SHARED_SWIGLU_LIMIT")
    else None
)
use_fp8 = os.environ.get("BENCH_FP8", "0") == "1"
_qbk_str = os.environ.get("BENCH_QBK", "128")
quant_block_k = None if _qbk_str.lower() == "none" else int(_qbk_str)
tune_mode = os.environ.get("BENCH_TUNE", "0") == "1"
tune_max_configs = int(os.environ.get("BENCH_MAX_CONFIGS", "48"))
tune_vmem_headroom = float(os.environ.get("BENCH_TUNE_VMEM_HEADROOM", "0.95"))
metrics_jsonl = os.environ.get("BENCH_JSONL")
benchmark_variant = os.environ.get("BENCH_VARIANT", "fused_moe_v2_tune")
benchmark_run_id = os.environ.get("BENCH_RUN_ID", f"run-{int(t0)}")
forward_mode_label = os.environ.get("BENCH_FORWARD_MODE", "SYNTHETIC").upper()
_valid_forward_modes = {
    "SYNTHETIC",
    "EXTEND",
    "DECODE",
    "MIXED",
    "IDLE",
    "TARGET_VERIFY",
    "DRAFT_EXTEND",
    "DUMMY_FIRST",
}
if forward_mode_label not in _valid_forward_modes:
    raise ValueError(
        f"Unsupported BENCH_FORWARD_MODE={forward_mode_label!r}; "
        f"expected one of {sorted(_valid_forward_modes)}"
    )
_workload_regime_value = os.environ.get("BENCH_WORKLOAD_REGIME", "auto")
if _workload_regime_value not in ("auto", "latency", "throughput"):
    raise ValueError(
        f"Unsupported BENCH_WORKLOAD_REGIME={_workload_regime_value!r}; "
        "expected auto, latency, or throughput"
    )
_explicit_block_shape = any(
    os.environ.get(k) is not None for k in ("BENCH_BT", "BENCH_BF", "BENCH_BTC", "BENCH_BTS")
)
auto_tuned_block = (not tune_mode) and (not _explicit_block_shape)
from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
    MoeWorkloadRegime,
    get_tuned_fused_moe_v2_block_config,
)

workload_regime = MoeWorkloadRegime(_workload_regime_value)
use_wall = os.environ.get("BENCH_WALL", "0") == "1"
use_split = os.environ.get("BENCH_SPLIT", "0") == "1"
direct_scaled_dot = os.environ.get("BENCH_DIRECT_SCALED_DOT", "0") == "1"
enable_act_quant = os.environ.get("BENCH_ACT_QUANT", "0") == "1"
enable_bt_scatter_overlap = os.environ.get("BENCH_BT_SCATTER_OVERLAP", "1") == "1"
_gb = os.environ.get("BENCH_GATHER_BANKS")
fixed_gather_banks = int(_gb) if _gb else 2
cross_expert_prefetch_modes = parse_csv_str("BENCH_CROSS_EXPERT_PREFETCH", ["full"])
valid_cross_expert_prefetch_modes = {"none", "full", "w13"}
invalid_modes = [
    mode for mode in cross_expert_prefetch_modes if mode not in valid_cross_expert_prefetch_modes
]
if invalid_modes:
    raise ValueError(
        f"Unsupported BENCH_CROSS_EXPERT_PREFETCH values {invalid_modes}; "
        "expected one of none, full, or w13."
    )
tail_aware_btc_values_raw = parse_csv_int("BENCH_TAIL_AWARE_BTC", [0])
if any(value not in (0, 1) for value in tail_aware_btc_values_raw):
    raise ValueError("BENCH_TAIL_AWARE_BTC accepts only comma-separated 0 or 1 values.")
tail_aware_btc_values = [bool(value) for value in tail_aware_btc_values_raw]
controlled_routing_modes = {"one_rank", "two_ranks", "four_ranks", "all_ranks"}
valid_routing_modes = {"random", "deterministic", "hot_expert", *controlled_routing_modes}
if routing_mode not in valid_routing_modes:
    raise ValueError(
        f"Unsupported BENCH_ROUTING_MODE={routing_mode!r}; "
        f"expected one of {sorted(valid_routing_modes)}."
    )
if use_split:
    timeit_fn = None
    timing_label = "split"
else:
    timeit_fn = wall_timeit if use_wall else trace_timeit
    timing_label = "wall" if use_wall else "trace"

if direct_scaled_dot:
    log("direct_scaled_dot=True (fp8 dot per quant group, scale after dot)")
if enable_bt_scatter_overlap:
    log("bt_scatter_overlap=True (next-BT scatter HBM bank overlap)")
log(f"cross_expert_prefetch={cross_expert_prefetch_modes}")
log(f"tail_aware_btc={tail_aware_btc_values}")
log(f"a2a_mode={a2a_mode}")
log(f"workload_regime={workload_regime}")
log(f"forward_mode={forward_mode_label}")

bt_candidates = parse_csv_int("BENCH_BT", [128])
bf_candidates = parse_csv_int("BENCH_BF", [256])
btc_candidates = parse_csv_int("BENCH_BTC", [128])
bts_candidates = parse_csv_int_or_none("BENCH_BTS")
token_candidates = parse_csv_int("BENCH_TOKENS", [4096])


# Tuned-first defaulting: when the user did NOT explicitly pin any block-shape
# param (BT/BF/BTC/BTS) and we're not sweeping (BENCH_TUNE), look the shape up in
# tuned_block_configs per token count and use that. Falls back to the BENCH_BT/BF
# defaults only when no tuned entry matches. Avoids silently benchmarking the
# default bf=256 when a tuned (e.g. bf=1024) config exists for the shape.
def _align_to(x, a):
    return ((x + a - 1) // a) * a


def _pow2_floor(x):
    if x <= 1:
        return 1
    return 1 << int(math.floor(math.log2(x)))


def _pow2_ceil(x):
    if x <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(x)))


def _ladder_div2(start):
    out = []
    v = int(start)
    while v > 0:
        out.append(v)
        if v == 1:
            break
        v //= 2
    return sorted(set(out), reverse=True)


def _aligned_divisors(n, alignment=8):
    """All divisors of n that are multiples of alignment, descending."""
    if n <= 0:
        return []
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            if i % alignment == 0:
                divs.add(i)
            j = n // i
            if j % alignment == 0:
                divs.add(j)
    return sorted(divs, reverse=True)


def _block_config_signature(config):
    """Compare configs by values even when direct execution loads two classes."""
    if config is None:
        return None
    return tuple(getattr(config, field) for field in ("bt", "bf", "btc", "bse", "bts"))


def _estimate_vmem_v2(
    *,
    bt,
    bf,
    btc,
    bse,
    bts,
    hidden_size,
    intermediate_size,
    num_experts,
    top_k,
    ep_size,
    num_tokens,
    use_fp8=False,
    quant_block_k=128,
    direct_scaled_dot=True,
    enable_bt_scatter_overlap=True,
    fixed_gather_banks=2,
    enable_act_quant=False,
    a2a_mode="expert",
    verbose=False,
):
    del bse, intermediate_size
    plan = build_fused_moe_v2_scratch_plan(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        ep_size=ep_size,
        bt=bt,
        bf=bf,
        bts=bts,
        input_bytes=2,
        enable_act_quant=enable_act_quant,
        has_weight_scales=use_fp8,
        quant_block_k=quant_block_k,
        enable_bt_scatter_overlap=enable_bt_scatter_overlap,
        fixed_gather_banks=fixed_gather_banks,
        a2a_mode=a2a_mode,
    )
    estimate = estimate_fused_moe_v2_vmem(
        plan,
        weight_bytes=1 if use_fp8 else 2,
        direct_scaled_dot=direct_scaled_dot,
    )

    if verbose:
        mb = lambda b: f"{b / (1024 * 1024):.2f}"
        log(f"    VMEM Breakdown (bt={bt} bf={bf} btc={btc} bts={bts}):")
        for name, size in estimate.components:
            log(f"      {name:<24} {mb(size)} MB")
        log(f"      {'Logical total':<24} {mb(estimate.total_bytes)} MB")
        log(
            f"      {'Mosaic scoped overhead':<24} "
            f"{mb(estimate.mosaic_scoped_overhead_bytes)} MB"
        )
        log(f"      {'Compiler budget':<24} {mb(estimate.compiler_budget_bytes)} MB")

    return estimate


def generate_tune_candidates(
    intermediate_size,
    hidden_size,
    local_num_tokens,
    ep_size,
    num_experts,
    top_k,
    *,
    use_fp8=False,
    quant_block_k=128,
    direct_scaled_dot=True,
    enable_bt_scatter_overlap=True,
    fixed_gather_banks=2,
    enable_act_quant=False,
    a2a_mode="expert",
    vmem_budget=FUSED_MOE_V2_VMEM_LIMIT_BYTES,
    vmem_headroom=0.95,
    max_configs=48,
    bse=256,
    use_shared_expert=False,
    se_inter=0,
    verbose=False,
):
    effective_budget = int(vmem_budget * vmem_headroom)

    bf_list = sorted(
        set(
            v
            for v in [128, 256, 512, 1024, 2048]
            if v <= intermediate_size and intermediate_size % v == 0
        )
    )

    bt_list = []
    for p_val in [2, 4]:
        if local_num_tokens == p_val:
            bt_list.append(p_val)
    p = 8
    while p <= local_num_tokens:
        if local_num_tokens % p == 0:
            bt_list.append(p)
        p *= 2
    if not bt_list:
        bt_list = [local_num_tokens]
    bt_list = sorted(set(bt_list))

    configs = []
    seen = set()
    first_verbose = True

    for bt in bt_list:
        max_bts = bt * ep_size
        expected = bt * ep_size * top_k / num_experts
        lo = _pow2_floor(expected)
        hi = _pow2_ceil(expected)
        # Non-power-of-2 candidates near expected, aligned to 8
        exp_floor8 = (int(expected) // 8) * 8
        exp_ceil8 = _align_to(int(math.ceil(expected)), 8)
        # 1.25x expected covers routing imbalance
        exp_hi8 = _align_to(int(math.ceil(expected * 1.25)), 8)
        bts_cands = sorted(
            {
                v
                for v in [bt, lo, hi, hi * 2, exp_floor8, exp_ceil8, exp_hi8]
                if 0 < v <= max_bts and v % 8 == 0
            }
        )
        if not bts_cands:
            bts_cands = [bt]
        if use_shared_expert:
            # in-kernel SE DMAs the full bt-token block into the bts-sized staging
            # buffer, so it requires bts >= bt.
            bts_cands = [v for v in bts_cands if v >= bt] or [bt]

        for bts_val in bts_cands:
            btc_cands = _aligned_divisors(bts_val, 8)
            if not btc_cands:
                continue

            for bf in bf_list:
                # Sweep the shared-expert block size bse (<= bf, divides se_inter)
                # only when SE is on; bse is irrelevant otherwise. Larger bse means
                # fewer SE blocks -> less per-block SE weight-DMA overhead.
                if use_shared_expert and se_inter > 0:
                    bse_cands = sorted(
                        {
                            v
                            for v in [128, 256, 512, 1024, 2048]
                            if v <= bf and v <= se_inter and se_inter % v == 0
                        }
                    ) or [min(bf, se_inter)]
                else:
                    bse_cands = [bse]
                for btc in btc_cands:
                    for bse_val in bse_cands:
                        bc = FusedMoEBlockConfig(
                            bt=bt,
                            bf=bf,
                            btc=btc,
                            bse=bse_val,
                            bts=bts_val,
                        )
                        num_tokens_total = local_num_tokens * ep_size
                        try:
                            bc_eff = bc.effective_for(
                                num_tokens=num_tokens_total,
                                ep_size=ep_size,
                            )
                        except ValueError:
                            continue

                        key = (bc_eff.bt, bc_eff.bf, bc_eff.btc, bc_eff.bts, bc_eff.bse)
                        if key in seen:
                            continue
                        seen.add(key)

                        estimate = _estimate_vmem_v2(
                            bt=bc_eff.bt,
                            bf=bc_eff.bf,
                            btc=bc_eff.btc,
                            bse=bc_eff.bse,
                            bts=bc_eff.bts,
                            hidden_size=hidden_size,
                            intermediate_size=intermediate_size,
                            num_experts=num_experts,
                            top_k=top_k,
                            ep_size=ep_size,
                            num_tokens=num_tokens_total,
                            use_fp8=use_fp8,
                            quant_block_k=quant_block_k,
                            direct_scaled_dot=direct_scaled_dot,
                            enable_bt_scatter_overlap=enable_bt_scatter_overlap,
                            fixed_gather_banks=fixed_gather_banks,
                            enable_act_quant=enable_act_quant,
                            a2a_mode=a2a_mode,
                            verbose=verbose and first_verbose,
                        )
                        first_verbose = False
                        if estimate.compiler_budget_bytes > effective_budget:
                            log(
                                f"  VMEM skip bt={bc_eff.bt},bf={bc_eff.bf},"
                                f"btc={bc_eff.btc},bts={bc_eff.bts},bse={bc_eff.bse}: "
                                f"{estimate.compiler_budget_bytes / (1024 * 1024):.1f}MB > "
                                f"{effective_budget / (1024 * 1024):.1f}MB"
                            )
                            continue
                        configs.append(bc)

    return prune_candidates(configs, max_configs, log=log)


if tune_mode:
    log(f"tune mode: auto-generating candidates for f={f}, ep={ep_size}")

ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

log(f"model: E={E} d={d} f={f} k={top_k} ep={ep_size} fp8={use_fp8}")
if routing_mode != "random":
    log(f"routing_mode={routing_mode}")
log(
    f"sweep: tokens={token_candidates} bt={bt_candidates} bf={bf_candidates} btc={btc_candidates} bts={bts_candidates}"
)

# --- Create weight arrays (shared across token counts) ---
key = jax.random.key(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)


def make_sharded(rng_key, shape, dtype, scale=1.0):
    local_shape = (shape[0] // num_devices, *shape[1:])
    per_device_arrays = []
    for i, dev in enumerate(jax.local_devices()):
        shard_key = jax.random.fold_in(rng_key, jax.process_index() * len(jax.local_devices()) + i)
        shard = jax.device_put(
            jax.random.normal(shard_key, local_shape, dtype=dtype) * scale,
            dev,
        )
        per_device_arrays.append(shard)
    return jax.make_array_from_single_device_arrays(shape, ep_sharding, per_device_arrays)


def make_deterministic_topk(num_tokens, top_k, num_experts):
    local_tokens = num_tokens // num_devices
    per_device_ids = []
    per_device_weights = []
    for i, dev in enumerate(jax.local_devices()):
        global_device_id = jax.process_index() * len(jax.local_devices()) + i
        token_start = global_device_id * local_tokens
        token_ids = jnp.arange(token_start, token_start + local_tokens, dtype=jnp.int32)
        k_offsets = jnp.arange(top_k, dtype=jnp.int32)
        ids = (token_ids[:, None] + k_offsets[None, :]) % jnp.int32(num_experts)
        weights = jnp.full((local_tokens, top_k), 1.0 / top_k, dtype=jnp.float32)
        per_device_ids.append(jax.device_put(ids, dev))
        per_device_weights.append(jax.device_put(weights, dev))
    topk_ids = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k),
        ep_sharding,
        per_device_ids,
    )
    topk_weights = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k),
        ep_sharding,
        per_device_weights,
    )
    return topk_weights, topk_ids


def make_hot_expert_topk(num_tokens, top_k, num_experts, hot_frac=0.1, hot_load=0.7):
    """Hot-expert routing: hot_load fraction of (token, k) slots route to
    hot_frac fraction of experts; rest uniformly to the cold tail."""
    local_tokens = num_tokens // num_devices
    num_hot = max(1, int(num_experts * hot_frac))
    per_device_ids = []
    per_device_weights = []
    rng = np.random.default_rng(42)
    for i, dev in enumerate(jax.local_devices()):
        is_hot = rng.random((local_tokens, top_k)) < hot_load
        hot_ids = rng.integers(0, num_hot, size=(local_tokens, top_k), dtype=np.int32)
        cold_ids = rng.integers(num_hot, num_experts, size=(local_tokens, top_k), dtype=np.int32)
        ids_np = np.where(is_hot, hot_ids, cold_ids).astype(np.int32)
        ids = jnp.array(ids_np)
        weights = jnp.full((local_tokens, top_k), 1.0 / top_k, dtype=jnp.float32)
        per_device_ids.append(jax.device_put(ids, dev))
        per_device_weights.append(jax.device_put(weights, dev))
    topk_ids = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k),
        ep_sharding,
        per_device_ids,
    )
    topk_weights = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k),
        ep_sharding,
        per_device_weights,
    )
    return topk_weights, topk_ids


def make_controlled_rank_topk(num_tokens, top_k, num_experts, pattern):
    ids_np, weights_np = make_controlled_topk(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        ep_size=ep_size,
        pattern=pattern,
        seed=routing_seed,
    )
    local_tokens = num_tokens // num_devices
    per_device_ids = []
    per_device_weights = []
    for i, dev in enumerate(jax.local_devices()):
        global_device_id = jax.process_index() * len(jax.local_devices()) + i
        token_start = global_device_id * local_tokens
        token_end = token_start + local_tokens
        per_device_ids.append(jax.device_put(jnp.asarray(ids_np[token_start:token_end]), dev))
        per_device_weights.append(
            jax.device_put(jnp.asarray(weights_np[token_start:token_end]), dev)
        )
    topk_ids = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k),
        ep_sharding,
        per_device_ids,
    )
    topk_weights = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k),
        ep_sharding,
        per_device_weights,
    )
    return topk_weights, topk_ids


def make_random_topk(num_tokens, top_k, num_experts):
    gating_local_shape = (num_tokens // num_devices, num_experts)
    gating_per_dev = []
    for i, dev in enumerate(jax.local_devices()):
        shard_key = jax.random.fold_in(k5, jax.process_index() * len(jax.local_devices()) + i)
        gating_per_dev.append(
            jax.device_put(
                jax.random.normal(shard_key, gating_local_shape, dtype=jnp.float32),
                dev,
            )
        )
    gating = jax.make_array_from_single_device_arrays(
        (num_tokens, num_experts),
        ep_sharding,
        gating_per_dev,
    )
    _, topk_ids = lax.top_k(gating, top_k)
    topk_logits = jnp.take_along_axis(gating, topk_ids, axis=-1)
    return jax.nn.softmax(topk_logits, axis=-1), topk_ids


def make_topk_for_routing_mode(num_tokens):
    if routing_mode == "deterministic":
        return make_deterministic_topk(num_tokens, top_k, E)
    if routing_mode == "hot_expert":
        return make_hot_expert_topk(num_tokens, top_k, E)
    if routing_mode in controlled_routing_modes:
        return make_controlled_rank_topk(num_tokens, top_k, E, routing_mode)
    return make_random_topk(num_tokens, top_k, E)


log("creating weight arrays...")
w1 = make_sharded(k2, (E, d, f), jnp.bfloat16, 0.01)
w2 = make_sharded(k3, (E, f, d), jnp.bfloat16, 0.01)
w3 = make_sharded(k4, (E, d, f), jnp.bfloat16, 0.01)

# Shared expert weights are dense (all tokens) and replicated across devices.
w1_shared = w2_shared = w3_shared = None
if use_shared_expert:
    log(f"creating shared expert weights (se_inter={se_inter}, replicated)...")
    repl_sharding = jax.sharding.NamedSharding(mesh, P())

    def make_replicated(rng_key, shape, dtype, scale=1.0):
        arrs = [
            jax.device_put(jax.random.normal(rng_key, shape, dtype=dtype) * scale, dev)
            for dev in jax.local_devices()
        ]
        return jax.make_array_from_single_device_arrays(shape, repl_sharding, arrs)

    ks1, ks2, ks3 = jax.random.split(jax.random.fold_in(key, 777), 3)
    w1_shared = make_replicated(ks1, (d, se_inter), jnp.bfloat16, 0.01)
    w3_shared = make_replicated(ks2, (d, se_inter), jnp.bfloat16, 0.01)
    w2_shared = make_replicated(ks3, (se_inter, d), jnp.bfloat16, 0.01)

w1_scale_s = w2_scale_s = w3_scale_s = None
qbk_arg = None
if use_fp8:
    log(f"quantizing weights to fp8 (quant_block_k={quant_block_k})...")

    if quant_block_k is None:

        @jax.jit
        @jax.shard_map(
            mesh=mesh,
            in_specs=(P(("data", "tensor")),),
            out_specs=(P(("data", "tensor")), P(("data", "tensor"))),
            check_vma=False,
        )
        def quantize_shard_map(w):
            local_w = w
            E_loc, K_dim, N_dim = local_w.shape
            w_f32 = local_w.astype(jnp.float32)
            amax = jnp.max(jnp.abs(w_f32), axis=1, keepdims=True)
            scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
            w_q = (w_f32 / scale).astype(jnp.float8_e4m3fn)
            scale = scale[:, :, None, :]  # (E, 1, N) -> (E, 1, 1, N)
            return w_q, scale.astype(jnp.float32)

    else:

        @jax.jit
        @jax.shard_map(
            mesh=mesh,
            in_specs=(P(("data", "tensor")),),
            out_specs=(P(("data", "tensor")), P(("data", "tensor"))),
            check_vma=False,
        )
        def quantize_shard_map(w):
            local_w = w
            E_loc, K_dim, N_dim = local_w.shape
            w_f32 = local_w.astype(jnp.float32).reshape(
                E_loc, K_dim // quant_block_k, quant_block_k, N_dim
            )
            amax = jnp.max(jnp.abs(w_f32), axis=2, keepdims=True)
            scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
            w_q = (w_f32 / scale).astype(jnp.float8_e4m3fn)
            w_q = w_q.reshape(E_loc, K_dim, N_dim)
            return w_q, scale.astype(jnp.float32)

    w1, w1_scale_s = quantize_shard_map(w1)
    w2, w2_scale_s = quantize_shard_map(w2)
    w3, w3_scale_s = quantize_shard_map(w3)
    qbk_arg = quant_block_k
    log("fp8 quantization done")

# Shared-expert fp8 quant (per-channel, replicated). Production uses per-channel
# shared-expert scales even when routed experts use block-wise FP8 scales. The
# in-kernel SE reuses the routed fp8 weight/token/output VMEM buffers, so SE
# weights must also be fp8 when activation quantization is enabled.
w1_shared_scale = w2_shared_scale = w3_shared_scale = None
if use_shared_expert and use_fp8:
    log("quantizing shared-expert weights to fp8 (per-channel, replicated)...")
    repl_sharding2 = jax.sharding.NamedSharding(mesh, P())

    def _quant_repl_pc(w):
        # w: (K, N) -> fp8 (K, N) + per-channel scale (1, N) over the K axis.
        w_f32 = w.astype(jnp.float32)
        amax = jnp.max(jnp.abs(w_f32), axis=0, keepdims=True)  # (1, N)
        scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
        w_q = (w_f32 / scale).astype(jnp.float8_e4m3fn)
        return w_q, scale.astype(jnp.float32)

    def _repl(x):
        arrs = [jax.device_put(x, dev) for dev in jax.local_devices()]
        return jax.make_array_from_single_device_arrays(x.shape, repl_sharding2, arrs)

    w1s_q, w1s_sc = _quant_repl_pc(jax.device_get(w1_shared))
    w3s_q, w3s_sc = _quant_repl_pc(jax.device_get(w3_shared))
    w2s_q, w2s_sc = _quant_repl_pc(jax.device_get(w2_shared))
    w1_shared, w1_shared_scale = _repl(np.asarray(w1s_q)), _repl(np.asarray(w1s_sc))
    w3_shared, w3_shared_scale = _repl(np.asarray(w3s_q)), _repl(np.asarray(w3s_sc))
    w2_shared, w2_shared_scale = _repl(np.asarray(w2s_q)), _repl(np.asarray(w2s_sc))
    log("shared-expert fp8 quantization done")

log("weights ready")

# --- Sweep ---
results: list[tuple[int, str, float, list[float]]] = []
runs_by_config: dict[tuple[int, str], dict[str, Any]] = {}
invocations_by_config: dict[tuple[int, str], tuple[FusedMoEBlockConfig, str, bool]] = {}
failure_records: list[dict[str, Any]] = []
routing_metrics_by_tokens = {}
routing_ids_by_tokens = {}


def _common_evidence_fields(num_tokens: int) -> dict[str, Any]:
    return {
        "variant": benchmark_variant,
        "run_id": benchmark_run_id,
        "tokens": num_tokens,
        "num_experts": E,
        "top_k": top_k,
        "hidden_size": d,
        "intermediate_size": f,
        "ep_size": ep_size,
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_kind": device_kind,
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "mesh_shape": {"data": 1, "tensor": num_devices},
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "activation_dtype": "bfloat16",
        "weight_dtype": "float8_e4m3fn" if use_fp8 else "bfloat16",
        "quant_block_k": quant_block_k,
        "use_shared_expert": use_shared_expert,
        "shared_expert_intermediate_size": se_inter if use_shared_expert else 0,
        "enable_act_quant": enable_act_quant,
        "enable_bt_scatter_overlap": enable_bt_scatter_overlap,
        "fixed_gather_banks": fixed_gather_banks,
        "direct_scaled_dot": direct_scaled_dot,
        "routing_mode": routing_mode,
        "a2a_mode": a2a_mode,
        "timing_method": timing_label,
        "swiglu_limit": SWIGLU_LIMIT,
        "shared_swiglu_limit": SHARED_SWIGLU_LIMIT,
        "forward_mode": forward_mode_label,
        "workload_regime": getattr(workload_regime, "value", workload_regime),
    }


for num_tokens in token_candidates:
    log(f"--- tokens={num_tokens} ---")

    production_cfg = None
    production_cfg_resolved = None
    try:
        production_cfg = get_tuned_fused_moe_v2_block_config(
            num_tokens=num_tokens,
            num_experts=E,
            top_k=top_k,
            hidden_size=d,
            intermediate_size=f,
            dtype=jnp.bfloat16,
            weight_dtype=(jnp.float8_e4m3fn if use_fp8 else jnp.bfloat16),
            ep_size=ep_size,
            use_shared_expert=use_shared_expert,
            use_grouped_topk=False,
            enable_act_quant=enable_act_quant,
            quant_mode=(
                "none" if not use_fp8 else ("per_channel" if quant_block_k is None else "blockwise")
            ),
            workload_regime=workload_regime,
        )
        local_nt = num_tokens // ep_size
        padded_nt = align_local_tokens_for_v2(local_nt) * ep_size
        production_cfg_resolved = production_cfg.effective_for(
            num_tokens=padded_nt,
            ep_size=ep_size,
        )
    except Exception as e:
        log(f"  production config lookup skipped ({e})")

    if tune_mode:
        local_nt = num_tokens // ep_size
        padded_local_nt = align_local_tokens_for_v2(local_nt)
        tune_configs = generate_tune_candidates(
            f,
            d,
            padded_local_nt,
            ep_size,
            E,
            top_k,
            use_fp8=use_fp8,
            quant_block_k=quant_block_k,
            direct_scaled_dot=direct_scaled_dot,
            enable_bt_scatter_overlap=enable_bt_scatter_overlap,
            fixed_gather_banks=fixed_gather_banks,
            enable_act_quant=enable_act_quant,
            a2a_mode=a2a_mode,
            bse=bse,
            use_shared_expert=use_shared_expert,
            se_inter=se_inter,
            vmem_headroom=tune_vmem_headroom,
            max_configs=tune_max_configs,
            verbose=(num_tokens == token_candidates[0]),
        )
        block_configs_to_try = tune_configs
    else:
        _default_cfgs = [
            FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts)
            for bt, bf, btc, bts in itertools.product(
                bt_candidates,
                bf_candidates,
                btc_candidates,
                bts_candidates,
            )
        ]
        if auto_tuned_block:
            if production_cfg is None:
                raise ValueError(
                    "Production block-config lookup failed; explicitly set BENCH_BT/BF/BTC/BTS "
                    "to run a non-production baseline."
                )
            log(
                f"  production config: bt={production_cfg.bt} bf={production_cfg.bf} "
                f"btc={production_cfg.btc} bse={production_cfg.bse} bts={production_cfg.bts}"
            )
            block_configs_to_try = [production_cfg]
        else:
            block_configs_to_try = _default_cfgs

    tokens = make_sharded(k1, (num_tokens, d), jnp.bfloat16)
    topk_wts, topk_idx = make_topk_for_routing_mode(num_tokens)

    if routing_mode in controlled_routing_modes:
        host_topk_ids, _ = make_controlled_topk(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=E,
            ep_size=ep_size,
            pattern=routing_mode,
            seed=routing_seed,
        )
    elif jax.process_count() == 1:
        host_topk_ids = np.asarray(jax.device_get(topk_idx))
    else:
        host_topk_ids = None

    if host_topk_ids is not None:
        routing_ids_by_tokens[num_tokens] = host_topk_ids
        routing_metrics = calculate_rank_routing_metrics(
            host_topk_ids,
            num_experts=E,
            ep_size=ep_size,
            hidden_size=d,
            activation_dtype_bytes=jnp.dtype(jnp.bfloat16).itemsize,
            source_ranks=infer_contiguous_source_ranks(num_tokens, ep_size),
        )
        routing_metrics_by_tokens[num_tokens] = routing_metrics
        remote_reduction = routing_metrics.remote_dispatch_payload_reduction_ratio
        remote_reduction_text = "n/a" if remote_reduction is None else f"{remote_reduction:.1%}"
        log(
            "  routing payload: "
            f"expert_routes={routing_metrics.valid_expert_routes} "
            f"unique_rank_routes={routing_metrics.unique_rank_routes} "
            f"mean_ranks/token={routing_metrics.mean_unique_ranks_per_token:.3f} "
            f"predicted_dispatch_reduction="
            f"{routing_metrics.dispatch_payload_reduction_ratio:.1%} "
            f"predicted_remote_reduction={remote_reduction_text}"
        )

    configs_to_try = [
        (bc_raw, *flags)
        for bc_raw in block_configs_to_try
        for flags in itertools.product(
            cross_expert_prefetch_modes,
            tail_aware_btc_values,
        )
    ]
    seen_resolved_configs = set()

    for (
        bc,
        xprefetch_mode,
        tail_aware_btc,
    ) in configs_to_try:
        bt, bf, btc, bts = bc.bt, bc.bf, bc.btc, bc.bts
        tag = (
            f"bt={bt},bf={bf},btc={btc},bts={bts},bse={bc.bse},"
            f"xprefetch={xprefetch_mode},"
            f"tail_btc={int(tail_aware_btc)},"
            f"direct={int(direct_scaled_dot)}"
        )

        padded_nt = num_tokens
        local_nt_raw = num_tokens // ep_size
        aligned_local_nt = align_local_tokens_for_v2(local_nt_raw)
        pad_local = aligned_local_nt - local_nt_raw
        if pad_local > 0:
            padded_nt = (local_nt_raw + pad_local) * ep_size

        try:
            bc_resolved = bc.effective_for(num_tokens=padded_nt, ep_size=ep_size)
        except ValueError as e:
            log(f"  SKIP {tag}: {e}")
            failure_records.append(
                {
                    **_common_evidence_fields(num_tokens),
                    "block_config": tag,
                    "requested_block_config": {
                        "bt": bc.bt,
                        "bf": bc.bf,
                        "btc": bc.btc,
                        "bts": bc.bts,
                        "bse": bc.bse,
                    },
                    "resolved_block_config": None,
                    "execution_status": "config_rejected",
                    "failure_stage": "config_resolution",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "correctness_status": "not_run",
                    "samples_ms": [],
                }
            )
            continue

        tag_resolved = (
            f"bt={bc_resolved.bt},bf={bc_resolved.bf},"
            f"btc={bc_resolved.btc},bts={bc_resolved.bts},bse={bc_resolved.bse},"
            f"xprefetch={xprefetch_mode},"
            f"tail_btc={int(tail_aware_btc)},"
            f"direct={int(direct_scaled_dot)}"
        )
        resolved_key = (
            bc_resolved.bt,
            bc_resolved.bf,
            bc_resolved.btc,
            bc_resolved.bts,
            bc_resolved.bse,
            xprefetch_mode,
            tail_aware_btc,
            direct_scaled_dot,
        )
        if resolved_key in seen_resolved_configs:
            log(f"  SKIP duplicate resolved config {tag} -> {tag_resolved}")
            continue
        seen_resolved_configs.add(resolved_key)

        vmem_estimate = _estimate_vmem_v2(
            bt=bc_resolved.bt,
            bf=bc_resolved.bf,
            btc=bc_resolved.btc,
            bse=bc_resolved.bse,
            bts=bc_resolved.bts,
            hidden_size=d,
            intermediate_size=f,
            num_experts=E,
            top_k=top_k,
            ep_size=ep_size,
            num_tokens=padded_nt,
            use_fp8=use_fp8,
            quant_block_k=quant_block_k,
            direct_scaled_dot=direct_scaled_dot,
            enable_bt_scatter_overlap=enable_bt_scatter_overlap,
            fixed_gather_banks=fixed_gather_banks,
            enable_act_quant=enable_act_quant,
            a2a_mode=a2a_mode,
        )
        runs_by_config[(num_tokens, tag_resolved)] = {
            "resolved_block_config": {
                "bt": bc_resolved.bt,
                "bf": bc_resolved.bf,
                "btc": bc_resolved.btc,
                "bts": bc_resolved.bts,
                "bse": bc_resolved.bse,
            },
            "cross_expert_prefetch_mode": xprefetch_mode,
            "tail_aware_btc": tail_aware_btc,
            "config_source": (
                "tune_candidate"
                if tune_mode
                else ("production_lookup" if auto_tuned_block else "explicit")
            ),
            "production_resolved_block_config": (
                None
                if production_cfg_resolved is None
                else {
                    "bt": production_cfg_resolved.bt,
                    "bf": production_cfg_resolved.bf,
                    "btc": production_cfg_resolved.btc,
                    "bts": production_cfg_resolved.bts,
                    "bse": production_cfg_resolved.bse,
                }
            ),
            "matches_production_block_config": (
                _block_config_signature(bc_resolved)
                == _block_config_signature(production_cfg_resolved)
            ),
            "logical_vmem_bytes": vmem_estimate.total_bytes,
            "estimated_vmem_bytes": vmem_estimate.compiler_budget_bytes,
            "vmem_limit_bytes": FUSED_MOE_V2_VMEM_LIMIT_BYTES,
            "vmem_headroom_bytes": (
                FUSED_MOE_V2_VMEM_LIMIT_BYTES - vmem_estimate.compiler_budget_bytes
            ),
            "vmem_estimate_scope": "logical_plus_calibrated_mosaic_scoped_working_set",
        }
        try:
            validate_fused_moe_v2_vmem_budget(vmem_estimate)
        except ValueError as e:
            log(f"  VMEM SKIP {tag_resolved}: {e}")
            failure_records.append(
                {
                    **_common_evidence_fields(num_tokens),
                    "block_config": tag_resolved,
                    **runs_by_config[(num_tokens, tag_resolved)],
                    "execution_status": "config_rejected",
                    "failure_stage": "vmem_preflight",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "correctness_status": "not_run",
                    "samples_ms": [],
                }
            )
            continue

        invocations_by_config[(num_tokens, tag_resolved)] = (
            bc,
            xprefetch_mode,
            tail_aware_btc,
        )

        def run_fn():
            return fused_ep_moe_v2(
                mesh,
                tokens,
                w1,
                w2,
                w3,
                topk_wts,
                topk_idx,
                top_k,
                block_config=bc,
                swiglu_limit=SWIGLU_LIMIT,
                shared_swiglu_limit=SHARED_SWIGLU_LIMIT,
                quant_block_k=qbk_arg,
                w1_scale=w1_scale_s,
                w2_scale=w2_scale_s,
                w3_scale=w3_scale_s,
                w1_shared=w1_shared,
                w2_shared=w2_shared,
                w3_shared=w3_shared,
                w1_shared_scale=w1_shared_scale,
                w2_shared_scale=w2_shared_scale,
                w3_shared_scale=w3_shared_scale,
                direct_scaled_dot=direct_scaled_dot,
                enable_act_quant=enable_act_quant,
                cross_expert_prefetch_mode=xprefetch_mode,
                tail_aware_btc=tail_aware_btc,
                fixed_gather_banks=fixed_gather_banks,
                enable_bt_scatter_overlap=enable_bt_scatter_overlap,
                a2a_mode=a2a_mode,
            )

        try:
            if use_split:
                dispatch_times, wait_times = split_timeit(run_fn, warmup=warmup, iters=iters)
                if dispatch_times:
                    d_avg = np.mean(dispatch_times)
                    w_avg = np.mean(wait_times)
                    wall_avg = d_avg + w_avg
                    log(
                        f"  {tag_resolved}: wall={wall_avg:.3f}ms = dispatch={d_avg:.3f}ms + wait={w_avg:.3f}ms"
                    )
                    log(f"    dispatch: {[round(t, 3) for t in dispatch_times]}")
                    log(f"    wait:     {[round(t, 3) for t in wait_times]}")
                    results.append(
                        (
                            num_tokens,
                            tag_resolved,
                            wall_avg,
                            [d + w for d, w in zip(dispatch_times, wait_times)],
                        )
                    )
            else:
                times = timeit_fn(run_fn, warmup=warmup, iters=iters)
                if times:
                    avg = np.mean(times)
                    log(
                        f"  {tag_resolved}: {avg:.3f} ms ({timing_label}) | samples={[round(t, 3) for t in times]}"
                    )
                    results.append((num_tokens, tag_resolved, avg, times))
                else:
                    log(f"  {tag_resolved}: no timing data")
                    failure_records.append(
                        {
                            **_common_evidence_fields(num_tokens),
                            "block_config": tag_resolved,
                            **runs_by_config[(num_tokens, tag_resolved)],
                            "execution_status": "no_timing_data",
                            "failure_stage": "timing",
                            "error_type": None,
                            "error_message": "Timing completed without device samples.",
                            "correctness_status": "not_run",
                            "samples_ms": [],
                        }
                    )
        except Exception as e:
            log(f"  FAIL {tag}: {e}")
            failure_records.append(
                {
                    **_common_evidence_fields(num_tokens),
                    "block_config": tag_resolved,
                    **runs_by_config[(num_tokens, tag_resolved)],
                    "execution_status": "failed",
                    "failure_stage": "compile_or_execute",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "correctness_status": "not_run",
                    "samples_ms": [],
                }
            )
            continue

# --- Summary ---
metrics_path = pathlib.Path(metrics_jsonl) if metrics_jsonl else None
if metrics_path is not None and jax.process_index() == 0 and failure_records:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        for record in failure_records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    log(f"wrote {len(failure_records)} failed/rejected candidates to {metrics_path}")

if not results:
    raise RuntimeError(
        "No fused MoE v2 configurations produced valid timings. "
        "Check the block configs and static kernel flags."
    )

if results:
    log("")
    log("=== Summary ===")
    by_tokens: dict[int, list[tuple[str, float, list[float]]]] = {}
    for nt, tag, avg, times in results:
        by_tokens.setdefault(nt, []).append((tag, avg, times))
    for nt in sorted(by_tokens.keys()):
        entries = sorted(by_tokens[nt], key=lambda x: float(np.median(x[2])))
        best_tag, best_avg, best_times = entries[0]
        best_p50 = float(np.median(best_times))
        log(f"  tokens={nt}: best_p50={best_p50:.3f}ms avg={best_avg:.3f}ms [{best_tag}]")
        if len(entries) > 1:
            for tag, avg, times in entries[1:]:
                log(f"    p50={np.median(times):.3f}ms avg={avg:.3f}ms [{tag}]")

    if metrics_path is not None and jax.process_index() == 0:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if failure_records else "w"
        with metrics_path.open(write_mode, encoding="utf-8") as fh:
            for nt in sorted(by_tokens):
                entries = sorted(by_tokens[nt], key=lambda x: float(np.median(x[2])))
                best_p50 = float(np.median(entries[0][2]))
                for tag, avg, times in entries:
                    run_metadata = runs_by_config[(nt, tag)]
                    p50 = float(np.median(times))
                    record = {
                        **_common_evidence_fields(nt),
                        "block_config": tag,
                        **run_metadata,
                        "execution_status": "success",
                        "correctness_status": "not_checked_per_config",
                        "latency_ms": float(avg),
                        "latency_p50_ms": p50,
                        "latency_p95_ms": float(np.percentile(times, 95)),
                        "latency_min_ms": float(np.min(times)),
                        "latency_std_ms": float(np.std(times)),
                        "samples_ms": [float(t) for t in times],
                        "is_best_for_tokens": bool(p50 == best_p50),
                    }
                    routing_metrics = routing_metrics_by_tokens.get(nt)
                    if routing_metrics is not None:
                        routing_record = routing_metrics.as_dict()
                        bt_match = re.search(r"(?:^|,)bt=(\d+)", tag)
                        if bt_match is not None:
                            resolved = run_metadata["resolved_block_config"]
                            btc_work = calculate_expert_btc_work(
                                routing_ids_by_tokens[nt],
                                num_experts=E,
                                ep_size=ep_size,
                                source_ranks=infer_contiguous_source_ranks(nt, ep_size),
                                local_block_tokens=int(bt_match.group(1)),
                                bts=int(resolved["bts"]),
                                btc=int(resolved["btc"]),
                            )
                            packed_launches, packed_remote_launches = (
                                calculate_packed_rank_dispatch_launches(
                                    routing_ids_by_tokens[nt],
                                    num_experts=E,
                                    ep_size=ep_size,
                                    source_ranks=infer_contiguous_source_ranks(nt, ep_size),
                                    local_block_tokens=int(bt_match.group(1)),
                                )
                            )
                            (
                                broadcast_launches,
                                broadcast_remote_launches,
                                broadcast_rows,
                                broadcast_remote_rows,
                            ) = calculate_rank_broadcast_traffic(
                                routing_ids_by_tokens[nt],
                                num_experts=E,
                                ep_size=ep_size,
                                source_ranks=infer_contiguous_source_ranks(nt, ep_size),
                                local_block_tokens=int(bt_match.group(1)),
                            )
                            gather_launches, gather_remote_launches = (
                                calculate_expert_gather_launches(
                                    routing_ids_by_tokens[nt],
                                    num_experts=E,
                                    ep_size=ep_size,
                                    source_ranks=infer_contiguous_source_ranks(nt, ep_size),
                                    local_block_tokens=int(bt_match.group(1)),
                                )
                            )
                            routing_record.update(
                                {
                                    **btc_work.as_dict(),
                                    "modeled_current_v2_dispatch_dma_launches": (
                                        routing_metrics.valid_expert_routes
                                    ),
                                    "modeled_current_v2_remote_dispatch_dma_launches": (
                                        routing_metrics.remote_expert_routes
                                    ),
                                    "modeled_unpacked_rank_dispatch_dma_launches": (
                                        routing_metrics.unique_rank_routes
                                    ),
                                    "modeled_unpacked_remote_rank_dispatch_dma_launches": (
                                        routing_metrics.unique_remote_rank_routes
                                    ),
                                    "modeled_packed_rank_dispatch_dma_launches": packed_launches,
                                    "modeled_packed_remote_rank_dispatch_dma_launches": (
                                        packed_remote_launches
                                    ),
                                    "modeled_rank_broadcast_dma_launches": (broadcast_launches),
                                    "modeled_remote_rank_broadcast_dma_launches": (
                                        broadcast_remote_launches
                                    ),
                                    "modeled_rank_broadcast_activation_rows": (broadcast_rows),
                                    "modeled_remote_rank_broadcast_activation_rows": (
                                        broadcast_remote_rows
                                    ),
                                    "modeled_current_v2_return_dma_launches": gather_launches,
                                    "modeled_current_v2_remote_return_dma_launches": (
                                        gather_remote_launches
                                    ),
                                    "modeled_packed_rank_combine_return_dma_launches": (
                                        packed_launches
                                    ),
                                    "modeled_packed_remote_rank_combine_return_dma_launches": (
                                        packed_remote_launches
                                    ),
                                    "dma_launch_measurement_scope": (
                                        "algorithmic_model_not_hardware_counter"
                                    ),
                                }
                            )
                        record["routing_metrics"] = routing_record
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
        log(f"wrote structured metrics to {metrics_path}")

missing_token_buckets = sorted(set(token_candidates) - set(by_tokens))
if missing_token_buckets:
    raise RuntimeError(
        "No fused MoE v2 configuration produced a valid timing for requested "
        f"token buckets {missing_token_buckets}. See the failure evidence above."
    )

# --- Correctness check (optional, winning measured config per token count) ---
if check_correctness:
    if jax.process_count() > 1:
        log("SKIP correctness check in multi-host mode (use a single-host TPU slice)")
    else:
        bench_tol = float(os.environ.get("BENCH_TOL", "0.05"))
        correctness_records = []
        correctness_failed = False
        for nt0 in sorted(by_tokens):
            best_tag = min(by_tokens[nt0], key=lambda entry: float(np.median(entry[2])))[0]
            bc0, xprefetch_mode, tail_aware_btc = invocations_by_config[(nt0, best_tag)]
            log(f"checking winning config for tokens={nt0}: {best_tag}")
            try:
                tokens_c = make_sharded(k1, (nt0, d), jnp.bfloat16)
                twts, tidx = make_topk_for_routing_mode(nt0)
                result = fused_ep_moe_v2(
                    mesh,
                    tokens_c,
                    w1,
                    w2,
                    w3,
                    twts,
                    tidx,
                    top_k,
                    block_config=bc0,
                    swiglu_limit=SWIGLU_LIMIT,
                    shared_swiglu_limit=SHARED_SWIGLU_LIMIT,
                    quant_block_k=qbk_arg,
                    w1_scale=w1_scale_s,
                    w2_scale=w2_scale_s,
                    w3_scale=w3_scale_s,
                    w1_shared=w1_shared,
                    w2_shared=w2_shared,
                    w3_shared=w3_shared,
                    w1_shared_scale=w1_shared_scale,
                    w2_shared_scale=w2_shared_scale,
                    w3_shared_scale=w3_shared_scale,
                    direct_scaled_dot=direct_scaled_dot,
                    enable_act_quant=enable_act_quant,
                    cross_expert_prefetch_mode=xprefetch_mode,
                    tail_aware_btc=tail_aware_btc,
                    fixed_gather_banks=fixed_gather_banks,
                    enable_bt_scatter_overlap=enable_bt_scatter_overlap,
                    a2a_mode=a2a_mode,
                )
                ref_kwargs = {
                    "swiglu_limit": SWIGLU_LIMIT,
                    "shared_swiglu_limit": SHARED_SWIGLU_LIMIT,
                }
                if use_fp8:
                    ref_kwargs["quant_block_k"] = quant_block_k
                    ref_kwargs["w1_scale"] = jax.device_get(w1_scale_s)
                    ref_kwargs["w2_scale"] = jax.device_get(w2_scale_s)
                    ref_kwargs["w3_scale"] = jax.device_get(w3_scale_s)
                if use_shared_expert:
                    ref_kwargs["w1_shared"] = jax.device_get(w1_shared)
                    ref_kwargs["w2_shared"] = jax.device_get(w2_shared)
                    ref_kwargs["w3_shared"] = jax.device_get(w3_shared)
                    if w1_shared_scale is not None:
                        ref_kwargs["w1_shared_scale"] = jax.device_get(w1_shared_scale)
                        ref_kwargs["w2_shared_scale"] = jax.device_get(w2_shared_scale)
                        ref_kwargs["w3_shared_scale"] = jax.device_get(w3_shared_scale)
                ref = ref_moe(
                    jax.device_get(tokens_c),
                    jax.device_get(w1),
                    jax.device_get(w2),
                    jax.device_get(w3),
                    jax.device_get(twts),
                    jax.device_get(tidx),
                    top_k,
                    **ref_kwargs,
                )
                result_f32 = jax.device_get(
                    jax.device_put(result, jax.sharding.NamedSharding(mesh, P()))
                ).astype(np.float32)
                ref_f32 = np.asarray(ref).astype(np.float32)
                max_err = float(np.max(np.abs(result_f32 - ref_f32)))
                rel_err = float(max_err / (np.max(np.abs(ref_f32)) + 1e-6))
                correctness_status = "passed" if rel_err <= bench_tol else "failed"
                error_type = None
                error_message = None
                log(
                    f"  {correctness_status.upper()}: max_abs_err={max_err:.4f}, "
                    f"rel_err={rel_err:.6f}, tolerance={bench_tol:.6f}"
                )
            except Exception as e:
                max_err = rel_err = None
                correctness_status = "failed"
                error_type = type(e).__name__
                error_message = str(e)
                log(f"  FAIL correctness execution: {e}")

            correctness_failed |= correctness_status == "failed"
            correctness_records.append(
                {
                    **_common_evidence_fields(nt0),
                    "block_config": best_tag,
                    **runs_by_config[(nt0, best_tag)],
                    "execution_status": "correctness_only",
                    "correctness_status": correctness_status,
                    "correctness_tolerance": bench_tol,
                    "max_abs_error": max_err,
                    "max_relative_error": rel_err,
                    "error_type": error_type,
                    "error_message": error_message,
                    "samples_ms": [],
                }
            )

        if metrics_path is not None and jax.process_index() == 0:
            with metrics_path.open("a", encoding="utf-8") as fh:
                for record in correctness_records:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
            log(f"appended {len(correctness_records)} correctness records to {metrics_path}")
        if correctness_failed:
            raise RuntimeError("At least one winning configuration failed correctness.")

log("done")
