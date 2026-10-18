#!/usr/bin/env bash
# Run the private fused-MoE rank-A2A correctness/performance matrix on one
# single-host TPU slice.  Each mode runs in a fresh Python process so compile
# time and executable state cannot leak between variants.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
bench_script="${repo_root}/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py"
phase="${1:-smoke}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
evidence_root="${EVIDENCE_ROOT:-/tmp/fused_moe_rank_a2a/${run_id}}"
read -r -a bench_modes <<< "${BENCH_MODES:-expert rank_direct rank_broadcast rank_dispatch rank_dispatch_combine}"

# An isolated Git worktree may intentionally reuse the base checkout's virtual
# environment.  Put this worktree first so imports cannot silently resolve to
# the editable-install source path from that base checkout.
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not found: ${python_bin}" >&2
  echo "Set PYTHON_BIN to the TPU checkout's virtual-environment Python." >&2
  exit 2
fi

if [[ ! -f "${bench_script}" ]]; then
  echo "Benchmark not found: ${bench_script}" >&2
  exit 2
fi

case "${phase}" in
  smoke|repeat|stress|all) ;;
  *)
    echo "Usage: $0 [smoke|repeat|stress|all]" >&2
    exit 2
    ;;
esac

if [[ ${#bench_modes[@]} -eq 0 ]]; then
  echo "BENCH_MODES must name at least one communication mode." >&2
  exit 2
fi

for mode in "${bench_modes[@]}"; do
  case "${mode}" in
    expert|rank_direct|rank_broadcast|rank_dispatch|rank_dispatch_combine) ;;
    *)
      echo "Unsupported BENCH_MODES entry: ${mode}" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${evidence_root}"

echo "run_id=${run_id}"
echo "phase=${phase}"
echo "repo_root=${repo_root}"
echo "python_bin=${python_bin}"
echo "pythonpath=${PYTHONPATH}"
echo "evidence_root=${evidence_root}"
echo "bench_modes=${bench_modes[*]}"
git -C "${repo_root}" status --short --branch
git -C "${repo_root}" rev-parse HEAD

"${python_bin}" - <<'PY'
import jax

devices = jax.devices()
print(f"jax={jax.__version__} backend={jax.default_backend()} devices={len(devices)}")
for device in devices:
    print(
        f"device id={device.id} kind={device.device_kind} "
        f"platform={device.platform} process={device.process_index}"
    )
if jax.default_backend() != "tpu" or len(devices) != 4:
    raise SystemExit("This runner requires one v6e-4 host with four TPU devices.")
PY

run_case() {
  local label="$1"
  local mode="$2"
  local routing="$3"
  local tokens="$4"
  local warmup="$5"
  local iterations="$6"
  local check="$7"
  local jsonl="${evidence_root}/${label}.jsonl"
  local log_file="${evidence_root}/${label}.log"

  echo "START ${label}: mode=${mode} routing=${routing} tokens=${tokens}"
  env \
    BENCH_SINGLE_HOST=1 \
    BENCH_D="${BENCH_D:-8192}" \
    BENCH_F="${BENCH_F:-1024}" \
    BENCH_E="${BENCH_E:-32}" \
    BENCH_TOPK="${BENCH_TOPK:-8}" \
    BENCH_TOKENS="${tokens}" \
    BENCH_BT="${BENCH_BT:-32}" \
    BENCH_BF="${BENCH_BF:-256}" \
    BENCH_BTC="${BENCH_BTC:-32}" \
    BENCH_BTS="${BENCH_BTS:-32}" \
    BENCH_A2A_MODE="${mode}" \
    BENCH_ROUTING_MODE="${routing}" \
    BENCH_WARMUP="${warmup}" \
    BENCH_ITERS="${iterations}" \
    BENCH_CHECK="${check}" \
    BENCH_RUN_ID="${run_id}" \
    BENCH_VARIANT="${label}" \
    BENCH_JSONL="${jsonl}" \
    "${python_bin}" "${bench_script}" 2>&1 | tee "${log_file}"
  echo "PASS ${label}"
}

run_smoke() {
  local mode
  local routing="${SMOKE_ROUTING:-one_rank}"
  local tokens="${SMOKE_TOKENS:-128}"
  local tokens_tag="${tokens//,/_}"
  local check="${SMOKE_CHECK:-1}"
  for mode in "${bench_modes[@]}"; do
    run_case \
      "smoke_${routing}_t${tokens_tag}_${mode}" \
      "${mode}" "${routing}" "${tokens}" 1 5 "${check}"
  done
}

run_repeat() {
  local repetition mode index
  local routing="${REPEAT_ROUTING:-one_rank}"
  local tokens="${REPEAT_TOKENS:-128}"
  local tokens_tag="${tokens//,/_}"
  local -a repeat_modes
  for repetition in 1 2 3; do
    repeat_modes=("${bench_modes[@]}")
    if ((repetition % 2 == 0)); then
      repeat_modes=()
      for ((index = ${#bench_modes[@]} - 1; index >= 0; index--)); do
        repeat_modes+=("${bench_modes[index]}")
      done
    fi
    for mode in "${repeat_modes[@]}"; do
      run_case \
        "repeat${repetition}_${routing}_t${tokens_tag}_${mode}" \
        "${mode}" "${routing}" "${tokens}" 2 10 "${REPEAT_CHECK:-0}"
    done
  done
}

run_stress() {
  local mode
  for mode in "${bench_modes[@]}"; do
    run_case "stress_four_ranks_t512_${mode}" "${mode}" four_ranks 512 2 50 1
  done
}

case "${phase}" in
  smoke) run_smoke ;;
  repeat) run_repeat ;;
  stress) run_stress ;;
  all)
    run_smoke
    run_repeat
    run_stress
    ;;
esac

echo "COMPLETE phase=${phase} evidence_root=${evidence_root}"
