"""CPU-only tests for the grouped fused MoE v2 correctness oracle."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.fused_moe.v2.kernel import activation_fn, ref_moe


def _tokenwise_reference(tokens, w1, w2, w3, weights, ids, top_k, act_fn):
    tokens_f32 = jnp.asarray(tokens, dtype=jnp.float32)
    output = jnp.zeros_like(tokens_f32)
    for token_id in range(tokens.shape[0]):
        for slot_id in range(top_k):
            expert_id = int(ids[token_id, slot_id])
            if not 0 <= expert_id < w1.shape[0]:
                continue
            x = tokens_f32[token_id : token_id + 1]
            gate = x @ jnp.asarray(w1[expert_id], dtype=jnp.float32)
            up = x @ jnp.asarray(w3[expert_id], dtype=jnp.float32)
            hidden = activation_fn(gate, up, act_fn)
            expert_output = hidden @ jnp.asarray(w2[expert_id], dtype=jnp.float32)
            output = output.at[token_id].add(expert_output[0] * float(weights[token_id, slot_id]))
    return output.astype(tokens.dtype)


@pytest.mark.parametrize("act_fn", ["silu", "gelu", "swigluoai"])
def test_grouped_reference_matches_tokenwise_routes(act_fn):
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(5, 4)).astype(np.float32)
    w1 = rng.normal(size=(3, 4, 3)).astype(np.float32)
    w2 = rng.normal(size=(3, 3, 4)).astype(np.float32)
    w3 = rng.normal(size=(3, 4, 3)).astype(np.float32)
    # Includes duplicate expert 1 for token 0 and an invalid padded route.
    ids = np.array(
        [[1, 1, -1], [0, 2, 1], [2, 0, 1], [1, 2, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    weights = rng.uniform(size=ids.shape).astype(np.float32)

    expected = _tokenwise_reference(tokens, w1, w2, w3, weights, ids, top_k=3, act_fn=act_fn)
    actual = ref_moe(tokens, w1, w2, w3, weights, ids, top_k=3, act_fn=act_fn)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(actual), dtype=np.float32),
        np.asarray(jax.device_get(expected), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
