"""Shared tensor-form rules for standalone native packs."""
from __future__ import annotations

import re

NOT_IN_PACK = frozenset({"per_layer_token_embd.weight"})
NATIVE_PLE_KEY_NAME = "blk.1.ple_key.weight"
NATIVE_PLE_KEY_TYPES = frozenset({"Q2_0", "Q8_0"})

# The float form read by the engine for each projection. Keep the packer and its
# verifier on this one table so conversions.json cannot redefine pack semantics.
FORM = {
    "ffn_gate_inp.weight": "BF16", "ffn_gate_inp_shexp.weight": "BF16",
    "hc_attn_down.weight": "BF16", "hc_attn_up.weight": "BF16", "hc_attn_inject.weight": "BF16",
    "hc_ffn_down.weight": "BF16", "hc_ffn_up.weight": "BF16", "hc_ffn_inject.weight": "BF16",
    "output_hc_down.weight": "BF16", "output_hc_up.weight": "BF16",
    "indexer.k_proj.weight": "BF16", "indexer.q_proj.weight": "BF16",
    "ssm_alpha.weight": "BF16", "ssm_beta.weight": "BF16",
    "ple_key.weight": "BF16", "ple_value.weight": "BF16", "ple_conv1d.weight": "F16",
    "attn_q_norm.weight": "F32", "attn_k_norm.weight": "F32", "hc_attn_norm.weight": "F32",
    "hc_ffn_norm.weight": "F32", "output_hc_norm.weight": "F32", "indexer.q_norm.weight": "F32",
    "indexer.k_norm.weight": "F32", "ple_norm_conv.weight": "F32", "ple_norm_key.weight": "F32",
    "ple_norm_query.weight": "F32", "ssm_a": "F32", "ssm_conv1d.weight": "F32", "ssm_dt.bias": "F32",
    "ssm_norm.weight": "F32",
}


def form_of(name: str) -> str | None:
    """Return the engine's expected float form for a tensor name, when specified."""
    return FORM.get(re.sub(r"^blk\.\d+\.", "", name))


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith("_exps.weight")
