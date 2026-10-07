#!/usr/bin/env python3
"""D_pack: the bytes iq_pack.py writes to dense.bin for one model (float tensors in engine form, ALIGN 64).
Usage: dpack.py shard1.json [shard2.json]"""
import json, re, sys
FLOAT = {"F32": 4, "F16": 2, "BF16": 2}
BF16 = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "hc_attn_down.weight", "hc_attn_up.weight",
        "hc_attn_inject.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
        "output_hc_down.weight", "output_hc_up.weight", "indexer.k_proj.weight", "indexer.q_proj.weight",
        "ssm_alpha.weight", "ssm_beta.weight", "ple_key.weight", "ple_value.weight"}
ALIGN = 64
n = tot = conv = conv_src = 0
for p in sys.argv[1:]:
    for t in json.load(open(p))["tensors"]:
        name = t["name"]
        if "_exps" in name or name == "per_layer_token_embd.weight" or t["type"] not in FLOAT:
            continue
        short = re.sub(r"^blk\.\d+\.", "", name)
        b = t["bytes"]
        if short in BF16 and t["type"] == "F32":
            conv += 1; conv_src += b; b //= 2
        n += 1; tot += b + (-b) % ALIGN
print(f"{n} float tensors, dense.bin {tot:,d} B = {tot/2**30:.3f} GiB; converted F32->BF16: {conv} ({conv_src:,d} B in the GGUF)")
