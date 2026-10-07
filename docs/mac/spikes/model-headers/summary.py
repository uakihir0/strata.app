#!/usr/bin/env python3
"""Per-file format map and per-pass byte counts from parsed GGUF headers (ggufhdr.py output)."""
import collections, json, re, sys

GDN = [l for l in range(48) if l % 4 != 3]
QSA = [l for l in range(48) if l % 4 == 3]

# per-pass components: (label, tensor suffixes)
COMP = [
    ("GDN projections (attn_qkv, attn_gate, ssm_out)", ["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"]),
    ("GDN small (ssm_alpha, ssm_beta, ssm_conv1d, ssm_a, ssm_dt.bias, ssm_norm)",
     ["ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight"]),
    ("QSA projections (attn_q, attn_k, attn_v, attn_output)",
     ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]),
    ("QSA indexer (k_proj, q_proj) and norms",
     ["indexer.k_proj.weight", "indexer.q_proj.weight", "indexer.k_norm.weight", "indexer.q_norm.weight",
      "attn_q_norm.weight", "attn_k_norm.weight"]),
    ("GR mixers (hc_attn_*, hc_ffn_*)",
     ["hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight", "hc_attn_norm.weight",
      "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight", "hc_ffn_norm.weight"]),
    ("Routers (ffn_gate_inp, ffn_gate_inp_shexp)", ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"]),
    ("Shared experts (ffn_{gate,up,down}_shexp)", ["ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"]),
    ("PLE block, layer 1 (ple_key, ple_value, norms, conv)",
     ["ple_key.weight", "ple_value.weight", "ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight",
      "ple_conv1d.weight"]),
    ("Head (output.weight, output_hc_*)", ["@output.weight", "@output_hc_down.weight", "@output_hc_up.weight",
                                          "@output_hc_norm.weight"]),
]


def main(paths):
    for p in paths:
        d = json.load(open(p))
        ts = d["tensors"]
        by = {t["name"]: t for t in ts}
        print("=" * 30, p, "tensors", len(ts), "data_start", d["data_start"], "total", sum(t["bytes"] for t in ts))
        print("experts", d["kv"].get("qwen4exp.expert_count"), d["kv"].get("general.size_label"))
        per = collections.defaultdict(list)
        for t in ts:
            m = re.match(r"blk\.(\d+)\.(.*)", t["name"])
            if m:
                per[m.group(2)].append((int(m.group(1)), t))
        for k in sorted(per):
            v = per[k]
            print(f"  {k:28s} n={len(v):2d} ne={v[0][1]['ne']} {dict(collections.Counter(t['type'] for _, t in v))}")
        for t in ts:
            if not t["name"].startswith("blk."):
                print(f"  G {t['name']:26s} ne={t['ne']} {t['type']} {t['bytes']}")
        # experts per layer
        ex = 0
        lay = []
        for l in range(48):
            g, u, dn = (by.get(f"blk.{l}.ffn_{r}_exps.weight") for r in ("gate", "up", "down"))
            if g is None:
                lay.append(None)
                continue
            ne = g["ne"][2]
            blob = (g["bytes"] + u["bytes"] + dn["bytes"]) // ne
            ex += g["bytes"] + u["bytes"] + dn["bytes"]
            lay.append((g["type"], u["type"], dn["type"], blob))
        print("  routed experts total", ex, "per-layer combos", dict(collections.Counter((a, b, c) for a, b, c, _ in filter(None, lay))))
        blobs = [x[3] for x in lay if x]
        print("  blob min/mean/max", min(blobs), sum(blobs) / len(blobs), max(blobs), "per token top-10:", 10 * sum(blobs))
        tot = 0
        for label, sfx in COMP:
            s = 0
            for x in sfx:
                if x.startswith("@"):
                    s += by[x[1:]]["bytes"]
                else:
                    s += sum(t["bytes"] for _, t in per.get(x, []))
            tot += s
            print(f"  pass {label:70s} {s:>14,d}")
        emb = by["token_embd.weight"]
        print(f"  pass total (dense, excluding token_embd rows) {tot:,d}; token_embd {emb['type']} {emb['bytes']:,d}; row {emb['bytes'] // emb['ne'][1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
