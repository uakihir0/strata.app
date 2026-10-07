import json, sys, collections
AB = {"Q2_0":"q2","IQ1_M":"1m","IQ2_XXS":"2xxs","IQ2_XS":"2xs","IQ2_S":"2s","IQ3_XXS":"3xxs","IQ3_S":"3s","IQ4_XS":"4xs","IQ4_NL":"4nl"}
for p in sys.argv[1:]:
    files = p.split("+")
    by = {}
    shard_of = {}
    for i, f in enumerate(files):
        d = json.load(open(f))
        for t in d["tensors"]:
            by[t["name"]] = t; shard_of[t["name"]] = i + 1
    out = []
    for l in range(48):
        g = by[f"blk.{l}.ffn_gate_exps.weight"]; dn = by[f"blk.{l}.ffn_down_exps.weight"]; u = by[f"blk.{l}.ffn_up_exps.weight"]
        assert g["type"] == u["type"]
        out.append(f"{l}:{g['type']}/{dn['type']}" + (f"@s{shard_of[g['name']]}" if len(files) > 1 else ""))
    print(p); print("  " + " ".join(out))
    # group by combination
    comb = collections.defaultdict(list)
    for l in range(48):
        g = by[f"blk.{l}.ffn_gate_exps.weight"]; dn = by[f"blk.{l}.ffn_down_exps.weight"]
        comb[(g["type"], dn["type"])].append(l)
    for k, v in sorted(comb.items(), key=lambda kv: kv[1][0]):
        print(f"  {k[0]}+{k[1]}: {len(v)} layers {v}")
