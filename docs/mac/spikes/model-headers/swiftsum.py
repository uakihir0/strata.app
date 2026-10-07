#!/usr/bin/env python3
"""Swift 1.5 files: which shard holds what, and the same byte groups as summary.py over both shards."""
import collections, json, os, re, sys, tempfile
import summary
for pair in sys.argv[1:]:
    a, b = pair.split("+")
    A, B = json.load(open(a)), json.load(open(b))
    print("=" * 20, pair)
    for i, d in enumerate((A, B), 1):
        ts = d["tensors"]
        print(f"  s{i}: tensors {len(ts)} data_start {d['data_start']} tensor bytes {sum(t['bytes'] for t in ts):,d} kv {len(d['kv'])}")
        nb = [t for t in ts if not t["name"].startswith("blk.")]
        print("     non-block:", [(t["name"], t["type"]) for t in nb])
        lay = collections.defaultdict(list)
        for t in ts:
            m = re.match(r"blk\.(\d+)\.(.*)", t["name"])
            if m: lay[int(m.group(1))].append(m.group(2))
        full = [l for l in sorted(lay) if len(lay[l]) >= 20 or (l % 4 != 3 and len(lay[l]) >= 20)]
        print("     layers present:", min(lay) if lay else None, "-", max(lay) if lay else None, "count", len(lay))
        for l in sorted(lay):
            ex = [x for x in lay[l] if "_exps" in x]
            if ex and len(ex) < 3: print(f"     layer {l} partial experts here: {ex}")
        exl = sorted({l for l in lay for x in lay[l] if "_exps" in x})
        print("     expert layers here:", exl[:3], "...", exl[-3:], len(exl))
        dl = sorted({l for l in lay for x in lay[l] if "_exps" not in x})
        print("     dense layers here:", (dl[:3], dl[-3:], len(dl)) if dl else None)
    # merged byte groups
    m = {"kv": A["kv"], "tensors": A["tensors"] + B["tensors"], "data_start": 0}
    path = os.path.join(tempfile.gettempdir(), "swift-merged.json")
    json.dump(m, open(path, "w"))
    summary.main([path])
