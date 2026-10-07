#!/usr/bin/env python3
"""Inventory of the MTP block in the BF16 checkpoint: every `mtp.*` tensor, its shard and its absolute byte range.
Reads model.safetensors.index.json and each needed shard's safetensors header by HTTP range requests.
Usage: mtpinv.py > mtp-inventory.json"""
import collections, json, struct, sys, urllib.request

REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/"

def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % rng} if rng else {})
    with urllib.request.urlopen(req) as r:
        if rng and r.status != 206:
            sys.exit("%s: expected 206, got %d" % (url, r.status))
        return r.read()

index = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
by_shard = collections.defaultdict(list)
for name, shard in index.items():
    if name.startswith("mtp."):
        by_shard[shard].append(name)
tensors = []
for shard in sorted(by_shard):
    n = struct.unpack("<Q", get(REPO + shard, (0, 7)))[0]
    hdr = json.loads(get(REPO + shard, (8, 8 + n - 1)))
    for name in sorted(by_shard[shard]):
        t = hdr[name]
        a, b = t["data_offsets"]
        tensors.append({"name": name, "shard": shard, "dtype": t["dtype"], "shape": t["shape"],
                        "start": 8 + n + a, "end": 8 + n + b - 1, "bytes": b - a})
tensors.sort(key=lambda t: t["name"])
json.dump({"repo": REPO, "total_bytes": sum(t["bytes"] for t in tensors), "tensors": tensors}, sys.stdout, indent=1)
