"""tools/make_profile.py - the expert-cache profile (`data/expert-profile.bin`): every (layer, expert) pair, ranked.

`--expert-cache auto` fills the VRAM slots it can afford with the profile's pairs in order, so the ranking decides
which experts start resident (the adaptive tier then swaps in what a conversation routes most).  A profile that
ranks fewer pairs than a card can hold caps the cache (issue #46: a 32 GB card stopped at 8,000 slots); this tool
writes one that ranks all 24,576.

The order: the base profile's ranking (default: the shipped data/expert-profile.bin), then the pairs your routing
traces used, most frequent first, then every pair still missing, interleaved across the layers.

    python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--out PATH]
                                 [--n-expert 256]      (a pruned model: GSQ-RCO Coder keeps 256 of 512)

A routing trace comes from a one-shot engine run with `--dump-routing FILE` (a prompt typical of your use; the
routed experts of every layer and position are written).  Point the model config's `--expert-profile` at the result.
"""
import argparse
import math
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512
MAX_ROUTES = 64
MAGIC, VERSION = b"STRP", 1


def read_profile(path, n_expert=N_EXPERT):
    blob = Path(path).read_bytes()
    if len(blob) < 24 or blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (nl, ne) != (N_LAYER, n_expert):
        raise SystemExit(f"{path}: {nl}x{ne}, not {N_LAYER}x{n_expert}")
    expected = 24 + 4 * n + N_LAYER * n_expert * 4
    if slots > N_LAYER * n_expert or n > slots or len(blob) != expected:
        raise SystemExit(f"{path}: profile size or slot count is inconsistent")
    ranked = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]
    if (len(set(ranked)) != n or
            any(layer >= N_LAYER or expert >= n_expert for layer, expert in ranked)):
        raise SystemExit(f"{path}: profile contains duplicate or out-of-range pairs")
    # The engine's STRP readers ignore this fixed-size table. It is retained
    # for compatibility but is not part of the resident ranking contract.
    return ranked


def iter_trace_records(path, n_expert=N_EXPERT):
    """Yield (layer, expert_ids) from complete --dump-routing records."""
    offset = 0
    with Path(path).open("rb") as trace:
        while True:
            header = trace.read(8)
            if not header:
                break
            if len(header) != 8:
                raise SystemExit(f"{path}: truncated routing record header at byte {offset}")
            layer, k = struct.unpack("<ii", header)
            if not 0 <= layer < N_LAYER:
                raise SystemExit(f"{path}: layer {layer} outside 0..{N_LAYER - 1} at byte {offset}")
            if not 1 <= k <= min(n_expert, MAX_ROUTES):
                raise SystemExit(
                    f"{path}: invalid route count {k}; expected 1..{min(n_expert, MAX_ROUTES)} "
                    f"at byte {offset}"
                )
            payload = trace.read(8 * k)
            if len(payload) != 8 * k:
                raise SystemExit(f"{path}: truncated routing ids/weights at byte {offset + 8}")
            experts = struct.unpack_from(f"<{k}i", payload)
            if any(not 0 <= expert < n_expert for expert in experts):
                raise SystemExit(
                    f"{path}: expert id outside 0..{n_expert - 1} at byte {offset + 8}"
                )
            if len(set(experts)) != k:
                raise SystemExit(f"{path}: duplicate expert id in routing record at byte {offset}")
            weights = struct.unpack_from(f"<{k}f", payload, 4 * k)
            if any(not math.isfinite(weight) or weight < 0 for weight in weights):
                raise SystemExit(
                    f"{path}: routing weights must be finite and non-negative at byte {offset + 8 + 4 * k}"
                )
            yield layer, experts
            offset += 8 + 8 * k                       # ids and weights


def read_trace(path, n_expert=N_EXPERT):
    """Count routed (layer, expert) pairs in a --dump-routing file."""
    freq = defaultdict(int)
    for layer, experts in iter_trace_records(path, n_expert):
        for expert in experts:
            freq[(layer, expert)] += 1
    return freq


def write_profile(path, ranked, n_expert=N_EXPERT):
    table = [[-1] * n_expert for _ in range(N_LAYER)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, N_LAYER, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(N_LAYER):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="routing traces from --dump-routing")
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"), help="ranking to keep first")
    ap.add_argument("--no-base", action="store_true", help="rank by the traces only")
    ap.add_argument("--out", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer (default 512)")
    a = ap.parse_args()

    ranked, seen = [], set()

    def take(pairs):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)

    ne = a.n_expert
    if not a.no_base:
        take(read_profile(a.base, ne))
    n_base = len(ranked)
    freq = defaultdict(int)
    for t in a.traces:
        for p, c in read_trace(t, ne).items():
            freq[p] += c
    take(p for p, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))
    n_trace = len(ranked) - n_base
    take((layer, e) for e in range(ne) for layer in range(N_LAYER))   # the rest, across the layers
    write_profile(a.out, ranked, ne)
    assert read_profile(a.out, ne) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {len(ranked)} ranked pairs ({n_base} from the base, {n_trace} from the traces, "
          f"{len(ranked) - n_base - n_trace} filled in)")


if __name__ == "__main__":
    main()
