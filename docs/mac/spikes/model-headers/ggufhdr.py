#!/usr/bin/env python3
"""Parse the header of a GGUF file (or a prefix of one) and print its metadata and tensor table as JSON."""
import json, struct, sys

# ggml type id -> (name, block elements, block bytes)
T = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
     7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
     12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
     16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
     20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
     24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
     29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 42: ("Q2_0", 64, 18)}


class R:
    def __init__(self, b): self.b, self.o = b, 0
    def u(self, f):
        v = struct.unpack_from("<" + f, self.b, self.o); self.o += struct.calcsize(f); return v[0]
    def s(self):
        n = self.u("Q"); v = self.b[self.o:self.o + n].decode("utf-8", "replace"); self.o += n; return v
    def val(self, t):
        if t in (0, 1): return self.u("B" if t == 0 else "b")
        if t in (2, 3): return self.u("H" if t == 2 else "h")
        if t in (4, 5): return self.u("I" if t == 4 else "i")
        if t == 6: return self.u("f")
        if t == 7: return bool(self.u("B"))
        if t == 8: return self.s()
        if t == 9:
            et = self.u("I"); n = self.u("Q")
            if n > 64:   # skip long arrays (tokenizer); keep their length
                for _ in range(n): self.val(et)
                return {"array_len": n, "type": et}
            return [self.val(et) for _ in range(n)]
        if t in (10, 11): return self.u("Q" if t == 10 else "q")
        if t == 12: return self.u("d")
        raise ValueError(t)


def parse(path):
    b = open(path, "rb").read()
    r = R(b)
    assert r.b[:4] == b"GGUF"; r.o = 4
    ver = r.u("I"); nt = r.u("Q"); nk = r.u("Q")
    kv = {}
    for _ in range(nk):
        k = r.s(); t = r.u("I"); kv[k] = r.val(t)
    align = kv.get("general.alignment", 32)
    ts = []
    for _ in range(nt):
        name = r.s(); nd = r.u("I"); ne = [r.u("Q") for _ in range(nd)]; ty = r.u("I"); off = r.u("Q")
        tn, be, bb = T[ty]
        n = 1
        for x in ne: n *= x
        ts.append({"name": name, "ne": ne, "type": tn, "id": ty, "offset": off, "bytes": n // be * bb})
    data_start = (r.o + align - 1) // align * align
    return {"version": ver, "kv": kv, "tensors": ts, "header_end": r.o, "data_start": data_start}


if __name__ == "__main__":
    json.dump(parse(sys.argv[1]), sys.stdout, indent=1)
