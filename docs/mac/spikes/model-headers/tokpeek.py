#!/usr/bin/env python3
"""Print the tokenizer keys of a GGUF header prefix: special ids and their strings, and the added (control) tokens."""
import struct, sys
b = open(sys.argv[1], "rb").read()
o = 4
def u(f):
    global o
    v = struct.unpack_from("<" + f, b, o); o += struct.calcsize(f); return v[0]
def s():
    global o
    n = u("Q"); v = b[o:o + n].decode("utf-8", "replace"); o += n; return v
def val(t):
    if t in (0, 1): return u("B" if t == 0 else "b")
    if t in (2, 3): return u("H" if t == 2 else "h")
    if t in (4, 5): return u("I" if t == 4 else "i")
    if t == 6: return u("f")
    if t == 7: return bool(u("B"))
    if t == 8: return s()
    if t == 9:
        et = u("I"); n = u("Q"); return [val(et) for _ in range(n)]
    if t in (10, 11): return u("Q" if t == 10 else "q")
    if t == 12: return u("d")
    raise ValueError(t)
ver = u("I"); nt = u("Q"); nk = u("Q")
kv = {}
for _ in range(nk):
    k = s(); t = u("I"); kv[k] = val(t)
toks = kv["tokenizer.ggml.tokens"]; tt = kv.get("tokenizer.ggml.token_type", [])
print("tokens", len(toks), "merges", len(kv.get("tokenizer.ggml.merges", [])), "types", len(tt))
for k, v in kv.items():
    if k.startswith("tokenizer.") and not isinstance(v, list):
        print(k, repr(v)[:120] if k != "tokenizer.chat_template" else f"<{len(v)} chars>")
from collections import Counter
print("token_type counts", Counter(tt))
for i, (t, ty) in enumerate(zip(toks, tt)):
    if ty != 1:
        print(i, ty, repr(t))
