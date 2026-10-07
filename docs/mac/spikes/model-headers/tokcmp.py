#!/usr/bin/env python3
"""Compare the tokenizer of several GGUF header prefixes: hashes of tokens, merges, types and the chat template."""
import hashlib, sys
sys.argv, files = sys.argv[:2], sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("tp", "tokpeek.py")
for f in files:
    import struct
    b = open(f, "rb").read(); o = [4]
    def u(fm):
        v = struct.unpack_from("<" + fm, b, o[0]); o[0] += struct.calcsize(fm); return v[0]
    def s():
        n = u("Q"); v = b[o[0]:o[0] + n].decode("utf-8", "replace"); o[0] += n; return v
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
    u("I"); u("Q"); nk = u("Q"); kv = {}
    for _ in range(nk):
        k = s(); t = u("I"); kv[k] = val(t)
    h = lambda x: hashlib.sha256(repr(x).encode()).hexdigest()[:12]
    print(f, "tokens", h(kv["tokenizer.ggml.tokens"]), "merges", h(kv["tokenizer.ggml.merges"]),
          "types", h(kv["tokenizer.ggml.token_type"]), "template", h(kv.get("tokenizer.chat_template")),
          len(kv.get("tokenizer.chat_template", "")), "add_bos", kv.get("tokenizer.ggml.add_bos_token"),
          "eos", kv.get("tokenizer.ggml.eos_token_id"), "bos", kv.get("tokenizer.ggml.bos_token_id"))
