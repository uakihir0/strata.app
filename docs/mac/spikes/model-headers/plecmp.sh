#!/bin/sh
# Compare PLE table rows of the ISTA shard 2 and the Swift IQ2_XS shard 1, and decode them.
# Row r is at data_start + tensor_offset + r * 90 (IQ4_NL, 160 elements = 5 blocks of 18 B).
set -e
I=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
S=https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/b22d729eae29b5796f76fb70f91aef549b9fc52c/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
for r in 0 12345 1000000 200000000 320001535; do
  a=$((192 + r * 90)); b=$((10967808 + 350863360 + r * 90))
  curl -sSL -r $a-$((a + 89)) -o ista-$r.bin "$I"
  curl -sSL -r $b-$((b + 89)) -o swift-$r.bin "$S"
  cmp -s ista-$r.bin swift-$r.bin && echo "row $r identical" || echo "row $r DIFFERS"
done
python3 - <<'PY'
import struct
kv = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
def dec(b):
    out = []
    for k in range(5):
        blk = b[k * 18:(k + 1) * 18]
        d = struct.unpack('<e', blk[:2])[0]
        qs = blk[2:]
        y = [0.0] * 32
        for j in range(16):
            y[j] = d * kv[qs[j] & 15]
            y[j + 16] = d * kv[qs[j] >> 4]
        out += y
    return out
for r in (0, 12345, 320001535):
    v = dec(open(f'swift-{r}.bin', 'rb').read())
    print(r, ['%.9e' % x for x in v[:8]], ['%.9e' % x for x in v[-8:]], 'L1 %.9e' % sum(abs(x) for x in v))
PY
