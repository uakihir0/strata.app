# Model file headers

Scripts and data behind the numbers in [03-model-reference.md](../../03-model-reference.md), sections 2-4 and 9.3.
Results are summarised in [appendix-a-measurements.md](../../appendix-a-measurements.md), section A.9. Nothing here
needs a model download: every byte came from HTTP range requests against the pinned Hugging Face revisions, and the
sizes came from the Hugging Face API. Run on the reference Mac with the system `python3` (3.9 or newer, standard
library only) and `curl`.

| Path | What it is |
| --- | --- |
| `ggufhdr.py` | Parses a GGUF header, or a prefix of a file that contains the header, and prints the metadata and the tensor table as JSON. Arrays longer than 64 entries (the tokenizer) are replaced by their length. |
| `headers/*.json` | `ggufhdr.py` output for the 7 shard 1 files of the Mac catalogue, the ISTA shard 2 (shared by the ISTA files and Coder) and both Swift shards 2 |
| `summary.py`, `summary.txt` | Per-file type map per tensor name, file totals and per-pass byte groups (doc 03 sections 3.8, 4.1 and 4.2) for the five ISTA and Coder files |
| `swiftsum.py`, `swift-summary.txt` | Which shard holds what in the Swift files, and the same byte groups over both shards (doc 03 section 3.9) |
| `perlayer.py` | Per-layer expert format combinations (doc 03 section 3.8) |
| `dpack.py` | The size of the `dense.bin` that `iq_pack.py` writes for a file (doc 03 section 4.5) |
| `tokpeek.py`, `tokcmp.py` | Special tokens of one file; hashes of tokens, merges, types and chat template across files (doc 03 section 2.6). They read the raw header prefix, not the JSON. |
| `mtpinv.py`, `mtp-inventory.json`, `mtp-inventory.md` | The 31 `mtp.*` tensors of the BF16 checkpoint, with their shards and absolute byte ranges (doc 03 sections 2.3 and 4.4) |
| `plecmp.sh` | Compares five PLE table rows of the ISTA shard 2 with the Swift IQ2_XS shard 1, and decodes three of them (doc 03 section 9.3) |

## Fetching a header

The URL of a file is `https://huggingface.co/<repo>/resolve/<revision>/<path>`, with the repositories, revisions and
paths of doc 03 section 2.2. Every shard 1 header ends before byte 11.1 M, so the first 16 MiB are enough. The ISTA
shard 2 header is 192 bytes and a Swift shard 2 header under 64 KiB.

```bash
R=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09
curl -sSL -r 0-16777215 -o q2_0-s1.head "$R/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
curl -sSL -r 0-1048575  -o q2_0-s2.head "$R/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
python3 ggufhdr.py q2_0-s1.head > q2_0-s1.json

C=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/5348543e0147355ac9cbcb031184a3546350988e
curl -sSL -r 0-16777215 -o coder-s1.head "$C/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"

S=https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/b22d729eae29b5796f76fb70f91aef549b9fc52c
for i in 1 2; do
  curl -sSL -r 0-16777215 -o swift-IQ2_XS-s$i.head "$S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000$i-of-00002.gguf"
done
```

The other ISTA quants (`IQ2_XS`, `IQ3_XXS`, `IQ3_S`) and Swift `IQ3_XXS` follow the same pattern. The headers are not
stored here because they are 16 MiB each; the JSON files hold everything the documents use except the tokenizer.

Shard sizes (doc 03 section 2.4) come from the API, one call per repository:

```bash
curl -s "https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09?recursive=true" \
  | python3 -c 'import json,sys; [print(e["size"], e["path"]) for e in json.load(sys.stdin) if e["path"].endswith(".gguf")]'
```

## Reproducing the summaries

```bash
cd headers
python3 ../summary.py q2_0-s1.json IQ2_XS-s1.json IQ3_XXS-s1.json IQ3_S-s1.json coder-s1.json   # = summary.txt
python3 ../swiftsum.py swift-IQ2_XS-s1.json+swift-IQ2_XS-s2.json swift-IQ3_XXS-s1.json+swift-IQ3_XXS-s2.json
python3 ../perlayer.py q2_0-s1.json IQ2_XS-s1.json swift-IQ3_XXS-s1.json+swift-IQ3_XXS-s2.json
python3 ../dpack.py q2_0-s1.json
```

`summary.py` output equals `summary.txt` once the `headers/` prefix is removed from the file names (checked
2026-10-03). `swiftsum.py` writes a merged table to the system temporary folder and summarises it.

## Single tensors

Two tensors were read whole to check claims in doc 03 section 3.7. The absolute offset of a tensor is
`data_start + offset` from the JSON.

```bash
# output_hc_norm.weight of the Q2_0 file: F32 [10240] at 11,024,672 + 443,596,800
curl -sSL -r 454621472-454662431 -o ohn.bin "$R/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
# blk.0.ffn_gate_inp.weight and blk.0.ffn_gate_inp_shexp.weight of Swift IQ2_XS, both F32
curl -sSL -r 30025918208-30031171327 -o swift-r0.bin "$S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
python3 - <<'EOF'
import array
n = array.array('f'); n.frombytes(open('ohn.bin', 'rb').read())
print('output_hc_norm min %.4f max %.4f mean %.4f' % (min(n), max(n), sum(n) / len(n)))
r = array.array('I'); r.frombytes(open('swift-r0.bin', 'rb').read())
print('Swift router words with nonzero low 16 bits:', sum(1 for x in r if x & 0xffff), 'of', len(r))
EOF
```

Results: the norm ranges from 0.6328 to 13.375 with mean 3.75, so the file stores the final multiplier, not `w`
(doc 03 section 3.7). All 1,313,280 words of the two Swift router tensors have zero low halves: they are BF16 values
stored as F32, so narrowing them to BF16 is exact.

## MTP inventory

```bash
python3 mtpinv.py > mtp-inventory.json
```

It reads `model.safetensors.index.json` of `Qwen/Qwen3.8-Flash-Next` at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, then the 8-byte length and the JSON header of each of the 28 shards that
hold an `mtp.*` tensor, and refuses any range response that is not `206`. `start` and `end` are absolute, inclusive
byte offsets in the shard, ready for a `Range` header. A rerun on 2026-10-03 gave the stored file exactly: 31
tensors, 5,214,301,696 bytes.

## PLE rows

```bash
sh plecmp.sh
```

Row `r` of `per_layer_token_embd.weight` is at `data_start + offset + 90 r`: 192 in the ISTA shard 2 and
10,967,808 + 350,863,360 in the Swift IQ2_XS shard 1. The five rows are byte-identical, and the decoded rows 0, 12345
and 320,001,535 reproduce `kProbeHead`, `kProbeTail` and `kProbeSum` of upstream `src/kernels/ple_oracle_vectors.inc`
(row 320,001,535 is all zeros). The decode uses the split-halves IQ4_NL rule of doc 03 section 6.8.
