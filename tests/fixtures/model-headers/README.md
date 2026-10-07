# Model header fixtures

These JSON files are the compact GGUF header snapshots from
`docs/mac/spikes/model-headers/headers/`. They contain tensor tables and
non-tokenizer metadata, not model weights.

Each supported model has 1,224 tensors across its shards. The five ISTA/Coder
shard 1 snapshots contain 1,223 tensors each and share the one-tensor
`q2_0-s2.json` snapshot. The Swift pairs contain 342 + 882 and 317 + 907
tensors.

From the repository root, run `python -m unittest tools.metal.test_model_headers`
to verify shard sizes and tensor byte totals against docs/mac/03-model-reference.md,
the required shard metadata and shared PLE tensor, reject duplicate JSON keys,
exercise `ggufhdr.py` with a generated GGUF header, and reproduce the stored
ISTA and Swift summaries.
